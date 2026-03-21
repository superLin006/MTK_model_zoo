/* moonshine_streaming.h - Moonshine MTK NPU Streaming Inference
 *
 * Sliding-window streaming ASR:
 *   Audio chunks → Encoder NPU (160 frames in / 40 frames out)
 *   → Accumulate encoder output → Decoder when enough frames collected
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

#include "executor/NeuronExecutor.h"
#include "vad/silero_vad_wrapper.h"

namespace moonshine {

// ===================== Streaming Constants =====================

static constexpr int CHUNK_FRAMES       = 160;  // encoder input frames per chunk
static constexpr int CHUNK_T_ENC        = 40;   // encoder output frames per chunk
static constexpr int STEP_FRAMES        = 40;   // sliding step (no overlap)
static constexpr int MAX_ENC_FRAMES     = 500;  // max accumulated encoder frames
static constexpr int TRIGGER_ENC_FRAMES = 500;  // fallback: trigger decoder at this many frames (max window)

// VAD parameters
static constexpr int   VAD_WINDOW_MS      = 32;    // VAD analysis window (32ms = 512 samples)
static constexpr float VAD_THRESHOLD      = 0.5f;  // speech probability threshold
static constexpr int   VAD_SILENCE_MS     = 400;   // silence duration to trigger decoder (ms)
static constexpr int   VAD_SPEECH_PAD_MS  = 30;    // speech padding
static constexpr int   VAD_MIN_SPEECH_MS  = 200;   // minimum speech before silence triggers

// Shared constants (same as offline)
static constexpr int FRAME_LEN_S      = 80;    // samples per frame (5ms @ 16kHz)
static constexpr int VOCAB_SIZE_S     = 32768;
static constexpr int ENCODER_HIDDEN_S = 620;
static constexpr int DECODER_HIDDEN_S = 512;
static constexpr int NUM_LAYERS_S     = 10;
static constexpr int ROPE_DIM_S       = 32;
static constexpr int MAX_DEC_LEN_S    = 128;
static constexpr int MAX_NEW_TOKENS_S = 120;
static constexpr int BOS_TOKEN_S      = 1;
static constexpr int EOS_TOKEN_S      = 2;
static constexpr float EPS_S          = 1e-6f;

/**
 * MoonshineStreamingEngine
 *
 * Processes audio in small chunks (0.8s per chunk).
 * Accumulates encoder outputs and triggers decoder when
 * enough frames (TRIGGER_ENC_FRAMES) are collected.
 *
 * Usage:
 *   engine.Initialize(...);
 *   // For each chunk of audio:
 *   std::string partial = engine.ProcessChunk(samples, num_samples);
 *   // At end of audio:
 *   std::string final = engine.Flush();
 */
class MoonshineStreamingEngine {
public:
    MoonshineStreamingEngine() = default;
    ~MoonshineStreamingEngine();

    /**
     * Initialize the streaming engine.
     *
     * @param encoder_chunk_dla  Path to moonshine_encoder_chunk.dla [1,160,80] -> [1,40,620]
     * @param decoder_dla        Path to moonshine_decoder.dla (same as offline)
     * @param embed_tokens_path  Path to embed_tokens.npy [32768, 512]
     * @param proj_weight_path   Path to proj_weight.npy [512, 620]
     * @param log_k_path         Path to log_k.npy [1] float32
     * @param vocab_path         Path to vocab.txt (id<TAB>piece)
     * @return true on success
     */
    bool Initialize(const std::string& encoder_chunk_dla,
                    const std::string& decoder_dla,
                    const std::string& embed_tokens_path,
                    const std::string& proj_weight_path,
                    const std::string& log_k_path,
                    const std::string& vocab_path);

    /**
     * Process a chunk of audio samples.
     *
     * @param audio_samples  Float32 PCM samples (16kHz mono)
     * @param num_samples    Number of samples
     * @return Transcription text if decoder was triggered, empty string otherwise
     */
    std::string ProcessChunk(const float* audio_samples, int num_samples);

    /**
     * Force-flush remaining audio and run decoder on any accumulated frames.
     * Call this when the audio stream ends.
     *
     * @return Final transcription text (empty if no buffered frames)
     */
    std::string Flush();

    /**
     * Reset internal state for a new utterance.
     */
    void Reset();

    // Performance statistics
    float encoder_total_ms  = 0.0f;
    float decoder_total_ms  = 0.0f;
    int   encoder_calls     = 0;
    int   decoder_calls     = 0;

private:
    // --- Internal helpers ---

    // Preprocess one frame of raw audio samples → CMVN + Asinh → 80 floats
    void PreprocessFrame(const float* samples, float* out_frame);

    // Run chunk encoder NPU: input [1,160,80] → output [1,40,620]
    // Returns false on failure
    bool RunChunkEncoder(const float* frame_data, float* enc_out);

    // Run adapter projection on accumulated encoder output
    // enc_buf [M, 620] → proj_out [M, 512]
    void RunAdapterProjection(const float* enc_buf, int num_frames,
                               std::vector<float>& proj_out);

    // Run decoder autoregressive loop on projected encoder output
    // Returns decoded text
    std::string RunDecoder(const std::vector<float>& encoder_proj, int num_enc_frames);

    // Single decoder step
    bool RunDecoderStep(const std::vector<float>& decoder_embed,
                        const std::vector<float>& encoder_proj,
                        const std::vector<float>& past_keys,
                        const std::vector<float>& past_values,
                        const std::vector<float>& cos_cur,
                        const std::vector<float>& sin_cur,
                        const std::vector<float>& attn_mask,
                        const std::vector<float>& enc_attn_mask,
                        std::vector<float>& logits,
                        std::vector<float>& new_keys,
                        std::vector<float>& new_values);

    // Token decode
    std::string DecodeToken(int token_id);
    std::string TokenIdsToText(const std::vector<int>& token_ids);

    // Load vocab
    bool LoadVocab(const std::string& vocab_path);

    // Trigger decoder on current enc_buffer_ contents
    std::string TriggerDecoder();

    // --- Weights ---
    float log_k_   = 0.0f;
    float k_scale_ = 1.0f;  // = exp(log_k_)
    std::vector<float> embed_tokens_;   // [32768 * 512]
    std::vector<float> proj_weight_;    // [512 * 620]

    // RoPE tables (precomputed)
    std::vector<float> rope_cos_;
    std::vector<float> rope_sin_;
    int rope_table_len_ = 0;

    // --- Tokenizer ---
    std::vector<std::string> vocab_;

    // --- NPU Executors ---
    std::unique_ptr<mtk::neuropilot::NeuronExecutor> encoder_executor_;
    std::unique_ptr<mtk::neuropilot::NeuronExecutor> decoder_executor_;

    bool initialized_ = false;

    // --- VAD ---
    SileroVadWrapper vad_{16000,
                          VAD_WINDOW_MS,
                          VAD_THRESHOLD,
                          VAD_SILENCE_MS,
                          VAD_SPEECH_PAD_MS,
                          VAD_MIN_SPEECH_MS};

    // --- Streaming Buffers ---

    // Preprocessed frames buffer: stores CMVN+Asinh frames
    // Capacity: CHUNK_FRAMES frames, each FRAME_LEN_S floats
    std::vector<float> frame_buffer_;   // [frame_count_ * FRAME_LEN_S]
    int frame_count_ = 0;

    // Audio sample remainder (not yet forming a full frame)
    std::vector<float> audio_remainder_;

    // Encoder output accumulation buffer: [enc_frame_count_ * ENCODER_HIDDEN_S]
    std::vector<float> enc_buffer_;
    int enc_frame_count_ = 0;
};

} // namespace moonshine
