/* moonshine_inference.h - Moonshine MTK NPU Inference
 *
 * Complete end-to-end ASR inference:
 *   Audio → Preprocess → Encoder NPU → Adapter Projection → Decoder Loop → Text
 */

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>

#include "executor/NeuronExecutor.h"

namespace moonshine {

// Forward declaration
class NeuronExecutorWrapper;

// Token streaming callback (optional)
// Called with each decoded text piece as it becomes available
using TokenCallback = std::function<void(const std::string&)>;

/**
 * Moonshine Inference Engine
 *
 * Usage:
 *   MoonshineInference engine;
 *   engine.Initialize(encoder_dla, decoder_dla, embed_tokens_npy,
 *                     pos_emb_npy, proj_weight_npy, log_k_npy,
 *                     tokenizer_json);
 *   std::string text = engine.Transcribe(audio_path);
 */
class MoonshineInference {
public:
    MoonshineInference() = default;
    ~MoonshineInference();

    /**
     * Initialize the inference engine.
     *
     * @param encoder_dla_path  Path to moonshine_encoder.dla
     * @param decoder_dla_path  Path to moonshine_decoder.dla
     * @param embed_tokens_path Path to embed_tokens.npy [32768, 512]
     * @param pos_emb_path      Path to pos_emb_weight.npy [4096, 620]
     * @param proj_weight_path  Path to proj_weight.npy [512, 620]
     * @param log_k_path        Path to log_k.npy [1] float32
     * @param tokenizer_path    Path to tokenizer.json
     * @return true on success
     */
    bool Initialize(const std::string& encoder_dla_path,
                    const std::string& decoder_dla_path,
                    const std::string& embed_tokens_path,
                    const std::string& pos_emb_path,
                    const std::string& proj_weight_path,
                    const std::string& log_k_path,
                    const std::string& vocab_path);

    /**
     * Transcribe an audio file.
     *
     * @param audio_path  Path to WAV file (16kHz mono)
     * @param callback    Optional: called with each decoded piece
     * @return Transcribed text
     */
    std::string Transcribe(const std::string& audio_path,
                           TokenCallback callback = nullptr);

    /**
     * Transcribe from raw float32 samples.
     *
     * @param samples   Float32 audio samples (16kHz mono)
     * @param callback  Optional streaming callback
     * @return Transcribed text
     */
    std::string TranscribeSamples(const std::vector<float>& samples,
                                  TokenCallback callback = nullptr);

    // Performance stats from last inference
    double encoder_time_ms = 0.0;
    double decoder_total_ms = 0.0;
    double decoder_per_step_ms = 0.0;
    int    num_tokens = 0;

private:
    // --- Internal methods ---
    bool LoadWeights(const std::string& embed_tokens_path,
                     const std::string& pos_emb_path,
                     const std::string& proj_weight_path,
                     const std::string& log_k_path);
    bool LoadVocab(const std::string& vocab_path);

    // Encoder: [1,2000,80] → [1,500,620]
    bool RunEncoder(const std::vector<float>& frames,
                    std::vector<float>& encoder_out);

    // Adapter: pos_emb + proj(620→512) → [1,500,512]
    void RunAdapterProjection(const std::vector<float>& encoder_out,
                               std::vector<float>& encoder_proj);

    // Decoder single step
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

    // Token decode helpers
    std::string DecodeToken(int token_id);
    std::string TokenIdsToText(const std::vector<int>& token_ids);

    // --- Weights ---
    float log_k_ = 0.0f;
    std::vector<float> embed_tokens_;  // [32768 * 512]
    std::vector<float> pos_emb_;       // [4096 * 620]
    std::vector<float> proj_weight_;   // [512 * 620]

    // RoPE tables (precomputed at init): [MAX_DEC_LEN+10, ROPE_DIM]
    std::vector<float> rope_cos_;
    std::vector<float> rope_sin_;
    int rope_table_len_ = 0;

    // --- Tokenizer ---
    // BPE vocab: id -> piece string
    std::vector<std::string> vocab_;

    // --- NPU Executors ---
    std::unique_ptr<mtk::neuropilot::NeuronExecutor> encoder_executor_;
    std::unique_ptr<mtk::neuropilot::NeuronExecutor> decoder_executor_;

    bool initialized_ = false;
};

} // namespace moonshine
