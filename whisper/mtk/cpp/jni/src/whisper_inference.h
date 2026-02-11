/**
 * Whisper Inference Engine for MTK NPU
 *
 * Main inference class for running Whisper speech recognition
 * on MTK MT8371 NPU platform
 */

#ifndef WHISPER_INFERENCE_H
#define WHISPER_INFERENCE_H

#include "mtk-npu/neuron_executor.h"
#include "utils/audio_utils.h"
#include <vector>
#include <string>
#include <memory>

// Whisper special tokens
#define WHISPER_SOT              50258   // <|startoftranscript|>
#define WHISPER_EOT              50257   // <|endoftext|>
#define WHISPER_SOT_LM           50360   // <|startoflm|>
#define WHISPER_TRANSCRIBE       50359   // <|transcribe|>
#define WHISPER_NO_TIMESTAMPS    50363   // <|notimestamps|>
#define WHISPER_TIMESTAMP_BEGIN  50364   // <|0.00|>
#define WHISPER_TASK_EN          50259   // <|en|>
#define WHISPER_TASK_ZH          50260   // <|zh|>

// Model variant selection - change this when switching models and recompile
// WHISPER_MODEL_LARGE_V3_TURBO: initial tokens = [SOT, lang, SOT_LM(50360), TIMESTAMP_BEGIN(50364)]
// WHISPER_MODEL_BASE:           initial tokens = [SOT, lang, TRANSCRIBE(50359), NO_TIMESTAMPS(50363)]
#define WHISPER_MODEL_LARGE_V3_TURBO  1
#define WHISPER_MODEL_BASE            2
#define WHISPER_MODEL_VARIANT  WHISPER_MODEL_LARGE_V3_TURBO  // <-- change here when switching models

/**
 * Whisper Inference Engine
 *
 * Manages encoder and decoder DLA models and performs
 * autoregressive speech recognition
 */
class WhisperInference {
public:
    WhisperInference();
    ~WhisperInference();

    /**
     * Initialize inference engine
     * @param model_dir Path to directory containing DLA models and resources
     * @return 0 on success, negative on error
     */
    int init(const char* model_dir);

    /**
     * Run inference on audio file
     * @param audio_file Path to audio file (WAV format)
     * @param language Language code ("en", "zh", or NULL for auto-detect)
     * @param task Task code ("transcribe" or "translate")
     * @return Recognized text
     */
    std::string run(const char* audio_file,
                   const char* language = nullptr,
                   const char* task = "transcribe");

    /**
     * Release resources
     */
    void release();

private:
    /**
     * Load mel filters from file
     */
    bool load_mel_filters(const std::string& path);

    /**
     * Load vocabulary from file
     */
    bool load_vocab(const std::string& path);

    /**
     * Load token embedding weights (for decoder)
     */
    bool load_token_embeddings(const std::string& path);

    /**
     * Load position embeddings (for decoder)
     */
    bool load_position_embeddings(const std::string& path);

    /**
     * Load encoder DLA model
     */
    bool load_encoder_dla(const std::string& path);

    /**
     * Load decoder DLA model
     */
    bool load_decoder_dla(const std::string& path);

    /**
     * Run encoder inference
     * @param mel_spec Input mel spectrogram [80, 3000]
     * @param encoder_output Output [1, 1500, 512]
     * @return true on success
     */
    bool run_encoder(const std::vector<float>& mel_spec,
                    std::vector<float>& encoder_output);

    /**
     * Run decoder inference (autoregressive)
     * @param encoder_output Encoder output [1, 1500, 512]
     * @param tokens Output token sequence
     * @param language_token Language token (e.g., WHISPER_TASK_EN or WHISPER_TASK_ZH)
     * @return true on success
     */
    bool run_decoder(const std::vector<float>& encoder_output,
                    std::vector<int>& tokens,
                    int language_token = WHISPER_TASK_EN);

    /**
     * Decode tokens to text using vocabulary
     * @param tokens Token sequence
     * @param task_code Task token (for language-specific processing)
     * @return Decoded text
     */
    std::string decode_tokens(const std::vector<int>& tokens, int task_code);

    /**
     * Lookup token embedding
     * @param token_id Token ID
     * @param embeddings Output embeddings buffer
     * @param offset Offset in buffer to write to
     */
    void lookup_embedding(int token_id, std::vector<float>& embeddings, size_t offset);

    /**
     * Reset KV cache for new inference
     */
    void reset_kv_cache();

    /**
     * Create self-attention mask for current cache length
     * @param cache_len Current cache length
     * @param mask Output mask buffer [1, 1, 1, 449]
     */
    void create_self_attn_mask(int cache_len, float* mask);

    /**
     * Get position embedding for given position
     * @param position Token position
     * @param output Output buffer [1, 1, 512]
     */
    void get_position_embedding(int position, float* output);

private:
    // Configuration
    bool initialized_ = false;
    bool debug_mode_ = false;  // Control debug output

    // Model configuration (can be set for different model sizes)
    int vocab_size_ = 51866;   // 51865 for base, 51866 for large-v3-turbo
    int d_model_ = 1280;       // n_text_state: 384(tiny), 512(base), 768(small), 1024(medium), 1280(large)
    int num_layers_ = 4;       // n_text_layer: 4(tiny/large-v3-turbo), 6(base), 12(small), 24(medium), 32(large)
    int max_cache_len_ = 448;  // n_text_ctx: always 448

    // Embedding weights (vocab_size x d_model)
    std::vector<float> token_embeddings_;

    // Mel filters
    std::vector<float> mel_filters_;
    bool mel_filters_loaded_ = false;

    // Vocabulary
    std::vector<VocabEntry> vocab_;
    bool vocab_loaded_ = false;

    // NPU executors
    std::unique_ptr<NeuronExecutor> encoder_executor_;
    std::unique_ptr<NeuronExecutor> decoder_executor_;

    // KV cache buffers: [num_layers, batch=1, seq_len, d_model]
    std::vector<float> past_self_keys_;     // [6, 1, 448, 512]
    std::vector<float> past_self_values_;   // [6, 1, 448, 512]
    std::vector<float> cached_cross_keys_;  // [6, 1, 1500, 512] - computed once
    std::vector<float> cached_cross_values_;// [6, 1, 1500, 512] - computed once

    // Position embeddings (loaded from Python or sinusoidal)
    std::vector<float> position_embeddings_; // [448, 512]
    bool position_embeddings_loaded_ = false;

    // Current cache state
    int cache_len_ = 0;
    bool cross_cache_initialized_ = false;
};

#endif // WHISPER_INFERENCE_H
