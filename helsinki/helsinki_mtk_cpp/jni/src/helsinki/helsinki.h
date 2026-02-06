/**
 * Helsinki Translation Model - MTK NPU Implementation with KV Cache
 *
 * Adapted from:
 * - helsinki-rknn (CPU/RKNN implementation)
 * - sensevoice_mtk_cpp (MTK NPU interface)
 *
 * Architecture:
 * - Embedding: CPU (GATHER not supported on MT8371)
 * - Encoder: NPU (DLA)
 * - Decoder: NPU (DLA) with KV Cache
 *
 * KV Cache Format (4D to avoid 5D tensor limitation):
 * - past_keys/values: [num_layers, batch, max_cache_len, d_model]
 * - new_keys/values: [num_layers, batch, 1, d_model]
 */

#ifndef HELSINKI_H
#define HELSINKI_H

#include <vector>
#include <string>
#include <cstdint>
#include "mtk-npu/neuron_executor.h"

// Performance statistics
typedef struct {
    double encoder_time_ms;
    double decoder_time_ms;
    double decoder_avg_time_ms;
    double total_time_ms;
    int total_steps;
    int output_tokens;
} helsinki_perf_stats_t;

class HelsinkiTranslator {
public:
    HelsinkiTranslator();
    ~HelsinkiTranslator();

    // Initialize model (load DLA, embeddings, etc.)
    int init(const char* model_dir);

    // Translate with token IDs (tokenization done externally)
    int translate(const int64_t* input_ids, int input_len,
                  int64_t* output_ids, int max_output_len,
                  helsinki_perf_stats_t* perf_stats = nullptr);

    // Utility functions
    void apply_repetition_penalty(float* logits, int vocab_size,
                                  const std::vector<int64_t>& generated_tokens,
                                  float penalty);
    void block_repeated_ngrams(float* logits, int vocab_size,
                               const std::vector<int64_t>& generated_tokens,
                               int ngram_size);

    void print_model_info();
    void release();

    // Getters for model config
    int get_vocab_size() const { return vocab_size_; }
    int get_d_model() const { return d_model_; }
    int get_src_seq_len() const { return src_seq_len_; }

private:
    // Load functions
    bool load_embeddings(const std::string& path);
    bool load_position_embeddings(const std::string& path);
    bool load_encoder_dla(const std::string& path);
    bool load_decoder_dla(const std::string& path);

    // Inference functions
    int run_encoder(const float* encoder_input, const float* encoder_attn_mask, float* encoder_output);
    int run_decoder(const float* decoder_embed, const float* encoder_hidden,
                    const float* past_keys, const float* past_values,
                    const float* position_embed,
                    const float* attn_mask, const float* encoder_attn_mask,
                    float* logits, float* new_keys, float* new_values);

    // Helper functions
    void embed_tokens(const int64_t* token_ids, int seq_len, float* output);
    void embed_single_token(int64_t token_id, int step, float* output);
    void get_position_embedding(int position, float* output);
    void create_attn_mask(int cache_len, float* output);
    void create_encoder_self_attn_mask(int actual_src_len, float* output);
    void create_encoder_attn_mask(int actual_src_len, float* output);
    int argmax(const float* logits, int size);
    void reset_kv_cache();
    void update_kv_cache(const float* new_keys, const float* new_values);

    // Neuron Runtime library functions (dynamically loaded)
    bool load_neuron_library();

    // Model configuration
    int vocab_size_ = 65001;
    int d_model_ = 512;
    int num_layers_ = 6;
    int num_heads_ = 8;
    int src_seq_len_ = 64;
    int max_cache_len_ = 64;
    int max_position_ = 512;

    // Special tokens
    int pad_token_id_ = 65000;
    int eos_token_id_ = 0;

    // Embedding weights (CPU)
    std::vector<float> embedding_weights_;    // [vocab_size, d_model]
    std::vector<float> position_embeddings_;  // [max_position, d_model]

    // KV Cache
    std::vector<float> past_keys_;   // [num_layers, 1, max_cache_len, d_model]
    std::vector<float> past_values_; // [num_layers, 1, max_cache_len, d_model]
    int cache_len_ = 0;

    // NPU Executors (new Neuron API)
    NeuronExecutor* encoder_executor_ = nullptr;
    NeuronExecutor* decoder_executor_ = nullptr;

    // Inference buffers
    std::vector<float> encoder_input_;   // [1, src_seq_len, d_model]
    std::vector<float> encoder_self_attn_mask_; // [1, 1, src_seq_len, src_seq_len]
    std::vector<float> encoder_output_;  // [1, src_seq_len, d_model]
    std::vector<float> decoder_embed_;   // [1, 1, d_model]
    std::vector<float> position_embed_;  // [1, 1, d_model]
    std::vector<float> attn_mask_;       // [1, 1, 1, max_cache_len+1]
    std::vector<float> encoder_attn_mask_;  // [1, 1, 1, src_seq_len]
    std::vector<float> logits_;          // [1, 1, vocab_size]
    std::vector<float> new_keys_;        // [num_layers, 1, 1, d_model]
    std::vector<float> new_values_;      // [num_layers, 1, 1, d_model]

    int actual_src_len_ = 0;  // Actual source sequence length (without padding)

    bool initialized_ = false;
    std::string model_dir_;
};

#endif // HELSINKI_H
