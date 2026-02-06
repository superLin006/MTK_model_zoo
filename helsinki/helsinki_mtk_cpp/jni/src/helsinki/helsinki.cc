/**
 * Helsinki Translation Model - MTK NPU Implementation with KV Cache
 */

#include "helsinki.h"
#include "mtk-npu/neuron_executor.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <dlfcn.h>

// ==================== Constructor / Destructor ====================

HelsinkiTranslator::HelsinkiTranslator() {}

HelsinkiTranslator::~HelsinkiTranslator() {
    release();
}

// ==================== Initialize ====================

int HelsinkiTranslator::init(const char* model_dir) {
    if (initialized_) {
        std::cerr << "[WARN] Already initialized" << std::endl;
        return 0;
    }

    model_dir_ = model_dir;
    std::cout << "[INFO] Initializing Helsinki Translator from: " << model_dir_ << std::endl;

    // Load embedding weights
    if (!load_embeddings(model_dir_ + "/embedding_weights.bin")) {
        return -1;
    }

    // Load position embeddings
    if (!load_position_embeddings(model_dir_ + "/position_embeddings.bin")) {
        return -1;
    }

    // Load encoder DLA
    if (!load_encoder_dla(model_dir_ + "/encoder_src64_MT8371.dla")) {
        return -1;
    }

    // Load decoder DLA
    if (!load_decoder_dla(model_dir_ + "/decoder_kv_src64_cache64_MT8371.dla")) {
        return -1;
    }

    // Allocate inference buffers
    encoder_input_.resize(1 * src_seq_len_ * d_model_);
    encoder_self_attn_mask_.resize(1 * 1 * src_seq_len_ * src_seq_len_);
    encoder_output_.resize(1 * src_seq_len_ * d_model_);
    decoder_embed_.resize(1 * 1 * d_model_);
    position_embed_.resize(1 * 1 * d_model_);
    attn_mask_.resize(1 * 1 * 1 * (max_cache_len_ + 1));
    encoder_attn_mask_.resize(1 * 1 * 1 * src_seq_len_);
    logits_.resize(1 * 1 * vocab_size_);
    new_keys_.resize(num_layers_ * 1 * 1 * d_model_);
    new_values_.resize(num_layers_ * 1 * 1 * d_model_);

    // Allocate KV cache
    size_t cache_size = num_layers_ * 1 * max_cache_len_ * d_model_;
    past_keys_.resize(cache_size, 0.0f);
    past_values_.resize(cache_size, 0.0f);

    initialized_ = true;
    std::cout << "[INFO] Initialization complete!" << std::endl;

    return 0;
}

// ==================== Load Embeddings ====================

bool HelsinkiTranslator::load_embeddings(const std::string& path) {
    std::cout << "[INFO] Loading embeddings: " << path << std::endl;

    // Read meta file
    std::string meta_path = path.substr(0, path.rfind('.')) + "_meta.txt";
    std::ifstream meta_file(meta_path);
    if (meta_file) {
        std::string line;
        while (std::getline(meta_file, line)) {
            if (line.find("vocab_size=") == 0) {
                vocab_size_ = std::stoi(line.substr(11));
            } else if (line.find("d_model=") == 0) {
                d_model_ = std::stoi(line.substr(8));
            }
        }
    }

    // Read embedding weights
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Failed to open: " << path << std::endl;
        return false;
    }

    size_t embed_size = vocab_size_ * d_model_;
    embedding_weights_.resize(embed_size);
    file.read(reinterpret_cast<char*>(embedding_weights_.data()), embed_size * sizeof(float));

    if (!file) {
        std::cerr << "[ERROR] Failed to read embeddings" << std::endl;
        return false;
    }

    std::cout << "[INFO]   Shape: [" << vocab_size_ << ", " << d_model_ << "]" << std::endl;
    return true;
}

bool HelsinkiTranslator::load_position_embeddings(const std::string& path) {
    std::cout << "[INFO] Loading position embeddings: " << path << std::endl;

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Failed to open: " << path << std::endl;
        return false;
    }

    size_t pos_size = max_position_ * d_model_;
    position_embeddings_.resize(pos_size);
    file.read(reinterpret_cast<char*>(position_embeddings_.data()), pos_size * sizeof(float));

    if (!file) {
        std::cerr << "[ERROR] Failed to read position embeddings" << std::endl;
        return false;
    }

    std::cout << "[INFO]   Shape: [" << max_position_ << ", " << d_model_ << "]" << std::endl;
    return true;
}

// ==================== Load DLA Models ====================

bool HelsinkiTranslator::load_encoder_dla(const std::string& path) {
    std::cout << "[INFO] Loading encoder DLA: " << path << std::endl;

    // Encoder input/output shapes
    // Input 0: input_embed [1, src_seq_len, d_model]
    // Input 1: attn_mask [1, 1, src_seq_len, src_seq_len]
    // Output: encoder_output [1, src_seq_len, d_model]

    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, (uint32_t)src_seq_len_, (uint32_t)d_model_},      // input_embed
        {1, 1, (uint32_t)src_seq_len_, (uint32_t)src_seq_len_}  // attn_mask
    };

    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, (uint32_t)src_seq_len_, (uint32_t)d_model_}  // encoder_output
    };

    encoder_executor_ = new NeuronExecutor(path, input_shapes, output_shapes, "Encoder");
    if (!encoder_executor_->Initialize()) {
        std::cerr << "[ERROR] Failed to initialize encoder executor" << std::endl;
        delete encoder_executor_;
        encoder_executor_ = nullptr;
        return false;
    }

    std::cout << "[INFO]   Encoder loaded successfully" << std::endl;
    return true;
}

bool HelsinkiTranslator::load_decoder_dla(const std::string& path) {
    std::cout << "[INFO] Loading decoder DLA: " << path << std::endl;

    // Decoder input/output shapes
    // Input 0: decoder_embed [1, 1, d_model]
    // Input 1: encoder_hidden [1, src_seq_len, d_model]
    // Input 2: past_keys [num_layers, 1, max_cache_len, d_model]
    // Input 3: past_values [num_layers, 1, max_cache_len, d_model]
    // Input 4: position_embed [1, 1, d_model]
    // Input 5: attn_mask [1, 1, 1, max_cache_len+1]
    // Input 6: encoder_attn_mask [1, 1, 1, src_seq_len]
    // Output 0: logits [1, 1, vocab_size]
    // Output 1: new_keys [num_layers, 1, 1, d_model]
    // Output 2: new_values [num_layers, 1, 1, d_model]

    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, 1, (uint32_t)d_model_},                                  // decoder_embed
        {1, (uint32_t)src_seq_len_, (uint32_t)d_model_},             // encoder_hidden
        {(uint32_t)num_layers_, 1, (uint32_t)max_cache_len_, (uint32_t)d_model_},  // past_keys
        {(uint32_t)num_layers_, 1, (uint32_t)max_cache_len_, (uint32_t)d_model_},  // past_values
        {1, 1, (uint32_t)d_model_},                                  // position_embed
        {1, 1, 1, (uint32_t)(max_cache_len_ + 1)},                   // attn_mask
        {1, 1, 1, (uint32_t)src_seq_len_}                            // encoder_attn_mask
    };

    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, 1, (uint32_t)vocab_size_},                               // logits
        {(uint32_t)num_layers_, 1, 1, (uint32_t)d_model_},          // new_keys
        {(uint32_t)num_layers_, 1, 1, (uint32_t)d_model_}           // new_values
    };

    decoder_executor_ = new NeuronExecutor(path, input_shapes, output_shapes, "Decoder");
    if (!decoder_executor_->Initialize()) {
        std::cerr << "[ERROR] Failed to initialize decoder executor" << std::endl;
        delete decoder_executor_;
        decoder_executor_ = nullptr;
        return false;
    }

    std::cout << "[INFO]   Decoder loaded successfully" << std::endl;
    return true;
}

// ==================== Embedding Functions ====================

void HelsinkiTranslator::embed_tokens(const int64_t* token_ids, int seq_len, float* output) {
    // Clear output
    std::fill(output, output + src_seq_len_ * d_model_, 0.0f);

    // Embed each token
    int actual_len = std::min(seq_len, src_seq_len_);
    for (int i = 0; i < actual_len; i++) {
        int64_t token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size_) {
            float* dst = output + i * d_model_;
            const float* src_embed = embedding_weights_.data() + token_id * d_model_;
            memcpy(dst, src_embed, d_model_ * sizeof(float));
        }
    }
}

void HelsinkiTranslator::embed_single_token(int64_t token_id, int step, float* output) {
    if (token_id >= 0 && token_id < vocab_size_) {
        const float* src_embed = embedding_weights_.data() + token_id * d_model_;
        memcpy(output, src_embed, d_model_ * sizeof(float));
    } else {
        std::fill(output, output + d_model_, 0.0f);
    }
}

void HelsinkiTranslator::get_position_embedding(int position, float* output) {
    if (position >= 0 && position < max_position_) {
        memcpy(output, position_embeddings_.data() + position * d_model_,
               d_model_ * sizeof(float));
    } else {
        std::fill(output, output + d_model_, 0.0f);
    }
}

void HelsinkiTranslator::create_attn_mask(int cache_len, float* output) {
    // Create attention mask for self-attention with KV cache
    // Shape: [1, 1, 1, max_cache_len + 1]
    // 0 for valid positions, -1e9 for invalid positions
    const float NEG_INF = -1e9f;
    int total_len = max_cache_len_ + 1;

    // Initialize all to invalid
    std::fill(output, output + total_len, NEG_INF);

    // Valid positions: [0, cache_len) in past cache
    for (int i = 0; i < cache_len; i++) {
        output[i] = 0.0f;
    }

    // Current position (last position) is always valid
    output[total_len - 1] = 0.0f;
}

void HelsinkiTranslator::create_encoder_self_attn_mask(int actual_src_len, float* output) {
    // Create encoder self-attention mask
    // Shape: [1, 1, src_seq_len, src_seq_len]
    // 0 for valid positions, -1e9 for padding positions
    const float NEG_INF = -1e9f;

    for (int r = 0; r < src_seq_len_; r++) {
        for (int c = 0; c < src_seq_len_; c++) {
            if (c < actual_src_len) {
                output[r * src_seq_len_ + c] = 0.0f;
            } else {
                output[r * src_seq_len_ + c] = NEG_INF;
            }
        }
    }
}

void HelsinkiTranslator::create_encoder_attn_mask(int actual_src_len, float* output) {
    // Create encoder attention mask for cross-attention
    // Shape: [1, 1, 1, src_seq_len]
    // 0 for valid positions, -1e9 for padding positions
    const float NEG_INF = -1e9f;

    for (int i = 0; i < src_seq_len_; i++) {
        if (i < actual_src_len) {
            output[i] = 0.0f;  // Valid
        } else {
            output[i] = NEG_INF;  // Padding
        }
    }
}

// ==================== KV Cache ====================

void HelsinkiTranslator::reset_kv_cache() {
    std::fill(past_keys_.begin(), past_keys_.end(), 0.0f);
    std::fill(past_values_.begin(), past_values_.end(), 0.0f);
    cache_len_ = 0;
}

void HelsinkiTranslator::update_kv_cache(const float* new_keys, const float* new_values) {
    if (cache_len_ >= max_cache_len_) {
        // Shift cache left (drop oldest)
        for (int layer = 0; layer < num_layers_; layer++) {
            size_t layer_offset = layer * max_cache_len_ * d_model_;
            // Shift positions 1..max_cache_len-1 to 0..max_cache_len-2
            memmove(past_keys_.data() + layer_offset,
                    past_keys_.data() + layer_offset + d_model_,
                    (max_cache_len_ - 1) * d_model_ * sizeof(float));
            memmove(past_values_.data() + layer_offset,
                    past_values_.data() + layer_offset + d_model_,
                    (max_cache_len_ - 1) * d_model_ * sizeof(float));
        }
        cache_len_ = max_cache_len_ - 1;
    }

    // Copy new keys/values to cache (at position cache_len_)
    for (int layer = 0; layer < num_layers_; layer++) {
        size_t src_offset = layer * d_model_;
        size_t dst_offset = layer * max_cache_len_ * d_model_ + cache_len_ * d_model_;

        memcpy(past_keys_.data() + dst_offset, new_keys + src_offset, d_model_ * sizeof(float));
        memcpy(past_values_.data() + dst_offset, new_values + src_offset, d_model_ * sizeof(float));
    }

    cache_len_++;
}

// ==================== NPU Inference ====================

int HelsinkiTranslator::run_encoder(const float* encoder_input, const float* encoder_attn_mask, float* encoder_output) {
    if (!encoder_executor_ || !encoder_executor_->IsInitialized()) {
        std::cerr << "[ERROR] Encoder executor not initialized" << std::endl;
        return -1;
    }

    // Prepare inputs
    std::vector<const void*> inputs = {
        encoder_input,
        encoder_attn_mask
    };

    // Prepare outputs
    std::vector<void*> outputs = {
        encoder_output
    };

    // Run inference
    if (!encoder_executor_->Run(inputs, outputs)) {
        std::cerr << "[ERROR] Encoder inference failed" << std::endl;
        return -1;
    }

    return 0;
}

int HelsinkiTranslator::run_decoder(const float* decoder_embed, const float* encoder_hidden,
                                     const float* past_keys, const float* past_values,
                                     const float* position_embed,
                                     const float* attn_mask, const float* encoder_attn_mask,
                                     float* logits, float* new_keys, float* new_values) {
    if (!decoder_executor_ || !decoder_executor_->IsInitialized()) {
        std::cerr << "[ERROR] Decoder executor not initialized" << std::endl;
        return -1;
    }

    // Prepare inputs (order must match model definition)
    std::vector<const void*> inputs = {
        decoder_embed,
        encoder_hidden,
        past_keys,
        past_values,
        position_embed,
        attn_mask,
        encoder_attn_mask
    };

    // Prepare outputs
    std::vector<void*> outputs = {
        logits,
        new_keys,
        new_values
    };

    // Run inference
    if (!decoder_executor_->Run(inputs, outputs)) {
        std::cerr << "[ERROR] Decoder inference failed" << std::endl;
        return -1;
    }

    return 0;
}

// ==================== Translate ====================

int HelsinkiTranslator::translate(const int64_t* input_ids, int input_len,
                                   int64_t* output_ids, int max_output_len,
                                   helsinki_perf_stats_t* perf_stats) {
    if (!initialized_) {
        std::cerr << "[ERROR] Not initialized" << std::endl;
        return -1;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // Store actual source length for encoder masks.
    // NOTE: main.cc passes padded length (src_seq_len). We must infer the real length
    // from PAD tokens, otherwise PAD will be treated as valid tokens and can cause
    // degenerate looping / mismatched outputs vs Python.
    int max_check = std::min(input_len, src_seq_len_);
    actual_src_len_ = max_check;
    for (int i = 0; i < max_check; i++) {
        if (input_ids[i] == pad_token_id_) {
            actual_src_len_ = i;
            break;
        }
    }
    if (actual_src_len_ < 1) {
        actual_src_len_ = 1;
    }

    // 1. Embed input tokens
    embed_tokens(input_ids, input_len, encoder_input_.data());

    // 1.1 Create encoder self-attention mask
    create_encoder_self_attn_mask(actual_src_len_, encoder_self_attn_mask_.data());

    // 2. Run encoder
    auto encoder_start = std::chrono::high_resolution_clock::now();
    int ret = run_encoder(encoder_input_.data(), encoder_self_attn_mask_.data(), encoder_output_.data());
    auto encoder_end = std::chrono::high_resolution_clock::now();

    if (ret != 0) {
        return -1;
    }

    double encoder_ms = std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();
    std::cout << "[PERF] Encoder: " << encoder_ms << " ms" << std::endl;

    // 3. Reset KV cache
    reset_kv_cache();

    // 4. Create encoder attention mask (once, as source doesn't change)
    create_encoder_attn_mask(actual_src_len_, encoder_attn_mask_.data());

    // 5. Autoregressive decoding
    std::vector<int64_t> generated_tokens;
    int current_token = pad_token_id_;  // Start token
    int out_len = 0;

    // Disable penalties while aligning with Python argmax baseline
    const float REPETITION_PENALTY = 1.0f;
    const int NO_REPEAT_NGRAM_SIZE = 0;

    auto decoder_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_output_len; step++) {
        // 5.1 Prepare Decoder inputs
        embed_single_token(current_token, step, decoder_embed_.data());
        get_position_embedding(step, position_embed_.data());
        create_attn_mask(cache_len_, attn_mask_.data());

        // 5.2 Run Decoder
        int ret = run_decoder(decoder_embed_.data(), encoder_output_.data(),
                              past_keys_.data(), past_values_.data(),
                              position_embed_.data(),
                              attn_mask_.data(), encoder_attn_mask_.data(),
                              logits_.data(), new_keys_.data(), new_values_.data());
        
        if (ret != 0) {
            std::cerr << "[ERROR] Decoder failed at step " << step << std::endl;
            break;
        }

        // 5.4 Update KV Cache
        update_kv_cache(new_keys_.data(), new_values_.data());

        // 5.5 Apply Penalties
        if (REPETITION_PENALTY != 1.0f) {
            apply_repetition_penalty(logits_.data(), vocab_size_, generated_tokens, REPETITION_PENALTY);
        }

        if (NO_REPEAT_NGRAM_SIZE > 0) {
            block_repeated_ngrams(logits_.data(), vocab_size_, generated_tokens, NO_REPEAT_NGRAM_SIZE);
        }

        // 5.6 Sample next token
        current_token = argmax(logits_.data(), vocab_size_);

        std::cout << "[DEBUG] Step " << step << ": token " << current_token;
        if (current_token == eos_token_id_) std::cout << " [EOS]";
        std::cout << std::endl;

        // Match Python greedy baseline: stop only on EOS.
        if (current_token == eos_token_id_) {
            break;
        }

        output_ids[out_len++] = current_token;
        generated_tokens.push_back(current_token);
    }

    auto decoder_end = std::chrono::high_resolution_clock::now();
    auto total_end = decoder_end;

    double decoder_ms = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Performance stats
    if (perf_stats != nullptr) {
        perf_stats->encoder_time_ms = encoder_ms;
        perf_stats->decoder_time_ms = decoder_ms;
        perf_stats->decoder_avg_time_ms = (out_len > 0) ? (decoder_ms / out_len) : 0.0;
        perf_stats->total_time_ms = total_ms;
        perf_stats->total_steps = out_len;
        perf_stats->output_tokens = out_len;
    }

    std::cout << "\n========== Performance ==========\n";
    std::cout << "[PERF] Encoder:     " << encoder_ms << " ms\n";
    std::cout << "[PERF] Decoder:     " << decoder_ms << " ms\n";
    std::cout << "[PERF] Avg/token:   " << (out_len > 0 ? decoder_ms / out_len : 0.0) << " ms\n";
    std::cout << "[PERF] Total:       " << total_ms << " ms\n";
    std::cout << "[PERF] Tokens:      " << out_len << "\n";
    std::cout << "================================\n\n";

    return out_len;
}

// ==================== Helper Functions ====================

int HelsinkiTranslator::argmax(const float* logits, int size) {
    int max_idx = 0;
    float max_val = logits[0];

    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    return max_idx;
}

void HelsinkiTranslator::apply_repetition_penalty(float* logits, int vocab_size,
                                                   const std::vector<int64_t>& generated_tokens,
                                                   float penalty) {
    if (penalty == 1.0f) return;

    for (int64_t token : generated_tokens) {
        if (token >= 0 && token < vocab_size) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}

void HelsinkiTranslator::block_repeated_ngrams(float* logits, int vocab_size,
                                                const std::vector<int64_t>& generated_tokens,
                                                int ngram_size) {
    if ((int)generated_tokens.size() < ngram_size - 1) return;

    // Get context (last ngram_size-1 tokens)
    std::vector<int64_t> context;
    int start_idx = generated_tokens.size() - (ngram_size - 1);
    for (size_t i = start_idx; i < generated_tokens.size(); i++) {
        context.push_back(generated_tokens[i]);
    }

    // Check each candidate
    for (int candidate = 0; candidate < vocab_size; candidate++) {
        std::vector<int64_t> test_ngram = context;
        test_ngram.push_back(candidate);

        if ((int)generated_tokens.size() >= ngram_size) {
            for (size_t i = 0; i <= generated_tokens.size() - ngram_size; i++) {
                bool match = true;
                for (int j = 0; j < ngram_size; j++) {
                    if (generated_tokens[i + j] != test_ngram[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    logits[candidate] = -1e9f;
                    break;
                }
            }
        }
    }
}

void HelsinkiTranslator::print_model_info() {
    std::cout << "\n========== Model Info ==========\n";
    std::cout << "vocab_size:     " << vocab_size_ << "\n";
    std::cout << "d_model:        " << d_model_ << "\n";
    std::cout << "num_layers:     " << num_layers_ << "\n";
    std::cout << "num_heads:      " << num_heads_ << "\n";
    std::cout << "src_seq_len:    " << src_seq_len_ << "\n";
    std::cout << "max_cache_len:  " << max_cache_len_ << "\n";
    std::cout << "pad_token_id:   " << pad_token_id_ << "\n";
    std::cout << "eos_token_id:   " << eos_token_id_ << "\n";
    std::cout << "================================\n\n";
}

void HelsinkiTranslator::release() {
    if (!initialized_) return;

    std::cout << "[INFO] Releasing resources..." << std::endl;

    if (encoder_executor_) {
        delete encoder_executor_;
        encoder_executor_ = nullptr;
    }

    if (decoder_executor_) {
        delete decoder_executor_;
        decoder_executor_ = nullptr;
    }

    embedding_weights_.clear();
    position_embeddings_.clear();
    past_keys_.clear();
    past_values_.clear();

    initialized_ = false;
    std::cout << "[INFO] Resources released" << std::endl;
}
