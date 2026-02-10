/**
 * Whisper Inference Engine Implementation for MTK NPU
 */

#include "whisper_inference.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <cstdlib>

// Debug output macro
#define DEBUG_LOG(msg) if (debug_mode_) { std::cout << "[DEBUG] " << msg << std::endl; }

// ==================== Constructor / Destructor ====================

WhisperInference::WhisperInference() {}

WhisperInference::~WhisperInference() {
    release();
}

// ==================== Initialize ====================

int WhisperInference::init(const char* model_dir) {
    if (initialized_) {
        std::cerr << "[WARN] Already initialized" << std::endl;
        return 0;
    }

    // Check for debug mode from environment variable
    const char* debug_env = std::getenv("WHISPER_DEBUG");
    debug_mode_ = (debug_env != nullptr && std::string(debug_env) == "1");

    std::cout << "[INFO] Initializing Whisper Inference from: "
              << model_dir << std::endl;
    if (debug_mode_) {
        std::cout << "[DEBUG] Debug mode enabled (set WHISPER_DEBUG=0 to disable)" << std::endl;
    }

    std::string base_dir(model_dir);

    // Load token embeddings
    if (!load_token_embeddings(base_dir + "/token_embedding.npy")) {
        std::cerr << "[ERROR] Failed to load token embeddings" << std::endl;
        return -1;
    }

    // Load position embeddings
    if (!load_position_embeddings(base_dir + "/position_embedding.npy")) {
        std::cerr << "[WARN] Failed to load position embeddings, using zeros" << std::endl;
        // Not critical, will use zeros
    }

    // Load mel filters
    if (!load_mel_filters(base_dir + "/mel_80_filters.txt")) {
        std::cerr << "[WARN] Failed to load mel filters, using placeholder" << std::endl;
        // Create placeholder mel filters (80 x 201)
        mel_filters_.resize(N_MELS * MELS_FILTERS_SIZE);
        for (int i = 0; i < N_MELS * MELS_FILTERS_SIZE; i++) {
            mel_filters_[i] = 0.01f;
        }
    } else {
        mel_filters_loaded_ = true;
    }

    // Load vocabulary
    if (!load_vocab(base_dir + "/vocab.txt")) {
        std::cerr << "[WARN] Failed to load vocabulary" << std::endl;
        // Don't fail - vocabulary is optional for basic functionality
    } else {
        vocab_loaded_ = true;
    }

    // Load encoder DLA
    if (!load_encoder_dla(base_dir + "/encoder.dla")) {
        std::cerr << "[ERROR] Failed to load encoder DLA" << std::endl;
        return -1;
    }

    // Load decoder DLA
    if (!load_decoder_dla(base_dir + "/decoder_kv.dla")) {
        std::cerr << "[ERROR] Failed to load decoder DLA" << std::endl;
        return -1;
    }

    initialized_ = true;
    std::cout << "[INFO] Whisper Inference initialized successfully!" << std::endl;

    return 0;
}

// ==================== Load Resources ====================

bool WhisperInference::load_token_embeddings(const std::string& path) {
    std::cout << "[INFO] Loading token embeddings: " << path << std::endl;

    // Load from .npy file
    // For our specific file, we know the header is 128 bytes
    // (magic 10 + version 2 + header_len 2 + header 118 = 132, padded to 128)
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        std::cerr << "[ERROR] Failed to open: " << path << std::endl;
        return false;
    }

    // Skip the header (128 bytes for this specific file)
    fseek(fp, 128, SEEK_SET);

    // Read data
    size_t embed_size = vocab_size_ * d_model_;
    token_embeddings_.resize(embed_size);

    size_t read_count = fread(token_embeddings_.data(), sizeof(float),
                             embed_size, fp);
    fclose(fp);

    if (read_count != embed_size) {
        std::cerr << "[ERROR] Failed to read embeddings (expected "
                  << embed_size << ", got " << read_count << ")" << std::endl;
        return false;
    }

    std::cout << "[INFO]   Loaded embeddings: [" << vocab_size_
              << ", " << d_model_ << "]" << std::endl;
    return true;
}

bool WhisperInference::load_position_embeddings(const std::string& path) {
    std::cout << "[INFO] Loading position embeddings: " << path << std::endl;

    // Load from .npy file
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        std::cerr << "[ERROR] Failed to open: " << path << std::endl;
        return false;
    }

    // Skip the header (128 bytes)
    fseek(fp, 128, SEEK_SET);

    // Read data: [448, 512]
    size_t pos_embed_size = max_cache_len_ * d_model_;
    position_embeddings_.resize(pos_embed_size);

    size_t read_count = fread(position_embeddings_.data(), sizeof(float),
                             pos_embed_size, fp);
    fclose(fp);

    if (read_count != pos_embed_size) {
        std::cerr << "[ERROR] Failed to read position embeddings (expected "
                  << pos_embed_size << ", got " << read_count << ")" << std::endl;
        return false;
    }

    position_embeddings_loaded_ = true;
    std::cout << "[INFO]   Loaded position embeddings: [" << max_cache_len_
              << ", " << d_model_ << "]" << std::endl;

    // Debug: Print first 10 values
    if (debug_mode_) {
        std::cout << "[DEBUG]   Position embed [0] first 10: ";
        for (int i = 0; i < 10; i++) {
            std::cout << position_embeddings_[i] << " ";
        }
        std::cout << std::endl;
    }

    return true;
}

bool WhisperInference::load_mel_filters(const std::string& path) {
    std::cout << "[INFO] Loading mel filters: " << path << std::endl;

    mel_filters_.resize(N_MELS * MELS_FILTERS_SIZE);

    if (read_mel_filters(path.c_str(), mel_filters_.data(),
                         N_MELS * MELS_FILTERS_SIZE) != 0) {
        return false;
    }

    mel_filters_loaded_ = true;
    return true;
}

bool WhisperInference::load_vocab(const std::string& path) {
    std::cout << "[INFO] Loading vocabulary: " << path << std::endl;

    vocab_.resize(VOCAB_NUM);

    if (read_vocab(path.c_str(), vocab_.data()) != 0) {
        return false;
    }

    vocab_loaded_ = true;
    return true;
}

bool WhisperInference::load_encoder_dla(const std::string& path) {
    std::cout << "[INFO] Loading encoder DLA: " << path << std::endl;

    // Encoder: Input [1, 80, 3000], Output [1, 1500, 512]
    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, 80, 3000}  // mel spectrogram
    };

    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, 1500, 512}  // encoder output
    };

    encoder_executor_ = std::make_unique<NeuronExecutor>(
        path, input_shapes, output_shapes, "Encoder");

    if (!encoder_executor_->Initialize()) {
        std::cerr << "[ERROR] Failed to initialize encoder executor" << std::endl;
        encoder_executor_.reset();
        return false;
    }

    std::cout << "[INFO]   Encoder loaded successfully" << std::endl;
    return true;
}

bool WhisperInference::load_decoder_dla(const std::string& path) {
    std::cout << "[INFO] Loading decoder DLA (with KV Cache): " << path << std::endl;

    // Decoder with KV Cache has 8 inputs and 5 outputs:
    // Inputs:
    //   0. token_embeddings: [1, 1, 512] - single token
    //   1. encoder_output: [1, 1500, 512]
    //   2. past_self_keys: [6, 1, 448, 512]
    //   3. past_self_values: [6, 1, 448, 512]
    //   4. position_embed: [1, 1, 512]
    //   5. self_attn_mask: [1, 1, 1, 449]
    //   6. cached_cross_keys: [6, 1, 1500, 512]
    //   7. cached_cross_values: [6, 1, 1500, 512]
    //
    // Outputs:
    //   0. logits: [1, 1, 51865]
    //   1. new_self_keys: [6, 1, 1, 512]
    //   2. new_self_values: [6, 1, 1, 512]
    //   3. new_cross_keys: [6, 1, 1500, 512]
    //   4. new_cross_values: [6, 1, 1500, 512]

    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, 1, (uint32_t)d_model_},                    // 0. token_embeddings
        {1, 1500, (uint32_t)d_model_},                 // 1. encoder_output
        {(uint32_t)num_layers_, 1, (uint32_t)max_cache_len_, (uint32_t)d_model_}, // 2. past_self_keys
        {(uint32_t)num_layers_, 1, (uint32_t)max_cache_len_, (uint32_t)d_model_}, // 3. past_self_values
        {1, 1, (uint32_t)d_model_},                    // 4. position_embed
        {1, 1, 1, (uint32_t)max_cache_len_ + 1},      // 5. self_attn_mask (449)
        {(uint32_t)num_layers_, 1, 1500, (uint32_t)d_model_}, // 6. cached_cross_keys
        {(uint32_t)num_layers_, 1, 1500, (uint32_t)d_model_}, // 7. cached_cross_values
    };

    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, 1, VOCAB_NUM},                             // 0. logits
        {(uint32_t)num_layers_, 1, 1, (uint32_t)d_model_},        // 1. new_self_keys
        {(uint32_t)num_layers_, 1, 1, (uint32_t)d_model_},        // 2. new_self_values
        {(uint32_t)num_layers_, 1, 1500, (uint32_t)d_model_},     // 3. new_cross_keys
        {(uint32_t)num_layers_, 1, 1500, (uint32_t)d_model_},     // 4. new_cross_values
    };

    decoder_executor_ = std::make_unique<NeuronExecutor>(
        path, input_shapes, output_shapes, "DecoderKV");

    if (!decoder_executor_->Initialize()) {
        std::cerr << "[ERROR] Failed to initialize decoder executor" << std::endl;
        decoder_executor_.reset();
        return false;
    }

    // Initialize KV cache buffers
    past_self_keys_.resize(num_layers_ * 1 * max_cache_len_ * d_model_, 0.0f);
    past_self_values_.resize(num_layers_ * 1 * max_cache_len_ * d_model_, 0.0f);
    cached_cross_keys_.resize(num_layers_ * 1 * 1500 * d_model_, 0.0f);
    cached_cross_values_.resize(num_layers_ * 1 * 1500 * d_model_, 0.0f);

    // Note: Position embeddings are loaded in init(), not here!

    std::cout << "[INFO]   Decoder KV Cache loaded successfully" << std::endl;
    std::cout << "[INFO]   Cache buffers: " << std::endl;
    std::cout << "[INFO]     past_self_keys/values: ["
              << num_layers_ << ", 1, " << max_cache_len_ << ", " << d_model_ << "]" << std::endl;
    std::cout << "[INFO]     cached_cross_keys/values: ["
              << num_layers_ << ", 1, 1500, " << d_model_ << "]" << std::endl;

    return true;
}

// ==================== Run Inference ====================

std::string WhisperInference::run(const char* audio_file,
                                 const char* language,
                                 const char* /*task*/) {
    if (!initialized_) {
        std::cerr << "[ERROR] Not initialized" << std::endl;
        return "";
    }

    std::cout << "\n[INFO] Running inference on: " << audio_file << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();
    auto t1 = total_start;

    // Load audio
    audio_buffer_t audio;
    if (load_audio(audio_file, &audio) != 0) {
        std::cerr << "[ERROR] Failed to load audio" << std::endl;
        return "";
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto audio_load_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    // Save audio info for performance report
    int audio_num_frames = audio.num_frames;
    int audio_sample_rate = audio.sample_rate;

    // Compute mel spectrogram
    std::vector<float> mel_spec;
    audio_preprocess(&audio, mel_filters_.data(), mel_spec);
    free_audio(&audio);
    auto t3 = std::chrono::high_resolution_clock::now();
    auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

    // Run encoder
    std::vector<float> encoder_output;
    if (!run_encoder(mel_spec, encoder_output)) {
        std::cerr << "[ERROR] Encoder inference failed" << std::endl;
        return "";
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto encoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

    // Determine language token
    int language_token = WHISPER_TASK_EN;
    if (language && strcmp(language, "zh") == 0) {
        language_token = WHISPER_TASK_ZH;
    }

    // Run decoder (autoregressive)
    std::vector<int> tokens;
    if (!run_decoder(encoder_output, tokens, language_token)) {
        std::cerr << "[ERROR] Decoder inference failed" << std::endl;
        return "";
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    auto decoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4);

    // Decode tokens to text
    std::string text = decode_tokens(tokens, language_token);
    auto t6 = std::chrono::high_resolution_clock::now();
    auto decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - total_start);

    // Calculate performance metrics
    float audio_duration_sec = audio_num_frames / (float)audio_sample_rate;
    float total_time_sec = total_time.count() / 1000.0f;
    float rtf = total_time_sec / audio_duration_sec;

    // Calculate memory usage
    size_t encoder_memory = 1 * 1500 * 512 * sizeof(float);  // encoder output
    size_t decoder_kv_memory = (
        2 * num_layers_ * 1 * max_cache_len_ * d_model_ * sizeof(float) +  // self K,V
        2 * num_layers_ * 1 * 1500 * d_model_ * sizeof(float)              // cross K,V
    );
    size_t total_memory = encoder_memory + decoder_kv_memory;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Performance Test Report" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[Audio Info]" << std::endl;
    std::cout << "  Duration:        " << audio_duration_sec << " seconds" << std::endl;
    std::cout << "  Sample rate:     " << audio_sample_rate << " Hz" << std::endl;
    std::cout << "  Samples:         " << audio_num_frames << std::endl;

    std::cout << "\n[Timing Breakdown]" << std::endl;
    std::cout << "  Audio loading:   " << audio_load_time.count() << " ms" << std::endl;
    std::cout << "  Preprocessing:   " << preprocess_time.count() << " ms" << std::endl;
    std::cout << "  Encoder:         " << encoder_time.count() << " ms" << std::endl;
    std::cout << "  Decoder:         " << decoder_time.count() << " ms ("
              << tokens.size() << " tokens)" << std::endl;
    std::cout << "  Token decoding:  " << decode_time.count() << " ms" << std::endl;
    std::cout << "  TOTAL:           " << total_time.count() << " ms ("
              << total_time_sec << " seconds)" << std::endl;

    std::cout << "\n[Performance Metrics]" << std::endl;
    std::cout << "  RTF (Real-Time Factor):  " << std::fixed << std::setprecision(3)
              << rtf << "x" << std::endl;
    std::cout << "  Throughput:              "
              << (rtf > 0 ? 1.0f / rtf : 0.0f) << "x realtime" << std::endl;
    std::cout << "  Tokens/second:           "
              << (total_time_sec > 0 ? tokens.size() / total_time_sec : 0.0f)
              << " tokens/s" << std::endl;
    std::cout << "  Avg time/token:          "
              << (tokens.size() > 0 ? decoder_time.count() / (float)tokens.size() : 0.0f)
              << " ms/token" << std::endl;

    std::cout << "\n[Memory Usage]" << std::endl;
    std::cout << "  Encoder output:   " << (encoder_memory / 1024.0f / 1024.0f)
              << " MB" << std::endl;
    std::cout << "  Decoder KV cache: " << (decoder_kv_memory / 1024.0f / 1024.0f)
              << " MB" << std::endl;
    std::cout << "  Total runtime:    " << (total_memory / 1024.0f / 1024.0f)
              << " MB" << std::endl;

    std::cout << "\n[Model Info]" << std::endl;
    std::cout << "  Model:            Whisper Base" << std::endl;
    std::cout << "  KV Cache:         Enabled" << std::endl;
    std::cout << "  Max cache length: " << max_cache_len_ << std::endl;
    std::cout << "  Decoder layers:   " << num_layers_ << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Recognition Result" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << text << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[INFO] Inference completed in " << total_time.count()
              << " ms (RTF: " << std::fixed << std::setprecision(3) << rtf << "x)" << std::endl;

    return text;
}

// ==================== Encoder Inference ====================

bool WhisperInference::run_encoder(const std::vector<float>& mel_spec,
                                  std::vector<float>& encoder_output) {
    std::cout << "[INFO] Running encoder..." << std::endl;

    // Prepare input
    std::vector<const void*> inputs = {mel_spec.data()};

    // Prepare output buffer
    encoder_output.resize(1 * 1500 * 512);
    std::vector<void*> outputs = {encoder_output.data()};

    // Run inference
    if (!encoder_executor_->Run(inputs, outputs)) {
        std::cerr << "[ERROR] Encoder execution failed" << std::endl;
        return false;
    }

    std::cout << "[INFO] Encoder output shape: [1, 1500, 512]" << std::endl;

    // Debug: Print encoder output statistics
    float min_val = encoder_output[0], max_val = encoder_output[0], sum = 0.0f;
    for (size_t i = 0; i < encoder_output.size(); i++) {
        min_val = std::min(min_val, encoder_output[i]);
        max_val = std::max(max_val, encoder_output[i]);
        sum += encoder_output[i];
    }
    float mean = sum / encoder_output.size();
// DEBUG_CONVERT:     std::cout << "[DEBUG] Encoder output: min=" << min_val << ", max=" << max_val
// DEBUG_CONVERT:               << ", mean=" << mean << std::endl;
// DEBUG_CONVERT:     std::cout << "[DEBUG] First 10 encoder values: ";
// DEBUG_CONVERT:     for (int i = 0; i < 10; i++) {
// DEBUG_CONVERT:         std::cout << encoder_output[i] << " ";
// DEBUG_CONVERT:     }
// DEBUG_CONVERT:     std::cout << std::endl;

    return true;
}

// ==================== Decoder Inference (with KV Cache) ====================

bool WhisperInference::run_decoder(const std::vector<float>& encoder_output,
                                  std::vector<int>& tokens,
                                  int language_token) {
    std::cout << "[INFO] Running decoder with KV Cache (autoregressive greedy decoding)..." << std::endl;

    // Reset KV cache for new inference
    reset_kv_cache();

    // Initialize token sequence with special tokens (Python style)
    // [SOT, LANGUAGE, TRANSCRIBE, NO_TIMESTAMPS]
    std::vector<int> initial_tokens;
    initial_tokens.push_back(WHISPER_SOT);              // 50258 <|startoftranscript|>
    initial_tokens.push_back(language_token);           // 50259 <|en|> or 50260 <|zh|>
    initial_tokens.push_back(WHISPER_TASK_TRANScribe);  // 50359 <|transcribe|>
    initial_tokens.push_back(WHISPER_SPEAKER_END);      // 50363 <|notimestamps|>

    const char* lang_name = (language_token == WHISPER_TASK_ZH) ? "zh" : "en";
    std::cout << "[INFO] Language: " << lang_name << " (token=" << language_token << ")" << std::endl;
    std::cout << "[INFO] Initial tokens: [" << initial_tokens[0] << ", "
              << initial_tokens[1] << ", " << initial_tokens[2] << ", "
              << initial_tokens[3] << "]" << std::endl;

    // Buffers for decoder input/output
    std::vector<float> token_embed(d_model_);           // [1, 1, 512]
    std::vector<float> position_embed(d_model_);        // [1, 1, 512]
    std::vector<float> self_attn_mask(max_cache_len_ + 1);  // [1, 1, 1, 449]
    std::vector<float> logits(VOCAB_NUM);               // [1, 1, 51865]
    std::vector<float> new_self_keys(num_layers_ * d_model_);    // [6, 1, 1, 512]
    std::vector<float> new_self_values(num_layers_ * d_model_);  // [6, 1, 1, 512]
    std::vector<float> new_cross_keys(num_layers_ * 1500 * d_model_);   // [6, 1, 1500, 512]
    std::vector<float> new_cross_values(num_layers_ * 1500 * d_model_); // [6, 1, 1500, 512]

    tokens.clear();

    auto decoder_start = std::chrono::high_resolution_clock::now();
    long long total_npu_time_us = 0;
    long long total_embed_time_us = 0;

    // Phase 1: Process initial tokens (SOT sequence)
    std::cout << "[INFO] Phase 1: Processing " << initial_tokens.size() << " initial tokens..." << std::endl;

    for (size_t i = 0; i < initial_tokens.size(); i++) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        int token = initial_tokens[i];

        // Get token embedding
        lookup_embedding(token, token_embed, 0);

        // Get position embedding
        get_position_embedding(cache_len_, position_embed.data());

        // Create self-attention mask
        create_self_attn_mask(cache_len_, self_attn_mask.data());

        // Debug: Print first decoder call details
        if (debug_mode_ && i == 0) {
            std::cout << "[DEBUG] First decoder call (token=" << token << "):" << std::endl;
            std::cout << "[DEBUG]   Token embed first 10: ";
            for (int j = 0; j < 10; j++) std::cout << token_embed[j] << " ";
            std::cout << std::endl;
            std::cout << "[DEBUG]   Position embed first 10: ";
            for (int j = 0; j < 10; j++) std::cout << position_embed[j] << " ";
            std::cout << std::endl;
            std::cout << "[DEBUG]   Self attn mask [0:10]: ";
            for (int j = 0; j < 10; j++) std::cout << self_attn_mask[j] << " ";
            std::cout << std::endl;
            std::cout << "[DEBUG]   Self attn mask [446:449]: ";
            for (int j = 446; j < 449; j++) std::cout << self_attn_mask[j] << " ";
            std::cout << std::endl;
        }

        auto embed_done = std::chrono::high_resolution_clock::now();
        total_embed_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            embed_done - iter_start).count();

        // Prepare decoder inputs (8 inputs)
        std::vector<const void*> inputs;
        inputs.push_back(token_embed.data());         // 0. token_embeddings [1, 1, 512]
        inputs.push_back(encoder_output.data());      // 1. encoder_output [1, 1500, 512]
        inputs.push_back(past_self_keys_.data());     // 2. past_self_keys [6, 1, 448, 512]
        inputs.push_back(past_self_values_.data());   // 3. past_self_values [6, 1, 448, 512]
        inputs.push_back(position_embed.data());      // 4. position_embed [1, 1, 512]
        inputs.push_back(self_attn_mask.data());      // 5. self_attn_mask [1, 1, 1, 449]
        inputs.push_back(cached_cross_keys_.data());  // 6. cached_cross_keys [6, 1, 1500, 512]
        inputs.push_back(cached_cross_values_.data());// 7. cached_cross_values [6, 1, 1500, 512]

        // Prepare outputs (5 outputs)
        std::vector<void*> outputs;
        outputs.push_back(logits.data());             // 0. logits [1, 1, 51865]
        outputs.push_back(new_self_keys.data());      // 1. new_self_keys [6, 1, 1, 512]
        outputs.push_back(new_self_values.data());    // 2. new_self_values [6, 1, 1, 512]
        outputs.push_back(new_cross_keys.data());     // 3. new_cross_keys [6, 1, 1500, 512]
        outputs.push_back(new_cross_values.data());   // 4. new_cross_values [6, 1, 1500, 512]

        // Run decoder
        auto npu_start = std::chrono::high_resolution_clock::now();
        if (!decoder_executor_->Run(inputs, outputs)) {
            std::cerr << "[ERROR] Decoder execution failed at initial token " << i << std::endl;
            return false;
        }
        auto npu_done = std::chrono::high_resolution_clock::now();
        total_npu_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            npu_done - npu_start).count();

        // Update self-attention KV cache
        for (int layer = 0; layer < num_layers_; layer++) {
            // Copy new key/value to cache at position cache_len_
            float* dst_key = past_self_keys_.data() +
                            layer * max_cache_len_ * d_model_ +
                            cache_len_ * d_model_;
            float* dst_value = past_self_values_.data() +
                              layer * max_cache_len_ * d_model_ +
                              cache_len_ * d_model_;

            const float* src_key = new_self_keys.data() + layer * d_model_;
            const float* src_value = new_self_values.data() + layer * d_model_;

            std::memcpy(dst_key, src_key, d_model_ * sizeof(float));
            std::memcpy(dst_value, src_value, d_model_ * sizeof(float));
        }

        // Initialize cross-attention cache on first iteration
        if (!cross_cache_initialized_) {
            std::memcpy(cached_cross_keys_.data(), new_cross_keys.data(),
                       num_layers_ * 1500 * d_model_ * sizeof(float));
            std::memcpy(cached_cross_values_.data(), new_cross_values.data(),
                       num_layers_ * 1500 * d_model_ * sizeof(float));
            cross_cache_initialized_ = true;
        }

        cache_len_++;
    }

    // Debug: Print first few logits after Phase 1
    if (debug_mode_) {
        std::cout << "[DEBUG] After Phase 1, first 10 logits: ";
        for (int i = 0; i < 10; i++) {
            std::cout << logits[i] << " ";
        }
        std::cout << std::endl;
    }

    // Find argmax and check specific tokens
    int argmax_idx = 0;
    float argmax_val = logits[0];
    for (int i = 1; i < VOCAB_NUM; i++) {
        if (logits[i] > argmax_val) {
            argmax_val = logits[i];
            argmax_idx = i;
        }
    }
// DEBUG_CONVERT:     std::cout << "[DEBUG] Argmax token: " << argmax_idx << ", value: " << argmax_val << std::endl;
// DEBUG_CONVERT:     std::cout << "[DEBUG] EOT token (50257) value: " << logits[50257] << std::endl;
// DEBUG_CONVERT:     std::cout << "[DEBUG] Token 370 value: " << logits[370] << std::endl;

    std::cout << "[INFO] Phase 2: Generating text tokens (max " << (max_cache_len_ - cache_len_) << ")..." << std::endl;

    // Phase 2: Autoregressive generation
    int max_iterations = max_cache_len_ - cache_len_;
    int iteration = 0;

    while (iteration < max_iterations) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        // Get next token from previous logits
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < VOCAB_NUM; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }

        int next_token = max_idx;

// DEBUG_CONVERT:         std::cout << "[DEBUG] Iteration " << iteration << ": next_token=" << next_token
// DEBUG_CONVERT:                   << ", logit_value=" << max_val << std::endl;

        // Check for end of transcript
        if (next_token == WHISPER_EOT) {
            std::cout << "[INFO] EOT token detected after " << iteration << " tokens" << std::endl;
            break;
        }

        tokens.push_back(next_token);
        iteration++;

        // Get token embedding
        lookup_embedding(next_token, token_embed, 0);

        // Get position embedding
        get_position_embedding(cache_len_, position_embed.data());

        // Create self-attention mask
        create_self_attn_mask(cache_len_, self_attn_mask.data());

        auto embed_done = std::chrono::high_resolution_clock::now();
        total_embed_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            embed_done - iter_start).count();

        // Prepare inputs (same as Phase 1)
        std::vector<const void*> inputs;
        inputs.push_back(token_embed.data());
        inputs.push_back(encoder_output.data());
        inputs.push_back(past_self_keys_.data());
        inputs.push_back(past_self_values_.data());
        inputs.push_back(position_embed.data());
        inputs.push_back(self_attn_mask.data());
        inputs.push_back(cached_cross_keys_.data());
        inputs.push_back(cached_cross_values_.data());

        // Prepare outputs
        std::vector<void*> outputs;
        outputs.push_back(logits.data());
        outputs.push_back(new_self_keys.data());
        outputs.push_back(new_self_values.data());
        outputs.push_back(new_cross_keys.data());
        outputs.push_back(new_cross_values.data());

        // Run decoder
        auto npu_start = std::chrono::high_resolution_clock::now();
        if (!decoder_executor_->Run(inputs, outputs)) {
            std::cerr << "[ERROR] Decoder execution failed at iteration " << iteration << std::endl;
            return false;
        }
        auto npu_done = std::chrono::high_resolution_clock::now();
        total_npu_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            npu_done - npu_start).count();

        // Update self-attention KV cache
        for (int layer = 0; layer < num_layers_; layer++) {
            float* dst_key = past_self_keys_.data() +
                            layer * max_cache_len_ * d_model_ +
                            cache_len_ * d_model_;
            float* dst_value = past_self_values_.data() +
                              layer * max_cache_len_ * d_model_ +
                              cache_len_ * d_model_;

            const float* src_key = new_self_keys.data() + layer * d_model_;
            const float* src_value = new_self_values.data() + layer * d_model_;

            std::memcpy(dst_key, src_key, d_model_ * sizeof(float));
            std::memcpy(dst_value, src_value, d_model_ * sizeof(float));
        }

        cache_len_++;

        // Safety check
        if (cache_len_ >= max_cache_len_) {
            std::cout << "[WARN] Reached max cache length (" << max_cache_len_ << ")" << std::endl;
            break;
        }
    }

    auto decoder_end = std::chrono::high_resolution_clock::now();
    auto total_decoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        decoder_end - decoder_start);

    std::cout << "[INFO] Decoder generated " << tokens.size()
              << " tokens in " << (initial_tokens.size() + iteration) << " total iterations" << std::endl;
    std::cout << "[PERF] Decoder breakdown:" << std::endl;
    std::cout << "  Total decoder time:  " << total_decoder_time.count() << " ms" << std::endl;
    std::cout << "  NPU inference time:  " << total_npu_time_us / 1000.0 << " ms ("
              << (total_npu_time_us / 1000.0 / (initial_tokens.size() + iteration)) << " ms/iter)" << std::endl;
    std::cout << "  Embedding lookup:    " << total_embed_time_us / 1000.0 << " ms" << std::endl;
    std::cout << "  Overhead:            "
              << (total_decoder_time.count() - total_npu_time_us / 1000.0 - total_embed_time_us / 1000.0)
              << " ms" << std::endl;

    return true;
}

// ==================== Decode Tokens ====================

std::string WhisperInference::decode_tokens(const std::vector<int>& tokens,
                                          int task_code) {
    if (!vocab_loaded_) {
        // No vocabulary loaded, return token IDs as string
        std::string result = "[Tokens: ";
        for (size_t i = 0; i < std::min(tokens.size(), size_t(10)); i++) {
            result += std::to_string(tokens[i]) + " ";
        }
        if (tokens.size() > 10) {
            result += "...]";
        } else {
            result += "]";
        }
        return result;
    }

    std::string text;

    // Decode each token (vocab tokens are base64 encoded)
    for (int token : tokens) {
        if (token >= 0 && token < VOCAB_NUM) {
            std::string token_str = vocab_[token].token;

            // Skip special tokens (they start with "<|")
            if (token_str.length() >= 2 && token_str[0] == '<' && token_str[1] == '|') {
                continue;  // Don't add special tokens to output
            }

            // Base64 decode the token
            std::string decoded = base64_decode(token_str);
            text += decoded;
        }
    }

    // Post-processing: BPE uses Ġ (U+0120) to represent spaces
    replace_substr(text, "\u0120", " ");
    replace_substr(text, "", "");
    replace_substr(text, "\n", "");

    return text;
}

// ==================== Embedding Lookup ====================

void WhisperInference::lookup_embedding(int token_id,
                                       std::vector<float>& embeddings,
                                       size_t offset) {
    if (token_id >= 0 && token_id < vocab_size_) {
        std::memcpy(embeddings.data() + offset,
                   token_embeddings_.data() + token_id * d_model_,
                   d_model_ * sizeof(float));
    } else {
        // Out of range, zero embedding
        std::fill(embeddings.data() + offset,
                 embeddings.data() + offset + d_model_, 0.0f);
    }
}

// ==================== KV Cache Management ====================

void WhisperInference::reset_kv_cache() {
    // Reset cache buffers to zero
    std::fill(past_self_keys_.begin(), past_self_keys_.end(), 0.0f);
    std::fill(past_self_values_.begin(), past_self_values_.end(), 0.0f);
    std::fill(cached_cross_keys_.begin(), cached_cross_keys_.end(), 0.0f);
    std::fill(cached_cross_values_.begin(), cached_cross_values_.end(), 0.0f);

    cache_len_ = 0;
    cross_cache_initialized_ = false;
}

void WhisperInference::create_self_attn_mask(int cache_len, float* mask) {
    // Self-attention mask: [1, 1, 1, 449]
    // Valid positions (past cache + current token) = 0.0
    // Invalid positions (unused cache) = -1e9

    int mask_len = max_cache_len_ + 1;  // 449

    // Set all to -1e9 (invalid)
    std::fill(mask, mask + mask_len, -1e9f);

    // Set valid positions to 0.0
    for (int i = 0; i < cache_len; i++) {
        mask[i] = 0.0f;  // Past cache positions
    }
    mask[max_cache_len_] = 0.0f;  // Current token position

    // Note: In practice, the model uses mask[cache_len] for the current token,
    // but we follow the Python implementation which uses a fixed position
}

void WhisperInference::get_position_embedding(int position, float* output) {
    // Get position embedding for the given position
    // Output: [1, 1, 512]

    if (position >= 0 && position < max_cache_len_ && position_embeddings_loaded_) {
        std::memcpy(output,
                   position_embeddings_.data() + position * d_model_,
                   d_model_ * sizeof(float));
    } else {
        // Debug: why are we here?
        if (debug_mode_ && position == 0) {  // Only print for first call in debug mode
            std::cerr << "[DEBUG] get_position_embedding returning zeros: position=" << position
                      << ", max_cache_len=" << max_cache_len_
                      << ", loaded=" << position_embeddings_loaded_
                      << ", vector_size=" << position_embeddings_.size() << std::endl;
        }
        // Out of range or not loaded, use zero
        std::fill(output, output + d_model_, 0.0f);
    }
}

// ==================== Release ====================

void WhisperInference::release() {
    encoder_executor_.reset();
    decoder_executor_.reset();
    token_embeddings_.clear();
    mel_filters_.clear();
    vocab_.clear();
    past_self_keys_.clear();
    past_self_values_.clear();
    cached_cross_keys_.clear();
    cached_cross_values_.clear();
    position_embeddings_.clear();
    initialized_ = false;

    std::cout << "[INFO] Whisper Inference released" << std::endl;
}
