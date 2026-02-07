/**
 * Whisper Inference Engine Implementation for MTK NPU
 */

#include "whisper_inference.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>

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

    std::cout << "[INFO] Initializing Whisper Inference from: "
              << model_dir << std::endl;

    std::string base_dir(model_dir);

    // Load token embeddings
    if (!load_token_embeddings(base_dir + "/token_embedding.npy")) {
        std::cerr << "[ERROR] Failed to load token embeddings" << std::endl;
        return -1;
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
    if (!load_encoder_dla(base_dir + "/encoder_base_80x3000_MT8371.dla")) {
        std::cerr << "[ERROR] Failed to load encoder DLA" << std::endl;
        return -1;
    }

    // Load decoder DLA
    if (!load_decoder_dla(base_dir + "/decoder_base_448_MT8371.dla")) {
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
    std::cout << "[INFO] Loading decoder DLA: " << path << std::endl;

    // Decoder expects token embeddings as input (embedding lookup done in C++)
    // Input 0: token_embeddings [1, 448, 512] (float32)
    // Input 1: encoder_output [1, 1500, 512] (float32)
    // Output: logits [1, 448, 51865] (float32)

    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, MAX_TOKENS, (uint32_t)d_model_},  // token embeddings
        {1, 1500, (uint32_t)d_model_}         // encoder output
    };

    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, MAX_TOKENS, VOCAB_NUM}  // logits
    };

    decoder_executor_ = std::make_unique<NeuronExecutor>(
        path, input_shapes, output_shapes, "Decoder");

    if (!decoder_executor_->Initialize()) {
        std::cerr << "[ERROR] Failed to initialize decoder executor" << std::endl;
        decoder_executor_.reset();
        return false;
    }

    std::cout << "[INFO]   Decoder loaded successfully" << std::endl;
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

    std::cout << "\n[PERF] Performance breakdown:" << std::endl;
    std::cout << "  Audio loading:   " << audio_load_time.count() << " ms" << std::endl;
    std::cout << "  Preprocessing:   " << preprocess_time.count() << " ms" << std::endl;
    std::cout << "  Encoder:         " << encoder_time.count() << " ms" << std::endl;
    std::cout << "  Decoder:         " << decoder_time.count() << " ms ("
              << tokens.size() << " tokens)" << std::endl;
    std::cout << "  Token decoding:  " << decode_time.count() << " ms" << std::endl;
    std::cout << "  TOTAL:           " << total_time.count() << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "[INFO] Inference completed in " << total_time.count()
              << " ms" << std::endl;
    std::cout << "[INFO] Result: \"" << text << "\"" << std::endl;

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
    return true;
}

// ==================== Decoder Inference ====================

bool WhisperInference::run_decoder(const std::vector<float>& encoder_output,
                                  std::vector<int>& tokens,
                                  int language_token) {
    std::cout << "[INFO] Running decoder (autoregressive greedy decoding)..." << std::endl;

    // Initialize token sequence with special tokens (Python style)
    // [SOT, LANGUAGE, TRANSCRIBE, NO_TIMESTAMPS]
    std::vector<int> token_sequence;
    token_sequence.push_back(WHISPER_SOT);           // 50258 <|startoftranscript|>
    token_sequence.push_back(language_token);        // 50259 <|en|> or 50260 <|zh|>
    token_sequence.push_back(WHISPER_TASK_TRANScribe); // 50359 <|transcribe|>
    token_sequence.push_back(WHISPER_SPEAKER_END);   // 50363 <|notimestamps|>

    const char* lang_name = (language_token == WHISPER_TASK_ZH) ? "zh" : "en";
    std::cout << "[INFO] Language: " << lang_name << " (token=" << language_token << ")" << std::endl;
    std::cout << "[INFO] Initial tokens: [" << token_sequence[0] << ", "
              << token_sequence[1] << ", " << token_sequence[2] << ", "
              << token_sequence[3] << "]" << std::endl;

    // Buffers for decoder input/output
    std::vector<float> token_embeddings_padded(MAX_TOKENS * d_model_, 0.0f);
    std::vector<float> logits(MAX_TOKENS * VOCAB_NUM);

    tokens.clear();
    int next_token = WHISPER_SOT;
    int iteration = 0;
    int max_iterations = MAX_TOKENS - 4;  // Max tokens to generate

    auto decoder_start = std::chrono::high_resolution_clock::now();
    long long total_npu_time_us = 0;
    long long total_embed_time_us = 0;

    while (next_token != WHISPER_EOT && iteration < max_iterations) {
        iteration++;

        auto iter_start = std::chrono::high_resolution_clock::now();

        // Get current sequence length
        int seq_len = token_sequence.size();

        // Get embeddings for current token sequence
        for (int i = 0; i < seq_len; i++) {
            lookup_embedding(token_sequence[i], token_embeddings_padded, i * d_model_);
        }

        // Zero-pad the rest
        if (seq_len < MAX_TOKENS) {
            std::fill(token_embeddings_padded.begin() + seq_len * d_model_,
                     token_embeddings_padded.end(), 0.0f);
        }

        auto embed_done = std::chrono::high_resolution_clock::now();
        total_embed_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            embed_done - iter_start).count();

        // Prepare decoder inputs
        std::vector<const void*> inputs;
        inputs.push_back(token_embeddings_padded.data());
        inputs.push_back(encoder_output.data());

        // Prepare output
        std::vector<void*> outputs;
        outputs.push_back(logits.data());

        // Run decoder
        auto npu_start = std::chrono::high_resolution_clock::now();
        if (!decoder_executor_->Run(inputs, outputs)) {
            std::cerr << "[ERROR] Decoder execution failed at iteration "
                      << iteration << std::endl;
            return false;
        }
        auto npu_done = std::chrono::high_resolution_clock::now();
        total_npu_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            npu_done - npu_start).count();

        // Get logits at current position (last real token position, not padded)
        // Python: logits[0, len(tokens)-1, :]
        int current_pos = seq_len - 1;
        float* current_logits = logits.data() + current_pos * VOCAB_NUM;

        // Find argmax
        int max_idx = 0;
        float max_val = current_logits[0];
        for (int i = 1; i < VOCAB_NUM; i++) {
            if (current_logits[i] > max_val) {
                max_val = current_logits[i];
                max_idx = i;
            }
        }

        next_token = max_idx;

        // Add to sequence
        token_sequence.push_back(next_token);
        tokens.push_back(next_token);

        // Check for end of transcript
        if (next_token == WHISPER_EOT) {
            std::cout << "[INFO] EOT token detected, stopping" << std::endl;
            break;
        }
    }

    auto decoder_end = std::chrono::high_resolution_clock::now();
    auto total_decoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        decoder_end - decoder_start);

    std::cout << "[INFO] Decoder generated " << tokens.size()
              << " tokens in " << iteration << " iterations" << std::endl;
    std::cout << "[PERF] Decoder breakdown:" << std::endl;
    std::cout << "  Total decoder time:  " << total_decoder_time.count() << " ms" << std::endl;
    std::cout << "  NPU inference time:  " << total_npu_time_us / 1000.0 << " ms ("
              << (total_npu_time_us / 1000.0 / iteration) << " ms/iter)" << std::endl;
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

// ==================== Release ====================

void WhisperInference::release() {
    encoder_executor_.reset();
    decoder_executor_.reset();
    token_embeddings_.clear();
    mel_filters_.clear();
    vocab_.clear();
    initialized_ = false;

    std::cout << "[INFO] Whisper Inference released" << std::endl;
}
