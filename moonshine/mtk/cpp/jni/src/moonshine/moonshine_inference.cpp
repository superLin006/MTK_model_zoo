/* moonshine_inference.cpp - Moonshine MTK NPU Inference Implementation */

#include "moonshine_inference.h"
#include "utils/audio_utils.h"
#include "executor/NeuronExecutor.h"
#include "common/Log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>

namespace moonshine {

// ===================== Destructor =====================

MoonshineInference::~MoonshineInference() {
    encoder_executor_.reset();
    decoder_executor_.reset();
}

// ===================== Initialize =====================

bool MoonshineInference::Initialize(
    const std::string& encoder_dla_path,
    const std::string& decoder_dla_path,
    const std::string& embed_tokens_path,
    const std::string& pos_emb_path,
    const std::string& proj_weight_path,
    const std::string& log_k_path,
    const std::string& vocab_path)
{
    LOG(INFO) << "[Moonshine] Initializing...";

    // Load weights
    if (!LoadWeights(embed_tokens_path, pos_emb_path, proj_weight_path, log_k_path)) {
        LOG(ERROR) << "[Moonshine] Failed to load weights";
        return false;
    }

    // Load vocab
    if (!LoadVocab(vocab_path)) {
        LOG(ERROR) << "[Moonshine] Failed to load vocab";
        return false;
    }

    // Precompute RoPE tables
    rope_table_len_ = MAX_NEW_TOKENS + 10;
    rope_cos_.resize(rope_table_len_ * ROPE_DIM);
    rope_sin_.resize(rope_table_len_ * ROPE_DIM);

    int half_rot = ROPE_DIM / 2;  // 16
    // inv_freq[i] = 1.0 / (10000^(2i/ROPE_DIM)), i=0..15
    for (int pos = 0; pos < rope_table_len_; pos++) {
        for (int i = 0; i < half_rot; i++) {
            float inv_freq = 1.0f / std::pow(10000.0f, 2.0f * i / ROPE_DIM);
            float freq = static_cast<float>(pos) * inv_freq;
            float c = std::cos(freq);
            float s = std::sin(freq);
            // interleaved: cos_half[i] repeated twice → positions 2i and 2i+1
            rope_cos_[pos * ROPE_DIM + 2 * i]     = c;
            rope_cos_[pos * ROPE_DIM + 2 * i + 1] = c;
            rope_sin_[pos * ROPE_DIM + 2 * i]     = s;
            rope_sin_[pos * ROPE_DIM + 2 * i + 1] = s;
        }
    }

    LOG(INFO) << "[Moonshine] RoPE table precomputed, len=" << rope_table_len_;

    // Initialize NPU executors
    std::string npu_options = "--apusys-config \"{ \\\"high_addr\\\": true }\"";

    encoder_executor_ = std::make_unique<mtk::neuropilot::NeuronExecutor>(
        "moonshine_encoder", encoder_dla_path, npu_options);

    if (!encoder_executor_->Initialized()) {
        LOG(ERROR) << "[Moonshine] Failed to initialize encoder executor";
        return false;
    }
    LOG(INFO) << "[Moonshine] Encoder executor initialized";

    decoder_executor_ = std::make_unique<mtk::neuropilot::NeuronExecutor>(
        "moonshine_decoder", decoder_dla_path, npu_options);

    if (!decoder_executor_->Initialized()) {
        LOG(ERROR) << "[Moonshine] Failed to initialize decoder executor";
        return false;
    }
    LOG(INFO) << "[Moonshine] Decoder executor initialized";

    initialized_ = true;
    LOG(INFO) << "[Moonshine] Initialization complete";
    return true;
}

// ===================== Load Weights =====================

bool MoonshineInference::LoadWeights(
    const std::string& embed_tokens_path,
    const std::string& pos_emb_path,
    const std::string& proj_weight_path,
    const std::string& log_k_path)
{
    LOG(INFO) << "[Moonshine] Loading embed_tokens from " << embed_tokens_path;
    if (!LoadNpy(embed_tokens_path, embed_tokens_)) return false;
    LOG(INFO) << "[Moonshine] embed_tokens: " << embed_tokens_.size()
              << " floats (" << embed_tokens_.size() / DECODER_HIDDEN << " tokens)";

    LOG(INFO) << "[Moonshine] Loading pos_emb from " << pos_emb_path;
    if (!LoadNpy(pos_emb_path, pos_emb_)) return false;
    LOG(INFO) << "[Moonshine] pos_emb: " << pos_emb_.size() << " floats";

    LOG(INFO) << "[Moonshine] Loading proj_weight from " << proj_weight_path;
    if (!LoadNpy(proj_weight_path, proj_weight_)) return false;
    LOG(INFO) << "[Moonshine] proj_weight: " << proj_weight_.size() << " floats";

    LOG(INFO) << "[Moonshine] Loading log_k from " << log_k_path;
    std::vector<float> log_k_vec;
    if (!LoadNpy(log_k_path, log_k_vec)) return false;
    log_k_ = log_k_vec[0];
    LOG(INFO) << "[Moonshine] log_k=" << log_k_ << " k=" << std::exp(log_k_);

    return true;
}

// ===================== Load Vocab =====================

bool MoonshineInference::LoadVocab(const std::string& vocab_path) {
    LOG(INFO) << "[Moonshine] Loading vocab from " << vocab_path;

    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        LOG(ERROR) << "[Moonshine] Cannot open vocab file: " << vocab_path;
        return false;
    }

    // Initialize vocab with empty strings
    vocab_.resize(VOCAB_SIZE, "");

    // Format: id<TAB>piece\n
    std::string line;
    int count = 0;
    while (std::getline(f, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        int id = std::stoi(line.substr(0, tab));
        std::string piece = line.substr(tab + 1);
        if (id >= 0 && id < VOCAB_SIZE) {
            vocab_[id] = piece;
            count++;
        }
    }

    LOG(INFO) << "[Moonshine] Vocab loaded: " << count << " tokens";
    return count > 0;
}

// ===================== Transcribe =====================

std::string MoonshineInference::Transcribe(const std::string& audio_path,
                                            TokenCallback callback) {
    if (!initialized_) {
        LOG(ERROR) << "[Moonshine] Not initialized";
        return "";
    }

    // Load audio
    std::vector<float> samples;
    int sample_rate = 0;
    if (!LoadWav(audio_path, samples, sample_rate)) {
        LOG(ERROR) << "[Moonshine] Failed to load audio: " << audio_path;
        return "";
    }

    LOG(INFO) << "[Moonshine] Audio: " << samples.size() << " samples @ "
              << sample_rate << " Hz (" << samples.size() / sample_rate << "s)";

    return TranscribeSamples(samples, callback);
}

std::string MoonshineInference::TranscribeSamples(const std::vector<float>& samples,
                                                    TokenCallback callback) {
    if (!initialized_) return "";

    // ======= Step 1: Preprocess =======
    std::vector<float> frames;
    PreprocessAudio(samples, log_k_, frames);
    LOG(INFO) << "[Moonshine] Preprocessed: " << frames.size()
              << " floats (" << NUM_FRAMES << "x" << FRAME_LEN << ")";

    // ======= Step 2: Encoder =======
    std::vector<float> encoder_out;  // [1, 500, 620]
    auto t_enc_start = std::chrono::high_resolution_clock::now();
    if (!RunEncoder(frames, encoder_out)) {
        LOG(ERROR) << "[Moonshine] Encoder failed";
        return "";
    }
    auto t_enc_end = std::chrono::high_resolution_clock::now();
    encoder_time_ms = std::chrono::duration<double, std::milli>(t_enc_end - t_enc_start).count();
    LOG(INFO) << "[Moonshine] Encoder done in " << encoder_time_ms << " ms";
    LOG(INFO) << "[Moonshine] Encoder output: " << encoder_out.size() << " floats";

    // ======= Step 3: Adapter Projection =======
    std::vector<float> encoder_proj;  // [1, 500, 512]
    RunAdapterProjection(encoder_out, encoder_proj);
    LOG(INFO) << "[Moonshine] Adapter projection done: " << encoder_proj.size() << " floats";

    // ======= Step 4: Prepare decoder =======
    // KV cache: [NUM_LAYERS, 1, MAX_DEC_LEN, DECODER_HIDDEN]
    size_t kv_size = NUM_LAYERS * 1 * MAX_DEC_LEN * DECODER_HIDDEN;
    std::vector<float> past_keys(kv_size, 0.0f);
    std::vector<float> past_values(kv_size, 0.0f);

    // Encoder attention mask: [1,1,1,T_ENC] all zeros (all positions valid)
    std::vector<float> enc_attn_mask(1 * 1 * 1 * T_ENC, 0.0f);

    // ======= Step 5: Decoder loop =======
    std::vector<int> token_ids = {BOS_TOKEN};
    int generated = 0;

    auto t_dec_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < MAX_NEW_TOKENS; step++) {
        int current_token = token_ids.back();

        // Embed token: [1, 1, 512]
        std::vector<float> decoder_embed(1 * 1 * DECODER_HIDDEN);
        const float* emb_row = embed_tokens_.data() + current_token * DECODER_HIDDEN;
        std::copy(emb_row, emb_row + DECODER_HIDDEN, decoder_embed.data());

        // RoPE cos/sin for current position: [1, 1, 32]
        std::vector<float> cos_cur(1 * 1 * ROPE_DIM);
        std::vector<float> sin_cur(1 * 1 * ROPE_DIM);
        if (step < rope_table_len_) {
            std::copy(rope_cos_.data() + step * ROPE_DIM,
                      rope_cos_.data() + (step + 1) * ROPE_DIM,
                      cos_cur.data());
            std::copy(rope_sin_.data() + step * ROPE_DIM,
                      rope_sin_.data() + (step + 1) * ROPE_DIM,
                      sin_cur.data());
        }

        // Self-attention mask: [1, 1, 1, MAX_DEC_LEN+1]
        // Positions 0..step-1 valid (history cache), last position valid (current token)
        // All others = -1e9
        int mask_len = MAX_DEC_LEN + 1;
        std::vector<float> attn_mask(1 * 1 * 1 * mask_len, -1e9f);
        for (int i = 0; i < step; i++) {
            attn_mask[i] = 0.0f;  // history KV cache positions
        }
        attn_mask[mask_len - 1] = 0.0f;  // current token position

        // Decoder NPU step
        std::vector<float> logits(1 * 1 * VOCAB_SIZE);
        // new_keys/new_values: [NUM_LAYERS, 1, 1, DECODER_HIDDEN]
        std::vector<float> new_keys(NUM_LAYERS * 1 * 1 * DECODER_HIDDEN);
        std::vector<float> new_values(NUM_LAYERS * 1 * 1 * DECODER_HIDDEN);

        if (!RunDecoderStep(decoder_embed, encoder_proj, past_keys, past_values,
                             cos_cur, sin_cur, attn_mask, enc_attn_mask,
                             logits, new_keys, new_values)) {
            LOG(ERROR) << "[Moonshine] Decoder step " << step << " failed";
            break;
        }

        // Update KV cache: past_keys[layer, 0, step, :] = new_keys[layer, 0, 0, :]
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            const float* nk = new_keys.data() + layer * DECODER_HIDDEN;
            const float* nv = new_values.data() + layer * DECODER_HIDDEN;
            float* pk = past_keys.data() + layer * (MAX_DEC_LEN * DECODER_HIDDEN) + step * DECODER_HIDDEN;
            float* pv = past_values.data() + layer * (MAX_DEC_LEN * DECODER_HIDDEN) + step * DECODER_HIDDEN;
            std::copy(nk, nk + DECODER_HIDDEN, pk);
            std::copy(nv, nv + DECODER_HIDDEN, pv);
        }

        // Greedy decode
        int next_token = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());

        token_ids.push_back(next_token);
        generated++;

        if (step == 0) {
            LOG(INFO) << "[Moonshine] Step 0: next_token=" << next_token;
        }

        if (next_token == EOS_TOKEN) {
            LOG(INFO) << "[Moonshine] EOS at step " << step;
            break;
        }

        // Optional streaming
        if (callback) {
            std::string piece = DecodeToken(next_token);
            if (!piece.empty()) {
                callback(piece);
            }
        }
    }

    auto t_dec_end = std::chrono::high_resolution_clock::now();
    decoder_total_ms = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();
    num_tokens = generated;
    decoder_per_step_ms = (generated > 0) ? (decoder_total_ms / generated) : 0.0;

    LOG(INFO) << "[Moonshine] Decoder done: " << generated << " tokens in "
              << decoder_total_ms << " ms (" << decoder_per_step_ms << " ms/token)";

    // Decode text
    return TokenIdsToText(token_ids);
}

// ===================== RunEncoder =====================

bool MoonshineInference::RunEncoder(const std::vector<float>& frames,
                                     std::vector<float>& encoder_out) {
    // Input: [1, 2000, 80]
    // Output: [1, 500, 620]
    size_t out_size = 1 * T_ENC * ENCODER_HIDDEN;
    encoder_out.resize(out_size);

    using namespace mtk::neuropilot;
    TensorBuffer input_buf{
        .data = const_cast<float*>(frames.data()),
        .bytes = frames.size() * sizeof(float),
        .type = kFloat32
    };
    TensorBuffer output_buf{
        .data = encoder_out.data(),
        .bytes = encoder_out.size() * sizeof(float),
        .type = kFloat32
    };

    return encoder_executor_->RunForMultipleInputsOutputs({input_buf}, {output_buf});
}

// ===================== RunAdapterProjection =====================

void MoonshineInference::RunAdapterProjection(const std::vector<float>& encoder_out,
                                               std::vector<float>& encoder_proj) {
    // encoder_out: [1, T_ENC, 620]
    // pos_emb_weight: [4096, 620] → take first T_ENC rows
    // proj_weight: [512, 620]  (Linear no bias)
    // Output: [1, T_ENC, 512]
    //
    // Step 1: enc = encoder_out + pos_emb[0:T_ENC, :]
    // Step 2: enc_proj = enc @ proj_weight.T  → [T_ENC, 512]

    encoder_proj.resize(1 * T_ENC * DECODER_HIDDEN, 0.0f);

    for (int t = 0; t < T_ENC; t++) {
        const float* enc_row = encoder_out.data() + t * ENCODER_HIDDEN;
        const float* pos_row = pos_emb_.data() + t * ENCODER_HIDDEN;
        float* out_row = encoder_proj.data() + t * DECODER_HIDDEN;

        // Add pos_emb: tmp[620] = enc_row + pos_row
        // Then multiply by proj_weight.T: out[512] = tmp[620] @ proj[512x620].T
        // proj_weight: [512, 620], so out[j] = sum(tmp[k] * proj_weight[j*620+k])

        for (int j = 0; j < DECODER_HIDDEN; j++) {
            float acc = 0.0f;
            const float* pw = proj_weight_.data() + j * ENCODER_HIDDEN;
            for (int k = 0; k < ENCODER_HIDDEN; k++) {
                acc += (enc_row[k] + pos_row[k]) * pw[k];
            }
            out_row[j] = acc;
        }
    }
}

// ===================== RunDecoderStep =====================

bool MoonshineInference::RunDecoderStep(
    const std::vector<float>& decoder_embed,  // [1, 1, 512]
    const std::vector<float>& encoder_proj,   // [1, 500, 512]
    const std::vector<float>& past_keys,      // [10, 1, 128, 512]
    const std::vector<float>& past_values,    // [10, 1, 128, 512]
    const std::vector<float>& cos_cur,        // [1, 1, 32]
    const std::vector<float>& sin_cur,        // [1, 1, 32]
    const std::vector<float>& attn_mask,      // [1, 1, 1, 129]
    const std::vector<float>& enc_attn_mask,  // [1, 1, 1, 500]
    std::vector<float>& logits,               // [1, 1, 32768]
    std::vector<float>& new_keys,             // [10, 1, 1, 512]
    std::vector<float>& new_values)           // [10, 1, 1, 512]
{
    using namespace mtk::neuropilot;

    // Input order (from model_info.json):
    // 0: decoder_embed    [1, 1, 512]
    // 1: encoder_out      [1, 500, 512]
    // 2: past_keys        [10, 1, 128, 512]
    // 3: past_values      [10, 1, 128, 512]
    // 4: cos              [1, 1, 32]
    // 5: sin              [1, 1, 32]
    // 6: attn_mask        [1, 1, 1, 129]
    // 7: enc_attn_mask    [1, 1, 1, 500]

    std::vector<TensorBuffer> inputs = {
        {const_cast<float*>(decoder_embed.data()),  decoder_embed.size()  * sizeof(float), kFloat32},
        {const_cast<float*>(encoder_proj.data()),   encoder_proj.size()   * sizeof(float), kFloat32},
        {const_cast<float*>(past_keys.data()),      past_keys.size()      * sizeof(float), kFloat32},
        {const_cast<float*>(past_values.data()),    past_values.size()    * sizeof(float), kFloat32},
        {const_cast<float*>(cos_cur.data()),        cos_cur.size()        * sizeof(float), kFloat32},
        {const_cast<float*>(sin_cur.data()),        sin_cur.size()        * sizeof(float), kFloat32},
        {const_cast<float*>(attn_mask.data()),      attn_mask.size()      * sizeof(float), kFloat32},
        {const_cast<float*>(enc_attn_mask.data()),  enc_attn_mask.size()  * sizeof(float), kFloat32},
    };

    // Output order:
    // 0: logits      [1, 1, 32768]
    // 1: new_keys    [10, 1, 1, 512]
    // 2: new_values  [10, 1, 1, 512]
    std::vector<TensorBuffer> outputs = {
        {logits.data(),     logits.size()     * sizeof(float), kFloat32},
        {new_keys.data(),   new_keys.size()   * sizeof(float), kFloat32},
        {new_values.data(), new_values.size() * sizeof(float), kFloat32},
    };

    return decoder_executor_->RunForMultipleInputsOutputs(inputs, outputs);
}

// ===================== Token Decoding =====================

std::string MoonshineInference::DecodeToken(int token_id) {
    if (token_id < 0 || token_id >= VOCAB_SIZE) return "";
    if (token_id == BOS_TOKEN || token_id == EOS_TOKEN) return "";

    const std::string& piece = vocab_[token_id];
    if (piece.empty()) return "";

    // Sentencepiece uses \u2581 (▁ = U+2581) as space prefix
    // Replace \u2581 (UTF-8: \xe2\x96\x81) with space
    std::string result;
    result.reserve(piece.size());
    for (size_t i = 0; i < piece.size(); ) {
        if (i + 2 < piece.size() &&
            (unsigned char)piece[i] == 0xe2 &&
            (unsigned char)piece[i+1] == 0x96 &&
            (unsigned char)piece[i+2] == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += piece[i];
            i++;
        }
    }
    return result;
}

std::string MoonshineInference::TokenIdsToText(const std::vector<int>& token_ids) {
    std::string text;
    // Skip BOS (first token), decode the rest
    for (size_t i = 1; i < token_ids.size(); i++) {
        int tid = token_ids[i];
        if (tid == EOS_TOKEN) break;
        text += DecodeToken(tid);
    }

    // Trim leading space
    if (!text.empty() && text[0] == ' ') {
        text = text.substr(1);
    }

    return text;
}

} // namespace moonshine
