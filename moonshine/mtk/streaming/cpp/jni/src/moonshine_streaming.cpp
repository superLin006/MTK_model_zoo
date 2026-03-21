/* moonshine_streaming.cpp - Moonshine MTK NPU Streaming Inference Implementation
 *
 * Sliding window strategy:
 *   - Accumulate raw audio samples into frame_buffer_ (80 samples/frame)
 *   - Every CHUNK_FRAMES=160 frames → run Encoder NPU → get 40 encoded frames
 *   - Accumulate encoder output in enc_buffer_
 *   - When enc_frame_count_ >= TRIGGER_ENC_FRAMES → run Decoder → return text
 */

#include "moonshine_streaming.h"
#include "utils/audio_utils.h"
#include "executor/NeuronExecutor.h"
#include "common/Log.h"
#include "vad/silero_vad_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>

namespace moonshine {

// ===================== Destructor =====================

MoonshineStreamingEngine::~MoonshineStreamingEngine() {
    encoder_executor_.reset();
    decoder_executor_.reset();
}

// ===================== Initialize =====================

bool MoonshineStreamingEngine::Initialize(
    const std::string& encoder_chunk_dla,
    const std::string& decoder_dla,
    const std::string& embed_tokens_path,
    const std::string& proj_weight_path,
    const std::string& log_k_path,
    const std::string& vocab_path)
{
    LOG(INFO) << "[Streaming] Initializing...";

    // Load embed_tokens
    LOG(INFO) << "[Streaming] Loading embed_tokens from " << embed_tokens_path;
    if (!LoadNpy(embed_tokens_path, embed_tokens_)) {
        LOG(ERROR) << "[Streaming] Failed to load embed_tokens";
        return false;
    }
    LOG(INFO) << "[Streaming] embed_tokens: " << embed_tokens_.size()
              << " floats (" << embed_tokens_.size() / DECODER_HIDDEN_S << " tokens)";

    // Load proj_weight
    LOG(INFO) << "[Streaming] Loading proj_weight from " << proj_weight_path;
    if (!LoadNpy(proj_weight_path, proj_weight_)) {
        LOG(ERROR) << "[Streaming] Failed to load proj_weight";
        return false;
    }
    LOG(INFO) << "[Streaming] proj_weight: " << proj_weight_.size() << " floats";

    // Load log_k
    LOG(INFO) << "[Streaming] Loading log_k from " << log_k_path;
    std::vector<float> log_k_vec;
    if (!LoadNpy(log_k_path, log_k_vec)) {
        LOG(ERROR) << "[Streaming] Failed to load log_k";
        return false;
    }
    log_k_   = log_k_vec[0];
    k_scale_ = std::exp(log_k_);
    LOG(INFO) << "[Streaming] log_k=" << log_k_ << " k=" << k_scale_;

    // Load vocab
    if (!LoadVocab(vocab_path)) {
        LOG(ERROR) << "[Streaming] Failed to load vocab";
        return false;
    }

    // Precompute RoPE tables
    rope_table_len_ = MAX_NEW_TOKENS_S + 10;
    rope_cos_.resize(rope_table_len_ * ROPE_DIM_S);
    rope_sin_.resize(rope_table_len_ * ROPE_DIM_S);
    int half_rot = ROPE_DIM_S / 2;  // 16
    for (int pos = 0; pos < rope_table_len_; pos++) {
        for (int i = 0; i < half_rot; i++) {
            float inv_freq = 1.0f / std::pow(10000.0f, 2.0f * i / ROPE_DIM_S);
            float freq = static_cast<float>(pos) * inv_freq;
            float c = std::cos(freq);
            float s = std::sin(freq);
            rope_cos_[pos * ROPE_DIM_S + 2 * i]     = c;
            rope_cos_[pos * ROPE_DIM_S + 2 * i + 1] = c;
            rope_sin_[pos * ROPE_DIM_S + 2 * i]     = s;
            rope_sin_[pos * ROPE_DIM_S + 2 * i + 1] = s;
        }
    }
    LOG(INFO) << "[Streaming] RoPE table precomputed, len=" << rope_table_len_;

    // Initialize NPU executors
    std::string npu_options = "--apusys-config \"{ \\\"high_addr\\\": true }\"";

    encoder_executor_ = std::make_unique<mtk::neuropilot::NeuronExecutor>(
        "moonshine_encoder_chunk", encoder_chunk_dla, npu_options);
    if (!encoder_executor_->Initialized()) {
        LOG(ERROR) << "[Streaming] Failed to initialize encoder executor";
        return false;
    }
    LOG(INFO) << "[Streaming] Chunk encoder executor initialized";

    decoder_executor_ = std::make_unique<mtk::neuropilot::NeuronExecutor>(
        "moonshine_decoder", decoder_dla, npu_options);
    if (!decoder_executor_->Initialized()) {
        LOG(ERROR) << "[Streaming] Failed to initialize decoder executor";
        return false;
    }
    LOG(INFO) << "[Streaming] Decoder executor initialized";

    // Reserve buffers
    frame_buffer_.reserve(CHUNK_FRAMES * FRAME_LEN_S);
    enc_buffer_.reserve(MAX_ENC_FRAMES * ENCODER_HIDDEN_S);
    audio_remainder_.reserve(FRAME_LEN_S * 2);

    // Check VAD is loaded (constructed with embedded model data)
    if (!vad_.IsLoaded()) {
        LOG(ERROR) << "[Streaming] Silero VAD failed to load";
        return false;
    }
    LOG(INFO) << "[Streaming] Silero VAD initialized"
              << " (window=" << VAD_WINDOW_MS << "ms"
              << " threshold=" << VAD_THRESHOLD
              << " silence=" << VAD_SILENCE_MS << "ms)";

    initialized_ = true;
    LOG(INFO) << "[Streaming] Initialization complete";
    return true;
}

// ===================== Reset =====================

void MoonshineStreamingEngine::Reset() {
    frame_buffer_.clear();
    frame_count_ = 0;
    audio_remainder_.clear();
    enc_buffer_.clear();
    enc_frame_count_ = 0;
    vad_.Reset();
    LOG(INFO) << "[Streaming] State reset";
}

// ===================== LoadVocab =====================

bool MoonshineStreamingEngine::LoadVocab(const std::string& vocab_path) {
    LOG(INFO) << "[Streaming] Loading vocab from " << vocab_path;
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        LOG(ERROR) << "[Streaming] Cannot open vocab file: " << vocab_path;
        return false;
    }
    vocab_.resize(VOCAB_SIZE_S, "");
    std::string line;
    int count = 0;
    while (std::getline(f, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        int id = std::stoi(line.substr(0, tab));
        std::string piece = line.substr(tab + 1);
        if (id >= 0 && id < VOCAB_SIZE_S) {
            vocab_[id] = piece;
            count++;
        }
    }
    LOG(INFO) << "[Streaming] Vocab loaded: " << count << " tokens";
    return count > 0;
}

// ===================== PreprocessFrame =====================

void MoonshineStreamingEngine::PreprocessFrame(const float* samples, float* out_frame) {
    // CMVN: compute mean
    float mean = 0.0f;
    for (int i = 0; i < FRAME_LEN_S; i++) {
        mean += samples[i];
    }
    mean /= FRAME_LEN_S;

    // CMVN: compute RMS
    float rms_sq = 0.0f;
    for (int i = 0; i < FRAME_LEN_S; i++) {
        float c = samples[i] - mean;
        rms_sq += c * c;
    }
    float rms = std::sqrt(rms_sq / FRAME_LEN_S + EPS_S);

    // Apply CMVN + AsinhCompression
    for (int i = 0; i < FRAME_LEN_S; i++) {
        float normed = (samples[i] - mean) / rms;
        out_frame[i] = std::asinh(k_scale_ * normed);
    }
}

// ===================== RunChunkEncoder =====================

bool MoonshineStreamingEngine::RunChunkEncoder(const float* frame_data, float* enc_out) {
    using namespace mtk::neuropilot;

    // Input: [1, CHUNK_FRAMES, FRAME_LEN_S] = [1, 160, 80]
    size_t in_bytes  = 1 * CHUNK_FRAMES * FRAME_LEN_S * sizeof(float);
    // Output: [1, CHUNK_T_ENC, ENCODER_HIDDEN_S] = [1, 40, 620]
    size_t out_bytes = 1 * CHUNK_T_ENC * ENCODER_HIDDEN_S * sizeof(float);

    TensorBuffer input_buf{
        .data  = const_cast<float*>(frame_data),
        .bytes = in_bytes,
        .type  = kFloat32
    };
    TensorBuffer output_buf{
        .data  = enc_out,
        .bytes = out_bytes,
        .type  = kFloat32
    };

    return encoder_executor_->RunForMultipleInputsOutputs({input_buf}, {output_buf});
}

// ===================== ProcessChunk =====================

std::string MoonshineStreamingEngine::ProcessChunk(const float* audio_samples, int num_samples) {
    if (!initialized_) {
        LOG(ERROR) << "[Streaming] Not initialized";
        return "";
    }

    // --- Step 1: Run VAD on raw PCM (parallel to encoder, independent) ---
    bool vad_triggered = vad_.ShouldTrigger(audio_samples, num_samples);

    // --- Step 2: Append new samples to remainder buffer for encoder ---
    for (int i = 0; i < num_samples; i++) {
        audio_remainder_.push_back(audio_samples[i]);
    }

    // --- Step 3: Process complete frames from remainder (encoder path) ---
    std::string result;

    while ((int)audio_remainder_.size() >= FRAME_LEN_S) {
        // Preprocess one frame
        float processed_frame[FRAME_LEN_S];
        PreprocessFrame(audio_remainder_.data(), processed_frame);

        // Remove processed samples from remainder
        audio_remainder_.erase(audio_remainder_.begin(),
                               audio_remainder_.begin() + FRAME_LEN_S);

        // Append to frame_buffer_
        for (int i = 0; i < FRAME_LEN_S; i++) {
            frame_buffer_.push_back(processed_frame[i]);
        }
        frame_count_++;

        // When we have CHUNK_FRAMES frames, run encoder
        if (frame_count_ >= CHUNK_FRAMES) {
            // Take the last CHUNK_FRAMES frames
            int start_frame = frame_count_ - CHUNK_FRAMES;
            const float* input_ptr = frame_buffer_.data() + start_frame * FRAME_LEN_S;

            // Allocate encoder output buffer
            float enc_out[CHUNK_T_ENC * ENCODER_HIDDEN_S];

            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = RunChunkEncoder(input_ptr, enc_out);
            auto t1 = std::chrono::high_resolution_clock::now();

            float enc_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            encoder_total_ms += enc_ms;
            encoder_calls++;

            if (!ok) {
                LOG(ERROR) << "[Streaming] Encoder call " << encoder_calls << " failed";
                // Clear frame buffer to avoid repeated failure on same data
                frame_buffer_.clear();
                frame_count_ = 0;
                continue;
            }

            LOG(INFO) << "[Streaming] Encoder call #" << encoder_calls
                      << " done in " << enc_ms << " ms"
                      << " (frame_count=" << frame_count_ << ")";

            // Append CHUNK_T_ENC frames to enc_buffer_
            int frames_to_add = CHUNK_T_ENC;
            // Don't exceed MAX_ENC_FRAMES
            if (enc_frame_count_ + frames_to_add > MAX_ENC_FRAMES) {
                frames_to_add = MAX_ENC_FRAMES - enc_frame_count_;
            }
            if (frames_to_add > 0) {
                for (int i = 0; i < frames_to_add * ENCODER_HIDDEN_S; i++) {
                    enc_buffer_.push_back(enc_out[i]);
                }
                enc_frame_count_ += frames_to_add;
            }

            // Slide: keep only last CHUNK_FRAMES frames in frame_buffer_
            // With STEP_FRAMES == CHUNK_FRAMES (no overlap), clear entirely
            if (STEP_FRAMES >= CHUNK_FRAMES) {
                frame_buffer_.clear();
                frame_count_ = 0;
            } else {
                // Partial overlap: keep last (CHUNK_FRAMES - STEP_FRAMES) frames
                int keep = CHUNK_FRAMES - STEP_FRAMES;
                int keep_offset = frame_count_ - keep;
                std::vector<float> new_buf(frame_buffer_.begin() + keep_offset * FRAME_LEN_S,
                                           frame_buffer_.end());
                frame_buffer_ = std::move(new_buf);
                frame_count_ = keep;
            }

            // Fallback: trigger decoder if encoder buffer is full (safety cap)
            if (result.empty() && enc_frame_count_ >= TRIGGER_ENC_FRAMES) {
                LOG(INFO) << "[Streaming] Fallback trigger: enc_frame_count_="
                          << enc_frame_count_;
                // Reset VAD state so it doesn't immediately re-trigger on the
                // tail of this segment after the fallback decode
                vad_.Reset();
                result = TriggerDecoder();
            }
        }
    }

    // --- Step 4: VAD-triggered decode (takes priority over fallback) ---
    if (vad_triggered && enc_frame_count_ > 0) {
        LOG(INFO) << "[Streaming] VAD trigger: enc_frame_count_=" << enc_frame_count_;
        result = TriggerDecoder();
    }

    return result;
}

// ===================== Flush =====================

std::string MoonshineStreamingEngine::Flush() {
    if (!initialized_) return "";

    LOG(INFO) << "[Streaming] Flush called, enc_frame_count_=" << enc_frame_count_
              << " frame_count_=" << frame_count_;

    // Process any remaining complete frames in audio_remainder_
    while ((int)audio_remainder_.size() >= FRAME_LEN_S) {
        float processed_frame[FRAME_LEN_S];
        PreprocessFrame(audio_remainder_.data(), processed_frame);
        audio_remainder_.erase(audio_remainder_.begin(),
                               audio_remainder_.begin() + FRAME_LEN_S);
        for (int i = 0; i < FRAME_LEN_S; i++) {
            frame_buffer_.push_back(processed_frame[i]);
        }
        frame_count_++;
    }

    // If we have accumulated frames in frame_buffer_ but not enough for a full chunk,
    // pad with zeros to form a full chunk and run encoder
    if (frame_count_ > 0 && frame_count_ < CHUNK_FRAMES) {
        LOG(INFO) << "[Streaming] Padding " << frame_count_ << " frames to " << CHUNK_FRAMES;
        // Pad with zero frames
        int pad_frames = CHUNK_FRAMES - frame_count_;
        for (int i = 0; i < pad_frames * FRAME_LEN_S; i++) {
            frame_buffer_.push_back(0.0f);
        }

        // Run encoder on padded chunk
        float enc_out[CHUNK_T_ENC * ENCODER_HIDDEN_S];
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = RunChunkEncoder(frame_buffer_.data(), enc_out);
        auto t1 = std::chrono::high_resolution_clock::now();

        float enc_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        encoder_total_ms += enc_ms;
        encoder_calls++;

        if (ok) {
            // Only add valid frames (proportional to actual frame_count_)
            // Use all CHUNK_T_ENC frames since they're for end of audio
            int valid_enc = std::max(1, (frame_count_ * CHUNK_T_ENC) / CHUNK_FRAMES);
            valid_enc = std::min(valid_enc, CHUNK_T_ENC);

            if (enc_frame_count_ + valid_enc > MAX_ENC_FRAMES) {
                valid_enc = MAX_ENC_FRAMES - enc_frame_count_;
            }
            if (valid_enc > 0) {
                for (int i = 0; i < valid_enc * ENCODER_HIDDEN_S; i++) {
                    enc_buffer_.push_back(enc_out[i]);
                }
                enc_frame_count_ += valid_enc;
            }
            LOG(INFO) << "[Streaming] Flush encoder: " << enc_ms << " ms, added "
                      << valid_enc << " enc frames";
        }

        frame_buffer_.clear();
        frame_count_ = 0;
    }

    // Trigger decoder if we have any encoder output
    if (enc_frame_count_ > 0) {
        return TriggerDecoder();
    }

    return "";
}

// ===================== TriggerDecoder =====================

std::string MoonshineStreamingEngine::TriggerDecoder() {
    if (enc_frame_count_ == 0) return "";

    int num_frames = std::min(enc_frame_count_, MAX_ENC_FRAMES);
    LOG(INFO) << "[Streaming] TriggerDecoder: " << num_frames << " enc frames";

    // Run adapter projection: [num_frames, 620] → [num_frames, 512]
    std::vector<float> encoder_proj;
    RunAdapterProjection(enc_buffer_.data(), num_frames, encoder_proj);

    // Run decoder
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string text = RunDecoder(encoder_proj, num_frames);
    auto t1 = std::chrono::high_resolution_clock::now();

    float dec_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    decoder_total_ms += dec_ms;
    decoder_calls++;

    LOG(INFO) << "[Streaming] Decoder call #" << decoder_calls
              << " done in " << dec_ms << " ms, text: '" << text << "'";

    // Reset encoder accumulation
    enc_buffer_.clear();
    enc_frame_count_ = 0;

    return text;
}

// ===================== RunAdapterProjection =====================

void MoonshineStreamingEngine::RunAdapterProjection(
    const float* enc_buf, int num_frames, std::vector<float>& proj_out)
{
    // enc_buf: [num_frames, ENCODER_HIDDEN_S=620]
    // proj_weight_: [DECODER_HIDDEN_S=512, ENCODER_HIDDEN_S=620]
    // Output: [MAX_ENC_FRAMES=500, DECODER_HIDDEN_S=512] (padded with zeros for decoder compatibility)
    //
    // The decoder DLA was compiled with fixed T_ENC=500.
    // We always output MAX_ENC_FRAMES frames (zero-padded), and pass enc_attn_mask
    // to indicate which frames are valid.

    proj_out.resize(MAX_ENC_FRAMES * DECODER_HIDDEN_S, 0.0f);

    // Only project valid frames
    int valid_frames = std::min(num_frames, MAX_ENC_FRAMES);
    for (int t = 0; t < valid_frames; t++) {
        const float* enc_row = enc_buf + t * ENCODER_HIDDEN_S;
        float* out_row = proj_out.data() + t * DECODER_HIDDEN_S;

        for (int j = 0; j < DECODER_HIDDEN_S; j++) {
            float acc = 0.0f;
            const float* pw = proj_weight_.data() + j * ENCODER_HIDDEN_S;
            for (int k = 0; k < ENCODER_HIDDEN_S; k++) {
                acc += enc_row[k] * pw[k];
            }
            out_row[j] = acc;
        }
    }
    // Remaining frames (valid_frames..MAX_ENC_FRAMES-1) are already 0.0f
}

// ===================== RunDecoder =====================

std::string MoonshineStreamingEngine::RunDecoder(
    const std::vector<float>& encoder_proj, int num_enc_frames)
{
    // KV cache: [NUM_LAYERS_S, 1, MAX_DEC_LEN_S, DECODER_HIDDEN_S]
    size_t kv_size = NUM_LAYERS_S * 1 * MAX_DEC_LEN_S * DECODER_HIDDEN_S;
    std::vector<float> past_keys(kv_size, 0.0f);
    std::vector<float> past_values(kv_size, 0.0f);

    // Encoder attention mask: [1, 1, 1, MAX_ENC_FRAMES=500]
    // Valid positions (0..num_enc_frames-1) = 0.0, padded positions = -1e9
    int valid_enc = std::min(num_enc_frames, MAX_ENC_FRAMES);
    std::vector<float> enc_attn_mask(MAX_ENC_FRAMES, -1e9f);
    for (int i = 0; i < valid_enc; i++) {
        enc_attn_mask[i] = 0.0f;
    }

    std::vector<int> token_ids = {BOS_TOKEN_S};
    int generated = 0;

    for (int step = 0; step < MAX_NEW_TOKENS_S; step++) {
        int current_token = token_ids.back();

        // Embed token: [1, 1, DECODER_HIDDEN_S]
        std::vector<float> decoder_embed(DECODER_HIDDEN_S);
        const float* emb_row = embed_tokens_.data() + current_token * DECODER_HIDDEN_S;
        std::copy(emb_row, emb_row + DECODER_HIDDEN_S, decoder_embed.data());

        // RoPE cos/sin: [1, 1, ROPE_DIM_S]
        std::vector<float> cos_cur(ROPE_DIM_S);
        std::vector<float> sin_cur(ROPE_DIM_S);
        if (step < rope_table_len_) {
            std::copy(rope_cos_.data() + step * ROPE_DIM_S,
                      rope_cos_.data() + (step + 1) * ROPE_DIM_S,
                      cos_cur.data());
            std::copy(rope_sin_.data() + step * ROPE_DIM_S,
                      rope_sin_.data() + (step + 1) * ROPE_DIM_S,
                      sin_cur.data());
        }

        // Self-attention mask: [1, 1, 1, MAX_DEC_LEN_S+1]
        int mask_len = MAX_DEC_LEN_S + 1;
        std::vector<float> attn_mask(mask_len, -1e9f);
        for (int i = 0; i < step; i++) {
            attn_mask[i] = 0.0f;
        }
        attn_mask[mask_len - 1] = 0.0f;

        // Decoder outputs
        std::vector<float> logits(VOCAB_SIZE_S);
        std::vector<float> new_keys(NUM_LAYERS_S * DECODER_HIDDEN_S);
        std::vector<float> new_values(NUM_LAYERS_S * DECODER_HIDDEN_S);

        if (!RunDecoderStep(decoder_embed, encoder_proj, past_keys, past_values,
                             cos_cur, sin_cur, attn_mask, enc_attn_mask,
                             logits, new_keys, new_values)) {
            LOG(ERROR) << "[Streaming] Decoder step " << step << " failed";
            break;
        }

        // Update KV cache
        for (int layer = 0; layer < NUM_LAYERS_S; layer++) {
            const float* nk = new_keys.data() + layer * DECODER_HIDDEN_S;
            const float* nv = new_values.data() + layer * DECODER_HIDDEN_S;
            float* pk = past_keys.data() + layer * (MAX_DEC_LEN_S * DECODER_HIDDEN_S) + step * DECODER_HIDDEN_S;
            float* pv = past_values.data() + layer * (MAX_DEC_LEN_S * DECODER_HIDDEN_S) + step * DECODER_HIDDEN_S;
            std::copy(nk, nk + DECODER_HIDDEN_S, pk);
            std::copy(nv, nv + DECODER_HIDDEN_S, pv);
        }

        // Greedy decode
        int next_token = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());

        token_ids.push_back(next_token);
        generated++;

        if (step == 0) {
            LOG(INFO) << "[Streaming] Decoder step 0: next_token=" << next_token;
        }

        if (next_token == EOS_TOKEN_S) {
            LOG(INFO) << "[Streaming] EOS at step " << step;
            break;
        }
    }

    return TokenIdsToText(token_ids);
}

// ===================== RunDecoderStep =====================

bool MoonshineStreamingEngine::RunDecoderStep(
    const std::vector<float>& decoder_embed,  // [1*1*512]
    const std::vector<float>& encoder_proj,   // [1*M*512]
    const std::vector<float>& past_keys,      // [10*1*128*512]
    const std::vector<float>& past_values,    // [10*1*128*512]
    const std::vector<float>& cos_cur,        // [1*1*32]
    const std::vector<float>& sin_cur,        // [1*1*32]
    const std::vector<float>& attn_mask,      // [1*1*1*129]
    const std::vector<float>& enc_attn_mask,  // [1*1*1*M]
    std::vector<float>& logits,               // [1*1*32768]
    std::vector<float>& new_keys,             // [10*1*1*512]
    std::vector<float>& new_values)           // [10*1*1*512]
{
    using namespace mtk::neuropilot;

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

    std::vector<TensorBuffer> outputs = {
        {logits.data(),     logits.size()     * sizeof(float), kFloat32},
        {new_keys.data(),   new_keys.size()   * sizeof(float), kFloat32},
        {new_values.data(), new_values.size() * sizeof(float), kFloat32},
    };

    return decoder_executor_->RunForMultipleInputsOutputs(inputs, outputs);
}

// ===================== Token Decoding =====================

std::string MoonshineStreamingEngine::DecodeToken(int token_id) {
    if (token_id < 0 || token_id >= VOCAB_SIZE_S) return "";
    if (token_id == BOS_TOKEN_S || token_id == EOS_TOKEN_S) return "";

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

std::string MoonshineStreamingEngine::TokenIdsToText(const std::vector<int>& token_ids) {
    std::string text;
    for (size_t i = 1; i < token_ids.size(); i++) {
        int tid = token_ids[i];
        if (tid == EOS_TOKEN_S) break;
        text += DecodeToken(tid);
    }
    // Trim leading space
    if (!text.empty() && text[0] == ' ') {
        text = text.substr(1);
    }
    return text;
}

} // namespace moonshine
