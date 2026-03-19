/*
 * Zipformer MTK NPU Inference Implementation
 */

#include "zipformer_inference.h"
#include "executor/ExecutorFactory.h"
#include "common/Log.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace zipformer {

// ---------------------------------------------------------------------------
// NPY file loader (for decoder_embedding.npy)
// Handles the numpy .npy format header and reads float32 data.
// Format: magic (6 bytes) + version (2 bytes) + header_len (2 or 4 bytes) + header + data
// ---------------------------------------------------------------------------
static bool load_npy_float32(const char* path, std::vector<float>& data,
                              int& rows, int& cols) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        LOG(ERROR) << "Cannot open npy file: " << path;
        return false;
    }

    // Read magic + version
    uint8_t magic[6];
    uint8_t version[2];
    if (fread(magic, 1, 6, fp) != 6 || fread(version, 1, 2, fp) != 2) {
        fclose(fp);
        return false;
    }

    // Check magic: \x93NUMPY
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        LOG(ERROR) << "Not a valid npy file: " << path;
        fclose(fp);
        return false;
    }

    // Read header length (2 bytes for v1, 4 bytes for v2+)
    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t hlen16;
        if (fread(&hlen16, 2, 1, fp) != 1) { fclose(fp); return false; }
        header_len = hlen16;
    } else {
        uint32_t hlen32;
        if (fread(&hlen32, 4, 1, fp) != 1) { fclose(fp); return false; }
        header_len = hlen32;
    }

    // Read header string
    std::vector<char> header_buf(header_len + 1, 0);
    if (fread(header_buf.data(), 1, header_len, fp) != header_len) {
        fclose(fp);
        return false;
    }
    std::string header(header_buf.data());

    // Parse shape from header: look for 'shape': (R, C)
    // We only handle 2D float32 arrays for decoder_embedding.npy
    size_t shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        shape_pos = header.find("\"shape\"");
    }
    if (shape_pos == std::string::npos) {
        LOG(ERROR) << "Cannot find shape in npy header";
        fclose(fp);
        return false;
    }

    // Find the tuple
    size_t lp = header.find('(', shape_pos);
    size_t rp = header.find(')', lp);
    if (lp == std::string::npos || rp == std::string::npos) {
        fclose(fp);
        return false;
    }

    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    // Remove whitespace
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

    // Parse two integers separated by comma
    size_t comma = shape_str.find(',');
    if (comma == std::string::npos) {
        fclose(fp);
        return false;
    }
    rows = std::stoi(shape_str.substr(0, comma));
    // After last comma may be trailing comma for 1-element tuples, skip
    std::string col_str = shape_str.substr(comma + 1);
    col_str.erase(std::remove(col_str.begin(), col_str.end(), ','), col_str.end());
    cols = std::stoi(col_str);

    // Read float32 data
    size_t n_floats = (size_t)rows * cols;
    data.resize(n_floats);
    size_t read = fread(data.data(), sizeof(float), n_floats, fp);
    fclose(fp);

    if (read != n_floats) {
        LOG(ERROR) << "npy read error: expected " << n_floats << " floats, got " << read;
        return false;
    }

    LOG(INFO) << "Loaded npy: " << path << " shape=[" << rows << "," << cols << "]";
    return true;
}

// ---------------------------------------------------------------------------
// Vocab loader
// Format: token id\n  (e.g. "▁ 1\n" or "<blank> 0\n")
// vocab[id] = token_string
// ---------------------------------------------------------------------------
static bool load_vocab(const char* path, std::vector<std::string>& vocab) {
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG(ERROR) << "Cannot open vocab file: " << path;
        return false;
    }

    int max_id = 0;
    std::vector<std::pair<int, std::string>> entries;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Find last space to split token and id
        size_t sp = line.rfind(' ');
        if (sp == std::string::npos) continue;
        std::string token = line.substr(0, sp);
        int id = std::stoi(line.substr(sp + 1));
        entries.push_back({id, token});
        if (id > max_id) max_id = id;
    }

    vocab.resize(max_id + 1);
    for (auto& e : entries) {
        vocab[e.first] = e.second;
    }

    LOG(INFO) << "Loaded vocab: " << path << ", max_id=" << max_id
              << ", total entries=" << entries.size();
    return true;
}

// ---------------------------------------------------------------------------
// Tokens to text
// Replace U+2581 (▁) with space, strip leading/trailing spaces
// ---------------------------------------------------------------------------
static std::string tokens_to_text(const std::vector<int>& token_ids,
                                   const std::vector<std::string>& vocab) {
    std::string text;
    for (int id : token_ids) {
        if (id >= 0 && id < (int)vocab.size()) {
            text += vocab[id];
        }
    }

    // Replace ▁ (UTF-8: 0xE2 0x96 0x81) with space
    const std::string from = "\xe2\x96\x81";
    const std::string to   = " ";
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }

    // Lowercase ASCII letters (A-Z → a-z), leave UTF-8 multibyte (Chinese) untouched
    for (char& c : text) {
        if ((unsigned char)c >= 'A' && (unsigned char)c <= 'Z') c = c - 'A' + 'a';
    }

    // Strip leading/trailing whitespace
    size_t start = text.find_first_not_of(' ');
    size_t end   = text.find_last_not_of(' ');
    if (start == std::string::npos) return "";
    return text.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// CPU embedding lookup with mask
// token_ids: array of CONTEXT_SIZE int values
// out: [CONTEXT_SIZE x emb_dim] float32 (will be wrapped as [1, CONTEXT_SIZE, emb_dim])
// ---------------------------------------------------------------------------
static void embed_tokens(const int* token_ids, int context_size,
                         const float* emb_weight, int emb_vocab, int emb_dim,
                         float* out) {
    for (int i = 0; i < context_size; ++i) {
        int tid = token_ids[i];
        float mask = (tid >= 0) ? 1.0f : 0.0f;
        int safe_id = (tid >= 0) ? tid : 0;
        safe_id = std::min(safe_id, emb_vocab - 1);
        const float* row = emb_weight + (size_t)safe_id * emb_dim;
        float* dst = out + i * emb_dim;
        for (int d = 0; d < emb_dim; ++d) {
            dst[d] = row[d] * mask;
        }
    }
}

// ---------------------------------------------------------------------------
// zipformer_init
// ---------------------------------------------------------------------------
bool zipformer_init(ZipformerMTK& ctx,
                    const char* enc_path,
                    const char* dec_path,
                    const char* joi_path,
                    const char* emb_path,
                    const char* vocab_path) {
    LOG(INFO) << "=== Zipformer MTK Init ===";
    LOG(INFO) << "Encoder: " << enc_path;
    LOG(INFO) << "Decoder: " << dec_path;
    LOG(INFO) << "Joiner:  " << joi_path;
    LOG(INFO) << "Embedding: " << emb_path;
    LOG(INFO) << "Vocab: " << vocab_path;

    mtk::neuropilot::ExecutorFactory factory;

    // Load encoder
    ctx.encoder_exec = factory.CreateExecutor(
        mtk::neuropilot::ExecutorType::NeuronRuntime,
        "ZipformerEncoder",
        std::string(enc_path)
    );
    if (!ctx.encoder_exec || !ctx.encoder_exec->Initialized()) {
        LOG(ERROR) << "Failed to init encoder executor";
        return false;
    }
    LOG(INFO) << "Encoder loaded OK";

    // Log encoder tensor sizes
    for (int i = 0; i < ENC_NUM_INPUTS; ++i) {
        size_t sz = ctx.encoder_exec->GetInputTensorSize(i);
        LOG(INFO) << "  Encoder input[" << i << "] = " << sz << " bytes";
    }
    for (int i = 0; i < ENC_NUM_OUTPUTS; ++i) {
        size_t sz = ctx.encoder_exec->GetOutputTensorSize(i);
        LOG(INFO) << "  Encoder output[" << i << "] = " << sz << " bytes";
    }

    // Load decoder
    ctx.decoder_exec = factory.CreateExecutor(
        mtk::neuropilot::ExecutorType::NeuronRuntime,
        "ZipformerDecoder",
        std::string(dec_path)
    );
    if (!ctx.decoder_exec || !ctx.decoder_exec->Initialized()) {
        LOG(ERROR) << "Failed to init decoder executor";
        return false;
    }
    LOG(INFO) << "Decoder loaded OK";

    // Load joiner
    ctx.joiner_exec = factory.CreateExecutor(
        mtk::neuropilot::ExecutorType::NeuronRuntime,
        "ZipformerJoiner",
        std::string(joi_path)
    );
    if (!ctx.joiner_exec || !ctx.joiner_exec->Initialized()) {
        LOG(ERROR) << "Failed to init joiner executor";
        return false;
    }
    LOG(INFO) << "Joiner loaded OK";

    // Load embedding
    int rows = 0, cols = 0;
    if (!load_npy_float32(emb_path, ctx.emb_weight, rows, cols)) {
        LOG(ERROR) << "Failed to load embedding npy";
        return false;
    }
    ctx.emb_vocab_size = rows;
    ctx.emb_dim = cols;
    LOG(INFO) << "Embedding: vocab=" << rows << " dim=" << cols;

    // Load vocab
    if (!load_vocab(vocab_path, ctx.vocab)) {
        LOG(ERROR) << "Failed to load vocab";
        return false;
    }

    // Reset cached states to zero
    ctx.cache.reset();

    ctx.initialized = true;
    LOG(INFO) << "=== Zipformer MTK Init Done ===";
    return true;
}

// ---------------------------------------------------------------------------
// zipformer_free
// ---------------------------------------------------------------------------
void zipformer_free(ZipformerMTK& ctx) {
    ctx.encoder_exec.reset();
    ctx.decoder_exec.reset();
    ctx.joiner_exec.reset();
    ctx.emb_weight.clear();
    ctx.vocab.clear();
    ctx.initialized = false;
}

// ---------------------------------------------------------------------------
// run_encoder
// Input:  x_chunk [1,103,80] (just the 103*80 floats in row-major)
//         ctx.cache contains current cached states
// Output: encoder_out [1,24,512] written to enc_out_buf (24*512 floats)
//         ctx.cache updated to new cached states
// ---------------------------------------------------------------------------
static bool run_encoder(ZipformerMTK& ctx,
                        const float* x_chunk,
                        float* enc_out_buf) {
    // Build input tensor buffers (36 total)
    std::vector<mtk::neuropilot::TensorBuffer> inputs(ENC_NUM_INPUTS);

    // Input 0: x [1,103,80]
    inputs[0].data = const_cast<float*>(x_chunk);
    inputs[0].bytes = (size_t)1 * SEGMENT * N_MELS * sizeof(float);
    inputs[0].type = mtk::neuropilot::kFloat32;

    // Inputs 1..35: cached states
    for (int i = 0; i < N_CACHED_STATES; ++i) {
        inputs[i + 1].data = ctx.cache.data(i);
        inputs[i + 1].bytes = ctx.cache.bytes(i);
        inputs[i + 1].type = mtk::neuropilot::kFloat32;
    }

    // Build output tensor buffers (36 total).
    // Use separate output buffers for cached states (cannot alias input buffers).
    EncoderCache new_cache;

    std::vector<mtk::neuropilot::TensorBuffer> outputs(ENC_NUM_OUTPUTS);

    // Output 0: encoder_out [1,24,256]
    outputs[0].data = enc_out_buf;
    outputs[0].bytes = (size_t)1 * ENCODER_OUT_T * ENC_DIM * sizeof(float);  // 24*256
    outputs[0].type = mtk::neuropilot::kFloat32;

    // Outputs 1..35: new cached states — write into separate new_cache buffers
    for (int i = 0; i < N_CACHED_STATES; ++i) {
        outputs[i + 1].data = new_cache.data(i);
        outputs[i + 1].bytes = new_cache.bytes(i);
        outputs[i + 1].type = mtk::neuropilot::kFloat32;
    }

    bool ok = ctx.encoder_exec->RunForMultipleInputsOutputs(inputs, outputs);
    if (!ok) {
        LOG(ERROR) << "Encoder inference failed";
        return false;
    }

    // Copy new cached states back into ctx.cache
    ctx.cache = std::move(new_cache);
    return true;
}

// ---------------------------------------------------------------------------
// run_decoder
// token_ids[CONTEXT_SIZE]: last 2 token IDs
// dec_out_buf: [1,512] output
// ---------------------------------------------------------------------------
static bool run_decoder(ZipformerMTK& ctx,
                        const int* token_ids,
                        float* dec_out_buf) {
    // CPU embedding lookup
    // emb_input: [1, CONTEXT_SIZE, emb_dim] = [1, 2, 512]
    std::vector<float> emb_input((size_t)CONTEXT_SIZE * ctx.emb_dim, 0.0f);
    embed_tokens(token_ids, CONTEXT_SIZE,
                 ctx.emb_weight.data(), ctx.emb_vocab_size, ctx.emb_dim,
                 emb_input.data());

    std::vector<mtk::neuropilot::TensorBuffer> inputs(1);
    inputs[0].data = emb_input.data();
    inputs[0].bytes = emb_input.size() * sizeof(float);
    inputs[0].type = mtk::neuropilot::kFloat32;

    std::vector<mtk::neuropilot::TensorBuffer> outputs(1);
    outputs[0].data = dec_out_buf;
    outputs[0].bytes = (size_t)DEC_DIM * sizeof(float);  // decoder_out [1,512]
    outputs[0].type = mtk::neuropilot::kFloat32;

    bool ok = ctx.decoder_exec->RunForMultipleInputsOutputs(inputs, outputs);
    if (!ok) {
        LOG(ERROR) << "Decoder inference failed";
    }
    return ok;
}

// ---------------------------------------------------------------------------
// run_joiner
// enc_frame: [1,512], dec_out: [1,512]
// logit_buf: [1,6254] output
// ---------------------------------------------------------------------------
static bool run_joiner(ZipformerMTK& ctx,
                       const float* enc_frame,
                       const float* dec_out,
                       float* logit_buf) {
    std::vector<mtk::neuropilot::TensorBuffer> inputs(2);
    inputs[0].data = const_cast<float*>(enc_frame);
    inputs[0].bytes = (size_t)ENC_DIM * sizeof(float);
    inputs[0].type = mtk::neuropilot::kFloat32;

    inputs[1].data = const_cast<float*>(dec_out);
    inputs[1].bytes = (size_t)DEC_DIM * sizeof(float);  // decoder_out [1,512]
    inputs[1].type = mtk::neuropilot::kFloat32;

    std::vector<mtk::neuropilot::TensorBuffer> outputs(1);
    outputs[0].data = logit_buf;
    outputs[0].bytes = (size_t)VOCAB_SIZE * sizeof(float);
    outputs[0].type = mtk::neuropilot::kFloat32;

    bool ok = ctx.joiner_exec->RunForMultipleInputsOutputs(inputs, outputs);
    if (!ok) {
        LOG(ERROR) << "Joiner inference failed";
    }
    return ok;
}

// ---------------------------------------------------------------------------
// zipformer_recognize
// ---------------------------------------------------------------------------
bool zipformer_recognize(ZipformerMTK& ctx,
                         const float* fbank,
                         int num_frames,
                         std::string& out_text,
                         std::vector<int>& out_tokens) {
    if (!ctx.initialized) {
        LOG(ERROR) << "Context not initialized";
        return false;
    }

    // Reset cached states
    ctx.cache.reset();

    // Initialize hypothesis with BLANK * CONTEXT_SIZE
    std::vector<int> hyp(CONTEXT_SIZE, BLANK_ID);

    // Initial decoder output
    std::vector<float> dec_out(DEC_DIM, 0.0f);
    {
        int init_ids[CONTEXT_SIZE];
        for (int i = 0; i < CONTEXT_SIZE; ++i) init_ids[i] = BLANK_ID;
        if (!run_decoder(ctx, init_ids, dec_out.data())) {
            LOG(ERROR) << "Initial decoder run failed";
            return false;
        }
    }

    std::vector<float> enc_out_buf((size_t)ENCODER_OUT_T * ENC_DIM);  // 24*256
    std::vector<float> logit_buf(VOCAB_SIZE);

    // Tail padding: add SEGMENT/100.0 * SAMPLE_RATE worth of silence = ~1.03s
    // This is done in Python by feeding extra silence frames.
    // For C++: we just process frames we have; if the last chunk is partial, skip it.
    // Python pads with 1.03s of silence which yields SEGMENT-extra frames.
    // To match exactly, compute total padded frames: padded_frames ~ num_frames + SEGMENT
    // We process while: num_processed + SEGMENT <= padded_frames
    // where padded_frames = num_frames (Python pads silence which may yield a few more frames)
    // Conservative: we process while num_processed + SEGMENT <= num_frames
    // (the tail pad in Python mainly ensures the last chunk is complete)
    // For simplicity: process all full chunks available from actual fbank.

    int num_processed = 0;
    int chunk_idx = 0;

    LOG(INFO) << "Starting greedy search: num_frames=" << num_frames
              << " segment=" << SEGMENT << " offset=" << OFFSET;

    while (num_processed + SEGMENT <= num_frames) {
        const float* chunk_ptr = fbank + (size_t)num_processed * N_MELS;

        // Run encoder
        if (!run_encoder(ctx, chunk_ptr, enc_out_buf.data())) {
            LOG(ERROR) << "Encoder failed at chunk " << chunk_idx;
            return false;
        }

        // Greedy search over T time steps
        for (int t = 0; t < ENCODER_OUT_T; ++t) {
            const float* cur_enc = enc_out_buf.data() + (size_t)t * ENC_DIM;

            if (!run_joiner(ctx, cur_enc, dec_out.data(), logit_buf.data())) {
                LOG(ERROR) << "Joiner failed at chunk " << chunk_idx << " t=" << t;
                return false;
            }

            // Argmax
            int y = 0;
            float best = logit_buf[0];
            for (int v = 1; v < VOCAB_SIZE; ++v) {
                if (logit_buf[v] > best) {
                    best = logit_buf[v];
                    y = v;
                }
            }

            if (y != BLANK_ID && y != UNK_ID) {
                hyp.push_back(y);

                // Streaming token output: lookup vocab, replace ▁ with space, lowercase ASCII
                if (y >= 0 && y < (int)ctx.vocab.size()) {
                    std::string word = ctx.vocab[y];
                    const std::string boundary = "\xe2\x96\x81";
                    size_t pos = 0;
                    while ((pos = word.find(boundary, pos)) != std::string::npos) {
                        word.replace(pos, boundary.size(), " ");
                        pos += 1;
                    }
                    for (char& c : word) {
                        if ((unsigned char)c >= 'A' && (unsigned char)c <= 'Z') c = c - 'A' + 'a';
                    }
                    LOG(INFO) << "token " << y << " -> [" << word << "]";
                    printf("%s", word.c_str());
                    fflush(stdout);
                }

                // Update decoder
                int ctx_ids[CONTEXT_SIZE];
                int hyp_sz = (int)hyp.size();
                for (int c = 0; c < CONTEXT_SIZE; ++c) {
                    ctx_ids[c] = hyp[hyp_sz - CONTEXT_SIZE + c];
                }
                if (!run_decoder(ctx, ctx_ids, dec_out.data())) {
                    LOG(ERROR) << "Decoder update failed";
                    return false;
                }
            }
        }

        num_processed += OFFSET;
        chunk_idx++;
        LOG(INFO) << "Chunk " << chunk_idx << ": processed=" << num_processed
                  << "/" << num_frames
                  << " tokens=" << (int)hyp.size() - CONTEXT_SIZE;
    }

    // Extract result (skip initial BLANK * CONTEXT_SIZE)
    out_tokens.assign(hyp.begin() + CONTEXT_SIZE, hyp.end());
    out_text = tokens_to_text(out_tokens, ctx.vocab);

    LOG(INFO) << "Greedy search done: " << chunk_idx << " chunks, "
              << out_tokens.size() << " tokens";
    LOG(INFO) << "Text: " << out_text;

    return true;
}

}  // namespace zipformer
