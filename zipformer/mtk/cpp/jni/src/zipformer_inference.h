/*
 * Zipformer MTK NPU Inference
 *
 * Streaming Transducer (Zipformer) inference on MTK DLA.
 *
 * Encoder inputs (36 total, in MODEL_CONFIG order):
 *   [0]  x             [1,103,80]       float32
 *   [1-5]  cached_len_{0-4}  [2,1]      float32
 *   [6-10] cached_avg_{0-4}  [2,1,256]  float32
 *   [11]   cached_key_0  [2,128,1,192] float32  (ds=1)
 *   [12]   cached_key_1  [2,64,1,192]          (ds=2)
 *   [13]   cached_key_2  [2,32,1,192]          (ds=4)
 *   [14]   cached_key_3  [2,16,1,192]          (ds=8)
 *   [15]   cached_key_4  [2,64,1,192]          (ds=2)
 *   [16]   cached_val_0  [2,128,1,96]          (ds=1)
 *   [17]   cached_val_1  [2,64,1,96]           (ds=2)
 *   [18]   cached_val_2  [2,32,1,96]           (ds=4)
 *   [19]   cached_val_3  [2,16,1,96]           (ds=8)
 *   [20]   cached_val_4  [2,64,1,96]           (ds=2)
 *   [21-25] cached_val2_{0-4}  (same shapes as val)
 *   [26-30] cached_conv1_{0-4}  [2,1,256,30]
 *   [31-35] cached_conv2_{0-4}  [2,1,256,30]
 *
 * Encoder outputs (36 total):
 *   [0]  encoder_out  [1,24,256]   float32
 *   [1-35] new_cached_{...}  same order & shapes as inputs[1-35]
 *
 * Decoder inputs: emb_input [1,2,512] float32
 * Decoder outputs: decoder_out [1,512] float32
 *
 * Joiner inputs: enc_out [1,256], dec_out [1,512]
 * Joiner outputs: logit [1,6254]
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "executor/Executor.h"

namespace zipformer {

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------
static constexpr int SAMPLE_RATE    = 16000;
static constexpr int N_MELS         = 80;
static constexpr int SEGMENT        = 103;   // encoder input frames per chunk
static constexpr int OFFSET         = 96;    // stride per chunk
static constexpr int ENCODER_OUT_T  = 24;    // encoder output time steps
static constexpr int ENC_DIM        = 256;   // encoder output dim
static constexpr int DEC_DIM        = 512;   // decoder output dim
static constexpr int VOCAB_SIZE     = 6254;
static constexpr int CONTEXT_SIZE   = 2;
static constexpr int BLANK_ID       = 0;
static constexpr int UNK_ID         = 2;

// Encoder: 36 inputs (x + 35 cached), 36 outputs (enc_out + 35 new cached)
static constexpr int ENC_NUM_INPUTS  = 36;
static constexpr int ENC_NUM_OUTPUTS = 36;
static constexpr int N_CACHED_STATES = 35;   // ENC_NUM_INPUTS - 1

// Sizes in floats for each cached state (order matches MODEL_CONFIG sans 'x'):
// left_context_len = decode_chunk_size(32) * num_left_chunks(4) = 128
// ds = zipformer_downsampling_factors = (1,2,4,8,2)
//
// State 0-4:   cached_len_{0-4}   [2,1]              = 2 each
// State 5-9:   cached_avg_{0-4}   [2,1,256]          = 512 each
// State 10:    cached_key_0       [2,128,1,192]  ds=1 = 49152
// State 11:    cached_key_1       [2,64,1,192]   ds=2 = 24576
// State 12:    cached_key_2       [2,32,1,192]   ds=4 = 12288
// State 13:    cached_key_3       [2,16,1,192]   ds=8 = 6144
// State 14:    cached_key_4       [2,64,1,192]   ds=2 = 24576
// State 15:    cached_val_0       [2,128,1,96]   ds=1 = 24576
// State 16:    cached_val_1       [2,64,1,96]    ds=2 = 12288
// State 17:    cached_val_2       [2,32,1,96]    ds=4 = 6144
// State 18:    cached_val_3       [2,16,1,96]    ds=8 = 3072
// State 19:    cached_val_4       [2,64,1,96]    ds=2 = 12288
// State 20:    cached_val2_0      [2,128,1,96]   ds=1 = 24576
// State 21:    cached_val2_1      [2,64,1,96]    ds=2 = 12288
// State 22:    cached_val2_2      [2,32,1,96]    ds=4 = 6144
// State 23:    cached_val2_3      [2,16,1,96]    ds=8 = 3072
// State 24:    cached_val2_4      [2,64,1,96]    ds=2 = 12288
// State 25-29: cached_conv1_{0-4} [2,1,256,30]       = 15360 each
// State 30-34: cached_conv2_{0-4} [2,1,256,30]       = 15360 each

static constexpr int CACHE_STATE_SIZES[N_CACHED_STATES] = {
    // cached_len (5)
    2, 2, 2, 2, 2,
    // cached_avg (5)
    512, 512, 512, 512, 512,
    // cached_key (5): left_ctx//ds * attention_dim * num_layers
    49152, 24576, 12288, 6144, 24576,
    // cached_val (5): left_ctx//ds * (attention_dim//2) * num_layers
    24576, 12288, 6144, 3072, 12288,
    // cached_val2 (5)
    24576, 12288, 6144, 3072, 12288,
    // cached_conv1 (5): num_layers * batch * d_model * (kernel-1)
    15360, 15360, 15360, 15360, 15360,
    // cached_conv2 (5)
    15360, 15360, 15360, 15360, 15360,
};

// ---------------------------------------------------------------------------
// Cached state storage
// ---------------------------------------------------------------------------
struct EncoderCache {
    std::vector<std::vector<float>> buf;  // buf[i] = flat float array for state i

    EncoderCache() { reset(); }

    void reset() {
        buf.resize(N_CACHED_STATES);
        for (int i = 0; i < N_CACHED_STATES; ++i) {
            buf[i].assign(CACHE_STATE_SIZES[i], 0.0f);
        }
    }

    float* data(int i) { return buf[i].data(); }
    const float* data(int i) const { return buf[i].data(); }
    size_t bytes(int i) const { return buf[i].size() * sizeof(float); }
};

// ---------------------------------------------------------------------------
// Vocab
// ---------------------------------------------------------------------------
struct VocabEntry {
    int id;
    std::string token;
};

// ---------------------------------------------------------------------------
// ZipformerMTK context
// ---------------------------------------------------------------------------
struct ZipformerMTK {
    // Executors
    std::unique_ptr<mtk::neuropilot::Executor> encoder_exec;
    std::unique_ptr<mtk::neuropilot::Executor> decoder_exec;
    std::unique_ptr<mtk::neuropilot::Executor> joiner_exec;

    // Embedding table [vocab_size x emb_dim]
    std::vector<float> emb_weight;
    int emb_vocab_size = 0;
    int emb_dim = 0;

    // Cached encoder states (zero-initialized at start)
    EncoderCache cache;

    // Vocab: vocab[id] = token_string
    std::vector<std::string> vocab;  // indexed by token id

    bool initialized = false;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/**
 * Initialize ZipformerMTK context.
 */
bool zipformer_init(ZipformerMTK& ctx,
                    const char* enc_path,
                    const char* dec_path,
                    const char* joi_path,
                    const char* emb_path,
                    const char* vocab_path);

/**
 * Free resources.
 */
void zipformer_free(ZipformerMTK& ctx);

/**
 * Run streaming greedy search on pre-computed fbank features.
 * @param fbank      Fbank features, row-major [num_frames x N_MELS]
 * @param num_frames Number of fbank frames
 * @param out_text   Recognized text (output)
 * @param out_tokens Token IDs excluding initial blanks (output)
 */
bool zipformer_recognize(ZipformerMTK& ctx,
                         const float* fbank,
                         int num_frames,
                         std::string& out_text,
                         std::vector<int>& out_tokens);

}  // namespace zipformer
