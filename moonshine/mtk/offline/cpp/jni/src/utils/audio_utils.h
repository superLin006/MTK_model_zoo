/* audio_utils.h - Moonshine Audio Preprocessing
 *
 * CPU-side audio preprocessing:
 *   1. Load WAV file (16kHz mono float32)
 *   2. Pad/trim to T_AUDIO_FIXED = 160000 samples
 *   3. Frame-level CMVN (mean centering + RMS normalization)
 *   4. AsinhCompression: y = asinh(k * x)
 *
 * Output: [1, NUM_FRAMES, FRAME_LEN] = [1, 2000, 80] float32
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace moonshine {

// ===================== Constants =====================
static constexpr int FRAME_LEN      = 80;        // 5ms @ 16kHz
static constexpr int T_AUDIO_FIXED  = 160000;    // 10s @ 16kHz
static constexpr int NUM_FRAMES     = T_AUDIO_FIXED / FRAME_LEN;  // 2000
static constexpr int T_ENC          = 500;       // encoder output frames
static constexpr int MAX_DEC_LEN    = 128;       // KV cache max length
static constexpr int VOCAB_SIZE     = 32768;
static constexpr int ENCODER_HIDDEN = 620;
static constexpr int DECODER_HIDDEN = 512;
static constexpr int NUM_LAYERS     = 10;
static constexpr int NUM_HEADS      = 8;
static constexpr int HEAD_DIM       = 64;
static constexpr int ROPE_DIM       = 32;        // partial_rotary_factor=0.5
static constexpr int BOS_TOKEN      = 1;
static constexpr int EOS_TOKEN      = 2;
static constexpr int MAX_NEW_TOKENS = 120;
static constexpr float EPS          = 1e-6f;

// ===================== WAV Loading =====================

/**
 * Load a WAV file (16-bit PCM, mono, 16kHz).
 * Returns raw float32 samples normalized to [-1, 1].
 * Also supports 32-bit float WAV.
 *
 * @param path         Path to the WAV file
 * @param samples_out  Output vector of float32 samples
 * @param sample_rate  Output: detected sample rate
 * @return true on success
 */
bool LoadWav(const std::string& path,
             std::vector<float>& samples_out,
             int& sample_rate);

// ===================== Preprocessing =====================

/**
 * Preprocess audio for Moonshine encoder (CPU).
 *
 * Steps:
 *   1. Pad/trim to T_AUDIO_FIXED samples
 *   2. Reshape to [NUM_FRAMES, FRAME_LEN]
 *   3. Frame-CMVN: center each frame and normalize by RMS
 *   4. AsinhCompression: y = asinh(k * x), k = exp(log_k)
 *
 * @param samples   Input audio samples (arbitrary length, will be padded/trimmed)
 * @param log_k     AsinhCompression parameter (from model)
 * @param frames    Output: [NUM_FRAMES * FRAME_LEN] float32 in row-major [2000, 80]
 */
void PreprocessAudio(const std::vector<float>& samples,
                     float log_k,
                     std::vector<float>& frames);

// ===================== NPY I/O =====================

/**
 * Load a .npy file into a float32 vector.
 * Supports only float32 (dtype='<f4') npy files.
 *
 * @param path   Path to the .npy file
 * @param data   Output float32 data
 * @return true on success
 */
bool LoadNpy(const std::string& path, std::vector<float>& data);

} // namespace moonshine
