/*
 * Audio utilities for Zipformer MTK
 *
 * WAV loading + Fbank extraction using kaldi-native-fbank.
 * Parameters must match Python test_pt.py exactly:
 *   samp_freq = 16000
 *   num_bins = 80
 *   high_freq = -400.0
 *   dither = 0.0
 *   snip_edges = false
 */

#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace zipformer {

/**
 * Load a WAV file (16-bit PCM, mono or stereo).
 * Converts stereo to mono by taking first channel.
 * Returns samples normalized to [-1, 1].
 */
bool load_wav(const std::string& path,
              std::vector<float>& samples,
              int32_t& sample_rate);

/**
 * Extract Fbank features from float32 audio samples.
 * Uses kaldi-native-fbank with parameters matching test_pt.py.
 *
 * @param samples     Audio samples (16kHz, float32, range [-1,1])
 * @param sample_rate Sample rate (must be 16000)
 * @param features    Output: [num_frames x 80] float32, row-major
 * @param num_frames  Output: number of frames
 * @return true on success
 */
bool extract_fbank(const std::vector<float>& samples,
                   int32_t sample_rate,
                   std::vector<float>& features,
                   int32_t& num_frames);

}  // namespace zipformer
