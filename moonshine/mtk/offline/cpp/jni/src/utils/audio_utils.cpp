/* audio_utils.cpp - Moonshine Audio Preprocessing Implementation */

#include "audio_utils.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace moonshine {

// ===================== WAV Loading =====================

struct WavHeader {
    char     riff[4];       // "RIFF"
    uint32_t chunk_size;
    char     wave[4];       // "WAVE"
    char     fmt[4];        // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;  // 1=PCM, 3=IEEE_FLOAT
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

bool LoadWav(const std::string& path,
             std::vector<float>& samples_out,
             int& sample_rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[audio_utils] Cannot open WAV file: " << path << std::endl;
        return false;
    }

    WavHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(WavHeader));
    if (!f) {
        std::cerr << "[audio_utils] Failed to read WAV header" << std::endl;
        return false;
    }

    // Validate RIFF/WAVE
    if (strncmp(hdr.riff, "RIFF", 4) != 0 || strncmp(hdr.wave, "WAVE", 4) != 0) {
        std::cerr << "[audio_utils] Not a valid WAV file: " << path << std::endl;
        return false;
    }

    sample_rate = static_cast<int>(hdr.sample_rate);

    // Skip any extra fmt bytes
    if (hdr.fmt_size > 16) {
        f.seekg(hdr.fmt_size - 16, std::ios::cur);
    }

    // Skip chunks until we find "data"
    char chunk_id[4];
    uint32_t chunk_size;
    while (true) {
        f.read(chunk_id, 4);
        f.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (!f) {
            std::cerr << "[audio_utils] Failed to find data chunk" << std::endl;
            return false;
        }
        if (strncmp(chunk_id, "data", 4) == 0) {
            break;
        }
        f.seekg(chunk_size, std::ios::cur);
    }

    int num_samples = static_cast<int>(chunk_size) / (hdr.bits_per_sample / 8);
    if (hdr.num_channels > 1) {
        // Take only first channel
        num_samples /= hdr.num_channels;
    }
    samples_out.resize(num_samples);

    if (hdr.audio_format == 1) {
        // PCM integer
        if (hdr.bits_per_sample == 16) {
            std::vector<int16_t> raw(num_samples * hdr.num_channels);
            f.read(reinterpret_cast<char*>(raw.data()),
                   num_samples * hdr.num_channels * sizeof(int16_t));
            for (int i = 0; i < num_samples; i++) {
                samples_out[i] = static_cast<float>(raw[i * hdr.num_channels]) / 32768.0f;
            }
        } else if (hdr.bits_per_sample == 32) {
            std::vector<int32_t> raw(num_samples * hdr.num_channels);
            f.read(reinterpret_cast<char*>(raw.data()),
                   num_samples * hdr.num_channels * sizeof(int32_t));
            for (int i = 0; i < num_samples; i++) {
                samples_out[i] = static_cast<float>(raw[i * hdr.num_channels]) / 2147483648.0f;
            }
        } else {
            std::cerr << "[audio_utils] Unsupported PCM bits: " << hdr.bits_per_sample << std::endl;
            return false;
        }
    } else if (hdr.audio_format == 3) {
        // IEEE Float
        if (hdr.bits_per_sample == 32) {
            std::vector<float> raw(num_samples * hdr.num_channels);
            f.read(reinterpret_cast<char*>(raw.data()),
                   num_samples * hdr.num_channels * sizeof(float));
            for (int i = 0; i < num_samples; i++) {
                samples_out[i] = raw[i * hdr.num_channels];
            }
        } else {
            std::cerr << "[audio_utils] Unsupported float bits: " << hdr.bits_per_sample << std::endl;
            return false;
        }
    } else {
        std::cerr << "[audio_utils] Unsupported audio format: " << hdr.audio_format << std::endl;
        return false;
    }

    return true;
}

// ===================== Preprocessing =====================

void PreprocessAudio(const std::vector<float>& samples,
                     float log_k,
                     std::vector<float>& frames) {
    // Step 1: Pad or trim to T_AUDIO_FIXED samples
    std::vector<float> padded(T_AUDIO_FIXED, 0.0f);
    int copy_len = static_cast<int>(std::min((int)samples.size(), T_AUDIO_FIXED));
    std::copy(samples.begin(), samples.begin() + copy_len, padded.begin());

    // Step 2: Reshape to [NUM_FRAMES, FRAME_LEN] and apply CMVN + Asinh
    // Output: frames[NUM_FRAMES * FRAME_LEN]
    frames.resize(NUM_FRAMES * FRAME_LEN);

    float k = std::exp(log_k);

    for (int f = 0; f < NUM_FRAMES; f++) {
        const float* src = padded.data() + f * FRAME_LEN;
        float* dst = frames.data() + f * FRAME_LEN;

        // CMVN: compute mean
        float mean = 0.0f;
        for (int i = 0; i < FRAME_LEN; i++) {
            mean += src[i];
        }
        mean /= FRAME_LEN;

        // CMVN: center
        float rms_sq = 0.0f;
        for (int i = 0; i < FRAME_LEN; i++) {
            float c = src[i] - mean;
            rms_sq += c * c;
        }
        float rms = std::sqrt(rms_sq / FRAME_LEN + EPS);

        // Apply CMVN + AsinhCompression
        for (int i = 0; i < FRAME_LEN; i++) {
            float normed = (src[i] - mean) / rms;
            dst[i] = std::asinh(k * normed);
        }
    }
}

// ===================== NPY I/O =====================

bool LoadNpy(const std::string& path, std::vector<float>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[audio_utils] Cannot open npy file: " << path << std::endl;
        return false;
    }

    // Read magic: \x93NUMPY
    char magic[6];
    f.read(magic, 6);
    if (!f || magic[0] != '\x93' || strncmp(magic + 1, "NUMPY", 5) != 0) {
        std::cerr << "[audio_utils] Not a valid npy file: " << path << std::endl;
        return false;
    }

    // Read version
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hlen;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        header_len = hlen;
    } else if (major == 2 || major == 3) {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        std::cerr << "[audio_utils] Unsupported npy version: " << (int)major << std::endl;
        return false;
    }

    // Read and skip header dict (we assume float32)
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // Check dtype is float32
    if (header.find("'<f4'") == std::string::npos &&
        header.find("\"<f4\"") == std::string::npos &&
        header.find("<f4") == std::string::npos) {
        std::cerr << "[audio_utils] Warning: npy dtype may not be float32. Header: " << header.substr(0, 100) << std::endl;
    }

    // Read all remaining data as float32
    auto start_pos = f.tellg();
    f.seekg(0, std::ios::end);
    auto end_pos = f.tellg();
    size_t data_bytes = static_cast<size_t>(end_pos - start_pos);
    f.seekg(start_pos);

    size_t num_floats = data_bytes / sizeof(float);
    data.resize(num_floats);
    f.read(reinterpret_cast<char*>(data.data()), data_bytes);

    if (!f) {
        std::cerr << "[audio_utils] Failed to read npy data from: " << path << std::endl;
        return false;
    }

    return true;
}

} // namespace moonshine
