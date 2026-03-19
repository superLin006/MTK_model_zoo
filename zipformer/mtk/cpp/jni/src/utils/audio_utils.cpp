/*
 * Audio utilities implementation
 */

#include "audio_utils.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace zipformer {

// ---------------------------------------------------------------------------
// WAV loader (no external dependencies)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct WavHeader {
    char     riff[4];           // "RIFF"
    uint32_t file_size;
    char     wave[4];           // "WAVE"
    char     fmt[4];            // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;      // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};
#pragma pack(pop)

bool load_wav(const std::string& path,
              std::vector<float>& samples,
              int32_t& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good()) {
        fprintf(stderr, "Failed to read WAV header\n");
        return false;
    }

    // Verify RIFF/WAVE
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a RIFF/WAVE file\n");
        return false;
    }

    // Skip extra fmt bytes if fmt_size > 16
    if (header.fmt_size > 16) {
        file.seekg(header.fmt_size - 16, std::ios::cur);
    }

    // Find data chunk
    char chunk_id[4];
    uint32_t chunk_size = 0;
    while (file.read(chunk_id, 4)) {
        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (std::strncmp(chunk_id, "data", 4) == 0) {
            break;
        }
        file.seekg(chunk_size, std::ios::cur);
    }

    if (file.eof() || file.fail()) {
        fprintf(stderr, "Cannot find data chunk in WAV\n");
        return false;
    }

    uint16_t bps = header.bits_per_sample;
    uint16_t nch = header.num_channels;
    int32_t n_samples = static_cast<int32_t>(chunk_size / (bps / 8) / nch);
    samples.resize(n_samples);

    if (bps == 16) {
        std::vector<int16_t> raw(n_samples * nch);
        file.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(n_samples * nch * sizeof(int16_t)));
        for (int32_t i = 0; i < n_samples; ++i) {
            samples[i] = raw[i * nch] / 32768.0f;
        }
    } else if (bps == 32) {
        // 32-bit float or int32
        if (header.audio_format == 3) {
            // IEEE float
            std::vector<float> raw(n_samples * nch);
            file.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(n_samples * nch * sizeof(float)));
            for (int32_t i = 0; i < n_samples; ++i) {
                samples[i] = raw[i * nch];
            }
        } else {
            std::vector<int32_t> raw(n_samples * nch);
            file.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(n_samples * nch * sizeof(int32_t)));
            for (int32_t i = 0; i < n_samples; ++i) {
                samples[i] = raw[i * nch] / 2147483648.0f;
            }
        }
    } else {
        fprintf(stderr, "Unsupported bits_per_sample: %d\n", bps);
        return false;
    }

    sample_rate = static_cast<int32_t>(header.sample_rate);
    fprintf(stdout, "Loaded WAV: %s, %d Hz, %d channels, %d samples (%.2f s)\n",
            path.c_str(), sample_rate, nch, n_samples,
            (float)n_samples / sample_rate);
    return true;
}

// ---------------------------------------------------------------------------
// Fbank extraction
// Parameters match Python test_pt.py:
//   samp_freq = 16000
//   num_bins = 80
//   high_freq = -400.0   (negative = Nyquist - 400)
//   dither = 0.0
//   snip_edges = false
// ---------------------------------------------------------------------------
bool extract_fbank(const std::vector<float>& samples,
                   int32_t sample_rate,
                   std::vector<float>& features,
                   int32_t& num_frames) {
    knf::FbankOptions opts;

    // Frame options
    opts.frame_opts.samp_freq     = static_cast<float>(sample_rate);
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.frame_opts.dither        = 0.0f;
    opts.frame_opts.snip_edges    = false;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.preemph_coeff = 0.97f;
    opts.frame_opts.window_type   = "povey";

    // Mel options
    opts.mel_opts.num_bins = 80;
    opts.mel_opts.low_freq = 20.0f;
    opts.mel_opts.high_freq = -400.0f;  // Must match Python: -400.0

    // Energy/fbank options
    opts.use_energy    = false;
    opts.use_log_fbank = true;
    opts.use_power     = true;

    knf::OnlineFbank fbank(opts);

    // Feed waveform
    fbank.AcceptWaveform(static_cast<float>(sample_rate),
                         samples.data(),
                         static_cast<int32_t>(samples.size()));

    // Add tail padding (matches Python: segment/100.0 * sample_rate = 1.03s of silence)
    // This ensures the last chunk can be completed
    float tail_len_s = static_cast<float>(103) / 100.0f;  // ~1.03s
    int tail_samples = static_cast<int>(tail_len_s * sample_rate);
    std::vector<float> silence(tail_samples, 0.0f);
    fbank.AcceptWaveform(static_cast<float>(sample_rate),
                         silence.data(),
                         static_cast<int32_t>(silence.size()));

    fbank.InputFinished();

    num_frames = fbank.NumFramesReady();
    if (num_frames == 0) {
        fprintf(stderr, "No fbank frames generated\n");
        return false;
    }

    features.resize((size_t)num_frames * 80);
    for (int32_t i = 0; i < num_frames; ++i) {
        const float* frame = fbank.GetFrame(i);
        std::copy(frame, frame + 80, features.begin() + i * 80);
    }

    fprintf(stdout, "Fbank: %d frames extracted (%.2f s audio + %.2f s padding)\n",
            num_frames,
            (float)samples.size() / sample_rate,
            tail_len_s);
    return true;
}

}  // namespace zipformer
