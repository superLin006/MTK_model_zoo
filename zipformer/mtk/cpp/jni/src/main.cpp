/*
 * Zipformer MTK NPU - Main test program
 *
 * Usage:
 *   zipformer_mtk_test encoder.dla decoder.dla joiner.dla decoder_embedding.npy test.wav vocab.txt
 */

#include "zipformer_inference.h"
#include "utils/audio_utils.h"
#include "common/Log.h"
#include "neuron/api/APUWareUtilsLib.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdio>

INITIALIZE_EASYLOGGINGPP

// Read RSS from /proc/self/status
static long read_rss_kb() {
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb;
        }
    }
    return -1;
}

static long read_peak_rss_kb() {
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmHWM: %ld", &kb);
            return kb;
        }
    }
    return -1;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s encoder.dla decoder.dla joiner.dla "
                "decoder_embedding.npy test.wav vocab.txt\n", argv[0]);
        return 1;
    }

    const char* enc_path   = argv[1];
    const char* dec_path   = argv[2];
    const char* joi_path   = argv[3];
    const char* emb_path   = argv[4];
    const char* audio_path = argv[5];
    const char* vocab_path = argv[6];

    std::cout << "=== Zipformer MTK NPU Test ===\n";
    std::cout << "Encoder:   " << enc_path   << "\n";
    std::cout << "Decoder:   " << dec_path   << "\n";
    std::cout << "Joiner:    " << joi_path   << "\n";
    std::cout << "Embedding: " << emb_path   << "\n";
    std::cout << "Audio:     " << audio_path << "\n";
    std::cout << "Vocab:     " << vocab_path << "\n";
    std::cout << "==============================\n\n";

    // APU power management (optional, best-effort)
    int32_t powerHalHandle = 0;
    ApuWareUtilsLib ApuLib;
    ApuLib.load();
    if (ApuLib.mEnable) {
        std::cout << "APU Power Management enabled\n";
        powerHalHandle = ApuLib.acquirePerfParamsLock(
            powerHalHandle, 30000,
            (int*)kFastSingleAnswerParams.data(),
            kFastSingleAnswerParams.size()
        );
    }

    // -----------------------------------------------------------------------
    // Step 1: Load audio
    // -----------------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> samples;
    int32_t sample_rate = 0;
    if (!zipformer::load_wav(std::string(audio_path), samples, sample_rate)) {
        std::cerr << "Failed to load WAV: " << audio_path << "\n";
        return 1;
    }

    if (sample_rate != zipformer::SAMPLE_RATE) {
        std::cerr << "ERROR: Sample rate " << sample_rate
                  << " != expected " << zipformer::SAMPLE_RATE << "\n";
        return 1;
    }

    float audio_duration_s = (float)samples.size() / sample_rate;

    // -----------------------------------------------------------------------
    // Step 2: Extract Fbank
    // -----------------------------------------------------------------------
    std::vector<float> fbank_features;
    int32_t num_fbank_frames = 0;

    if (!zipformer::extract_fbank(samples, sample_rate, fbank_features, num_fbank_frames)) {
        std::cerr << "Failed to extract Fbank features\n";
        return 1;
    }

    auto t_audio_done = std::chrono::high_resolution_clock::now();
    auto audio_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_audio_done - t0).count();
    std::cout << "Audio processing: " << audio_ms << " ms\n";
    std::cout << "Audio duration: " << audio_duration_s << " s\n";
    std::cout << "Fbank frames: " << num_fbank_frames << "\n\n";

    // -----------------------------------------------------------------------
    // Step 3: Init Zipformer
    // -----------------------------------------------------------------------
    zipformer::ZipformerMTK ctx;

    auto t_init_start = std::chrono::high_resolution_clock::now();
    if (!zipformer::zipformer_init(ctx, enc_path, dec_path, joi_path, emb_path, vocab_path)) {
        std::cerr << "Failed to initialize Zipformer\n";
        if (ApuLib.mEnable) ApuLib.releasePerformanceLock(powerHalHandle);
        return 1;
    }
    auto t_init_done = std::chrono::high_resolution_clock::now();
    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_init_done - t_init_start).count();
    std::cout << "Init time: " << init_ms << " ms\n";

    long rss_after_init = read_rss_kb();
    long peak_after_init = read_peak_rss_kb();

    // -----------------------------------------------------------------------
    // Step 4: Run inference
    // -----------------------------------------------------------------------
    std::string result_text;
    std::vector<int> result_tokens;

    auto t_infer_start = std::chrono::high_resolution_clock::now();
    bool ok = zipformer::zipformer_recognize(ctx,
                                             fbank_features.data(),
                                             num_fbank_frames,
                                             result_text,
                                             result_tokens);
    auto t_infer_done = std::chrono::high_resolution_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_infer_done - t_infer_start).count();

    long rss_after_infer = read_rss_kb();
    long peak_final = read_peak_rss_kb();

    // -----------------------------------------------------------------------
    // Step 5: Print results
    // -----------------------------------------------------------------------
    std::cout << "\n";
    std::cout << "=== TRANSCRIPTION ===\n";
    if (ok && !result_text.empty()) {
        std::cout << result_text << "\n";
    } else {
        std::cout << "(empty or failed)\n";
    }
    std::cout << "=====================\n\n";

    std::cout << "=== PERFORMANCE ===\n";
    std::cout << "Audio duration:  " << audio_duration_s << " s\n";
    std::cout << "Audio process:   " << audio_ms << " ms\n";
    std::cout << "Init time:       " << init_ms << " ms\n";
    std::cout << "Inference time:  " << infer_ms << " ms\n";
    float rtf = (float)infer_ms / 1000.0f / audio_duration_s;
    std::cout << "RTF:             " << rtf << "x\n";
    if (rss_after_init >= 0)
        std::cout << "RSS after init:  " << rss_after_init / 1024.0 << " MB"
                  << " (peak: " << peak_after_init / 1024.0 << " MB)\n";
    if (rss_after_infer >= 0)
        std::cout << "RSS after infer: " << rss_after_infer / 1024.0 << " MB\n";
    if (peak_final >= 0)
        std::cout << "Peak RSS:        " << peak_final / 1024.0 << " MB\n";
    std::cout << "===================\n\n";

    if (!result_tokens.empty()) {
        std::cout << "Token IDs (" << result_tokens.size() << "): ";
        for (size_t i = 0; i < result_tokens.size(); ++i) {
            std::cout << result_tokens[i];
            if (i + 1 < result_tokens.size()) std::cout << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    zipformer::zipformer_free(ctx);

    if (ApuLib.mEnable) {
        ApuLib.releasePerformanceLock(powerHalHandle);
    }

    return ok ? 0 : 1;
}
