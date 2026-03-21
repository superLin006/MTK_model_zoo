/* main.cpp - Moonshine MTK NPU Streaming Inference Demo
 *
 * Simulates streaming by feeding audio in chunks of CHUNK_AUDIO_SAMPLES.
 *
 * Usage:
 *   ./moonshine_streaming_test <encoder_chunk.dla> <decoder.dla>
 *       <embed_tokens.npy> <proj_weight.npy> <log_k.npy> <vocab.txt> <audio.wav>
 */

#include "moonshine_streaming.h"
#include "utils/audio_utils.h"
#include "common/Log.h"
#include "neuron/api/APUWareUtilsLib.h"

#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <vector>

INITIALIZE_EASYLOGGINGPP

// Number of raw audio samples per chunk: CHUNK_FRAMES * FRAME_LEN_S = 160 * 80 = 12800
static constexpr int CHUNK_AUDIO_SAMPLES = moonshine::CHUNK_FRAMES * moonshine::FRAME_LEN_S;

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

void PrintUsage(const char* prog) {
    std::cout << "Moonshine MTK NPU Streaming Speech Recognition\n\n";
    std::cout << "Usage: " << prog
              << " <encoder_chunk.dla> <decoder.dla>"
                 " <embed_tokens.npy> <proj_weight.npy> <log_k.npy>"
                 " <vocab.txt> <audio.wav>\n\n";
    std::cout << "Chunk size: " << CHUNK_AUDIO_SAMPLES << " samples ("
              << (float)CHUNK_AUDIO_SAMPLES / 16000.0f << "s)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::string encoder_chunk_dla = argv[1];
    std::string decoder_dla       = argv[2];
    std::string embed_tokens      = argv[3];
    std::string proj_weight       = argv[4];
    std::string log_k             = argv[5];
    std::string vocab_path        = argv[6];
    std::string audio_path        = argv[7];

    LOG(INFO) << "=======================================================";
    LOG(INFO) << "Moonshine MTK NPU Streaming Speech Recognition";
    LOG(INFO) << "=======================================================";
    LOG(INFO) << "Encoder (chunk): " << encoder_chunk_dla;
    LOG(INFO) << "Decoder:         " << decoder_dla;
    LOG(INFO) << "Audio:           " << audio_path;
    LOG(INFO) << "Chunk size:      " << CHUNK_AUDIO_SAMPLES << " samples ("
              << (float)CHUNK_AUDIO_SAMPLES / 16000.0f << "s)";
    LOG(INFO) << "Trigger at:      " << moonshine::TRIGGER_ENC_FRAMES << " enc frames";

    // APU power management
    int32_t powerHalHandle = 0;
    ApuWareUtilsLib ApuLib;
    ApuLib.load();
    if (ApuLib.mEnable) {
        LOG(INFO) << "APU Power Management enabled";
        powerHalHandle = ApuLib.acquirePerfParamsLock(
            powerHalHandle, 30000,
            (int*)kFastSingleAnswerParams.data(),
            kFastSingleAnswerParams.size()
        );
    }

    // Initialize streaming engine
    moonshine::MoonshineStreamingEngine engine;

    LOG(INFO) << "Initializing streaming engine...";
    auto init_start = std::chrono::high_resolution_clock::now();

    if (!engine.Initialize(encoder_chunk_dla, decoder_dla,
                            embed_tokens, proj_weight, log_k, vocab_path)) {
        LOG(ERROR) << "Failed to initialize streaming engine";
        if (ApuLib.mEnable) ApuLib.releasePerformanceLock(powerHalHandle);
        return 1;
    }

    auto init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();
    LOG(INFO) << "Initialization done in " << init_ms << " ms";

    long rss_after_init = read_rss_kb();

    // Load audio
    std::vector<float> audio_samples;
    int sample_rate = 0;
    if (!moonshine::LoadWav(audio_path, audio_samples, sample_rate)) {
        LOG(ERROR) << "Failed to load audio: " << audio_path;
        if (ApuLib.mEnable) ApuLib.releasePerformanceLock(powerHalHandle);
        return 1;
    }

    int total_samples = static_cast<int>(audio_samples.size());
    float audio_duration_s = static_cast<float>(total_samples) / sample_rate;
    LOG(INFO) << "Audio: " << total_samples << " samples @ "
              << sample_rate << " Hz (" << audio_duration_s << "s)";

    // Simulate streaming: feed audio in CHUNK_AUDIO_SAMPLES-sized chunks
    LOG(INFO) << "-------------------------------------------------------";
    LOG(INFO) << "Starting streaming inference...";

    std::vector<std::string> partial_results;
    auto infer_start = std::chrono::high_resolution_clock::now();

    int chunk_idx = 0;
    for (int i = 0; i < total_samples; i += CHUNK_AUDIO_SAMPLES) {
        int chunk_size = std::min(CHUNK_AUDIO_SAMPLES, total_samples - i);
        LOG(INFO) << "  Chunk " << chunk_idx << ": samples [" << i
                  << ", " << (i + chunk_size) << ") size=" << chunk_size;

        std::string partial = engine.ProcessChunk(audio_samples.data() + i, chunk_size);
        if (!partial.empty()) {
            partial_results.push_back(partial);
            std::cout << "[Partial " << partial_results.size() << "] " << partial << "\n";
            std::cout.flush();
        }
        chunk_idx++;
    }

    // Flush remaining
    LOG(INFO) << "Flushing remaining audio...";
    std::string final_text = engine.Flush();
    if (!final_text.empty()) {
        partial_results.push_back(final_text);
        std::cout << "[Final] " << final_text << "\n";
        std::cout.flush();
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

    long rss_after_infer = read_rss_kb();
    long peak_rss = read_peak_rss_kb();

    // Combine all partial results
    std::string combined_text;
    for (size_t i = 0; i < partial_results.size(); i++) {
        if (i > 0 && !partial_results[i].empty() && !combined_text.empty()) {
            combined_text += " ";
        }
        combined_text += partial_results[i];
    }

    LOG(INFO) << "-------------------------------------------------------";

    if (!combined_text.empty()) {
        std::cout << "\n";
        std::cout << "=== TRANSCRIPTION ===\n";
        std::cout << combined_text << "\n";
        std::cout << "=====================\n\n";
    } else {
        LOG(WARNING) << "No transcription result";
        std::cout << "\n[Warning] No transcription result\n\n";
    }

    std::cout << "=== PERFORMANCE ===\n";
    std::cout << "Init Time:           " << init_ms << " ms\n";
    std::cout << "Total Infer Time:    " << infer_ms << " ms\n";
    std::cout << "Audio Duration:      " << (audio_duration_s * 1000.0f) << " ms\n";
    std::cout << "RTF:                 " << (infer_ms / (audio_duration_s * 1000.0f)) << "\n";
    std::cout << "--- Encoder ---\n";
    std::cout << "  Total calls:       " << engine.encoder_calls << "\n";
    std::cout << "  Total time:        " << engine.encoder_total_ms << " ms\n";
    if (engine.encoder_calls > 0) {
        std::cout << "  Per call:          "
                  << (engine.encoder_total_ms / engine.encoder_calls) << " ms\n";
    }
    std::cout << "--- Decoder ---\n";
    std::cout << "  Total calls:       " << engine.decoder_calls << "\n";
    std::cout << "  Total time:        " << engine.decoder_total_ms << " ms\n";
    if (rss_after_init >= 0)
        std::cout << "RSS after init:   " << rss_after_init / 1024.0 << " MB\n";
    if (rss_after_infer >= 0)
        std::cout << "RSS after infer:  " << rss_after_infer / 1024.0 << " MB\n";
    if (peak_rss >= 0)
        std::cout << "Peak RSS:         " << peak_rss / 1024.0 << " MB\n";
    std::cout << "===================\n\n";

    LOG(INFO) << "=======================================================";

    if (ApuLib.mEnable) ApuLib.releasePerformanceLock(powerHalHandle);

    return combined_text.empty() ? 1 : 0;
}
