/* main.cpp - Moonshine MTK NPU Inference Demo
 *
 * Usage:
 *   ./moonshine_test <encoder.dla> <decoder.dla> <embed_tokens.npy>
 *                   <pos_emb.npy> <proj_weight.npy> <log_k.npy>
 *                   <vocab.txt> <audio.wav>
 */

#include "moonshine_inference.h"
#include "common/Log.h"
#include "neuron/api/APUWareUtilsLib.h"

#include <iostream>
#include <string>
#include <chrono>
#include <fstream>

INITIALIZE_EASYLOGGINGPP

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

void PrintUsage(const char* program_name) {
    std::cout << "Moonshine MTK NPU Speech Recognition\n\n";
    std::cout << "Usage: " << program_name
              << " <encoder.dla> <decoder.dla> <embed_tokens.npy>"
                 " <pos_emb.npy> <proj_weight.npy> <log_k.npy>"
                 " <tokenizer.json> <audio.wav>\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  encoder.dla      Moonshine encoder DLA model\n";
    std::cout << "  decoder.dla      Moonshine decoder DLA model\n";
    std::cout << "  embed_tokens.npy Token embedding matrix [32768, 512]\n";
    std::cout << "  pos_emb.npy      Positional embedding [4096, 620]\n";
    std::cout << "  proj_weight.npy  Adapter projection weight [512, 620]\n";
    std::cout << "  log_k.npy        AsinhCompression log_k parameter\n";
    std::cout << "  vocab.txt        Vocabulary file (id<TAB>piece per line)\n";
    std::cout << "  audio.wav        Input WAV file (16kHz mono)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::string encoder_dla  = argv[1];
    std::string decoder_dla  = argv[2];
    std::string embed_tokens = argv[3];
    std::string pos_emb      = argv[4];
    std::string proj_weight  = argv[5];
    std::string log_k        = argv[6];
    std::string vocab_path   = argv[7];
    std::string audio_path   = argv[8];

    LOG(INFO) << "=======================================================";
    LOG(INFO) << "Moonshine MTK NPU Speech Recognition";
    LOG(INFO) << "=======================================================";
    LOG(INFO) << "Encoder:    " << encoder_dla;
    LOG(INFO) << "Decoder:    " << decoder_dla;
    LOG(INFO) << "Audio:      " << audio_path;

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

    // Initialize
    moonshine::MoonshineInference engine;

    LOG(INFO) << "Initializing Moonshine engine...";
    auto init_start = std::chrono::high_resolution_clock::now();

    if (!engine.Initialize(encoder_dla, decoder_dla, embed_tokens,
                            pos_emb, proj_weight, log_k, vocab_path)) {
        LOG(ERROR) << "Failed to initialize Moonshine engine";
        if (ApuLib.mEnable) ApuLib.releasePerformanceLock(powerHalHandle);
        return 1;
    }

    auto init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();
    LOG(INFO) << "Initialization done in " << init_ms << " ms";

    long rss_after_init = read_rss_kb();

    // Run inference
    LOG(INFO) << "-------------------------------------------------------";
    LOG(INFO) << "Starting transcription...";

    auto infer_start = std::chrono::high_resolution_clock::now();
    std::string text = engine.Transcribe(audio_path);
    auto infer_end = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

    long rss_after_infer = read_rss_kb();
    long peak_rss = read_peak_rss_kb();

    LOG(INFO) << "-------------------------------------------------------";

    if (!text.empty()) {
        std::cout << "\n";
        std::cout << "=== TRANSCRIPTION ===\n";
        std::cout << text << "\n";
        std::cout << "=====================\n\n";
    } else {
        LOG(WARNING) << "No transcription result";
    }

    std::cout << "=== PERFORMANCE ===\n";
    std::cout << "Init Time:          " << init_ms << " ms\n";
    std::cout << "Total Infer Time:   " << infer_ms << " ms\n";
    std::cout << "  Encoder:          " << engine.encoder_time_ms << " ms\n";
    std::cout << "  Decoder total:    " << engine.decoder_total_ms << " ms\n";
    std::cout << "  Decoder per step: " << engine.decoder_per_step_ms << " ms\n";
    std::cout << "  Num tokens:       " << engine.num_tokens << "\n";
    if (rss_after_init >= 0)
        std::cout << "RSS after init:  " << rss_after_init / 1024.0 << " MB\n";
    if (rss_after_infer >= 0)
        std::cout << "RSS after infer: " << rss_after_infer / 1024.0 << " MB\n";
    if (peak_rss >= 0)
        std::cout << "Peak RSS:        " << peak_rss / 1024.0 << " MB\n";
    std::cout << "===================\n\n";

    LOG(INFO) << "=======================================================";

    if (ApuLib.mEnable) ApuLib.releasePerformanceLock(powerHalHandle);

    return text.empty() ? 1 : 0;
}
