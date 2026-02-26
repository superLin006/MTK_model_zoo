/* SenseVoice Main - Speech Recognition Demo
 *
 * Usage: sensevoice_main <model.dla> <tokens.txt> <audio.wav> [language] [text_norm]
 *
 * Language options: auto, zh, en, yue, ja, ko
 * Text norm options: with_itn, without_itn
 */

#include "sensevoice.h"
#include "common/Log.h"
#include "neuron/api/APUWareUtilsLib.h"

#include <iostream>
#include <string>
#include <chrono>
#include <fstream>

INITIALIZE_EASYLOGGINGPP

// 读取进程 RSS 内存（KB），读取 /proc/self/status
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

// 读取进程峰值内存 VmHWM（KB）
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
    std::cout << "SenseVoice Speech Recognition for MTK NPU\n\n";
    std::cout << "Usage: " << program_name << " <model.dla> <tokens.txt> <audio.wav> [language] [text_norm]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  model.dla    Path to SenseVoice DLA model file\n";
    std::cout << "  tokens.txt   Path to tokens file\n";
    std::cout << "  audio.wav    Path to audio file (WAV or PCM, 16kHz mono)\n";
    std::cout << "  language     Language hint: auto, zh, en, yue, ja, ko (default: auto)\n";
    std::cout << "  text_norm    Text normalization: with_itn (punctuation), without_itn (default: with_itn)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " sensevoice.dla tokens.txt test.wav\n";
    std::cout << "  " << program_name << " sensevoice.dla tokens.txt test.wav zh\n";
    std::cout << "  " << program_name << " sensevoice.dla tokens.txt test.wav auto without_itn\n";
}

sensevoice::Language ParseLanguage(const std::string& lang_str) {
    if (lang_str == "zh" || lang_str == "chinese") {
        return sensevoice::Language::Chinese;
    } else if (lang_str == "en" || lang_str == "english") {
        return sensevoice::Language::English;
    } else if (lang_str == "yue" || lang_str == "cantonese") {
        return sensevoice::Language::Cantonese;
    } else if (lang_str == "ja" || lang_str == "japanese") {
        return sensevoice::Language::Japanese;
    } else if (lang_str == "ko" || lang_str == "korean") {
        return sensevoice::Language::Korean;
    } else {
        return sensevoice::Language::Auto;
    }
}

sensevoice::TextNorm ParseTextNorm(const std::string& norm_str) {
    if (norm_str == "with_itn" || norm_str == "itn") {
        return sensevoice::TextNorm::WithITN;
    } else {
        return sensevoice::TextNorm::WithoutITN;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string tokens_path = argv[2];
    std::string audio_path = argv[3];
    std::string language_str = (argc > 4) ? argv[4] : "auto";
    std::string text_norm_str = (argc > 5) ? argv[5] : "with_itn";  // Default: enable punctuation

    sensevoice::Language language = ParseLanguage(language_str);
    sensevoice::TextNorm text_norm = ParseTextNorm(text_norm_str);

    LOG(INFO) << "=======================================================";
    LOG(INFO) << "SenseVoice Speech Recognition for MTK NPU";
    LOG(INFO) << "=======================================================";
    LOG(INFO) << "Model: " << model_path;
    LOG(INFO) << "Tokens: " << tokens_path;
    LOG(INFO) << "Audio: " << audio_path;
    LOG(INFO) << "Language: " << language_str;
    LOG(INFO) << "Text Norm: " << text_norm_str;
    LOG(INFO) << "=======================================================";

    // Initialize APU power management
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

    // Initialize SenseVoice
    sensevoice::SenseVoice sv;

    LOG(INFO) << "Initializing SenseVoice...";
    auto init_start = std::chrono::high_resolution_clock::now();

    if (!sv.Initialize(model_path, tokens_path)) {
        LOG(ERROR) << "Failed to initialize SenseVoice";
        if (ApuLib.mEnable) {
            ApuLib.releasePerformanceLock(powerHalHandle);
        }
        return 1;
    }

    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        init_end - init_start).count();
    LOG(INFO) << "Initialization completed in " << init_duration << " ms";

    // Memory after initialization
    long rss_after_init = read_rss_kb();
    long peak_after_init = read_peak_rss_kb();

    // Recognize speech
    LOG(INFO) << "-------------------------------------------------------";
    LOG(INFO) << "Starting recognition...";

    auto infer_start = std::chrono::high_resolution_clock::now();
    sensevoice::RecognitionResult result = sv.RecognizeFile(audio_path, language, text_norm);
    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        infer_end - infer_start).count();

    long rss_after_infer = read_rss_kb();
    long peak_final = read_peak_rss_kb();

    LOG(INFO) << "-------------------------------------------------------";
    LOG(INFO) << "Recognition Result:";
    LOG(INFO) << "-------------------------------------------------------";

    if (!result.text.empty()) {
        std::cout << "\n";
        std::cout << "=== TRANSCRIPTION ===\n";
        std::cout << result.text << "\n";
        std::cout << "=====================\n\n";

        // Print metadata if available
        if (!result.language.empty()) {
            LOG(INFO) << "Detected Language: " << result.language;
        }
        if (!result.emotion.empty()) {
            LOG(INFO) << "Detected Emotion: " << result.emotion;
        }
        if (!result.event.empty()) {
            LOG(INFO) << "Detected Event: " << result.event;
        }

        // Print token details if verbose
        if (result.tokens.size() <= 50) {  // Only print if not too many
            LOG(INFO) << "";
            LOG(INFO) << "Tokens (" << result.tokens.size() << "):";
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                LOG(INFO) << "  [" << i << "] t=" << result.timestamps[i]
                          << "s: " << result.tokens[i];
            }
        }
    } else {
        LOG(WARNING) << "No speech detected or recognition failed";
    }

    // Performance summary (stdout, easy to parse)
    std::cout << "=== PERFORMANCE ===\n";
    std::cout << "Init Time:       " << init_duration << " ms\n";
    std::cout << "Inference Time:  " << infer_duration << " ms\n";
    if (rss_after_init >= 0)
        std::cout << "RSS after init:  " << rss_after_init / 1024.0 << " MB"
                  << " (peak: " << peak_after_init / 1024.0 << " MB)\n";
    if (rss_after_infer >= 0)
        std::cout << "RSS after infer: " << rss_after_infer / 1024.0 << " MB\n";
    if (peak_final >= 0)
        std::cout << "Peak RSS:        " << peak_final / 1024.0 << " MB\n";
    std::cout << "===================\n\n";

    LOG(INFO) << "=======================================================";

    // Cleanup
    if (ApuLib.mEnable) {
        ApuLib.releasePerformanceLock(powerHalHandle);
    }

    return result.text.empty() ? 1 : 0;
}
