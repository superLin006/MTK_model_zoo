/**
 * Whisper MTK NPU - Main Entry Point
 *
 * Command-line tool for running Whisper speech recognition
 * on MTK MT8371 NPU platform
 */

#include "whisper_inference.h"
#include <iostream>
#include <string>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <model_dir> <audio_file> [language] [task]\n"
              << "\n"
              << "Arguments:\n"
              << "  model_dir   Path to directory containing DLA models and resources\n"
              << "              Required files:\n"
              << "                - encoder_base_80x3000_MT8371.dla\n"
              << "                - decoder_base_448_MT8371.dla\n"
              << "                - token_embedding.npy\n"
              << "                - mel_filters.txt (optional)\n"
              << "                - vocab.txt (optional)\n"
              << "  audio_file  Path to audio file (WAV format, 16kHz mono)\n"
              << "  language    Language code: 'en', 'zh', or 'auto' (default: 'auto')\n"
              << "  task        Task type: 'transcribe' or 'translate' (default: 'transcribe')\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog_name << " /data/models/whisper /data/audio/test_en.wav\n"
              << "  " << prog_name << " /data/models/whisper /data/audio/test_zh.wav zh transcribe\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================\n"
              << "  Whisper MTK NPU Inference Tool\n"
              << "========================================\n"
              << std::endl;

    // Parse arguments
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* audio_file = argv[2];
    const char* language = (argc >= 4) ? argv[3] : "auto";
    const char* task = (argc >= 5) ? argv[4] : "transcribe";

    std::cout << "[CONFIG] Model directory: " << model_dir << "\n"
              << "[CONFIG] Audio file: " << audio_file << "\n"
              << "[CONFIG] Language: " << language << "\n"
              << "[CONFIG] Task: " << task << "\n"
              << std::endl;

    // Create inference engine
    WhisperInference whisper;

    // Initialize
    std::cout << "\n=== Initialization ===" << std::endl;
    if (whisper.init(model_dir) != 0) {
        std::cerr << "\n[ERROR] Initialization failed!" << std::endl;
        return 1;
    }

    // Run inference
    std::cout << "\n=== Inference ===" << std::endl;
    std::string result = whisper.run(audio_file, language, task);

    std::cout << "\n[INFO] Done!" << std::endl;
    return 0;
}
