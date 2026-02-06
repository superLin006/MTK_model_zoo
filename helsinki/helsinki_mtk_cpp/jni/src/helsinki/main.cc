/**
 * Helsinki Translation Demo - MTK NPU with KV Cache
 */

#include "helsinki.h"
#include "sp_tokenizer.h"
#include <iostream>
#include <string>
#include <chrono>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <model_dir> [vocab_dir]" << std::endl;
    std::cout << std::endl;
    std::cout << "model_dir: Directory containing DLA models and embeddings" << std::endl;
    std::cout << "vocab_dir: Directory containing source.spm, target.spm, vocab.txt" << std::endl;
    std::cout << "           (default: model_dir/../../models/Helsinki-NLP/opus-mt-en-zh)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[1];
    std::string vocab_dir = (argc > 2) ? argv[2] :
                             model_dir + "/../../models/Helsinki-NLP/opus-mt-en-zh";

    std::cout << "================================================" << std::endl;
    std::cout << "Helsinki Translation - MTK NPU (KV Cache)" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "Vocab dir: " << vocab_dir << std::endl;

    // Initialize translator
    HelsinkiTranslator translator;
    std::cout << "\n[1/2] Loading NPU models..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    if (translator.init(model_dir.c_str()) != 0) {
        std::cerr << "Failed to initialize translator!" << std::endl;
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "NPU models loaded in " << init_time << " ms" << std::endl;

    translator.print_model_info();

    // Initialize tokenizer
    SPTokenizer src_tokenizer, tgt_tokenizer;
    std::cout << "[2/2] Loading tokenizers..." << std::endl;

    std::string src_spm = vocab_dir + "/source.spm";
    std::string tgt_spm = vocab_dir + "/target.spm";
    std::string vocab_txt = vocab_dir + "/vocab.txt";

    // Try to load tokenizers (optional - can work without)
    bool tokenizer_loaded = false;
    if (src_tokenizer.load_model(src_spm.c_str(), vocab_txt.c_str()) &&
        tgt_tokenizer.load_model(tgt_spm.c_str(), vocab_txt.c_str())) {
        tokenizer_loaded = true;
        std::cout << "Tokenizers loaded successfully" << std::endl;
    } else {
        std::cout << "[WARN] Tokenizers not loaded - using manual token input mode" << std::endl;
    }

    // Test translations
    std::cout << "\n================================================" << std::endl;
    std::cout << "Translation Tests" << std::endl;
    std::cout << "================================================" << std::endl;

    if (tokenizer_loaded) {
        std::vector<std::string> test_sentences = {
            "The rapid advancement of artificial intelligence has transformed many industries around the world.",
            "Scientists have discovered a new species of deep-sea fish that can survive in extreme conditions.",
            "The international conference on climate change brought together experts from over fifty countries.",
            "Modern smartphones have become essential tools for communication, entertainment, and productivity.",
            "The ancient city was buried under volcanic ash for thousands of years before archaeologists found it.",
            "Researchers are developing new renewable energy technologies to reduce our dependence on fossil fuels.",
            "The global economy is experiencing significant changes due to digital transformation and automation.",
            "Students from different cultural backgrounds can learn a lot from each other in international schools.",
            "The documentary film explores the impact of plastic pollution on marine ecosystems worldwide.",
            "Advances in medical science have led to breakthrough treatments for previously incurable diseases."
        };

        for (const auto& sentence : test_sentences) {
            std::cout << "\n---------------------------------" << std::endl;
            std::cout << "Input:  " << sentence << std::endl;

            // Tokenize
            std::vector<int64_t> input_ids = src_tokenizer.encode(sentence, true);
            std::vector<int64_t> padded_ids = src_tokenizer.pad_sequence(
                input_ids, translator.get_src_seq_len(), src_tokenizer.get_pad_token_id());

            // Translate
            std::vector<int64_t> output_ids(64);
            helsinki_perf_stats_t perf_stats;

            start = std::chrono::high_resolution_clock::now();
            int out_len = translator.translate(padded_ids.data(), padded_ids.size(),
                                                output_ids.data(), 64, &perf_stats);
            end = std::chrono::high_resolution_clock::now();

            if (out_len > 0) {
                output_ids.resize(out_len);
                std::cout << "Output IDs: ";
                for (size_t i = 0; i < output_ids.size(); i++) {
                    std::cout << output_ids[i];
                    if (i + 1 < output_ids.size()) std::cout << " ";
                }
                std::cout << std::endl;
                std::string result = tgt_tokenizer.decode(output_ids, true);
                std::cout << "Output: " << result << std::endl;
                std::cout << "Time:   " << perf_stats.total_time_ms << " ms" << std::endl;
            } else {
                std::cerr << "Translation failed!" << std::endl;
            }
        }
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "================================================" << std::endl;
    return 0;
}
