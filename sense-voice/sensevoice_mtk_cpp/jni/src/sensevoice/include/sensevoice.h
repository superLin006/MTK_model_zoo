/* SenseVoice - Complete Speech Recognition Pipeline
 *
 * End-to-end speech recognition for MTK NPU.
 * Audio input -> Text output
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "sensevoice_config.h"
#include "audio_frontend.h"
#include "tokenizer.h"
#include "sensevoice_model.h"

namespace sensevoice {

class SenseVoice {
public:
    SenseVoice();
    ~SenseVoice();

    // Initialize with configuration
    bool Initialize(const SenseVoiceConfig& config);

    // Initialize with paths (convenience method)
    bool Initialize(const std::string& model_path,
                    const std::string& tokens_path);

    // Check if initialized
    bool IsInitialized() const;

    // Recognize speech from audio samples
    // Input: audio samples (float, normalized to [-1, 1]), 16kHz mono
    // Output: recognition result with text, tokens, and timestamps
    RecognitionResult Recognize(const std::vector<float>& samples,
                                Language language = Language::Auto,
                                TextNorm text_norm = TextNorm::WithoutITN);

    // Recognize speech from audio file (WAV or PCM)
    RecognitionResult RecognizeFile(const std::string& audio_path,
                                    Language language = Language::Auto,
                                    TextNorm text_norm = TextNorm::WithoutITN);

    // Get configuration
    const SenseVoiceConfig& GetConfig() const { return config_; }

    // Get audio frontend (for advanced usage)
    AudioFrontend* GetAudioFrontend() { return audio_frontend_.get(); }

    // Get tokenizer (for advanced usage)
    Tokenizer* GetTokenizer() { return tokenizer_.get(); }

    // Get model (for advanced usage)
    SenseVoiceModel* GetModel() { return model_.get(); }

private:
    SenseVoiceConfig config_;
    std::unique_ptr<AudioFrontend> audio_frontend_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<SenseVoiceModel> model_;
    bool initialized_ = false;
};

}  // namespace sensevoice
