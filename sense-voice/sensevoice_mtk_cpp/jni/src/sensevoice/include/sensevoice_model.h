/* SenseVoice Model for MTK NPU
 *
 * Wrapper for loading and running SenseVoice DLA model on MTK NPU.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "sensevoice_config.h"

namespace sensevoice {

class SenseVoiceModel {
public:
    SenseVoiceModel();
    ~SenseVoiceModel();

    // Initialize model from DLA file
    bool Initialize(const ModelConfig& config);

    // Check if model is initialized
    bool IsInitialized() const { return initialized_; }

    // Run inference
    // Input: LFR features [num_frames, 560]
    // Output: logits [num_frames + 4, vocab_size]
    std::vector<float> Run(const std::vector<float>& features,
                           int32_t num_frames,
                           Language language = Language::Auto,
                           TextNorm text_norm = TextNorm::WithoutITN);

    // Get model metadata
    const ModelConfig& GetConfig() const { return config_; }

    // Get expected input size for given number of frames
    size_t GetInputSize(int32_t num_frames) const {
        return num_frames * config_.input_feat_dim;
    }

    // Get expected output size for given number of input frames
    size_t GetOutputSize(int32_t num_frames) const {
        return (num_frames + kNumPromptTokens) * config_.vocab_size;
    }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    ModelConfig config_;
    bool initialized_ = false;

    static constexpr int32_t kNumPromptTokens = 4;  // language, event, event_type, text_norm
};

}  // namespace sensevoice
