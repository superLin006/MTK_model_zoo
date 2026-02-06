/* SenseVoice Model Implementation for MTK NPU
 *
 * Uses NeuronUsdk executor for DLA model inference.
 */

#include "sensevoice_model.h"
#include "executor/ExecutorFactory.h"
#include "executor/Executor.h"
#include "common/Log.h"

#include <cstring>
#include <algorithm>

namespace sensevoice {

// Fixed model input size (as compiled in DLA)
static constexpr int32_t kModelInputFrames = 166;
static constexpr int32_t kModelOutputFrames = 170;  // 166 + 4 prompt tokens

class SenseVoiceModel::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    bool Initialize(const ModelConfig& config) {
        config_ = config;

        // Create executor using factory
        mtk::neuropilot::ExecutorFactory factory;
        executor_ = factory.CreateExecutor(
            mtk::neuropilot::ExecutorType::NeuronUsdk,
            "SenseVoice",
            config.model_path
        );

        if (!executor_ || !executor_->Initialized()) {
            LOG(ERROR) << "Failed to initialize SenseVoice executor";
            return false;
        }

        LOG(INFO) << "SenseVoice model initialized successfully";
        LOG(INFO) << "  Model path: " << config.model_path;
        LOG(INFO) << "  Vocab size: " << config.vocab_size;
        LOG(INFO) << "  Input dim: " << config.input_feat_dim;
        LOG(INFO) << "  Fixed input frames: " << kModelInputFrames;

        // Log tensor sizes
        for (int i = 0; i < 5; ++i) {
            size_t size = executor_->GetInputTensorSize(i);
            if (size != SIZE_MAX) {
                LOG(INFO) << "  Input[" << i << "] size: " << size << " bytes";
            }
        }
        size_t out_size = executor_->GetOutputTensorSize(0);
        if (out_size != SIZE_MAX) {
            LOG(INFO) << "  Output[0] size: " << out_size << " bytes";
        }

        return true;
    }

    std::vector<float> Run(const std::vector<float>& features,
                           int32_t num_frames,
                           Language language,
                           TextNorm text_norm) {
        if (!executor_ || !executor_->Initialized()) {
            LOG(ERROR) << "Executor not initialized";
            return {};
        }

        // Pad or truncate features to match model's fixed input size
        std::vector<float> padded_features(kModelInputFrames * config_.input_feat_dim, 0.0f);

        int32_t frames_to_copy = std::min(num_frames, kModelInputFrames);
        size_t bytes_to_copy = frames_to_copy * config_.input_feat_dim * sizeof(float);

        // Debug: check input data
        LOG(INFO) << "Debug: features.size()=" << features.size()
                  << ", num_frames=" << num_frames
                  << ", input_feat_dim=" << config_.input_feat_dim
                  << ", expected=" << (num_frames * config_.input_feat_dim);

        // Validate input size
        size_t expected_size = static_cast<size_t>(num_frames) * config_.input_feat_dim;
        if (features.size() < expected_size) {
            LOG(ERROR) << "Features size mismatch! Got " << features.size()
                       << " but expected " << expected_size;
            return {};
        }

        // Debug: print feature values at different positions
        LOG(INFO) << "Debug: Feature values at different frame positions:";

        // Check first frame (start of audio)
        LOG(INFO) << "  Frame 0 (first 5 dims):";
        for (int d = 0; d < 5 && d < config_.input_feat_dim; ++d) {
            LOG(INFO) << "    [" << d << "] = " << features[d];
        }

        // Check middle frame
        int mid_frame = num_frames / 2;
        int mid_offset = mid_frame * config_.input_feat_dim;
        LOG(INFO) << "  Frame " << mid_frame << " (middle, first 5 dims):";
        for (int d = 0; d < 5 && d < config_.input_feat_dim; ++d) {
            LOG(INFO) << "    [" << d << "] = " << features[mid_offset + d];
        }

        // Check frame 100 (if exists)
        if (num_frames > 100) {
            int frame100_offset = 100 * config_.input_feat_dim;
            LOG(INFO) << "  Frame 100 (first 5 dims):";
            for (int d = 0; d < 5 && d < config_.input_feat_dim; ++d) {
                LOG(INFO) << "    [" << d << "] = " << features[frame100_offset + d];
            }
        }

        // Check frame 166 (cutoff point, if exists)
        if (num_frames > 166) {
            int frame166_offset = 166 * config_.input_feat_dim;
            LOG(INFO) << "  Frame 166 (first 5 dims) - this is where truncation happens:";
            for (int d = 0; d < 5 && d < config_.input_feat_dim; ++d) {
                LOG(INFO) << "    [" << d << "] = " << features[frame166_offset + d];
            }
        }

        // Check for NaN or Inf in features
        int nan_count = 0, inf_count = 0;
        float min_val = features[0], max_val = features[0];
        for (size_t i = 0; i < features.size(); ++i) {
            if (std::isnan(features[i])) nan_count++;
            if (std::isinf(features[i])) inf_count++;
            if (features[i] < min_val) min_val = features[i];
            if (features[i] > max_val) max_val = features[i];
        }
        LOG(INFO) << "Debug: Feature stats: min=" << min_val << ", max=" << max_val
                  << ", NaN=" << nan_count << ", Inf=" << inf_count;

        std::memcpy(padded_features.data(), features.data(), bytes_to_copy);

        if (num_frames < kModelInputFrames) {
            LOG(INFO) << "Input padded from " << num_frames << " to " << kModelInputFrames << " frames";
        } else if (num_frames > kModelInputFrames) {
            LOG(WARNING) << "Input truncated from " << num_frames << " to " << kModelInputFrames << " frames";
            LOG(WARNING) << "Audio longer than ~10s will be truncated. Consider processing in chunks.";
        }

        // Prepare prompt tokens (as float for compatibility)
        std::vector<float> language_tensor = {static_cast<float>(GetLanguageId(language))};
        std::vector<float> event_tensor = {1.0f};       // Fixed event ID
        std::vector<float> event_type_tensor = {2.0f};  // Fixed event type ID
        std::vector<float> text_norm_tensor = {static_cast<float>(GetTextNormId(text_norm))};

        // Prepare output buffer (fixed size based on model)
        std::vector<float> output(kModelOutputFrames * config_.vocab_size, 0.0f);

        // Create tensor buffers
        std::vector<mtk::neuropilot::TensorBuffer> inputs(5);

        // Input 0: Audio features [166, 560]
        inputs[0].data = padded_features.data();
        inputs[0].bytes = padded_features.size() * sizeof(float);
        inputs[0].type = mtk::neuropilot::kFloat32;

        // Input 1: Language ID
        inputs[1].data = language_tensor.data();
        inputs[1].bytes = language_tensor.size() * sizeof(float);
        inputs[1].type = mtk::neuropilot::kFloat32;

        // Input 2: Event ID
        inputs[2].data = event_tensor.data();
        inputs[2].bytes = event_tensor.size() * sizeof(float);
        inputs[2].type = mtk::neuropilot::kFloat32;

        // Input 3: Event Type ID
        inputs[3].data = event_type_tensor.data();
        inputs[3].bytes = event_type_tensor.size() * sizeof(float);
        inputs[3].type = mtk::neuropilot::kFloat32;

        // Input 4: Text Norm ID
        inputs[4].data = text_norm_tensor.data();
        inputs[4].bytes = text_norm_tensor.size() * sizeof(float);
        inputs[4].type = mtk::neuropilot::kFloat32;

        // Output buffer
        std::vector<mtk::neuropilot::TensorBuffer> outputs(1);
        outputs[0].data = output.data();
        outputs[0].bytes = output.size() * sizeof(float);
        outputs[0].type = mtk::neuropilot::kFloat32;

        // Run inference
        bool success = executor_->RunForMultipleInputsOutputs(inputs, outputs);
        if (!success) {
            LOG(ERROR) << "Inference failed";
            return {};
        }

        // Debug: check raw output values
        LOG(INFO) << "Debug: Raw output buffer stats:";
        int out_nan_count = 0, out_inf_count = 0;
        float out_min = output[0], out_max = output[0];
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::isnan(output[i])) out_nan_count++;
            if (std::isinf(output[i])) out_inf_count++;
            if (!std::isnan(output[i]) && !std::isinf(output[i])) {
                if (output[i] < out_min) out_min = output[i];
                if (output[i] > out_max) out_max = output[i];
            }
        }
        LOG(INFO) << "  Output size: " << output.size() << " elements";
        LOG(INFO) << "  Output stats: min=" << out_min << ", max=" << out_max
                  << ", NaN=" << out_nan_count << ", Inf=" << out_inf_count;

        // Debug: check first few output values of frame 0
        LOG(INFO) << "  Frame 0 first 10 logits:";
        for (int i = 0; i < 10 && i < config_.vocab_size; ++i) {
            LOG(INFO) << "    [" << i << "] = " << output[i];
        }

        // Return only the valid portion of output based on actual input frames
        // Output frames = min(input_frames, kModelInputFrames) + 4 prompt tokens
        int32_t valid_output_frames = frames_to_copy + kNumPromptTokens;
        std::vector<float> valid_output(valid_output_frames * config_.vocab_size);
        std::memcpy(valid_output.data(), output.data(),
                    valid_output_frames * config_.vocab_size * sizeof(float));

        return valid_output;
    }

    int32_t GetMaxInputFrames() const {
        return kModelInputFrames;
    }

private:
    ModelConfig config_;
    std::unique_ptr<mtk::neuropilot::Executor> executor_;
};

SenseVoiceModel::SenseVoiceModel() : impl_(std::make_unique<Impl>()) {}
SenseVoiceModel::~SenseVoiceModel() = default;

bool SenseVoiceModel::Initialize(const ModelConfig& config) {
    config_ = config;
    initialized_ = impl_->Initialize(config);
    return initialized_;
}

std::vector<float> SenseVoiceModel::Run(const std::vector<float>& features,
                                        int32_t num_frames,
                                        Language language,
                                        TextNorm text_norm) {
    if (!initialized_) {
        LOG(ERROR) << "Model not initialized";
        return {};
    }
    return impl_->Run(features, num_frames, language, text_norm);
}

}  // namespace sensevoice
