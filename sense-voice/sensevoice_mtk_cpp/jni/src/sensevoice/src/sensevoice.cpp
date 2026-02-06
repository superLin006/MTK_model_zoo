/* SenseVoice Implementation
 *
 * Complete speech recognition pipeline.
 */

#include "sensevoice.h"
#include "common/Log.h"

#include <chrono>

namespace sensevoice {

SenseVoice::SenseVoice()
    : audio_frontend_(nullptr),
      tokenizer_(nullptr),
      model_(nullptr) {
}

SenseVoice::~SenseVoice() = default;

bool SenseVoice::Initialize(const SenseVoiceConfig& config) {
    config_ = config;

    // Initialize audio frontend
    audio_frontend_ = std::make_unique<AudioFrontend>(config.audio);
    if (!audio_frontend_) {
        LOG(ERROR) << "Failed to create audio frontend";
        return false;
    }
    LOG(INFO) << "Audio frontend initialized";

    // Initialize tokenizer
    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->Load(config.model.tokens_path)) {
        LOG(ERROR) << "Failed to load tokens from: " << config.model.tokens_path;
        return false;
    }
    LOG(INFO) << "Tokenizer loaded with " << tokenizer_->VocabSize() << " tokens";

    // Initialize model
    model_ = std::make_unique<SenseVoiceModel>();
    if (!model_->Initialize(config.model)) {
        LOG(ERROR) << "Failed to initialize model";
        return false;
    }
    LOG(INFO) << "Model initialized";

    initialized_ = true;
    return true;
}

bool SenseVoice::Initialize(const std::string& model_path,
                            const std::string& tokens_path) {
    SenseVoiceConfig config;
    config.model.model_path = model_path;
    config.model.tokens_path = tokens_path;
    return Initialize(config);
}

bool SenseVoice::IsInitialized() const {
    return initialized_;
}

RecognitionResult SenseVoice::Recognize(const std::vector<float>& samples,
                                        Language language,
                                        TextNorm text_norm) {
    RecognitionResult result;

    if (!initialized_) {
        LOG(ERROR) << "SenseVoice not initialized";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Extract features
    LOG(INFO) << "Processing audio: " << samples.size() << " samples ("
              << (samples.size() / 16000.0f) << " seconds)";

    int32_t num_lfr_frames = 0;
    std::vector<float> features = audio_frontend_->Process(samples, &num_lfr_frames);

    if (features.empty() || num_lfr_frames == 0) {
        LOG(ERROR) << "Failed to extract features";
        return result;
    }

    auto feature_time = std::chrono::high_resolution_clock::now();
    auto feature_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        feature_time - start_time).count();
    LOG(INFO) << "Feature extraction: " << num_lfr_frames << " frames, "
              << feature_duration << " ms";

    // Step 2: Run model inference
    std::vector<float> logits = model_->Run(features, num_lfr_frames, language, text_norm);

    if (logits.empty()) {
        LOG(ERROR) << "Inference failed";
        return result;
    }

    auto inference_time = std::chrono::high_resolution_clock::now();
    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        inference_time - feature_time).count();

    // Calculate actual output frames based on model's fixed input size
    static constexpr int32_t kModelInputFrames = 166;
    int32_t actual_input_frames = std::min(num_lfr_frames, kModelInputFrames);
    int32_t output_frames = actual_input_frames + 4;  // +4 for prompt tokens
    LOG(INFO) << "Inference: " << output_frames << " output frames, "
              << inference_duration << " ms";

    // Debug: print first few frames' argmax
    LOG(INFO) << "Debug: First 10 frames argmax:";
    for (int f = 0; f < 10 && f < output_frames; ++f) {
        const float* frame_logits = logits.data() + f * config_.model.vocab_size;
        int max_idx = 0;
        float max_val = frame_logits[0];
        for (int v = 1; v < config_.model.vocab_size; ++v) {
            if (frame_logits[v] > max_val) {
                max_val = frame_logits[v];
                max_idx = v;
            }
        }
        LOG(INFO) << "  Frame " << f << ": argmax=" << max_idx << ", value=" << max_val;
    }

    // Step 3: Decode CTC output
    result = tokenizer_->Decode(
        logits.data(),
        output_frames,
        config_.model.vocab_size,
        config_.audio.frame_shift_ms,
        config_.model.lfr_window_shift
    );

    auto decode_time = std::chrono::high_resolution_clock::now();
    auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        decode_time - inference_time).count();
    LOG(INFO) << "Decoding: " << result.tokens.size() << " tokens, "
              << decode_duration << " ms";

    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        decode_time - start_time).count();
    float audio_duration = samples.size() / 16000.0f;
    float rtf = total_duration / 1000.0f / audio_duration;

    LOG(INFO) << "Total time: " << total_duration << " ms, RTF: " << rtf;
    LOG(INFO) << "Result: " << result.text;

    return result;
}

RecognitionResult SenseVoice::RecognizeFile(const std::string& audio_path,
                                            Language language,
                                            TextNorm text_norm) {
    RecognitionResult result;

    // Determine file type and load
    std::vector<float> samples;
    int32_t sample_rate;

    // Try WAV first
    if (audio_path.size() > 4 &&
        (audio_path.substr(audio_path.size() - 4) == ".wav" ||
         audio_path.substr(audio_path.size() - 4) == ".WAV")) {
        if (!LoadWavFile(audio_path, &samples, &sample_rate)) {
            LOG(ERROR) << "Failed to load WAV file: " << audio_path;
            return result;
        }

        // Resample if needed
        if (sample_rate != config_.audio.sample_rate) {
            LOG(WARNING) << "Sample rate mismatch: file=" << sample_rate
                         << ", expected=" << config_.audio.sample_rate;
            // Simple decimation/interpolation for common rates
            // TODO: Implement proper resampling
        }
    }
    // Try PCM
    else if (audio_path.size() > 4 &&
             (audio_path.substr(audio_path.size() - 4) == ".pcm" ||
              audio_path.substr(audio_path.size() - 4) == ".raw")) {
        if (!LoadPcmFile(audio_path, &samples, config_.audio.sample_rate)) {
            LOG(ERROR) << "Failed to load PCM file: " << audio_path;
            return result;
        }
    }
    else {
        // Try WAV format first
        if (LoadWavFile(audio_path, &samples, &sample_rate)) {
            // OK
        } else if (LoadPcmFile(audio_path, &samples, config_.audio.sample_rate)) {
            // OK
        } else {
            LOG(ERROR) << "Failed to load audio file: " << audio_path;
            return result;
        }
    }

    if (samples.empty()) {
        LOG(ERROR) << "Empty audio file: " << audio_path;
        return result;
    }

    LOG(INFO) << "Loaded audio file: " << audio_path;
    LOG(INFO) << "  Samples: " << samples.size();
    LOG(INFO) << "  Duration: " << (samples.size() / 16000.0f) << " seconds";

    return Recognize(samples, language, text_norm);
}

}  // namespace sensevoice
