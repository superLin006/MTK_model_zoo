/**
 * Whisper Inference Engine for MTK NPU
 *
 * Main inference class for running Whisper speech recognition
 * on MTK MT8371 NPU platform
 */

#ifndef WHISPER_INFERENCE_H
#define WHISPER_INFERENCE_H

#include "mtk-npu/neuron_executor.h"
#include "utils/audio_utils.h"
#include <vector>
#include <string>
#include <memory>

// Whisper special tokens
#define WHISPER_SOT              50258   // Start of transcript
#define WHISPER_EOT              50257   // End of transcript
#define WHISPER_SPEAKER_START    50359
#define WHISPER_SPEAKER_END      50363
#define WHISPER_TASK_TRANScribe  50359
#define WHISPER_TASK_TRANSLATE   50360
#define WHISPER_TIMESTAMP_BEGIN  50364
#define WHISPER_TASK_EN          50259
#define WHISPER_TASK_ZH          50260

/**
 * Whisper Inference Engine
 *
 * Manages encoder and decoder DLA models and performs
 * autoregressive speech recognition
 */
class WhisperInference {
public:
    WhisperInference();
    ~WhisperInference();

    /**
     * Initialize inference engine
     * @param model_dir Path to directory containing DLA models and resources
     * @return 0 on success, negative on error
     */
    int init(const char* model_dir);

    /**
     * Run inference on audio file
     * @param audio_file Path to audio file (WAV format)
     * @param language Language code ("en", "zh", or NULL for auto-detect)
     * @param task Task code ("transcribe" or "translate")
     * @return Recognized text
     */
    std::string run(const char* audio_file,
                   const char* language = nullptr,
                   const char* task = "transcribe");

    /**
     * Release resources
     */
    void release();

private:
    /**
     * Load mel filters from file
     */
    bool load_mel_filters(const std::string& path);

    /**
     * Load vocabulary from file
     */
    bool load_vocab(const std::string& path);

    /**
     * Load token embedding weights (for decoder)
     */
    bool load_token_embeddings(const std::string& path);

    /**
     * Load encoder DLA model
     */
    bool load_encoder_dla(const std::string& path);

    /**
     * Load decoder DLA model
     */
    bool load_decoder_dla(const std::string& path);

    /**
     * Run encoder inference
     * @param mel_spec Input mel spectrogram [80, 3000]
     * @param encoder_output Output [1, 1500, 512]
     * @return true on success
     */
    bool run_encoder(const std::vector<float>& mel_spec,
                    std::vector<float>& encoder_output);

    /**
     * Run decoder inference (autoregressive)
     * @param encoder_output Encoder output [1, 1500, 512]
     * @param tokens Output token sequence
     * @return true on success
     */
    bool run_decoder(const std::vector<float>& encoder_output,
                    std::vector<int>& tokens);

    /**
     * Decode tokens to text using vocabulary
     * @param tokens Token sequence
     * @param task_code Task token (for language-specific processing)
     * @return Decoded text
     */
    std::string decode_tokens(const std::vector<int>& tokens, int task_code);

    /**
     * Lookup token embedding
     * @param token_id Token ID
     * @param embeddings Output embeddings buffer
     * @param offset Offset in buffer to write to
     */
    void lookup_embedding(int token_id, std::vector<float>& embeddings, size_t offset);

private:
    // Configuration
    bool initialized_ = false;

    // Embedding weights (vocab_size x d_model)
    std::vector<float> token_embeddings_;
    int vocab_size_ = 51865;
    int d_model_ = 512;

    // Mel filters
    std::vector<float> mel_filters_;
    bool mel_filters_loaded_ = false;

    // Vocabulary
    std::vector<VocabEntry> vocab_;
    bool vocab_loaded_ = false;

    // NPU executors
    std::unique_ptr<NeuronExecutor> encoder_executor_;
    std::unique_ptr<NeuronExecutor> decoder_executor_;
};

#endif // WHISPER_INFERENCE_H
