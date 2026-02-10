/**
 * Audio Processing Utilities for Whisper MTK NPU
 *
 * Ported from RKNN Whisper implementation with MTK adaptations
 */

#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <vector>
#include <string>
#include <cstdint>

// Whisper audio configuration constants
#define N_MELS 128                     // Number of mel frequency bins (80 for base, 128 for large-v3-turbo)
#define N_FFT 400                      // FFT window size
#define HOP_LENGTH 160                 // Hop length for STFT
#define MELS_FILTERS_SIZE 201          // Number of mel filters
#define MAX_AUDIO_LENGTH 480000        // 30 seconds at 16kHz (30 * 16000)
#define MAX_TOKENS 448                 // Maximum number of tokens
#define VOCAB_NUM 51866                // Vocabulary size (51865 for base, 51866 for large-v3-turbo)

// Audio buffer structure
typedef struct {
    float* data;
    int num_frames;
    int sample_rate;
} audio_buffer_t;

// Vocabulary entry structure
typedef struct {
    std::string token;
    int index;
} VocabEntry;

// ==================== Audio Loading ====================

/**
 * Load audio file (WAV format)
 * @param filename Path to audio file
 * @param audio Output audio buffer
 * @return 0 on success, -1 on error
 */
int load_audio(const char* filename, audio_buffer_t* audio);

/**
 * Free audio buffer
 * @param audio Audio buffer to free
 */
void free_audio(audio_buffer_t* audio);

// ==================== Audio Preprocessing ====================

/**
 * Compute log mel spectrogram from audio
 * @param audio_data Audio samples
 * @param audio_length Number of samples
 * @param mel_filters Mel filter bank (80 x 201)
 * @param mel_spec Output mel spectrogram (80 x 3000 for 30s)
 */
void audio_preprocess(audio_buffer_t* audio, float* mel_filters,
                      std::vector<float>& mel_spec);

// ==================== Mel Filters ====================

/**
 * Load mel filter bank from file
 * @param filename Path to mel filters file
 * @param data Output buffer (size: N_MELS * MELS_FILTERS_SIZE)
 * @return 0 on success, -1 on error
 */
int read_mel_filters(const char* filename, float* data, int max_lines);

// ==================== Vocabulary ====================

/**
 * Load vocabulary from file
 * @param filename Path to vocabulary file
 * @param vocab Output vocabulary array (size: VOCAB_NUM)
 * @return 0 on success, -1 on error
 */
int read_vocab(const char* filename, VocabEntry* vocab);

// ==================== Text Processing ====================

/**
 * Replace substring in string
 * @param str String to modify
 * @param from Substring to replace
 * @param to Replacement substring
 */
void replace_substr(std::string& str, const std::string& from,
                    const std::string& to);

/**
 * Decode base64 string (for Chinese text)
 * @param encoded_string Base64 encoded string
 * @return Decoded string
 */
std::string base64_decode(const std::string& encoded_string);

/**
 * Find argmax of array
 * @param array Input array
 * @return Index of maximum value
 */
int argmax(float* array);

#endif // AUDIO_UTILS_H
