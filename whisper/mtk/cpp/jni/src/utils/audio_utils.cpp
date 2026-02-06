/**
 * Audio Processing Utilities Implementation for Whisper MTK NPU
 *
 * Ported from RKNN Whisper implementation with FFTW integration
 */

#include "audio_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>

// Include FFTW
#include <fftw3.h>

// ==================== Audio Loading ====================

int load_audio(const char* filename, audio_buffer_t* audio) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "[ERROR] Failed to open audio file: " << filename << std::endl;
        return -1;
    }

    // Simple WAV parsing (16-bit PCM, mono, 16kHz)
    char header[44];
    fread(header, 1, 44, fp);

    // Check for RIFF header
    if (strncmp(header, "RIFF", 4) != 0) {
        std::cerr << "[ERROR] Not a valid WAV file" << std::endl;
        fclose(fp);
        return -1;
    }

    // Get data size
    int data_size = *(int*)&header[40];
    int num_samples = data_size / 2;  // 16-bit = 2 bytes per sample

    // Allocate buffer
    audio->data = (float*)malloc(num_samples * sizeof(float));
    if (!audio->data) {
        std::cerr << "[ERROR] Failed to allocate audio buffer" << std::endl;
        fclose(fp);
        return -1;
    }

    // Read 16-bit PCM samples and convert to float
    short* temp_buffer = (short*)malloc(data_size);
    fread(temp_buffer, 1, data_size, fp);
    fclose(fp);

    for (int i = 0; i < num_samples; i++) {
        audio->data[i] = (float)temp_buffer[i] / 32768.0f;
    }

    free(temp_buffer);

    audio->num_frames = num_samples;
    audio->sample_rate = 16000;  // Whisper expects 16kHz

    std::cout << "[INFO] Loaded audio: " << num_samples << " samples ("
              << (num_samples / 16000.0) << "s)" << std::endl;

    return 0;
}

void free_audio(audio_buffer_t* audio) {
    if (audio->data) {
        free(audio->data);
        audio->data = nullptr;
    }
    audio->num_frames = 0;
}

// ==================== Mel Filters ====================

int read_mel_filters(const char* filename, float* data, int max_lines) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("[ERROR] Error opening mel filters file");
        return -1;
    }

    int line_count = 0;
    while (line_count < max_lines && fscanf(file, "%f", &data[line_count]) == 1) {
        line_count++;
    }

    fclose(file);
    std::cout << "[INFO] Loaded " << line_count << " mel filter coefficients" << std::endl;

    return 0;
}

// ==================== Audio Preprocessing ====================

static void hann_window(std::vector<float>& window, int length) {
    for (int i = 0; i < length; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (length - 1)));
    }
}

static void reflect_pad(const std::vector<float>& audio,
                        std::vector<float>& padded_audio, int pad_width) {
    // Reflect padding
    std::copy(audio.begin(), audio.end(), padded_audio.begin() + pad_width);
    std::reverse_copy(audio.begin(), audio.begin() + pad_width, padded_audio.begin());
    std::reverse_copy(audio.end() - pad_width, audio.end(),
                     padded_audio.end() - pad_width);
}

// STFT using FFTW
static void compute_stft(const std::vector<float>& audio,
                        int audio_length,
                        int window_length,
                        int hop_length,
                        const std::vector<float>& window,
                        fftwf_complex* stft_result,
                        int num_frames) {
    // FFTW planning
    float* in = (float*)fftwf_malloc(sizeof(float) * window_length);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (window_length / 2 + 1));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(window_length, in, out, FFTW_ESTIMATE);

    for (int i = 0; i < num_frames; i++) {
        int start = i * hop_length;

        // Apply window and prepare input
        for (int j = 0; j < window_length; j++) {
            if (start + j < audio_length) {
                in[j] = audio[start + j] * window[j];
            } else {
                in[j] = 0.0f;
            }
        }

        // Execute FFT
        fftwf_execute(plan);

        // Copy result (only magnitude needed)
        for (int j = 0; j < window_length / 2 + 1; j++) {
            stft_result[i * (window_length / 2 + 1) + j][0] = out[j][0];
            stft_result[i * (window_length / 2 + 1) + j][1] = out[j][1];
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

// Compute magnitude (single value)
static float compute_magnitude(const fftwf_complex& value) {
    return value[0] * value[0] + value[1] * value[1];
}

// Compute magnitudes from transposed STFT (RK style)
// Input: stft_result_t is [num_freqs x num_frames] after transpose
// Output: magnitudes is [num_freqs x (num_frames-1)] - discards last frame!
static void compute_magnitudes(fftwf_complex* stft_result_t, int num_freqs,
                              int num_frames, std::vector<float>& magnitudes) {
    int k = 0;
    for (int i = 0; i < num_freqs; i++) {
        for (int j = 0; j < num_frames - 1; j++) {  // Note: num_frames - 1
            magnitudes[k] = compute_magnitude(stft_result_t[i * num_frames + j]);
            k++;
        }
    }
}

static void clamp_and_log_max(std::vector<float>& mel_spec, int rows, int cols) {
    float min_val = 1e-10f;
    float scaling_factor = 1.0f / 4.0f;
    float shift_value = 4.0f;

    // Apply log10 first, then find max
    float max_val = -1e10f;  // Start with very small value
    for (int i = 0; i < rows * cols; ++i) {
        float value = mel_spec[i];
        value = (value < min_val) ? min_val : value;
        mel_spec[i] = log10f(value);

        if (mel_spec[i] > max_val) {
            max_val = mel_spec[i];
        }
    }

    // Clamp and scale
    float threshold = max_val - 8.0f;
    for (int i = 0; i < rows * cols; ++i) {
        mel_spec[i] = (std::max(mel_spec[i], threshold) + shift_value) * scaling_factor;
    }
}

// Transpose complex STFT array (RK style)
static void transpose_complex(fftwf_complex* input, int input_rows, int input_cols,
                             fftwf_complex* output) {
    for (int i = 0; i < input_rows; ++i) {
        for (int j = 0; j < input_cols; ++j) {
            int input_index = i * input_cols + j;
            int output_index = j * input_rows + i;

            output[output_index][0] = input[input_index][0];
            output[output_index][1] = input[input_index][1];
        }
    }
}

// Simple matrix multiplication
static void matmul(float* A, float* B, std::vector<float>& C,
                   int ROWS_A, int COLS_A, int COLS_B) {
    for (int i = 0; i < ROWS_A; i++) {
        for (int j = 0; j < COLS_B; j++) {
            float sum = 0.0f;
            for (int k = 0; k < COLS_A; k++) {
                sum += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            C[i * COLS_B + j] = sum;
        }
    }
}

static void pad_x_mel(const std::vector<float> input,
                     int rows_input, int cols_input,
                     std::vector<float>& output, int cols_output) {
    for (int i = 0; i < rows_input; ++i) {
        std::copy(input.begin() + i * cols_input,
                 input.begin() + (i + 1) * cols_input,
                 output.begin() + i * cols_output);
    }
}

void audio_preprocess(audio_buffer_t* audio, float* mel_filters,
                     std::vector<float>& x_mel) {
    int audio_length = audio->num_frames;

    // Create hann window
    std::vector<float> window(N_FFT);
    hann_window(window, N_FFT);

    // Convert audio data to vector
    std::vector<float> audio_vec(audio->data, audio->data + audio_length);

    // Padding (reflect pad)
    int padded_size = audio_length + N_FFT;
    std::vector<float> padded_audio(padded_size);
    reflect_pad(audio_vec, padded_audio, N_FFT / 2);

    // Compute number of STFT frames
    int cur_num_frames_of_stfts = (padded_size - N_FFT) / HOP_LENGTH + 1;

    // Step 1: Compute STFT as [num_frames x num_freqs]
    int num_freqs = N_FFT / 2 + 1;  // 201 (MELS_FILTERS_SIZE)
    fftwf_complex* stfts_result = (fftwf_complex*)fftwf_malloc(
        sizeof(fftwf_complex) * num_freqs * cur_num_frames_of_stfts);

    compute_stft(padded_audio, padded_size, N_FFT, HOP_LENGTH,
                window, stfts_result, cur_num_frames_of_stfts);

    // Step 2: Transpose STFT to [num_freqs x num_frames]
    fftwf_complex* stfts_result_t = (fftwf_complex*)fftwf_malloc(
        sizeof(fftwf_complex) * num_freqs * cur_num_frames_of_stfts);

    transpose_complex(stfts_result, cur_num_frames_of_stfts, num_freqs, stfts_result_t);

    // Step 3: Compute magnitudes [num_freqs x (num_frames-1)] - discards last frame
    std::vector<float> magnitudes(num_freqs * (cur_num_frames_of_stfts - 1));
    compute_magnitudes(stfts_result_t, num_freqs, cur_num_frames_of_stfts, magnitudes);

    // Free FFTW memory
    fftwf_free(stfts_result);
    fftwf_free(stfts_result_t);

    // Step 4: Apply mel filters
    // mel_filters: [N_MELS x num_freqs], magnitudes: [num_freqs x (num_frames-1)]
    int ROWS_A = N_MELS;
    int COLS_A = num_freqs;
    int COLS_B = cur_num_frames_of_stfts - 1;

    std::vector<float> cur_x_mel(ROWS_A * COLS_B);
    matmul(mel_filters, magnitudes.data(), cur_x_mel, ROWS_A, COLS_A, COLS_B);

    // Step 5: Apply clamp and log transform
    clamp_and_log_max(cur_x_mel, ROWS_A, COLS_B);

    // Step 6: Pad to 3000 columns
    int target_cols = MAX_AUDIO_LENGTH / HOP_LENGTH;  // 3000
    x_mel.resize(N_MELS * target_cols, 0.0f);
    pad_x_mel(cur_x_mel, N_MELS, COLS_B, x_mel, target_cols);

    std::cout << "[INFO] Computed mel spectrogram: ["
              << N_MELS << " x " << target_cols << "]" << std::endl;
}

// ==================== Vocabulary ====================

int read_vocab(const char* filename, VocabEntry* vocab) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("[ERROR] Error opening vocabulary file");
        return -1;
    }

    char line[512];
    int count = 0;

    while (fgets(line, sizeof(line), fp) && count < VOCAB_NUM) {
        // Parse line format: "index token"
        char* space = strchr(line, ' ');
        if (space) {
            *space = '\0';
            vocab[count].index = atoi(line);
            vocab[count].token = std::string(space + 1);

            // Remove trailing newline
            if (!vocab[count].token.empty() &&
                vocab[count].token.back() == '\n') {
                vocab[count].token.pop_back();
            }

            count++;
        }
    }

    fclose(fp);
    std::cout << "[INFO] Loaded " << count << " vocabulary entries" << std::endl;

    return 0;
}

// ==================== Text Processing ====================

void replace_substr(std::string& str, const std::string& from,
                   const std::string& to) {
    if (from.empty()) return;

    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

static int32_t get_char_index(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    } else if (c >= 'a' && c <= 'z') {
        return c - 'a' + ('Z' - 'A') + 1;
    } else if (c >= '0' && c <= '9') {
        return c - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
    } else if (c == '+') {
        return 62;
    } else if (c == '/') {
        return 63;
    }

    std::cerr << "[WARN] Unknown character in base64: "
              << static_cast<int>(c) << std::endl;
    return -1;
}

std::string base64_decode(const std::string& encoded_string) {
    if (encoded_string.empty()) {
        std::cerr << "[ERROR] Empty base64 string" << std::endl;
        return "";
    }

    int32_t output_length = static_cast<int32_t>(encoded_string.size()) / 4 * 3;
    std::string decoded_string;
    decoded_string.reserve(output_length);

    int32_t index = 0;
    while (index < static_cast<int32_t>(encoded_string.size())) {
        if (encoded_string[index] == '=') {
            return " ";
        }

        int32_t first_byte = (get_char_index(encoded_string[index]) << 2) +
                            ((get_char_index(encoded_string[index + 1]) & 0x30) >> 4);
        decoded_string.push_back(static_cast<char>(first_byte));

        if (index + 2 < static_cast<int32_t>(encoded_string.size()) &&
            encoded_string[index + 2] != '=') {
            int32_t second_byte = ((get_char_index(encoded_string[index + 1]) & 0x0f) << 4) +
                                 ((get_char_index(encoded_string[index + 2]) & 0x3c) >> 2);
            decoded_string.push_back(static_cast<char>(second_byte));

            if (index + 3 < static_cast<int32_t>(encoded_string.size()) &&
                encoded_string[index + 3] != '=') {
                int32_t third_byte = ((get_char_index(encoded_string[index + 2]) & 0x03) << 6) +
                                    get_char_index(encoded_string[index + 3]);
                decoded_string.push_back(static_cast<char>(third_byte));
            }
        }
        index += 4;
    }

    return decoded_string;
}

int argmax(float* array) {
    int start_index = (MAX_TOKENS - 1) * 1 * VOCAB_NUM;
    int max_index = start_index;
    float max_value = array[start_index];

    for (int i = start_index + 1; i < start_index + VOCAB_NUM; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }

    int relative_index = max_index - start_index;
    return relative_index;
}
