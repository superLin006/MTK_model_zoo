/**
 * RCAN Super-Resolution - MTK NPU Implementation
 * Using official Neuron API (NeuronModel, NeuronCompilation, NeuronExecution)
 */

#include "rcan.h"
#include "mtk_npu/neuron_executor.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

// ==================== Preprocessing ====================

/**
 * @brief Preprocess image: RGB uint8 [H,W,3] -> float [1,3,H,W] (subtract mean)
 */
static void preprocess_image(const unsigned char* input_rgb,
                             float* output,
                             int height,
                             int width,
                             const float* rgb_mean,
                             float rgb_range) {
    // input_rgb: [H, W, 3] row-major
    // output: [1, 3, H, W] NCHW format

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int in_idx = h * width * 3 + w * 3 + c;
                int out_idx = c * height * width + h * width + w;

                float pixel = (float)input_rgb[in_idx];
                output[out_idx] = pixel - rgb_mean[c] * rgb_range;
            }
        }
    }
}

/**
 * @brief Resize image using bilinear interpolation
 */
static unsigned char* resize_image(const unsigned char* input,
                                    int in_width, int in_height,
                                    int out_width, int out_height) {
    unsigned char* output = (unsigned char*)malloc(out_width * out_height * 3);
    if (!output) {
        std::cerr << "[ERROR] Failed to allocate memory for resize" << std::endl;
        return nullptr;
    }

    // Simple bilinear interpolation
    float x_ratio = (float)in_width / out_width;
    float y_ratio = (float)in_height / out_height;

    for (int y = 0; y < out_height; y++) {
        for (int x = 0; x < out_width; x++) {
            float px = x * x_ratio;
            float py = y * y_ratio;

            int x0 = (int)px;
            int y0 = (int)py;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x1 >= in_width) x1 = in_width - 1;
            if (y1 >= in_height) y1 = in_height - 1;

            float dx = px - x0;
            float dy = py - y0;

            for (int c = 0; c < 3; c++) {
                int idx = y * out_width * 3 + x * 3 + c;

                float p00 = input[y0 * in_width * 3 + x0 * 3 + c];
                float p01 = input[y0 * in_width * 3 + x1 * 3 + c];
                float p10 = input[y1 * in_width * 3 + x0 * 3 + c];
                float p11 = input[y1 * in_width * 3 + x1 * 3 + c];

                float val = (1 - dx) * (1 - dy) * p00 +
                           dx * (1 - dy) * p01 +
                           (1 - dx) * dy * p10 +
                           dx * dy * p11;

                output[idx] = (unsigned char)(val + 0.5f);
            }
        }
    }

    return output;
}

// ==================== Postprocessing ====================

/**
 * @brief Postprocess image: float [1,3,H,W] -> RGB uint8 [H,W,3] (add mean)
 */
static void postprocess_image(const float* input,
                              unsigned char* output_rgb,
                              int height,
                              int width,
                              const float* rgb_mean,
                              float rgb_range) {
    // input: [1, 3, H, W] NCHW format
    // output_rgb: [H, W, 3] row-major

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int in_idx = c * height * width + h * width + w;
                int out_idx = h * width * 3 + w * 3 + c;

                float val = input[in_idx] + rgb_mean[c] * rgb_range;
                val = std::max(0.0f, std::min(255.0f, val));
                output_rgb[out_idx] = (unsigned char)(val + 0.5f);
            }
        }
    }
}

// ==================== Public API Implementation ====================

int rcan_init(const char* model_path, rcan_context_t* ctx) {
    if (!model_path || !ctx) {
        std::cerr << "[ERROR] Invalid parameters" << std::endl;
        return -1;
    }

    memset(ctx, 0, sizeof(rcan_context_t));

    std::cout << "[INFO] Initializing RCAN..." << std::endl;
    std::cout << "[INFO] Model path: " << model_path << std::endl;

    // Create NeuronExecutor
    // Input shape: [1, 3, 339, 510] (NCHW format)
    // Output shape: [1, 3, 1356, 2040]
    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, 3, 339, 510}  // [N, C, H, W]
    };

    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, 3, 1356, 2040}  // [N, C, H, W]
    };

    std::unique_ptr<NeuronExecutor> executor(
        new NeuronExecutor(model_path, input_shapes, output_shapes, "RCAN"));

    if (!executor->Initialize()) {
        std::cerr << "[ERROR] Failed to initialize NeuronExecutor" << std::endl;
        return -1;
    }

    // Store executor in context (as void* to avoid exposing NeuronExecutor in header)
    ctx->neuron_executor = executor.release();

    // Set model parameters
    ctx->input_width = 510;
    ctx->input_height = 339;
    ctx->input_channels = 3;
    ctx->output_width = 2040;
    ctx->output_height = 1356;
    ctx->output_channels = 3;
    ctx->scale_factor = 4;

    // Set MeanShift parameters (from training config)
    ctx->rgb_mean[0] = 0.4488f;
    ctx->rgb_mean[1] = 0.4371f;
    ctx->rgb_mean[2] = 0.4040f;
    ctx->rgb_range = 255.0f;

    ctx->initialized = 1;

    std::cout << "[INFO] RCAN initialized successfully!" << std::endl;
    std::cout << "[INFO]   Input:  " << ctx->input_width << "x" << ctx->input_height << std::endl;
    std::cout << "[INFO]   Output: " << ctx->output_width << "x" << ctx->output_height << std::endl;
    std::cout << "[INFO]   Scale:  " << ctx->scale_factor << "x" << std::endl;

    return 0;
}

int rcan_release(rcan_context_t* ctx) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }

    std::cout << "[INFO] Releasing RCAN resources..." << std::endl;

    if (ctx->neuron_executor) {
        NeuronExecutor* executor = static_cast<NeuronExecutor*>(ctx->neuron_executor);
        delete executor;
        ctx->neuron_executor = nullptr;
    }

    ctx->initialized = 0;

    return 0;
}

int rcan_inference(rcan_context_t* ctx,
                   const unsigned char* input_rgb,
                   unsigned char** output_rgb) {
    return rcan_inference_ex(ctx, input_rgb, ctx->input_width, ctx->input_height,
                            output_rgb, nullptr, nullptr);
}

int rcan_inference_ex(rcan_context_t* ctx,
                      const unsigned char* input_rgb,
                      int input_width,
                      int input_height,
                      unsigned char** output_rgb,
                      int* output_width,
                      int* output_height) {
    if (!ctx || !ctx->initialized || !input_rgb || !output_rgb) {
        std::cerr << "[ERROR] Invalid parameters" << std::endl;
        return -1;
    }

    std::cout << "[INFO] Running RCAN inference..." << std::endl;
    std::cout << "[INFO]   Input size: " << input_width << "x" << input_height << std::endl;

    NeuronExecutor* executor = static_cast<NeuronExecutor*>(ctx->neuron_executor);

    // Step 1: Resize input if needed
    unsigned char* resized_input = nullptr;
    if (input_width != ctx->input_width || input_height != ctx->input_height) {
        std::cout << "[INFO]   Resizing input to " << ctx->input_width << "x" << ctx->input_height << std::endl;
        resized_input = resize_image(input_rgb, input_width, input_height,
                                     ctx->input_width, ctx->input_height);
        if (!resized_input) {
            std::cerr << "[ERROR] Failed to resize image" << std::endl;
            return -1;
        }
    } else {
        // Use input directly (need to copy as it's const)
        resized_input = (unsigned char*)malloc(input_width * input_height * 3);
        memcpy(resized_input, input_rgb, input_width * input_height * 3);
    }

    // Step 2: Preprocess (RGB -> float, subtract mean)
    float* preprocessed = (float*)malloc(ctx->input_width * ctx->input_height * 3 * sizeof(float));
    if (!preprocessed) {
        std::cerr << "[ERROR] Failed to allocate preprocessing buffer" << std::endl;
        free(resized_input);
        return -1;
    }

    preprocess_image(resized_input, preprocessed, ctx->input_height, ctx->input_width,
                     ctx->rgb_mean, ctx->rgb_range);

    free(resized_input);

    // Step 3: Run NPU inference
    std::cout << "[INFO]   Running NPU inference..." << std::endl;

    // Prepare input and output vectors
    std::vector<const void*> inputs = {preprocessed};

    float* inference_output = (float*)malloc(ctx->output_width * ctx->output_height * 3 * sizeof(float));
    if (!inference_output) {
        std::cerr << "[ERROR] Failed to allocate output buffer" << std::endl;
        free(preprocessed);
        return -1;
    }

    std::vector<void*> outputs = {inference_output};

    if (!executor->Run(inputs, outputs)) {
        std::cerr << "[ERROR] NPU inference failed" << std::endl;
        free(preprocessed);
        free(inference_output);
        return -1;
    }

    // Step 4: Postprocess (float -> RGB, add mean)
    *output_rgb = (unsigned char*)malloc(ctx->output_height * ctx->output_width * 3);
    if (!*output_rgb) {
        std::cerr << "[ERROR] Failed to allocate output RGB buffer" << std::endl;
        free(preprocessed);
        free(inference_output);
        return -1;
    }

    postprocess_image(inference_output, *output_rgb, ctx->output_height, ctx->output_width,
                     ctx->rgb_mean, ctx->rgb_range);

    // Cleanup
    free(preprocessed);
    free(inference_output);

    if (output_width) *output_width = ctx->output_width;
    if (output_height) *output_height = ctx->output_height;

    std::cout << "[INFO]   Inference complete!" << std::endl;
    std::cout << "[INFO]   Output size: " << ctx->output_width << "x" << ctx->output_height << std::endl;

    return 0;
}
