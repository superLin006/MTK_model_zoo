/**
 * Real-ESRGAN Super-Resolution - MTK NPU Implementation
 * Using official Neuron API (NeuronModel, NeuronCompilation, NeuronExecution)
 *
 * Real-ESRGAN特点：
 * - 输入输出都在[0,1]范围，不需要MeanShift归一化
 * - 前处理：uint8 [0,255] -> float32 [0,1]
 * - 后处理：float32 [0,1] -> uint8 [0,255]
 */

#include "realesrgan.h"
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
 * @brief Preprocess image: RGB uint8 [H,W,3] -> float32 [1,3,H,W] (normalize to [0,1])
 *
 * Real-ESRGAN的输入需要在[0,1]范围内，所以只需除以255.0
 */
static void preprocess_image(const unsigned char* input_rgb,
                             float* output,
                             int height,
                             int width) {
    // input_rgb: [H, W, 3] row-major (RGB)
    // output: [1, 3, H, W] NCHW format

    float scale = 1.0f / 255.0f;

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int in_idx = h * width * 3 + w * 3 + c;
                int out_idx = c * height * width + h * width + w;

                // Normalize to [0, 1]
                output[out_idx] = (float)input_rgb[in_idx] * scale;
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
 * @brief Postprocess image: float32 [1,3,H,W] -> RGB uint8 [H,W,3] (denormalize from [0,1])
 *
 * Real-ESRGAN的输出在[0,1]范围内，需要乘以255.0并clip到[0,255]
 */
static void postprocess_image(const float* input,
                               unsigned char* output,
                               int height,
                               int width) {
    // input: [1, 3, H, W] NCHW format
    // output: [H, W, 3] row-major (RGB)

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int in_idx = c * height * width + h * width + w;
                int out_idx = h * width * 3 + w * 3 + c;

                // Denormalize from [0, 1] to [0, 255] and clip
                float val = input[in_idx];
                val = (val < 0.0f) ? 0.0f : (val > 1.0f) ? 1.0f : val;
                output[out_idx] = (unsigned char)(val * 255.0f + 0.5f);
            }
        }
    }
}

// ==================== Real-ESRGAN API ====================

/**
 * @brief Initialize Real-ESRGAN model
 */
int realesrgan_init(const char* model_path, realesrgan_context_t* ctx) {
    if (!model_path || !ctx) {
        std::cerr << "[ERROR] Invalid parameters" << std::endl;
        return -1;
    }

    memset(ctx, 0, sizeof(realesrgan_context_t));

    // 默认参数（339x510输入，4倍超分 -> 1356x2040输出）
    ctx->input_height = 339;
    ctx->input_width = 510;
    ctx->input_channels = 3;
    ctx->scale_factor = 4;
    ctx->output_height = ctx->input_height * ctx->scale_factor;
    ctx->output_width = ctx->input_width * ctx->scale_factor;
    ctx->output_channels = 3;

    // 计算buffer大小
    ctx->input_size = ctx->input_height * ctx->input_width * ctx->input_channels * sizeof(float);
    ctx->output_size = ctx->output_height * ctx->output_width * ctx->output_channels * sizeof(float);

    std::cout << "[INFO] Real-ESRGAN Model Configuration:" << std::endl;
    std::cout << "[INFO]   Input:  " << ctx->input_width << "x" << ctx->input_height << "x" << ctx->input_channels << std::endl;
    std::cout << "[INFO]   Output: " << ctx->output_width << "x" << ctx->output_height << "x" << ctx->output_channels << std::endl;
    std::cout << "[INFO]   Scale:  x" << ctx->scale_factor << std::endl;

    // 创建Neuron执行器
    // 输入形状: [1, 3, 339, 510]
    // 输出形状: [1, 3, 1356, 2040]
    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, 3, (uint32_t)ctx->input_height, (uint32_t)ctx->input_width}
    };
    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, 3, (uint32_t)ctx->output_height, (uint32_t)ctx->output_width}
    };

    NeuronExecutor* executor = new NeuronExecutor(model_path, input_shapes, output_shapes, "RealESRGAN");
    if (!executor) {
        std::cerr << "[ERROR] Failed to create NeuronExecutor" << std::endl;
        return -1;
    }

    if (!executor->Initialize()) {
        std::cerr << "[ERROR] Failed to initialize NeuronExecutor" << std::endl;
        delete executor;
        return -1;
    }

    ctx->neuron_executor = (void*)executor;
    ctx->initialized = 1;

    std::cout << "[INFO] Real-ESRGAN model initialized successfully" << std::endl;

    return 0;
}

/**
 * @brief Release Real-ESRGAN model resources
 */
int realesrgan_release(realesrgan_context_t* ctx) {
    if (!ctx || !ctx->initialized) {
        return -1;
    }

    if (ctx->neuron_executor) {
        NeuronExecutor* executor = (NeuronExecutor*)ctx->neuron_executor;
        delete executor;
        ctx->neuron_executor = nullptr;
    }

    ctx->initialized = 0;

    std::cout << "[INFO] Real-ESRGAN model released" << std::endl;

    return 0;
}

/**
 * @brief Run super-resolution inference (assumes input matches model size)
 */
int realesrgan_inference(realesrgan_context_t* ctx,
                         const unsigned char* input_rgb,
                         unsigned char** output_rgb) {
    if (!ctx || !ctx->initialized || !input_rgb || !output_rgb) {
        std::cerr << "[ERROR] Invalid parameters" << std::endl;
        return -1;
    }

    NeuronExecutor* executor = (NeuronExecutor*)ctx->neuron_executor;

    // 分配输入buffer（float32）
    std::vector<float> input_buffer(ctx->input_height * ctx->input_width * 3);

    // 前处理：uint8 [0,255] -> float32 [0,1]
    preprocess_image(input_rgb, input_buffer.data(), ctx->input_height, ctx->input_width);

    // 分配输出buffer（float32）
    std::vector<float> output_buffer(ctx->output_height * ctx->output_width * 3);

    // 准备输入输出指针
    std::vector<const void*> inputs = {input_buffer.data()};
    std::vector<void*> outputs = {output_buffer.data()};

    // 运行推理
    if (!executor->Run(inputs, outputs)) {
        std::cerr << "[ERROR] Inference failed" << std::endl;
        return -1;
    }

    // 分配输出图像buffer
    *output_rgb = (unsigned char*)malloc(ctx->output_height * ctx->output_width * 3);
    if (!*output_rgb) {
        std::cerr << "[ERROR] Failed to allocate output buffer" << std::endl;
        return -1;
    }

    // 后处理：float32 [0,1] -> uint8 [0,255]
    postprocess_image(output_buffer.data(), *output_rgb, ctx->output_height, ctx->output_width);

    return 0;
}

/**
 * @brief Run super-resolution with explicit dimensions (includes resize if needed)
 */
int realesrgan_inference_ex(realesrgan_context_t* ctx,
                            const unsigned char* input_rgb,
                            int input_width,
                            int input_height,
                            unsigned char** output_rgb,
                            int* output_width,
                            int* output_height) {
    if (!ctx || !ctx->initialized || !input_rgb || !output_rgb || !output_width || !output_height) {
        std::cerr << "[ERROR] Invalid parameters" << std::endl;
        return -1;
    }

    // 检查是否需要resize
    const unsigned char* processed_input = input_rgb;

    unsigned char* resized_input = nullptr;
    if (input_width != ctx->input_width || input_height != ctx->input_height) {
        std::cout << "[INFO] Resizing input from " << input_width << "x" << input_height
                  << " to " << ctx->input_width << "x" << ctx->input_height << std::endl;

        resized_input = resize_image(input_rgb, input_width, input_height,
                                     ctx->input_width, ctx->input_height);
        if (!resized_input) {
            std::cerr << "[ERROR] Failed to resize input image" << std::endl;
            return -1;
        }
        processed_input = resized_input;
    }

    // 运行推理
    int ret = realesrgan_inference(ctx, processed_input, output_rgb);

    // 释放resize后的buffer
    if (resized_input) {
        free(resized_input);
    }

    if (ret == 0) {
        *output_width = ctx->output_width;
        *output_height = ctx->output_height;
    }

    return ret;
}
