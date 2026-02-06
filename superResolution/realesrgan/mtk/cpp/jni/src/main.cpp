/**
 * Real-ESRGAN Super-Resolution - Main Program
 * Usage: realesrgan_inference <model_path> <input_image> <output_image>
 */

#include "realesrgan.h"
#include <iostream>
#include <chrono>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <input_image> <output_image>" << std::endl;
        std::cout << "Example: " << argv[0] << " RealESRGAN_x4plus_339x510_MT8371.dla input.png output.png" << std::endl;
        return -1;
    }

    const char* model_path = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];

    std::cout << "========================================" << std::endl;
    std::cout << "Real-ESRGAN Super-Resolution (MTK NPU)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model:   " << model_path << std::endl;
    std::cout << "Input:   " << input_path << std::endl;
    std::cout << "Output:  " << output_path << std::endl;
    std::cout << "========================================" << std::endl;

    int ret;

    // Initialize Real-ESRGAN
    realesrgan_context_t realesrgan_ctx;
    memset(&realesrgan_ctx, 0, sizeof(realesrgan_ctx));

    auto start_init = std::chrono::high_resolution_clock::now();
    ret = realesrgan_init(model_path, &realesrgan_ctx);
    auto end_init = std::chrono::high_resolution_clock::now();

    if (ret != 0) {
        std::cerr << "Failed to initialize Real-ESRGAN model!" << std::endl;
        return -1;
    }

    double init_time = std::chrono::duration<double, std::milli>(end_init - start_init).count();
    std::cout << "[INFO] Initialization time: " << init_time << " ms" << std::endl;

    // Load input image
    std::cout << "\n[INFO] Loading input image..." << std::endl;
    int width, height, channels;
    unsigned char* input_image = stbi_load(input_path, &width, &height, &channels, 3);
    if (!input_image) {
        std::cerr << "[ERROR] Failed to load input image: " << input_path << std::endl;
        realesrgan_release(&realesrgan_ctx);
        return -1;
    }

    std::cout << "[INFO]   Original size: " << width << "x" << height << "x" << channels << std::endl;

    // Run inference
    std::cout << "\n[INFO] Running super-resolution..." << std::endl;
    unsigned char* output_image = nullptr;
    int out_width, out_height;

    auto start_inference = std::chrono::high_resolution_clock::now();
    ret = realesrgan_inference_ex(&realesrgan_ctx, input_image, width, height,
                                  &output_image, &out_width, &out_height);
    auto end_inference = std::chrono::high_resolution_clock::now();

    if (ret != 0) {
        std::cerr << "[ERROR] Inference failed!" << std::endl;
        stbi_image_free(input_image);
        realesrgan_release(&realesrgan_ctx);
        return -1;
    }

    double inference_time = std::chrono::duration<double, std::milli>(end_inference - start_inference).count();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Input:    " << width << "x" << height << std::endl;
    std::cout << "  Output:   " << out_width << "x" << out_height << std::endl;
    std::cout << "  Scale:    x" << realesrgan_ctx.scale_factor << std::endl;
    std::cout << "  Time:     " << inference_time << " ms" << std::endl;
    std::cout << "  FPS:      " << (1000.0 / inference_time) << std::endl;
    std::cout << "========================================" << std::endl;

    // Save output image
    std::cout << "\n[INFO] Saving output image..." << std::endl;
    int save_result = stbi_write_png(output_path, out_width, out_height, 3,
                                     output_image, out_width * 3);
    if (save_result == 0) {
        std::cerr << "[ERROR] Failed to save output image: " << output_path << std::endl;
        stbi_image_free(input_image);
        free(output_image);
        realesrgan_release(&realesrgan_ctx);
        return -1;
    }

    std::cout << "[INFO] Output saved to: " << output_path << std::endl;

    // Cleanup
    stbi_image_free(input_image);
    free(output_image);
    realesrgan_release(&realesrgan_ctx);

    std::cout << "\n[INFO] Done!" << std::endl;
    return 0;
}
