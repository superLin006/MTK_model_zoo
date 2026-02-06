// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "realesrgan.h"
#include "image_utils.h"
#include "file_utils.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <model_path> <image_path>\n", argv[0]);
        printf("Example: %s model/RealESRGAN_x4plus_510x339_fp16.rknn model/input_510x339.png\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    int ret;
    rknn_realesrgan_context_t realesrgan_ctx;
    memset(&realesrgan_ctx, 0, sizeof(rknn_realesrgan_context_t));

    // Initialize model
    printf("========================================\n");
    printf("Real-ESRGAN Super-Resolution Demo\n");
    printf("========================================\n");
    ret = init_realesrgan_model(model_path, &realesrgan_ctx);
    if (ret != 0)
    {
        printf("init_realesrgan_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    // Read input image
    image_buffer_t src_image;
    image_buffer_t dst_image;
    image_buffer_t preprocessed_image;
    bool need_free_preprocessed = false;
    struct timeval start_time, stop_time;
    float elapsed_ms;
    const char *output_path = "output_sr.png";

    memset(&src_image, 0, sizeof(image_buffer_t));
    memset(&dst_image, 0, sizeof(image_buffer_t));
    memset(&preprocessed_image, 0, sizeof(image_buffer_t));

    ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    // Fix: read_image_stb doesn't set size field, calculate it manually
    if (src_image.size == 0)
    {
        int channels = 3;  // RGB
        if (src_image.format == IMAGE_FORMAT_RGBA8888) channels = 4;
        else if (src_image.format == IMAGE_FORMAT_GRAY8) channels = 1;
        src_image.size = src_image.width * src_image.height * channels;
    }

    printf("\nInput image loaded:\n");
    printf("  Path: %s\n", image_path);
    printf("  Size: %dx%d\n", src_image.width, src_image.height);
    printf("  Format: %d\n", src_image.format);
    printf("  Buffer size: %d bytes\n", src_image.size);

    if (src_image.size == 0 || src_image.virt_addr == NULL)
    {
        printf("ERROR: Invalid image buffer!\n");
        ret = -1;
        goto out;
    }

    // Convert RGBA to RGB if needed

    if (src_image.format == IMAGE_FORMAT_RGBA8888)
    {
        printf("Converting RGBA to RGB...\n");
        preprocessed_image.width = src_image.width;
        preprocessed_image.height = src_image.height;
        preprocessed_image.format = IMAGE_FORMAT_RGB888;
        preprocessed_image.size = src_image.width * src_image.height * 3;
        preprocessed_image.virt_addr = (unsigned char*)malloc(preprocessed_image.size);

        if (preprocessed_image.virt_addr == NULL)
        {
            printf("ERROR: Failed to allocate memory for RGB conversion!\n");
            ret = -1;
            goto out;
        }
        need_free_preprocessed = true;

        // Convert RGBA to RGB
        unsigned char* src_ptr = (unsigned char*)src_image.virt_addr;
        unsigned char* dst_ptr = preprocessed_image.virt_addr;
        for (int i = 0; i < src_image.width * src_image.height; i++)
        {
            dst_ptr[i * 3 + 0] = src_ptr[i * 4 + 0];  // R
            dst_ptr[i * 3 + 1] = src_ptr[i * 4 + 1];  // G
            dst_ptr[i * 3 + 2] = src_ptr[i * 4 + 2];  // B
            // Skip alpha channel
        }
    }
    else if (src_image.format == IMAGE_FORMAT_RGB888)
    {
        // Already RGB888, use as-is
        preprocessed_image = src_image;
    }
    else
    {
        printf("ERROR: Unsupported image format: %d\n", src_image.format);
        printf("       Only RGB888 and RGBA8888 are supported.\n");
        ret = -1;
        goto out;
    }

    // Check if resize is needed
    if (preprocessed_image.width != realesrgan_ctx.model_width ||
        preprocessed_image.height != realesrgan_ctx.model_height)
    {
        printf("WARNING: Input image size (%dx%d) doesn't match model size (%dx%d)\n",
               preprocessed_image.width, preprocessed_image.height,
               realesrgan_ctx.model_width, realesrgan_ctx.model_height);
        printf("         Model expects exactly %dx%d input.\n",
               realesrgan_ctx.model_width, realesrgan_ctx.model_height);
        printf("         Please resize your input image before running inference.\n");
        ret = -1;
        if (need_free_preprocessed) free(preprocessed_image.virt_addr);
        goto out;
    }

    // Run inference with timing
    gettimeofday(&start_time, NULL);

    ret = inference_realesrgan_model(&realesrgan_ctx, &preprocessed_image, &dst_image);

    gettimeofday(&stop_time, NULL);
    if (ret != 0)
    {
        printf("inference_realesrgan_model fail! ret=%d\n", ret);
        if (need_free_preprocessed) free(preprocessed_image.virt_addr);
        goto out;
    }

    // Calculate inference time
    elapsed_ms = (stop_time.tv_sec - start_time.tv_sec) * 1000.0f +
                 (stop_time.tv_usec - start_time.tv_usec) / 1000.0f;

    printf("\n========================================\n");
    printf("Inference Results:\n");
    printf("  Input:  %dx%d\n", src_image.width, src_image.height);
    printf("  Output: %dx%d\n", dst_image.width, dst_image.height);
    printf("  Scale:  %dx\n", realesrgan_ctx.scale_factor);
    printf("  Time:   %.2f ms\n", elapsed_ms);
    printf("  FPS:    %.2f\n", 1000.0f / elapsed_ms);
    printf("========================================\n");

    // Save output
    ret = write_image(output_path, &dst_image);
    if (ret == 0)
    {
        printf("\nSuper-resolved image saved to: %s\n", output_path);
    }
    else
    {
        printf("\nFailed to save output image as PNG!\n");
        // Fallback: Save as PPM format (simple binary format)
        char ppm_path[256];
        snprintf(ppm_path, sizeof(ppm_path), "output_sr.ppm");
        FILE *fp = fopen(ppm_path, "wb");
        if (fp)
        {
            fprintf(fp, "P6\n%d %d\n255\n", dst_image.width, dst_image.height);
            fwrite(dst_image.virt_addr, 1, dst_image.size, fp);
            fclose(fp);
            printf("Saved output as PPM: %s\n", ppm_path);
        }
    }

out:
    // Clean up
    if (need_free_preprocessed && preprocessed_image.virt_addr != NULL)
    {
        free(preprocessed_image.virt_addr);
    }
    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    if (dst_image.virt_addr != NULL)
    {
        free(dst_image.virt_addr);
    }

    ret = release_realesrgan_model(&realesrgan_ctx);
    if (ret != 0)
    {
        printf("release_realesrgan_model fail! ret=%d\n", ret);
    }

    return 0;
}
