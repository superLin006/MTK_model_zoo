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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "rcan.h"
#include "image_utils.h"
#include "file_utils.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <model_path> <image_path>\n", argv[0]);
        printf("Example: %s model/rcan_x4_fp.rknn model/test_input_256x256.png\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    int ret;
    rknn_rcan_context_t rcan_ctx;
    memset(&rcan_ctx, 0, sizeof(rknn_rcan_context_t));

    // Initialize model
    printf("========================================\n");
    printf("RCAN Super-Resolution Demo\n");
    printf("========================================\n");
    printf("DEBUG: About to init model...\n");
    fflush(stdout);
    ret = init_rcan_model(model_path, &rcan_ctx);
    printf("DEBUG: init_rcan_model returned %d\n", ret);
    fflush(stdout);
    if (ret != 0)
    {
        printf("init_rcan_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }
    printf("DEBUG: Model initialized successfully\n");
    fflush(stdout);

    // Read input image
    image_buffer_t src_image;
    image_buffer_t dst_image;
    struct timeval start_time, stop_time;
    float elapsed_ms;
    const char *output_path = "output_sr.png";

    memset(&src_image, 0, sizeof(image_buffer_t));
    memset(&dst_image, 0, sizeof(image_buffer_t));

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
    printf("  Buffer addr: %p\n", src_image.virt_addr);

    if (src_image.size == 0 || src_image.virt_addr == NULL)
    {
        printf("ERROR: Invalid image buffer!\n");
        ret = -1;
        goto out;
    }

    // Run inference with timing
    gettimeofday(&start_time, NULL);

    ret = inference_rcan_model(&rcan_ctx, &src_image, &dst_image);

    gettimeofday(&stop_time, NULL);
    if (ret != 0)
    {
        printf("inference_rcan_model fail! ret=%d\n", ret);
        goto out;
    }

    // Calculate inference time
    elapsed_ms = (stop_time.tv_sec - start_time.tv_sec) * 1000.0f +
                 (stop_time.tv_usec - start_time.tv_usec) / 1000.0f;

    printf("\n========================================\n");
    printf("Inference Results:\n");
    printf("  Input:  %dx%d\n", src_image.width, src_image.height);
    printf("  Output: %dx%d\n", dst_image.width, dst_image.height);
    printf("  Scale:  %dx\n", rcan_ctx.scale_factor);
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
        printf("\nFailed to save output image!\n");
    }

out:
    // Clean up
    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    if (dst_image.virt_addr != NULL)
    {
        free(dst_image.virt_addr);
    }

    ret = release_rcan_model(&rcan_ctx);
    if (ret != 0)
    {
        printf("release_rcan_model fail! ret=%d\n", ret);
    }

    return 0;
}
