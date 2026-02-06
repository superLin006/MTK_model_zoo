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

#ifndef _RKNN_DEMO_REALESRGAN_H_
#define _RKNN_DEMO_REALESRGAN_H_

#include "rknn_api.h"
#include "common.h"

// Real-ESRGAN application context
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;

    // Model input/output dimensions
    int model_channel;      // Input channels (should be 3 for RGB)
    int model_width;        // Input width (e.g., 510)
    int model_height;       // Input height (e.g., 339)
    int output_width;       // Output width (e.g., 2040 for 4x)
    int output_height;      // Output height (e.g., 1356 for 4x)
    int scale_factor;       // Super-resolution scale (e.g., 4)

    // Quantization flag
    bool is_quant;
} rknn_realesrgan_context_t;


/**
 * @brief Initialize Real-ESRGAN model
 *
 * @param model_path Path to RKNN model file
 * @param app_ctx Application context to be initialized
 * @return 0 on success, negative on failure
 */
int init_realesrgan_model(const char* model_path, rknn_realesrgan_context_t* app_ctx);

/**
 * @brief Release Real-ESRGAN model resources
 *
 * @param app_ctx Application context
 * @return 0 on success, negative on failure
 */
int release_realesrgan_model(rknn_realesrgan_context_t* app_ctx);

/**
 * @brief Run super-resolution inference
 *
 * @param app_ctx Application context
 * @param src_img Input low-resolution image (RGB888 format)
 * @param dst_img Output high-resolution image (RGB888 format, will be allocated)
 * @return 0 on success, negative on failure
 */
int inference_realesrgan_model(rknn_realesrgan_context_t* app_ctx,
                               image_buffer_t* src_img,
                               image_buffer_t* dst_img);

#endif //_RKNN_DEMO_REALESRGAN_H_
