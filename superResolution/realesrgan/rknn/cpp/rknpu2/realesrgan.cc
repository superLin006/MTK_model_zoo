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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "realesrgan.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

// FP16 to float conversion
static inline float fp16_to_float(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t fraction = h & 0x3ff;

    if (exponent == 0)
    {
        if (fraction == 0)
        {
            // Zero
            return sign ? -0.0f : 0.0f;
        }
        else
        {
            // Denormalized number
            float f = fraction / 1024.0f;
            f = f / 16384.0f;
            return sign ? -f : f;
        }
    }
    else if (exponent == 31)
    {
        if (fraction == 0)
        {
            // Infinity
            return sign ? -INFINITY : INFINITY;
        }
        else
        {
            // NaN
            return NAN;
        }
    }
    else
    {
        // Normalized number
        uint32_t f32_exp = exponent - 15 + 127;
        uint32_t f32_frac = fraction << 13;
        uint32_t f32 = (sign << 31) | (f32_exp << 23) | f32_frac;
        return *(float *)&f32;
    }
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_realesrgan_model(const char *model_path, rknn_realesrgan_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    printf("Loading Real-ESRGAN RKNN model...\n");

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Check quantization
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        (output_attrs[0].type == RKNN_TENSOR_INT8 || output_attrs[0].type == RKNN_TENSOR_UINT8))
    {
        app_ctx->is_quant = true;
        printf("Model is quantized (INT8)\n");
    }
    else
    {
        app_ctx->is_quant = false;
        printf("Model is FP16/FP32\n");
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // Parse input dimensions
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }

    // Parse output dimensions
    if (output_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        app_ctx->output_height = output_attrs[0].dims[2];
        app_ctx->output_width = output_attrs[0].dims[3];
    }
    else
    {
        app_ctx->output_height = output_attrs[0].dims[1];
        app_ctx->output_width = output_attrs[0].dims[2];
    }

    // Calculate scale factor
    app_ctx->scale_factor = app_ctx->output_height / app_ctx->model_height;

    printf("Model configuration:\n");
    printf("  Input:  %dx%dx%d (HxWxC)\n", app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
    printf("  Output: %dx%dx%d (HxWxC)\n", app_ctx->output_height, app_ctx->output_width, app_ctx->model_channel);
    printf("  Scale:  %dx\n", app_ctx->scale_factor);

    return 0;
}

int release_realesrgan_model(rknn_realesrgan_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_realesrgan_model(rknn_realesrgan_context_t *app_ctx, image_buffer_t *src_img, image_buffer_t *dst_img)
{
    int ret;
    int model_in_size;
    int model_out_size;
    float *input_data = NULL;
    float *output_data = NULL;
    unsigned char *dst_ptr;
    rknn_input inputs[1];
    rknn_output outputs[1];

    if ((!app_ctx) || (!src_img) || (!dst_img))
    {
        printf("Invalid parameters!\n");
        return -1;
    }

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    printf("Running Real-ESRGAN inference...\n");
    printf("Model quantization: %s\n", app_ctx->is_quant ? "INT8" : "FP16/FP32");

    // Validate input image size
    if (src_img->width != app_ctx->model_width || src_img->height != app_ctx->model_height)
    {
        printf("ERROR: Input size mismatch!\n");
        printf("  Expected: %dx%d\n", app_ctx->model_width, app_ctx->model_height);
        printf("  Got:      %dx%d\n", src_img->width, src_img->height);
        printf("  Please resize the input image to match the model's expected size.\n");
        return -1;
    }

    // Validate input format (must be RGB888)
    if (src_img->format != IMAGE_FORMAT_RGB888)
    {
        printf("ERROR: Input format must be RGB888! Got format=%d\n", src_img->format);
        return -1;
    }

    // Prepare input: Convert RGB uint8 [0, 255] to float [0, 1]
    // This matches Python test_onnx.py preprocessing:
    //   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    //   img = img.astype(np.float32) / 255.0
    //   img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    //   img = np.expand_dims(img, axis=0)   # Add batch
    model_in_size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    input_data = (float *)malloc(model_in_size * sizeof(float));
    if (input_data == NULL)
    {
        printf("malloc input buffer fail!\n");
        return -1;
    }

    unsigned char *src_ptr = (unsigned char *)src_img->virt_addr;

    // Normalization: Convert uint8 [0, 255] to float [0, 1]
    // This applies to BOTH FP16 and INT8 models
    // For INT8: RKNN will quantize [0, 1] to INT8 based on calibration parameters
    // For FP16: RKNN will use [0, 1] directly
    float normalization_factor = 1.0f / 255.0f;

    if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("Converting RGB to NCHW format [0, 1]...\n");
        // Convert HWC RGB to NCHW float
        for (int c = 0; c < app_ctx->model_channel; c++)
        {
            for (int h = 0; h < app_ctx->model_height; h++)
            {
                for (int w = 0; w < app_ctx->model_width; w++)
                {
                    int src_idx = (h * app_ctx->model_width + w) * app_ctx->model_channel + c;
                    int dst_idx = c * (app_ctx->model_height * app_ctx->model_width) + h * app_ctx->model_width + w;
                    input_data[dst_idx] = src_ptr[src_idx] * normalization_factor;
                }
            }
        }
    }
    else
    {
        printf("Converting RGB to NHWC format [0, 1]...\n");
        // Convert HWC RGB to NHWC float
        for (int i = 0; i < model_in_size; i++)
        {
            input_data[i] = src_ptr[i] * normalization_factor;
        }
    }

    // Set input data
    inputs[0].index = 0;
    inputs[0].buf = input_data;
    inputs[0].size = model_in_size * sizeof(float);
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;  // We provide FP32, RKNN will convert to FP16 if needed
    inputs[0].fmt = app_ctx->input_attrs[0].fmt;  // Use model's expected format (NHWC or NCHW)

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        free(input_data);
        return -1;
    }

    // Run inference
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        free(input_data);
        return -1;
    }

    // Get output
    outputs[0].index = 0;
    // For FP16/FP32 models, we want float output directly
    // For INT8 models, we also want RKNN to auto-dequantize to float
    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        free(input_data);
        return -1;
    }

    // Process output: Convert model output to RGB uint8
    // This matches Python test_onnx.py postprocessing:
    //   output = np.clip(output, 0, 1)
    //   output = (output * 255.0).round().astype(np.uint8)
    //   output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    //   output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    dst_img->width = app_ctx->output_width;
    dst_img->height = app_ctx->output_height;
    dst_img->format = IMAGE_FORMAT_RGB888;
    dst_img->size = app_ctx->output_width * app_ctx->output_height * 3;
    dst_img->virt_addr = (unsigned char *)malloc(dst_img->size);

    if (dst_img->virt_addr == NULL)
    {
        printf("malloc output buffer fail!\n");
        rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
        free(input_data);
        return -1;
    }

    dst_ptr = dst_img->virt_addr;
    model_out_size = app_ctx->output_width * app_ctx->output_height;

    // Process output as float (already dequantized by RKNN if INT8 model)
    printf("Processing output (already float, dequantized by RKNN if needed)...\n");
    float *float_data = (float *)outputs[0].buf;

    // Debug: Check output range
    float fmin = float_data[0];
    float fmax = float_data[0];
    double fsum = 0.0;
    for (int i = 0; i < model_out_size * 3; i++)
    {
        if (float_data[i] < fmin) fmin = float_data[i];
        if (float_data[i] > fmax) fmax = float_data[i];
        fsum += float_data[i];
    }
    float fmean = (float)(fsum / (model_out_size * 3));
    printf("Output range: [%.6f, %.6f], mean: %.6f\n", fmin, fmax, fmean);

    // Check if INT8 quantization has wrong scale (output range >> 1.0)
    // For Real-ESRGAN, expected output range is approximately [-0.15, 1.31]
    // If max > 10, it indicates wrong quantization parameters
    bool need_rescale = false;
    float rescale_factor = 1.0f;
    if (app_ctx->is_quant && fmax > 10.0f)
    {
        need_rescale = true;
        // Calculate rescale factor based on actual vs expected range
        // Expected output range: [-0.15, 1.31], range = 1.46
        // Observed output range: [fmin, fmax]
        float expected_range = 1.31f - (-0.15f);  // = 1.46
        float actual_range = fmax - fmin;
        rescale_factor = expected_range / actual_range;

        printf("WARNING: INT8 model output range is abnormal!\n");
        printf("         Expected: [-0.15, 1.31] (range: %.2f)\n", expected_range);
        printf("         Got:      [%.2f, %.2f] (range: %.2f)\n", fmin, fmax, actual_range);
        printf("         Applying rescale factor: %.6f\n", rescale_factor);
        printf("         This indicates quantization parameters are incorrect.\n");
    }

    // Check output format and convert
    if (app_ctx->output_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        // NCHW format: Convert to HWC RGB uint8
        printf("Converting NCHW float to RGB uint8...\n");

        for (int h = 0; h < app_ctx->output_height; h++)
        {
            for (int w = 0; w < app_ctx->output_width; w++)
            {
                int idx = (h * app_ctx->output_width + w) * 3;  // RGB output index
                int hw_idx = h * app_ctx->output_width + w;      // HW plane index

                // Read from NCHW and convert to RGB uint8
                // Direct mapping: c0=R, c1=G, c2=B
                float c0 = float_data[hw_idx];                      // Channel 0 (R)
                float c1 = float_data[model_out_size + hw_idx];     // Channel 1 (G)
                float c2 = float_data[2 * model_out_size + hw_idx]; // Channel 2 (B)

                // Apply rescale if needed (for INT8 with wrong quantization params)
                if (need_rescale)
                {
                    c0 *= rescale_factor;
                    c1 *= rescale_factor;
                    c2 *= rescale_factor;
                }

                // Apply postprocessing: clip to [0, 1], scale to [0, 255], convert to uint8
                // This matches Python: output = np.clip(output, 0, 1)
                //                      output = (output * 255.0).round().astype(np.uint8)
                dst_ptr[idx + 0] = (unsigned char)(fminf(fmaxf(c0 * 255.0f, 0.0f), 255.0f));  // R
                dst_ptr[idx + 1] = (unsigned char)(fminf(fmaxf(c1 * 255.0f, 0.0f), 255.0f));  // G
                dst_ptr[idx + 2] = (unsigned char)(fminf(fmaxf(c2 * 255.0f, 0.0f), 255.0f));  // B
            }
        }
    }
    else
    {
        // NHWC format: Already in HWC order
        printf("Converting NHWC float to RGB uint8...\n");

        for (int i = 0; i < model_out_size * 3; i++)
        {
            float val = float_data[i];

            // Apply rescale if needed (for INT8 with wrong quantization params)
            if (need_rescale)
            {
                val *= rescale_factor;
            }

            dst_ptr[i] = (unsigned char)(fminf(fmaxf(val * 255.0f, 0.0f), 255.0f));
        }
    }

    printf("Super-resolution completed: %dx%d -> %dx%d\n",
           src_img->width, src_img->height,
           dst_img->width, dst_img->height);

    // Release resources
    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
    free(input_data);

    return 0;
}
