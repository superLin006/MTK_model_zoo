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

#include "rcan.h"
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

int init_rcan_model(const char *model_path, rknn_rcan_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    printf("Loading RCAN RKNN model...\n");

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

    // Set to context
    app_ctx->rknn_ctx = ctx;

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

int release_rcan_model(rknn_rcan_context_t *app_ctx)
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

int inference_rcan_model(rknn_rcan_context_t *app_ctx, image_buffer_t *src_img, image_buffer_t *dst_img)
{
    int ret;
    int hw;
    image_buffer_t input_img;
    rknn_input inputs[1];
    rknn_output outputs[1];
    float *output_data;
    unsigned char *dst_ptr;
    unsigned char *bgr_buffer = NULL;
    uint16_t *fp16_data = NULL;

    if ((!app_ctx) || (!src_img) || (!dst_img))
    {
        printf("Invalid parameters!\n");
        return -1;
    }

    memset(&input_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process: Resize input to model size if needed
    if (src_img->width != app_ctx->model_width || src_img->height != app_ctx->model_height)
    {
        printf("Resizing input from %dx%d to %dx%d\n",
               src_img->width, src_img->height,
               app_ctx->model_width, app_ctx->model_height);

        input_img.width = app_ctx->model_width;
        input_img.height = app_ctx->model_height;
        input_img.format = IMAGE_FORMAT_RGB888;
        input_img.size = get_image_size(&input_img);
        input_img.virt_addr = (unsigned char *)malloc(input_img.size);
        if (input_img.virt_addr == NULL)
        {
            printf("malloc buffer size:%d fail!\n", input_img.size);
            return -1;
        }

        // For now, just use the input directly with warning
        // TODO: Implement proper image resize using RGA or libpng
        printf("WARNING: Input size mismatch, using input directly (may cause issues)\n");
        free(input_img.virt_addr);
        input_img = *src_img;
    }
    else
    {
        // Use input directly
        input_img = *src_img;
    }

    // Debug: Print model expected input format
    printf("Model input attr: fmt=%s, type=%s, dims=[%d,%d,%d,%d]\n",
           get_format_string(app_ctx->input_attrs[0].fmt),
           get_type_string(app_ctx->input_attrs[0].type),
           app_ctx->input_attrs[0].dims[0],
           app_ctx->input_attrs[0].dims[1],
           app_ctx->input_attrs[0].dims[2],
           app_ctx->input_attrs[0].dims[3]);

    // Set Input Data - NHWC format as expected by RKNN model
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = input_img.size;
    inputs[0].buf = input_img.virt_addr;
    inputs[0].pass_through = 0;  // Let RKNN handle conversion

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run inference
    printf("Running RCAN super-resolution inference...\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    outputs[0].index = 0;
    if (app_ctx->is_quant)
    {
        // For INT8 quantized models, request float output (RKNN will dequantize)
        outputs[0].want_float = 1;
    }
    else
    {
        // For FP16 models, get native FP16 data
        outputs[0].want_float = 0;
    }
    outputs[0].is_prealloc = 0;  // Let RKNN allocate buffer
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // Post Process: Convert output to RGB888
    printf("Processing output...\n");
    dst_img->width = app_ctx->output_width;
    dst_img->height = app_ctx->output_height;
    dst_img->format = IMAGE_FORMAT_RGB888;
    dst_img->size = app_ctx->output_width * app_ctx->output_height * 3;
    dst_img->virt_addr = (unsigned char *)malloc(dst_img->size);

    if (dst_img->virt_addr == NULL)
    {
        printf("malloc output buffer fail!\n");
        rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
        ret = -1;
        goto out;
    }

    // Convert output tensor to image
    dst_ptr = dst_img->virt_addr;
    hw = app_ctx->output_height * app_ctx->output_width;

    if (app_ctx->is_quant)
    {
        // INT8 model: RKNN already converted to float for us
        float *float_data = (float *)outputs[0].buf;

        printf("Debug - Output format: %s\n",
               app_ctx->output_attrs[0].fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");

        // Convert NCHW to HWC (RGB interleaved)
        printf("Converting NCHW to RGB format...\n");
        for (int h = 0; h < app_ctx->output_height; h++)
        {
            for (int w = 0; w < app_ctx->output_width; w++)
            {
                int idx = (h * app_ctx->output_width + w) * 3;  // RGB output index
                int hw_idx = h * app_ctx->output_width + w;      // HW plane index

                // Read from NCHW: R plane, G plane, B plane
                float r = float_data[hw_idx];           // R channel
                float g = float_data[hw + hw_idx];      // G channel
                float b = float_data[2*hw + hw_idx];    // B channel

                // Write as RGB
                dst_ptr[idx + 0] = (unsigned char)(fminf(fmaxf(r, 0.0f), 255.0f));
                dst_ptr[idx + 1] = (unsigned char)(fminf(fmaxf(g, 0.0f), 255.0f));
                dst_ptr[idx + 2] = (unsigned char)(fminf(fmaxf(b, 0.0f), 255.0f));
            }
        }
    }
    else
    {
        // FP16 model: Convert FP16 to uint8
        fp16_data = (uint16_t *)outputs[0].buf;

        printf("Debug - Output format: %s\n",
               app_ctx->output_attrs[0].fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");

        // Convert NCHW to HWC (RGB interleaved)
        printf("Converting NCHW FP16 to RGB format...\n");
        for (int h = 0; h < app_ctx->output_height; h++)
        {
            for (int w = 0; w < app_ctx->output_width; w++)
            {
                int idx = (h * app_ctx->output_width + w) * 3;  // RGB output index
                int hw_idx = h * app_ctx->output_width + w;      // HW plane index

                // Read from NCHW: R plane, G plane, B plane
                float r = fp16_to_float(fp16_data[hw_idx]);           // R channel
                float g = fp16_to_float(fp16_data[hw + hw_idx]);      // G channel
                float b = fp16_to_float(fp16_data[2*hw + hw_idx]);    // B channel

                // Write as RGB
                dst_ptr[idx + 0] = (unsigned char)(fminf(fmaxf(r, 0.0f), 255.0f));
                dst_ptr[idx + 1] = (unsigned char)(fminf(fmaxf(g, 0.0f), 255.0f));
                dst_ptr[idx + 2] = (unsigned char)(fminf(fmaxf(b, 0.0f), 255.0f));
            }
        }
    }

    printf("Super-resolution completed: %dx%d -> %dx%d\n",
           src_img->width, src_img->height,
           dst_img->width, dst_img->height);

    // Release outputs
    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);

out:
    // Free BGR buffer
    if (bgr_buffer != NULL)
    {
        free(bgr_buffer);
    }

    // Free temporary buffer if we allocated it
    if (input_img.virt_addr != src_img->virt_addr && input_img.virt_addr != NULL)
    {
        free(input_img.virt_addr);
    }

    return ret;
}
