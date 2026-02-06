/**
 * EDSR Super-Resolution - MTK NPU Implementation
 */

#ifndef _EDSR_MTK_H_
#define _EDSR_MTK_H_

#include <cstdint>
#include <cstdlib>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
class NeuronExecutor;

// EDSR context structure
typedef struct {
    // MTK Neuron executor
    void* neuron_executor;   // NeuronExecutor handle (opaque pointer)

    // Model dimensions
    int input_width;         // Input width (e.g., 256)
    int input_height;        // Input height (e.g., 256)
    int input_channels;      // Input channels (3 for RGB)
    int output_width;        // Output width (e.g., 1024 for 4x)
    int output_height;       // Output height (e.g., 1024 for 4x)
    int output_channels;     // Output channels (3 for RGB)
    int scale_factor;        // Super-resolution scale (e.g., 4)

    // MeanShift parameters
    float rgb_mean[3];       // RGB mean values
    float rgb_range;         // RGB range (255.0)

    // Buffer sizes
    size_t input_size;       // Input buffer size in bytes
    size_t output_size;      // Output buffer size in bytes

    // Initialization flag
    int initialized;
} edsr_context_t;

/**
 * @brief Initialize EDSR model
 *
 * @param model_path Path to DLA model file
 * @param ctx EDSR context to be initialized
 * @return 0 on success, negative on failure
 */
int edsr_init(const char* model_path, edsr_context_t* ctx);

/**
 * @brief Release EDSR model resources
 *
 * @param ctx EDSR context
 * @return 0 on success, negative on failure
 */
int edsr_release(edsr_context_t* ctx);

/**
 * @brief Run super-resolution inference
 *
 * @param ctx EDSR context
 * @param input_rgb Input image RGB data [H*W*3] (uint8)
 * @param output_rgb Output image RGB data [outH*outW*3] (will be allocated)
 * @return 0 on success, negative on failure
 */
int edsr_inference(edsr_context_t* ctx,
                   const unsigned char* input_rgb,
                   unsigned char** output_rgb);

/**
 * @brief Run super-resolution with explicit dimensions
 *
 * @param ctx EDSR context
 * @param input_rgb Input image RGB data [H*W*3] (uint8)
 * @param input_width Input image width
 * @param input_height Input image height
 * @param output_rgb Output image RGB data (will be allocated)
 * @param output_width Output image width (will be set)
 * @param output_height Output image height (will be set)
 * @return 0 on success, negative on failure
 */
int edsr_inference_ex(edsr_context_t* ctx,
                      const unsigned char* input_rgb,
                      int input_width,
                      int input_height,
                      unsigned char** output_rgb,
                      int* output_width,
                      int* output_height);

#ifdef __cplusplus
}
#endif

#endif // _EDSR_MTK_H_
