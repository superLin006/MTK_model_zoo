/* vad_ort_utils.h - Minimal ORT utilities for Silero VAD (Android standalone build)
 *
 * Replaces ort-utils.h from moonshine/core/ort-utils/ with a self-contained
 * version that does not depend on debug-utils.h or filesystem.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "onnxruntime_c_api.h"

// Logging macros (simple fprintf to stderr)
#define LOGF(format, ...)                                    \
  do {                                                       \
    fprintf(stderr, "[VAD] %s:%d: " format "\n",            \
            __func__, __LINE__, ##__VA_ARGS__);              \
  } while (0)

#define LOG_ORT_ERROR(ort_api, expr)                          \
  do {                                                        \
    OrtStatus *_onnx_status = (expr);                         \
    if (_onnx_status != NULL) {                               \
      const char *_msg = ort_api->GetErrorMessage(_onnx_status); \
      fprintf(stderr, "[VAD] ORT Error: %s\n", _msg);        \
      ort_api->ReleaseStatus(_onnx_status);                   \
    }                                                         \
  } while (0)

#define RETURN_ON_ORT_ERROR(ort_api, expr)                    \
  do {                                                        \
    OrtStatus *_onnx_status = (expr);                         \
    if (_onnx_status != NULL) {                               \
      const char *_msg = ort_api->GetErrorMessage(_onnx_status); \
      fprintf(stderr, "[VAD] ORT Error: %s\n", _msg);        \
      ort_api->ReleaseStatus(_onnx_status);                   \
      return -1;                                              \
    }                                                         \
  } while (0)

#define RETURN_ON_NULL(ptr)                                   \
  do {                                                        \
    if ((ptr) == nullptr) {                                   \
      fprintf(stderr, "[VAD] Error: " #ptr " is nullptr\n"); \
      return -1;                                              \
    }                                                         \
  } while (0)

// Load ONNX session from memory buffer
static inline int ort_session_from_memory(
    const OrtApi *ort_api, OrtEnv *env, OrtSessionOptions *session_options,
    const uint8_t *data, size_t data_size, OrtSession **session)
{
    RETURN_ON_NULL(ort_api);
    RETURN_ON_NULL(data);
    RETURN_ON_ORT_ERROR(
        ort_api,
        ort_api->CreateSessionFromArray(env, data, data_size,
                                        session_options, session));
    return 0;
}
