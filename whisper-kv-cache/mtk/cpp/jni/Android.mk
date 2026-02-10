# Whisper MTK NPU - Android Makefile
LOCAL_PATH := $(call my-dir)

# ========================================
# Prebuilt FFTW3F Library
# ========================================
include $(CLEAR_VARS)
LOCAL_MODULE := fftw3f
LOCAL_SRC_FILES := /home/xh/projects/MTK/1_third_party/fftw/Android/arm64-v8a/libfftw3f.a
LOCAL_EXPORT_C_INCLUDES := /home/xh/projects/MTK/1_third_party/fftw/include
include $(PREBUILT_STATIC_LIBRARY)

# ========================================
# Audio Utils Library
# ========================================
include $(CLEAR_VARS)

LOCAL_MODULE := audio_utils
LOCAL_SRC_FILES := src/utils/audio_utils.cpp
LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/src \
    $(LOCAL_PATH)/src/utils \
    /home/xh/projects/MTK/1_third_party/fftw/include

LOCAL_CPPFLAGS := -Wall -Wextra -O2

include $(BUILD_STATIC_LIBRARY)

# ========================================
# Neuron Executor Library
# ========================================
include $(CLEAR_VARS)

LOCAL_MODULE := neuron_executor
LOCAL_SRC_FILES := src/mtk-npu/neuron_executor.cpp
LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/src \
    $(LOCAL_PATH)/src/mtk-npu

LOCAL_CPPFLAGS := -Wall -Wextra -O2

include $(BUILD_STATIC_LIBRARY)

# ========================================
# Whisper Inference Library
# ========================================
include $(CLEAR_VARS)

LOCAL_MODULE := whisper_inference
LOCAL_SRC_FILES := src/whisper_inference.cpp
LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/src \
    $(LOCAL_PATH)/src/utils \
    $(LOCAL_PATH)/src/mtk-npu

LOCAL_CPPFLAGS := -Wall -Wextra -O2
LOCAL_STATIC_LIBRARIES := audio_utils neuron_executor

include $(BUILD_STATIC_LIBRARY)

# ========================================
# Main Executable
# ========================================
include $(CLEAR_VARS)

LOCAL_MODULE := whisper_kv_test
LOCAL_SRC_FILES := src/main.cpp
LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/src \
    $(LOCAL_PATH)/src/utils \
    $(LOCAL_PATH)/src/mtk-npu \
    /home/xh/projects/MTK/1_third_party/fftw/include

LOCAL_CPPFLAGS := -Wall -Wextra -O2
LOCAL_LDLIBS := -ldl -lm -llog
LOCAL_STATIC_LIBRARIES := whisper_inference audio_utils neuron_executor fftw3f

include $(BUILD_EXECUTABLE)
