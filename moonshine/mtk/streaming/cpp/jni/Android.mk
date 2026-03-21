# Moonshine MTK NPU Streaming Inference - Standalone Build

LOCAL_PATH := $(call my-dir)

# Global include paths
GLOBAL_C_INCLUDES := $(LOCAL_PATH)/src \
                     $(LOCAL_PATH)/third_party/easyloggingpp/include

#######################
# Third-party libraries
#######################

# easyloggingpp
EASYLOGGINGPP_ROOT := $(LOCAL_PATH)/third_party/easyloggingpp
include $(EASYLOGGINGPP_ROOT)/Android.mk

#######################
# Neuron runtime library
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := neuron

LOCAL_SRC_FILES := src/neuron/NeuronRuntimeLibrary.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES) \
                    $(LOCAL_PATH)/src/neuron \
                    $(LOCAL_PATH)/src/neuron/api

LOCAL_CFLAGS := $(APP_CPPFLAGS)

include $(BUILD_STATIC_LIBRARY)

#######################
# Profiler library
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := profiler

LOCAL_SRC_FILES := src/trace/ScopeProfiler.cpp \
                   src/trace/Stopwatch.cpp \
                   src/trace/Trace.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES)

LOCAL_CFLAGS := $(APP_CPPFLAGS)

include $(BUILD_STATIC_LIBRARY)

#######################
# Utils library
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := utils

LOCAL_SRC_FILES := src/utils/DumpWorker.cpp \
                   src/utils/MemAllocator.cpp \
                   src/utils/Utils.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES)

LOCAL_CFLAGS := $(APP_CPPFLAGS)

include $(BUILD_STATIC_LIBRARY)

#######################
# Executor library
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := executor

LOCAL_SRC_FILES := src/executor/ExecutorFactory.cpp \
                   src/executor/NeuronExecutor.cpp \
                   src/executor/NeuronUsdkExecutor.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES) \
                    $(LOCAL_PATH)/src/neuron \
                    $(LOCAL_PATH)/src/neuron/api

LOCAL_CFLAGS := $(APP_CPPFLAGS)

LOCAL_STATIC_LIBRARIES := neuron profiler utils

include $(BUILD_STATIC_LIBRARY)

#######################
# OnnxRuntime prebuilt shared library (required by Silero VAD)
#######################

ONNXRUNTIME_DIR := /home/xh/Android/OnnxRuntime/onnxruntime-android-1.17.1

include $(CLEAR_VARS)
LOCAL_MODULE := onnxruntime
LOCAL_SRC_FILES := $(ONNXRUNTIME_DIR)/jni/arm64-v8a/libonnxruntime.so
include $(PREBUILT_SHARED_LIBRARY)

#######################
# Silero VAD library
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := silero_vad

# silero-vad-model-data.h is ~190k lines; compilation will be slow, this is expected
LOCAL_SRC_FILES := src/vad/silero-vad.cpp \
                   src/vad/silero_vad_wrapper.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES) \
                    $(LOCAL_PATH)/src \
                    $(LOCAL_PATH)/src/vad \
                    $(ONNXRUNTIME_DIR)/headers

LOCAL_CFLAGS := $(APP_CPPFLAGS)

LOCAL_SHARED_LIBRARIES := onnxruntime

include $(BUILD_STATIC_LIBRARY)

#######################
# Moonshine streaming core library
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := moonshine_streaming_core

LOCAL_SRC_FILES := src/moonshine_streaming.cpp \
                   src/utils/audio_utils.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES) \
                    $(LOCAL_PATH)/src \
                    $(LOCAL_PATH)/src/vad \
                    $(ONNXRUNTIME_DIR)/headers

LOCAL_CFLAGS := $(APP_CPPFLAGS)

LOCAL_STATIC_LIBRARIES := silero_vad

include $(BUILD_STATIC_LIBRARY)

#######################
# Moonshine streaming executable
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := moonshine_streaming_test

LOCAL_SRC_FILES := src/main.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES) \
                    $(LOCAL_PATH)/src/neuron/api \
                    $(LOCAL_PATH)/src \
                    $(LOCAL_PATH)/src/vad \
                    $(ONNXRUNTIME_DIR)/headers

LOCAL_CFLAGS := $(APP_CPPFLAGS)

LOCAL_LDLIBS := -llog \
                -landroid \
                -ldl

LOCAL_STATIC_LIBRARIES := moonshine_streaming_core \
                          silero_vad \
                          easyloggingpp \
                          executor \
                          utils \
                          neuron \
                          profiler

LOCAL_SHARED_LIBRARIES := onnxruntime

include $(BUILD_EXECUTABLE)
