# Zipformer MTK NPU Inference - Android.mk

LOCAL_PATH := $(call my-dir)

# Global includes
GLOBAL_C_INCLUDES := $(LOCAL_PATH)/src \
                     $(LOCAL_PATH)/third_party/easyloggingpp/include

#######################
# Third-party: kaldi-native-fbank (prebuilt)
#######################
KALDI_FBANK_PATH := /home/xh/projects/MTK_models_zoo/1_third_party/kaldi_native_fbank/Android

include $(CLEAR_VARS)
LOCAL_MODULE := kaldi-native-fbank-core
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
    LOCAL_SRC_FILES := $(KALDI_FBANK_PATH)/arm64-v8a/libkaldi-native-fbank-core.a
else ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
    LOCAL_SRC_FILES := $(KALDI_FBANK_PATH)/armeabi-v7a/libkaldi-native-fbank-core.a
endif
LOCAL_EXPORT_C_INCLUDES := $(KALDI_FBANK_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := kissfft-float
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
    LOCAL_SRC_FILES := $(KALDI_FBANK_PATH)/arm64-v8a/libkissfft-float.a
else ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
    LOCAL_SRC_FILES := $(KALDI_FBANK_PATH)/armeabi-v7a/libkissfft-float.a
endif
include $(PREBUILT_STATIC_LIBRARY)

#######################
# easyloggingpp
#######################
EASYLOGGINGPP_ROOT := $(LOCAL_PATH)/third_party/easyloggingpp
include $(EASYLOGGINGPP_ROOT)/Android.mk

#######################
# Neuron runtime (static)
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
# Profiler (static)
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
# Utils (static)
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
# Executor (static)
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
# Zipformer main executable
#######################
include $(CLEAR_VARS)

LOCAL_MODULE := zipformer_mtk_test

LOCAL_SRC_FILES := src/main.cpp \
                   src/zipformer_inference.cpp \
                   src/utils/audio_utils.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES) \
                    $(LOCAL_PATH)/src/neuron/api \
                    $(KALDI_FBANK_PATH)/include

LOCAL_CFLAGS := $(APP_CPPFLAGS)

LOCAL_LDLIBS := -llog \
                -landroid \
                -ldl

LOCAL_STATIC_LIBRARIES := easyloggingpp \
                           executor \
                           utils \
                           neuron \
                           profiler \
                           kaldi-native-fbank-core \
                           kissfft-float

include $(BUILD_EXECUTABLE)
