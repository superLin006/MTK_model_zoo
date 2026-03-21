# Moonshine MTK NPU Inference - Standalone Build

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
# Moonshine core library
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := moonshine_core

LOCAL_SRC_FILES := src/moonshine/moonshine_inference.cpp \
                   src/utils/audio_utils.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES)

LOCAL_CFLAGS := $(APP_CPPFLAGS)

include $(BUILD_STATIC_LIBRARY)

#######################
# Moonshine main executable
#######################

include $(CLEAR_VARS)

LOCAL_MODULE := moonshine_test

LOCAL_SRC_FILES := src/moonshine/main.cpp

LOCAL_C_INCLUDES := $(GLOBAL_C_INCLUDES) \
                    $(LOCAL_PATH)/src/neuron/api

LOCAL_CFLAGS := $(APP_CPPFLAGS)

LOCAL_LDLIBS := -llog \
                -landroid \
                -ldl

LOCAL_STATIC_LIBRARIES := moonshine_core \
                          easyloggingpp \
                          executor \
                          utils \
                          neuron \
                          profiler

include $(BUILD_EXECUTABLE)
