#
# RCAN Super-Resolution - Android.mk
#

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := rcan_inference

LOCAL_SRC_FILES := src/main.cpp \
                   src/rcan.cpp \
                   src/mtk_npu/neuron_executor.cpp

LOCAL_C_INCLUDES := $(LOCAL_PATH)/src \
                    $(LOCAL_PATH)/src/mtk_npu \
                    $(LOCAL_PATH)/../third_party/stb

LOCAL_LDLIBS := -ldl -static-libstdc++

LOCAL_STATIC_LIBRARIES :=

# Link MTK NeuroPilot runtime
LOCAL_LDFLAGS := -L$(MTK_NEUROPILOT_SDK)/target/$(TARGET_ARCH_ABI)

LOCAL_CFLAGS := -O3 -Wall
LOCAL_CPPFLAGS := -std=c++17 -frtti -fexceptions

LOCAL_ARM_MODE := arm
LOCAL_ARM_NEON := true

include $(BUILD_EXECUTABLE)
