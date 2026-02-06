# kaldi-native-fbank Android build
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := kaldi-native-fbank-core

# kaldi-native-fbank 源文件 (需要根据实际情况调整)
# 这里假设源代码在某个位置，如果只有预编译库，则使用 PREBUILT
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
    LOCAL_SRC_FILES := ../Linux/aarch64/libkaldi-native-fbank-core.a
else ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
    LOCAL_SRC_FILES := ../Linux/armhf/libkaldi-native-fbank-core.a
endif

LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../include

include $(PREBUILT_STATIC_LIBRARY)
