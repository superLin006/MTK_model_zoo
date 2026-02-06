# Helsinki Translation - MTK NPU with KV Cache
# Android.mk for NDK build

LOCAL_PATH := $(call my-dir)

# ==================== SentencePiece Library ====================
include $(CLEAR_VARS)
LOCAL_MODULE := sentencepiece
LOCAL_SRC_FILES := third_party/sentencepiece/lib/$(TARGET_ARCH_ABI)/libsentencepiece.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/third_party/sentencepiece/include
include $(PREBUILT_STATIC_LIBRARY)

# ==================== Helsinki Core Library ====================
include $(CLEAR_VARS)

LOCAL_MODULE := helsinki_core

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/src/helsinki \
    $(LOCAL_PATH)/src/tokenizer \
    $(LOCAL_PATH)/third_party/sentencepiece/include

LOCAL_SRC_FILES := \
    src/helsinki/helsinki.cc \
    src/helsinki/mtk-npu/neuron_executor.cpp \
    src/tokenizer/sp_tokenizer.cc

LOCAL_CFLAGS := -O3 -DNDEBUG -Wall
LOCAL_CPPFLAGS := -std=c++17 -frtti -fexceptions

LOCAL_STATIC_LIBRARIES := sentencepiece

include $(BUILD_STATIC_LIBRARY)

# ==================== Helsinki Main Executable ====================
include $(CLEAR_VARS)

LOCAL_MODULE := helsinki_translate

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/src/helsinki \
    $(LOCAL_PATH)/src/tokenizer \
    $(LOCAL_PATH)/third_party/sentencepiece/include

LOCAL_SRC_FILES := \
    src/helsinki/main.cc

LOCAL_CFLAGS := -O3 -DNDEBUG -Wall
LOCAL_CPPFLAGS := -std=c++17 -frtti -fexceptions

LOCAL_LDLIBS := -llog -ldl

LOCAL_STATIC_LIBRARIES := helsinki_core sentencepiece

include $(BUILD_EXECUTABLE)
