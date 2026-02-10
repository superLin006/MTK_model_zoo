#!/bin/bash
# Real-ESRGAN Super-Resolution - Build Script
# Cross-compile for Android arm64-v8a using NDK

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Real-ESRGAN Build Script"
echo "=========================================="

# 配置
ANDROID_NDK="/home/xh/Android/Ndk/android-ndk-r25c"
MTK_NEUROPILOT_SDK="/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
TARGET_ABI="arm64-v8a"
TARGET_ARCH="arm64"

# 检查NDK
if [ ! -d "$ANDROID_NDK" ]; then
    echo -e "${RED}[ERROR] Android NDK not found: $ANDROID_NDK${NC}"
    exit 1
fi

# 检查MTK SDK
if [ ! -d "$MTK_NEUROPILOT_SDK" ]; then
    echo -e "${RED}[ERROR] MTK NeuroPilot SDK not found: $MTK_NEUROPILOT_SDK${NC}"
    exit 1
fi

echo -e "${GREEN}[INFO] Android NDK: $ANDROID_NDK${NC}"
echo -e "${GREEN}[INFO] MTK SDK: $MTK_NEUROPILOT_SDK${NC}"
echo -e "${GREEN}[INFO] Target ABI: $TARGET_ABI${NC}"

# 设置环境变量
export MTK_NEUROPILOT_SDK="$MTK_NEUROPILOT_SDK"

# 创建输出目录
mkdir -p build/$TARGET_ABI

# 编译
echo ""
echo "=========================================="
echo "Building..."
echo "=========================================="

$ANDROID_NDK/ndk-build \
    NDK_PROJECT_PATH=. \
    APP_BUILD_SCRIPT=./jni/Android.mk \
    NDK_APPLICATION_MK=./jni/Application.mk \
    MTK_NEUROPILOT_SDK=$MTK_NEUROPILOT_SDK \
    NDK_LIBS_OUT=build/$TARGET_ABI \
    -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Build Successful!"
    echo "==========================================${NC}"
    echo ""
    echo "Output: build/$TARGET_ABI/realesrgan_inference"
    ls -lh build/$TARGET_ABI/realesrgan_inference
else
    echo ""
    echo -e "${RED}=========================================="
    echo "Build Failed!"
    echo "==========================================${NC}"
    exit 1
fi
