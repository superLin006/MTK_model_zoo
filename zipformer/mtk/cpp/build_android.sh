#!/bin/bash
#
# Build script for Zipformer MTK NPU inference
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check NDK
if [ -z "$ANDROID_NDK" ]; then
    export ANDROID_NDK=/home/xh/Android/Ndk/android-ndk-r25c
fi

echo "==================================="
echo "Building Zipformer MTK NPU"
echo "==================================="
echo "NDK: $ANDROID_NDK"
echo "Project: $SCRIPT_DIR"

# Clean
rm -rf obj libs

# Build
"$ANDROID_NDK/ndk-build" -j$(nproc) \
    NDK_PROJECT_PATH=. \
    NDK_APPLICATION_MK=jni/Application.mk

echo ""
echo "==================================="
echo "Build completed!"
echo "==================================="
echo ""

if [ -f "libs/arm64-v8a/zipformer_mtk_test" ]; then
    echo "Output: libs/arm64-v8a/zipformer_mtk_test"
    ls -la libs/arm64-v8a/
else
    echo "ERROR: executable not found"
    exit 1
fi
