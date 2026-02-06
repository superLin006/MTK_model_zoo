#!/bin/bash
#
# Build script for SenseVoice MTK NPU inference
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check NDK path
if [ -z "$ANDROID_NDK" ]; then
    # Try common locations
    if [ -d "/home/xh/Android/Ndk/android-ndk-r25c" ]; then
        export ANDROID_NDK="/home/xh/Android/Ndk/android-ndk-r25c"
    else
        echo "Error: ANDROID_NDK not set and not found in common locations"
        exit 1
    fi
fi

echo "==================================="
echo "Building SenseVoice MTK NPU"
echo "==================================="
echo "NDK: $ANDROID_NDK"
echo "Project: $SCRIPT_DIR"

# Clean previous build
rm -rf obj libs

# Build
"$ANDROID_NDK/ndk-build" -j$(nproc) NDK_PROJECT_PATH=. NDK_APPLICATION_MK=jni/Application.mk

echo ""
echo "==================================="
echo "Build completed successfully!"
echo "==================================="
echo ""
echo "Output: libs/arm64-v8a/sensevoice_main"
echo ""

# Check output
if [ -f "libs/arm64-v8a/sensevoice_main" ]; then
    ls -la libs/arm64-v8a/
else
    echo "Error: Build failed, executable not found"
    exit 1
fi
