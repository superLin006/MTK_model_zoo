#!/bin/bash

# Whisper MTK NPU - Android Build Script
# Cross-compiles for MT8371 (ARM64-v8a)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_NDK="${HOME}/Android/Ndk/android-ndk-r25c"
BUILD_DIR="${SCRIPT_DIR}/build_android"
JNI_DIR="${SCRIPT_DIR}/jni"

echo "========================================"
echo "  Whisper MTK NPU - Android Build"
echo "========================================"
echo ""

# Check NDK
if [ ! -d "${ANDROID_NDK}" ]; then
    echo "[ERROR] Android NDK not found at: ${ANDROID_NDK}"
    echo "        Please update ANDROID_NDK path in this script"
    exit 1
fi

echo "[CONFIG] Android NDK: ${ANDROID_NDK}"
echo "[CONFIG] Target: arm64-v8a (MT8371)"
echo "[CONFIG] Platform: android-28"
echo ""

# Clean previous build
if [ -d "${BUILD_DIR}" ]; then
    echo "[INFO] Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
fi

# Build
echo "[INFO] Building with ndk-build..."
cd "${JNI_DIR}"

${ANDROID_NDK}/ndk-build \
    NDK_PROJECT_PATH=. \
    APP_BUILD_SCRIPT=./Android.mk \
    NDK_APPLICATION_MK=./Application.mk \
    APP_ABI=arm64-v8a \
    APP_PLATFORM=android-28 \
    -j$(nproc)

cd "${SCRIPT_DIR}"

# Check if executable was created
EXECUTABLE="${JNI_DIR}/libs/arm64-v8a/whisper_test"

if [ -f "${EXECUTABLE}" ]; then
    echo ""
    echo "========================================"
    echo "  Build Successful!"
    echo "========================================"
    echo ""
    echo "Executable: ${EXECUTABLE}"
    ls -lh "${EXECUTABLE}"
    echo ""
    echo "Binary information:"
    file "${EXECUTABLE}"
    echo ""
    echo "To deploy to device:"
    echo "  adb push ${EXECUTABLE} /data/local/tmp/"
    echo "  adb shell chmod +x /data/local/tmp/whisper_test"
    echo ""
else
    echo ""
    echo "[ERROR] Build failed - executable not found"
    exit 1
fi
