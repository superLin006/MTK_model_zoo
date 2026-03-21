#!/bin/bash
# Build Moonshine MTK NPU Inference for Android (arm64-v8a)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDK_ROOT=/home/xh/Android/Ndk/android-ndk-r25c

echo "========================================"
echo "Building Moonshine MTK NPU"
echo "NDK: $NDK_ROOT"
echo "========================================"

if [ ! -d "$NDK_ROOT" ]; then
    echo "ERROR: NDK not found at $NDK_ROOT"
    exit 1
fi

cd "$SCRIPT_DIR"

# Clean previous build
rm -rf obj libs

# Build
"$NDK_ROOT/ndk-build" \
    -C "$SCRIPT_DIR/jni" \
    NDK_PROJECT_PATH="$SCRIPT_DIR" \
    APP_BUILD_SCRIPT="$SCRIPT_DIR/jni/Android.mk" \
    NDK_APPLICATION_MK="$SCRIPT_DIR/jni/Application.mk" \
    NDK_LIBS_OUT="$SCRIPT_DIR/libs" \
    NDK_OUT="$SCRIPT_DIR/obj" \
    -j8

echo ""
echo "========================================"
if [ -f "$SCRIPT_DIR/libs/arm64-v8a/moonshine_test" ]; then
    echo "BUILD SUCCESS"
    ls -lh "$SCRIPT_DIR/libs/arm64-v8a/"
else
    echo "BUILD FAILED: moonshine_test not found"
    exit 1
fi
echo "========================================"
