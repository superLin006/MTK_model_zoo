#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDK_ROOT=/home/xh/Android/Ndk/android-ndk-r25c

echo "=== Building Moonshine Streaming for Android arm64-v8a ==="
echo "NDK: $NDK_ROOT"
echo "Project: $SCRIPT_DIR"

if [ ! -d "$NDK_ROOT" ]; then
    echo "ERROR: NDK not found at $NDK_ROOT"
    exit 1
fi

cd "$SCRIPT_DIR"

# Clean previous build
rm -rf obj libs

"$NDK_ROOT/ndk-build" \
    -C "$SCRIPT_DIR/jni" \
    NDK_PROJECT_PATH="$SCRIPT_DIR" \
    APP_BUILD_SCRIPT="$SCRIPT_DIR/jni/Android.mk" \
    NDK_APPLICATION_MK="$SCRIPT_DIR/jni/Application.mk" \
    NDK_LIBS_OUT="$SCRIPT_DIR/libs" \
    NDK_OUT="$SCRIPT_DIR/obj" \
    -j8

echo ""
echo "=== Build Result ==="
if [ -f "$SCRIPT_DIR/libs/arm64-v8a/moonshine_streaming_test" ]; then
    echo "BUILD SUCCESS"
    ls -lh "$SCRIPT_DIR/libs/arm64-v8a/"
else
    echo "BUILD FAILED: moonshine_streaming_test not found"
    exit 1
fi
