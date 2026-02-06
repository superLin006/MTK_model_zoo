#!/bin/bash
# Build Helsinki NPU for Android

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for NDK
if [ -z "$NDK_ROOT" ]; then
    NDK_PATHS=(
        "$HOME/Android/Ndk/android-ndk-r25c"
        "$HOME/Android/Sdk/ndk/25.1.8937393"
        "$HOME/Android/Sdk/ndk/25.2.9519653"
        "$HOME/Android/Sdk/ndk-bundle"
        "/opt/android-ndk"
    )

    for path in "${NDK_PATHS[@]}"; do
        if [ -d "$path" ]; then
            export NDK_ROOT="$path"
            break
        fi
    done

    if [ -z "$NDK_ROOT" ]; then
        echo -e "${RED}Error: NDK_ROOT not set${NC}"
        echo "Please set NDK_ROOT environment variable"
        exit 1
    fi
fi

echo -e "${GREEN}Using NDK: $NDK_ROOT${NC}"

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
rm -rf "$SCRIPT_DIR/libs"
rm -rf "$SCRIPT_DIR/obj"

# Build
echo -e "${GREEN}Building Helsinki NPU (ARM64)...${NC}"
$NDK_ROOT/ndk-build \
    -C "$SCRIPT_DIR" \
    NDK_PROJECT_PATH="$SCRIPT_DIR" \
    APP_BUILD_SCRIPT="$SCRIPT_DIR/jni/Android.mk" \
    NDK_APPLICATION_MK="$SCRIPT_DIR/jni/Application.mk" \
    -j$(nproc)

# Check result
if [ -f "$SCRIPT_DIR/libs/arm64-v8a/helsinki_translate" ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Executable: libs/arm64-v8a/helsinki_translate"
    SIZE=$(du -h "$SCRIPT_DIR/libs/arm64-v8a/helsinki_translate" | cut -f1)
    echo "Size: $SIZE"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
