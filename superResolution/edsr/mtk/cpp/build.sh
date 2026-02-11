#!/bin/bash
#
# EDSR Super-Resolution - Build Script
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Toolchains（优先使用环境变量，否则使用默认路径）
ANDROID_NDK="${ANDROID_NDK:-/path/to/android-ndk}"
MTK_SDK="${MTK_SDK:-$SCRIPT_DIR/../../../../0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk}"

# Export paths
export MTK_NEUROPILOT_SDK="${MTK_SDK}"

echo "========================================"
echo "EDSR Super-Resolution - Build Script"
echo "========================================"
echo "NDK: ${ANDROID_NDK}"
echo "MTK SDK: ${MTK_SDK}"
echo "========================================"

# Clean previous build
echo "[INFO] Cleaning previous build..."
rm -rf "${SCRIPT_DIR}/libs"
rm -rf "${SCRIPT_DIR}/obj"

# Build
echo "[INFO] Building..."
cd "${SCRIPT_DIR}/jni"

${ANDROID_NDK}/ndk-build \
    NDK_PROJECT_PATH=. \
    APP_BUILD_SCRIPT=./Android.mk \
    NDK_APPLICATION_MK=./Application.mk \
    -j4

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed!"
    exit 1
fi

cd "${SCRIPT_DIR}"

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo "Executable: libs/arm64-v8a/edsr_inference"
echo ""
ls -lh libs/arm64-v8a/edsr_inference
echo ""

echo "Next step:"
echo "  ./deploy_and_test.sh"
