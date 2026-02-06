#!/bin/bash
#
# EDSR Super-Resolution - Deploy and Test Script
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Device paths
DEVICE_DIR="/data/local/tmp/edsr_test"
DEVICE_MODEL="${DEVICE_DIR}/edsr_model.dla"
DEVICE_INPUT="${DEVICE_DIR}/input.png"
DEVICE_OUTPUT="${DEVICE_DIR}/output.png"
DEVICE_EXEC="${DEVICE_DIR}/edsr_inference"

# Local paths
LOCAL_EXEC="${SCRIPT_DIR}/libs/arm64-v8a/edsr_inference"
LOCAL_MODEL="${PROJECT_ROOT}/models/EDSR_x4_256x256_MT8371.dla"
LOCAL_TEST_IMAGE="${PROJECT_ROOT}/../test_data/0853x4.png"

echo "========================================"
echo "EDSR Super-Resolution - Deploy & Test"
echo "========================================"

# Check if device is connected
echo "[INFO] Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo "[ERROR] No Android device found!"
    echo "Please connect your MT8371 device and enable USB debugging."
    exit 1
fi

DEVICE=$(adb devices | grep "device$" | head -1 | awk '{print $1}')
echo "[INFO] Device found: ${DEVICE}"

# Create test directory on device
echo ""
echo "[INFO] Creating test directory on device..."
adb shell "mkdir -p ${DEVICE_DIR}"

# Push executable
echo ""
echo "[INFO] Pushing executable..."
adb push "${LOCAL_EXEC}" "${DEVICE_EXEC}"
adb shell "chmod 755 ${DEVICE_EXEC}"

# Push model
echo ""
echo "[INFO] Pushing DLA model..."
if [ ! -f "${LOCAL_MODEL}" ]; then
    echo "[ERROR] Model file not found: ${LOCAL_MODEL}"
    exit 1
fi
adb push "${LOCAL_MODEL}" "${DEVICE_MODEL}"

# Push test image
echo ""
echo "[INFO] Pushing test image..."
if [ ! -f "${LOCAL_TEST_IMAGE}" ]; then
    echo "[ERROR] Test image not found: ${LOCAL_TEST_IMAGE}"
    echo "Usage: $0 [input_image_path]"
    exit 1
fi
adb push "${LOCAL_TEST_IMAGE}" "${DEVICE_INPUT}"

# Run inference
echo ""
echo "========================================"
echo "Running EDSR inference..."
echo "========================================"
adb shell "cd ${DEVICE_DIR} && LD_LIBRARY_PATH=${MTK_NEUROPILOT_SDK}/target/lib64 ./edsr_inference edsr_model.dla input.png output.png"

# Pull result
echo ""
echo "[INFO] Pulling output image..."
adb pull "${DEVICE_OUTPUT}" "${SCRIPT_DIR}/output_edsr.png"

# Show results
echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"
echo "Output saved to: ${SCRIPT_DIR}/output_edsr.png"
ls -lh "${SCRIPT_DIR}/output_edsr.png"

echo ""
echo "To view the result, you can:"
echo "  1. Open ${SCRIPT_DIR}/output_edsr.png"
echo "  2. Compare with input: ${LOCAL_TEST_IMAGE}"
