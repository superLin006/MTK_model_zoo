#!/bin/bash
#
# RCAN Deployment Script - With SDK Library
#
# This script deploys RCAN inference binary with MTK Neuron Runtime SDK library
# to ensure consistent library version across different devices.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
ANDROID_NDK="/home/xh/Android/Ndk/android-ndk-r25c"
MTK_SDK="/home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
SDK_LIB="$MTK_SDK/mt8371/libneuron_runtime.8.so"

BINARY="$SCRIPT_DIR/jni/libs/arm64-v8a/rcan_inference"
DLA_MODEL="$PROJECT_ROOT/models/RCAN_BIX4_128x128_MT8371.dla"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "RCAN Deployment Script (with SDK lib)"
echo "========================================"
echo "NDK: ${ANDROID_NDK}"
echo "MTK SDK: ${MTK_SDK}"
echo "========================================"

# Check if device is connected
echo -e "\n[1/6] Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo -e "${RED}[ERROR]${NC} No Android device found!"
    echo "Please connect your device and enable USB debugging."
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Device connected"

# Check if binary exists
echo -e "\n[2/6] Checking files..."
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}[ERROR]${NC} Binary not found: $BINARY"
    echo "Please run: ./build.sh"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Binary found"

if [ ! -f "$SDK_LIB" ]; then
    echo -e "${YELLOW}[WARNING]${NC} SDK library not found: $SDK_LIB"
    echo "Will use system library instead."
    USE_SDK_LIB=false
else
    echo -e "${GREEN}[OK]${NC} SDK library found"
    USE_SDK_LIB=true
fi

# Deploy files
echo -e "\n[3/6] Deploying files to device..."

if [ "$USE_SDK_LIB" = true ]; then
    echo "  Pushing SDK library (3.3 MB)..."
    adb push "$SDK_LIB" /data/local/tmp/libneuron_runtime.8.so > /dev/null
    echo -e "  ${GREEN}✓${NC} SDK library deployed"
fi

echo "  Pushing RCAN binary..."
adb push "$BINARY" /data/local/tmp/rcan_inference > /dev/null
echo -e "  ${GREEN}✓${NC} Binary deployed"

if [ -f "$DLA_MODEL" ]; then
    echo "  Pushing DLA model (33.6 MB)..."
    adb push "$DLA_MODEL" /data/local/tmp/ > /dev/null
    echo -e "  ${GREEN}✓${NC} DLA model deployed"
else
    echo -e "${YELLOW}[WARNING]${NC} DLA model not found at: $DLA_MODEL"
    echo "  Skipping DLA model deployment"
fi

# Display usage
echo -e "\n[4/6] Deployment completed!"
echo ""
echo "Usage:"
echo "  adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./rcan_inference <model.dla> <input.png> <output.png>'"
echo ""
echo "Or run test:"
if [ -f "$DLA_MODEL" ]; then
    echo "  $0 --test"
fi

# Run test if requested
if [ "$1" = "--test" ] && [ -f "$DLA_MODEL" ]; then
    echo -e "\n[5/6] Running test..."

    # Find a test image
    TEST_IMAGE="$PROJECT_ROOT/test_data/0853x4.png"
    if [ ! -f "$TEST_IMAGE" ]; then
        echo -e "${YELLOW}[WARNING]${NC} Test image not found: $TEST_IMAGE"
        echo "Skipping test."
    else
        echo "  Pushing test image..."
        adb push "$TEST_IMAGE" /data/local/tmp/ > /dev/null

        # Create 128x128 input by downsampling
        echo "  Creating 128x128 input image..."
        adb shell "cd /data/local/tmp && convert 0853x4.png -resize 128x128 input_128x128.png" 2>/dev/null || echo "  Note: ImageMagick not available on device, using original image"

        echo "  Running inference..."
        adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./rcan_inference RCAN_BIX4_128x128_MT8371.dla input_128x128.png output.png" 2>&1 | tee /tmp/rcan_test.log

        # Check if test succeeded
        if grep -q "Done!" /tmp/rcan_test.log; then
            echo -e "\n  ${GREEN}[SUCCESS]${NC} Test passed!"

            # Pull output
            OUTPUT_FILE="$PROJECT_ROOT/test_data/output_$(date +%Y%m%d_%H%M%S).png"
            adb pull /data/local/tmp/output.png "$OUTPUT_FILE" > /dev/null 2>&1
            if [ -f "$OUTPUT_FILE" ]; then
                echo "  Output saved to: $OUTPUT_FILE"
            fi
        else
            echo -e "\n  ${RED}[FAILED]${NC} Test failed! Check log above."
        fi
    fi
fi

echo -e "\n[6/6] Done!"
echo "========================================"
