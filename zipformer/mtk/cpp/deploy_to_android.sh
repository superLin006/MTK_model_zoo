#!/bin/bash
#
# Deploy and test Zipformer MTK NPU on Android device
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE_DIR=/data/local/tmp/zipformer_mtk_test
MTK_LIB=/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/mt8371/lib
MODEL_DIR=/home/xh/projects/MTK_models_zoo/zipformer-mtk/mtk/python/models
TEST_DATA=/home/xh/projects/MTK_models_zoo/zipformer-mtk/mtk/test_data
BINARY=$SCRIPT_DIR/libs/arm64-v8a/zipformer_mtk_test

echo "=== Zipformer MTK Deploy ==="
echo "Device dir: $DEVICE_DIR"
echo ""

# Check device
echo "Checking ADB devices..."
adb devices

if ! adb get-state > /dev/null 2>&1; then
    echo "ERROR: No ADB device connected"
    exit 1
fi

echo ""
echo "Creating device directory..."
adb shell "mkdir -p $DEVICE_DIR"

echo "Pushing binary..."
adb push "$BINARY" "$DEVICE_DIR/"
adb push "$SCRIPT_DIR/libs/arm64-v8a/libc++_shared.so" "$DEVICE_DIR/"

echo "Pushing MTK runtime libs..."
adb push "$MTK_LIB"/*.so "$DEVICE_DIR/" || true

echo "Pushing models..."
adb push "$MODEL_DIR/encoder.dla" "$DEVICE_DIR/"
adb push "$MODEL_DIR/decoder_npu.dla" "$DEVICE_DIR/"
adb push "$MODEL_DIR/joiner.dla" "$DEVICE_DIR/"
adb push "$MODEL_DIR/decoder_embedding_weight.npy" "$DEVICE_DIR/"

echo "Pushing test data..."
adb push "$TEST_DATA/test.wav" "$DEVICE_DIR/"
adb push "$TEST_DATA/vocab.txt" "$DEVICE_DIR/"

echo ""
echo "=== Running inference ==="
adb shell "cd $DEVICE_DIR && chmod +x ./zipformer_mtk_test && \
    LD_LIBRARY_PATH=. ./zipformer_mtk_test \
    encoder.dla decoder_npu.dla joiner.dla \
    decoder_embedding_weight.npy test.wav vocab.txt"

echo ""
echo "=== Done ==="
