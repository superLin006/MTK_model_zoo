#!/bin/bash
# Deploy Moonshine MTK NPU Inference to Android device

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_DIR=/data/local/tmp/moonshine_test
MODELS_DIR="$SCRIPT_DIR/../python/models"
MODEL_DIR="$SCRIPT_DIR/../models/moonshine-streaming-small"
TEST_DATA="$SCRIPT_DIR/../test_data"

echo "========================================"
echo "Deploying Moonshine to Android"
echo "Device directory: $DEVICE_DIR"
echo "========================================"

# Check device
adb devices || { echo "No ADB device found"; exit 1; }

# Create directory
adb shell "mkdir -p $DEVICE_DIR"

# Push executable
echo "Pushing executable..."
adb push "$SCRIPT_DIR/libs/arm64-v8a/moonshine_test" "$DEVICE_DIR/"
adb push "$SCRIPT_DIR/libs/arm64-v8a/libc++_shared.so" "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/moonshine_test"

# Push DLA models
echo "Pushing DLA models..."
adb push "$MODELS_DIR/moonshine_encoder.dla" "$DEVICE_DIR/"
adb push "$MODELS_DIR/moonshine_decoder.dla" "$DEVICE_DIR/"

# Push weight files
echo "Pushing weight files..."
adb push "$MODELS_DIR/embed_tokens.npy" "$DEVICE_DIR/"
adb push "$MODELS_DIR/pos_emb_weight.npy" "$DEVICE_DIR/"
adb push "$MODELS_DIR/proj_weight.npy" "$DEVICE_DIR/"
adb push "$MODELS_DIR/log_k.npy" "$DEVICE_DIR/"
adb push "$MODELS_DIR/vocab.txt" "$DEVICE_DIR/"

# Push test audio
echo "Pushing test audio..."
adb push "$TEST_DATA/test_en.wav" "$DEVICE_DIR/"

# Copy runtime libs from existing test dir (if available)
adb shell "ls /data/local/tmp/zipformer_mtk_test/*.so 2>/dev/null && \
  cp /data/local/tmp/zipformer_mtk_test/*.so $DEVICE_DIR/ 2>/dev/null || true"

echo ""
echo "========================================"
echo "Deploy complete!"
echo ""
echo "Run with:"
echo "  adb shell 'cd $DEVICE_DIR && export LD_LIBRARY_PATH=. && \\"
echo "  ./moonshine_test moonshine_encoder.dla moonshine_decoder.dla \\"
echo "  embed_tokens.npy pos_emb_weight.npy proj_weight.npy log_k.npy \\"
echo "  vocab.txt test_en.wav'"
echo "========================================"
