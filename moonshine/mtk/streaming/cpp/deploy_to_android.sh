#!/bin/bash
# Deploy Moonshine Streaming to Android device

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_DIR=/data/local/tmp/moonshine_streaming_test
DELIVER="$SCRIPT_DIR/deliver"
LIBS="$SCRIPT_DIR/libs/arm64-v8a"

echo "=== Deploying Moonshine Streaming to Android ==="
echo "Device dir: $DEVICE_DIR"

# Create device directory
adb shell "mkdir -p $DEVICE_DIR"

# Push binary and shared lib
echo "Pushing binary..."
adb push "$LIBS/moonshine_streaming_test" "$DEVICE_DIR/"
adb push "$LIBS/libc++_shared.so"         "$DEVICE_DIR/"

# Push chunk encoder DLA
echo "Pushing models..."
adb push "$DELIVER/models/moonshine_encoder_chunk.dla" "$DEVICE_DIR/"
adb push "$DELIVER/models/moonshine_decoder.dla"       "$DEVICE_DIR/"
adb push "$DELIVER/models/embed_tokens.npy"            "$DEVICE_DIR/"
adb push "$DELIVER/models/proj_weight.npy"             "$DEVICE_DIR/"
adb push "$DELIVER/models/log_k.npy"                   "$DEVICE_DIR/"
adb push "$DELIVER/models/vocab.txt"                   "$DEVICE_DIR/"

# Push test audio
echo "Pushing test audio..."
adb push "$DELIVER/test_data/test_en.wav" "$DEVICE_DIR/"

# Copy MTK runtime libraries from existing zipformer test dir
echo "Copying MTK runtime libraries..."
adb shell "cp /data/local/tmp/zipformer_mtk_test/*.so $DEVICE_DIR/ 2>/dev/null || true"

# Set permissions
adb shell "chmod +x $DEVICE_DIR/moonshine_streaming_test"

echo ""
echo "=== Running test ==="
adb shell "cd $DEVICE_DIR && export LD_LIBRARY_PATH=. && \
  ./moonshine_streaming_test \
    moonshine_encoder_chunk.dla \
    moonshine_decoder.dla \
    embed_tokens.npy \
    proj_weight.npy \
    log_k.npy \
    vocab.txt \
    test_en.wav"
