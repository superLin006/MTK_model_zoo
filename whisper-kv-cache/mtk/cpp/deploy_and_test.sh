#!/bin/bash
#
# Whisper KV Cache - Deploy and Test Script
# Usage:
#   ./deploy_and_test.sh [audio_file] [language] [debug]
#
# Examples:
#   ./deploy_and_test.sh                        # Deploy only
#   ./deploy_and_test.sh test_en.wav en         # Deploy and test (no debug)
#   ./deploy_and_test.sh test_en.wav en debug   # Deploy and test (with debug)
#

echo "========================================"
echo "Whisper KV Cache - Deploy and Test"
echo "========================================"

# Configuration
DEVICE_DIR="/data/local/tmp/whisper_kv_test"
MODELS_DIR="/home/xh/projects/MTK_models_zoo/whisper-kv-cache/mtk/python/models_large_turbo"
TEST_DATA_DIR="/home/xh/projects/MTK_models_zoo/whisper-kv-cache/mtk/test_data"
MTK_LIB_DIR="/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/host/lib"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "[ERROR] No Android device connected"
    exit 1
fi

echo "[INFO] Creating device directory..."
adb shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR"

echo "[INFO] Pushing executable..."
adb push jni/libs/arm64-v8a/whisper_kv_test $DEVICE_DIR/

echo "[INFO] Pushing DLA models..."
adb push $MODELS_DIR/encoder_large-v3-turbo_128x3000_MT8371.dla $DEVICE_DIR/
adb push $MODELS_DIR/decoder_large-v3-turbo_448_MT8371.dla $DEVICE_DIR/

echo "[INFO] Pushing embeddings..."
adb push $MODELS_DIR/token_embedding.npy $DEVICE_DIR/
adb push $MODELS_DIR/position_embedding.npy $DEVICE_DIR/

echo "[INFO] Pushing mel filters and vocab..."
adb push $MODELS_DIR/mel_128_filters.txt $DEVICE_DIR/
adb push $MODELS_DIR/vocab.txt $DEVICE_DIR/

echo "[INFO] Pushing test audio files..."
adb push $TEST_DATA_DIR/test_en.wav $DEVICE_DIR/
adb push $TEST_DATA_DIR/test_zh.wav $DEVICE_DIR/
adb push $TEST_DATA_DIR/jfk.flac $DEVICE_DIR/
# Push split test files if they exist
if [ -f "$TEST_DATA_DIR/test_part1.wav" ]; then
    adb push $TEST_DATA_DIR/test_part1.wav $DEVICE_DIR/
fi
if [ -f "$TEST_DATA_DIR/test_part2.wav" ]; then
    adb push $TEST_DATA_DIR/test_part2.wav $DEVICE_DIR/
fi

echo "[INFO] Pushing MTK runtime libraries..."
adb push $MTK_LIB_DIR/libneuron_runtime.so $DEVICE_DIR/
adb push $MTK_LIB_DIR/libneuronusdk_adapter.so $DEVICE_DIR/
adb push $MTK_LIB_DIR/libc++_shared.so $DEVICE_DIR/

echo "[INFO] Setting permissions..."
adb shell "chmod +x $DEVICE_DIR/whisper_kv_test"

echo ""
echo "========================================"
echo "Deployment completed!"
echo "========================================"
echo ""

# Check if test parameters are provided
if [ $# -ge 2 ]; then
    AUDIO_FILE=$1
    LANGUAGE=$2
    DEBUG_MODE=${3:-""}

    echo "========================================"
    echo "Running inference test..."
    echo "Audio: $AUDIO_FILE"
    echo "Language: $LANGUAGE"
    if [ "$DEBUG_MODE" = "debug" ]; then
        echo "Mode: Debug (showing all messages)"
    else
        echo "Mode: Normal (hiding debug messages)"
    fi
    echo "========================================"
    echo ""

    # Build command with WHISPER_DEBUG environment variable
    if [ "$DEBUG_MODE" = "debug" ]; then
        CMD="cd $DEVICE_DIR && export LD_LIBRARY_PATH=. && export WHISPER_DEBUG=1 && ./whisper_kv_test . $AUDIO_FILE $LANGUAGE"
    else
        CMD="cd $DEVICE_DIR && export LD_LIBRARY_PATH=. && export WHISPER_DEBUG=0 && ./whisper_kv_test . $AUDIO_FILE $LANGUAGE"
    fi

    # Execute command
    adb shell "$CMD" 2>&1
else
    # Show usage instructions
    echo "Usage:"
    echo "  1. Deploy only:"
    echo "     ./deploy_and_test.sh"
    echo ""
    echo "  2. Deploy and test (no debug):"
    echo "     ./deploy_and_test.sh test_en.wav en"
    echo ""
    echo "  3. Deploy and test (with debug):"
    echo "     ./deploy_and_test.sh test_en.wav en debug"
    echo ""
    echo "  4. Manual test:"
    echo "     adb shell \"cd $DEVICE_DIR && export LD_LIBRARY_PATH=. && export WHISPER_DEBUG=0 && ./whisper_kv_test . test_en.wav en\""
    echo ""
fi
