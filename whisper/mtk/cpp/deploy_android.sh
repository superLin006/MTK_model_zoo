#!/bin/bash

# Whisper MTK NPU - Android Deploy and Test Script
# Deploys models and executable to MT8371 device

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
EXECUTABLE="${SCRIPT_DIR}/jni/libs/arm64-v8a/whisper_test"
MODEL_DIR="${PROJECT_ROOT}/python/models"
TEST_DATA_DIR="${PROJECT_ROOT}/test_data"

# Remote paths
REMOTE_BASE="/data/local/tmp/whisper_mtk"
REMOTE_EXEC="${REMOTE_BASE}/whisper_test"
REMOTE_MODELS="${REMOTE_BASE}/models"
REMOTE_AUDIO="${REMOTE_BASE}/audio"

echo "========================================"
echo "  Whisper MTK NPU - Deploy to Device"
echo "========================================"
echo ""

# Check device connection
echo "[INFO] Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo "[ERROR] No Android device found"
    echo "        Please connect device and enable USB debugging"
    exit 1
fi

DEVICE=$(adb devices | grep "device$" | head -1 | awk '{print $1}')
echo "[INFO] Device found: ${DEVICE}"
echo ""

# Create remote directories
echo "[INFO] Creating remote directories..."
adb shell "mkdir -p ${REMOTE_BASE}"
adb shell "mkdir -p ${REMOTE_MODELS}"
adb shell "mkdir -p ${REMOTE_AUDIO}"

# Deploy executable
echo "[INFO] Deploying executable..."
if [ -f "${EXECUTABLE}" ]; then
    adb push "${EXECUTABLE}" "${REMOTE_EXEC}"
    adb shell "chmod +x ${REMOTE_EXEC}"
    echo "  ✓ Pushed whisper_test"
else
    echo "[ERROR] Executable not found: ${EXECUTABLE}"
    echo "        Run ./build_android.sh first"
    exit 1
fi

# Deploy models
echo "[INFO] Deploying models..."
MODELS=(
    "encoder_base_80x3000_MT8371.dla"
    "decoder_base_448_MT8371.dla"
    "token_embedding.npy"
    "vocab.txt"
    "mel_80_filters.txt"
)

TOTAL_SIZE=0
for model in "${MODELS[@]}"; do
    if [ "$model" == "vocab.txt" ] || [ "$model" == "mel_80_filters.txt" ]; then
        MODEL_PATH="${SCRIPT_DIR}/models/${model}"
    else
        MODEL_PATH="${MODEL_DIR}/${model}"
    fi

    if [ -f "${MODEL_PATH}" ]; then
        SIZE=$(stat -f%z "${MODEL_PATH}" 2>/dev/null || stat -c%s "${MODEL_PATH}" 2>/dev/null)
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))

        echo "  Pushing ${model} ($(( SIZE / 1024 ))KB)..."
        adb push "${MODEL_PATH}" "${REMOTE_MODELS}/"
    else
        echo "  [WARN] Model not found: ${model}"
    fi
done

echo ""
echo "  Total model size: $((TOTAL_SIZE / 1024 / 1024))MB"

# Deploy test audio
echo "[INFO] Deploying test audio..."
AUDIO_FILES=(
    "test_en.wav"
    "test_zh.wav"
)

TEST_DATA_ALT="${SCRIPT_DIR}/test_data"

for audio in "${AUDIO_FILES[@]}"; do
    # Try test_data directory first
    AUDIO_PATH="${TEST_DATA_ALT}/${audio}"
    if [ ! -f "${AUDIO_PATH}" ]; then
        # Try project root test_data
        AUDIO_PATH="${PROJECT_ROOT}/test_data/${audio}"
    fi

    if [ -f "${AUDIO_PATH}" ]; then
        echo "  Pushing ${audio}..."
        adb push "${AUDIO_PATH}" "${REMOTE_AUDIO}/"
    else
        echo "  [WARN] Audio not found: ${audio}"
    fi
done

# Print summary
echo ""
echo "========================================"
echo "  Deployment Complete!"
echo "========================================"
echo ""
echo "Remote location: ${DEVICE}:${REMOTE_BASE}"
echo ""
echo "Deployed files:"
adb shell "ls -lh ${REMOTE_EXEC}"
adb shell "ls -lh ${REMOTE_MODELS}/"
adb shell "ls -lh ${REMOTE_AUDIO}/"
echo ""
echo "To run tests:"
echo "  adb shell"
echo "  cd ${REMOTE_BASE}"
echo "  ./whisper_test ./models ./audio/test_en.wav en"
echo ""
echo "Or use the test script:"
echo "  ./run_android_tests.sh"
echo ""
