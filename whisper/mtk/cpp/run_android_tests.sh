#!/bin/bash

# Whisper MTK NPU - Android Test Runner
# Runs inference tests on connected Android device

set -e

REMOTE_BASE="/data/local/tmp/whisper_mtk"
REMOTE_EXEC="${REMOTE_BASE}/whisper_test"

echo "========================================"
echo "  Whisper MTK NPU - Android Tests"
echo "========================================"
echo ""

# Check device
if ! adb devices | grep -q "device$"; then
    echo "[ERROR] No Android device connected"
    exit 1
fi

DEVICE=$(adb devices | grep "device$" | head -1 | awk '{print $1}')
echo "[INFO] Device: ${DEVICE}"
echo ""

# Test 1: English audio
echo "=== Test 1: English Audio ==="
echo "Command: ./whisper_test ./models ./audio/test_en.wav en transcribe"
echo ""
adb shell "cd ${REMOTE_BASE} && ${REMOTE_EXEC} ./models ./audio/test_en.wav en transcribe" 2>&1 | tail -20
echo ""

# Test 2: Chinese audio
echo "=== Test 2: Chinese Audio ==="
echo "Command: ./whisper_test ./models ./audio/test_zh.wav zh transcribe"
echo ""
adb shell "cd ${REMOTE_BASE} && ${REMOTE_EXEC} ./models ./audio/test_zh.wav zh transcribe" 2>&1 | tail -20
echo ""

echo "========================================"
echo "  Tests Complete"
echo "========================================"
