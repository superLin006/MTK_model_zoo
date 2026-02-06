#!/bin/bash
#
# Deploy and test SenseVoice on Android device
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE_DIR="/data/local/tmp/sensevoice"
DLA_FILE="$SCRIPT_DIR/../SenseVoice_workspace/compile/sensevoice_MT8371.dla"
TOKENS_FILE="$SCRIPT_DIR/../SenseVoice_workspace/models/sensevoice-small/tokens.txt"
AUDIO_DIR="$SCRIPT_DIR/../SenseVoice_workspace/audios"

echo "==================================="
echo "Deploying SenseVoice to device"
echo "==================================="

# Create device directory
adb shell "mkdir -p $DEVICE_DIR"

# Push executable
echo "Pushing executable..."
adb push libs/arm64-v8a/sensevoice_main $DEVICE_DIR/
adb shell "chmod +x $DEVICE_DIR/sensevoice_main"

# Push C++ shared library
echo "Pushing libc++_shared.so..."
NDK_STL_LIB=$(find /home/xh/Android/Sdk/ndk -name "libc++_shared.so" -path "*arm64-v8a*" | head -1)
if [ -n "$NDK_STL_LIB" ]; then
    adb push "$NDK_STL_LIB" $DEVICE_DIR/
fi

# Push model files
echo "Pushing model files..."
if [ -f "$DLA_FILE" ]; then
    adb push "$DLA_FILE" $DEVICE_DIR/
else
    echo "Warning: DLA file not found at $DLA_FILE"
fi

if [ -f "$TOKENS_FILE" ]; then
    adb push "$TOKENS_FILE" $DEVICE_DIR/
else
    echo "Warning: Tokens file not found at $TOKENS_FILE"
fi

# List deployed files
echo ""
echo "Deployed files:"
adb shell "ls -la $DEVICE_DIR/"

echo ""
echo "==================================="
echo "Deployment completed!"
echo "==================================="
echo ""

# Test function
run_test() {
    local audio_file=$1
    local audio_name=$(basename "$audio_file")

    echo ""
    echo "-----------------------------------"
    echo "Testing: $audio_name"
    echo "-----------------------------------"

    # Push audio file
    adb push "$audio_file" $DEVICE_DIR/

    # Run inference
    adb shell "cd $DEVICE_DIR && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$DEVICE_DIR && ./sensevoice_main sensevoice_MT8371.dla tokens.txt $audio_name"
}

# Run tests if audio files exist
if [ "$1" == "--test" ]; then
    shift
    if [ -n "$1" ]; then
        # Test specific file(s)
        for audio_file in "$@"; do
            if [ -f "$audio_file" ]; then
                run_test "$audio_file"
            else
                echo "Warning: Audio file not found: $audio_file"
            fi
        done
    else
        # Test default files
        for audio_file in "$AUDIO_DIR/test_zh.wav" "$AUDIO_DIR/audio4.wav"; do
            if [ -f "$audio_file" ]; then
                run_test "$audio_file"
            fi
        done
    fi
fi

echo ""
echo "To run a test manually:"
echo "  adb shell \"cd $DEVICE_DIR && export LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH:$DEVICE_DIR && ./sensevoice_main sensevoice_MT8371.dla tokens.txt <audio.wav>\""
