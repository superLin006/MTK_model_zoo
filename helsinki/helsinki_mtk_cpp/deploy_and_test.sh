#!/bin/bash
# Deploy and test Helsinki NPU on Android device

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR="/data/local/tmp/helsinki"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/../helsinki_workspace/model_prepare/model_kvcache}"
NEURON_LIB_DIR="${NEURON_LIB_DIR:-$SCRIPT_DIR/../../0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/mt8371}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check device connection
echo -e "${YELLOW}Checking device connection...${NC}"
if ! adb devices | grep -q "device$"; then
    echo -e "${RED}Error: No Android device found${NC}"
    exit 1
fi

DEVICE=$(adb devices | grep "device$" | head -1 | awk '{print $1}')
echo -e "${GREEN}Found device: $DEVICE${NC}"

# Check executable
if [ ! -f "$SCRIPT_DIR/libs/arm64-v8a/helsinki_translate" ]; then
    echo -e "${RED}Error: Executable not found. Run ./build.sh first${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Deploying Helsinki NPU to Android Device${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Create device directory
adb shell "mkdir -p $DEVICE_DIR"

# Push executable
echo "Pushing executable..."
adb push "$SCRIPT_DIR/libs/arm64-v8a/helsinki_translate" "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/helsinki_translate"

# Note: Don't push Neuron Runtime library - use system library instead
# System library is at /vendor/lib64/mt8189/libneuron_runtime.8.so
echo "Using system Neuron Runtime library (not pushing local version)"

# Push model files
echo "Pushing model files..."
adb push "$MODEL_DIR/encoder_src64_MT8371.dla" "$DEVICE_DIR/"
adb push "$MODEL_DIR/decoder_kv_src64_cache64_MT8371.dla" "$DEVICE_DIR/"
adb push "$MODEL_DIR/embedding_weights.bin" "$DEVICE_DIR/"
adb push "$MODEL_DIR/embedding_weights_meta.txt" "$DEVICE_DIR/"
adb push "$MODEL_DIR/position_embeddings.bin" "$DEVICE_DIR/"

# Push tokenizer files
if [ -f "$MODEL_DIR/source.spm" ]; then
    echo "Pushing tokenizer files..."
    adb push "$MODEL_DIR/source.spm" "$DEVICE_DIR/"
    adb push "$MODEL_DIR/target.spm" "$DEVICE_DIR/"
    adb push "$MODEL_DIR/vocab.txt" "$DEVICE_DIR/"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Running Translation Test${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Run test
adb shell "cd $DEVICE_DIR && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$DEVICE_DIR && ./helsinki_translate . ."

echo ""
echo -e "${GREEN}Done!${NC}"
