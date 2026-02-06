#!/bin/bash
set -ex

# Compile SenseVoice TFLite to DLA format
# Usage: ./compile_sensevoice_fp.sh <TFLITE_PATH> <PLATFORM> <NEURON_SDK_PATH>

TFLITE_PATH=${1:-"../model_prepare/model/sensevoice_complete.tflite"}
PLATFORM=${2:-"MT6899"}
NEURON_SDK_PATH=${3:-"/home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"}

echo "================================================================"
echo "SenseVoice TFLite -> DLA Compilation"
echo "================================================================"
echo "TFLite Path: ${TFLITE_PATH}"
echo "Platform: ${PLATFORM}"
echo "Neuron SDK: ${NEURON_SDK_PATH}"
echo "================================================================"

# Check if file exists
if [ ! -f "${TFLITE_PATH}" ]; then
    echo "Error: TFLite file not found: ${TFLITE_PATH}"
    exit 1
fi

# Set up environment
export PATH="${NEURON_SDK_PATH}/host/bin:${PATH}"
export LD_LIBRARY_PATH="${NEURON_SDK_PATH}/host/lib:${LD_LIBRARY_PATH}"

# Platform-specific settings
case ${PLATFORM} in
    "MT6899")
        ARCH="mdla5.5"
        L1_SIZE="2048"
        NUM_MDLA="2"
        ;;
    "MT6991")
        ARCH="mdla5.5"
        L1_SIZE="7168"
        NUM_MDLA="4"
        ;;
    "MT8371")
        ARCH="mdla5.3,edma3.6"
        L1_SIZE="256"
        NUM_MDLA="1"
        ;;
    *)
        echo "Warning: Unknown platform ${PLATFORM}, using default settings"
        ARCH="mdla5.5"
        L1_SIZE="2048"
        NUM_MDLA="2"
        ;;
esac

echo "Architecture: ${ARCH}"
echo "L1 Cache Size: ${L1_SIZE}KB"
echo "NUM MDLA: ${NUM_MDLA}"
echo ""

# Output file
OUTPUT_FILE="sensevoice_${PLATFORM}.dla"

# Compile command
ncc-tflite \
    --arch=${ARCH} \
    -O3 \
    --l1-size-kb=${L1_SIZE} \
    --num-mdla=${NUM_MDLA} \
    --show-memory-summary \
    --relax-fp32 \
    --opt-accuracy \
    --opt-footprint \
    --fc-to-conv \
    -d ${OUTPUT_FILE} \
    ${TFLITE_PATH} \
    2>&1 | tee compile_sensevoice_${PLATFORM}.log

echo ""
echo "================================================================"
echo "Compilation completed successfully!"
echo "Output: ${OUTPUT_FILE}"
echo "Log: compile_sensevoice_${PLATFORM}.log"
echo "================================================================"
