#!/bin/bash
# Helsinki Model DLA Compilation Script
# 将TFLite模型编译为MediaTek DLA格式

set -e

# 使用方法
usage() {
    echo "Usage: $0 <tflite_file> <platform> [neuropilot_sdk_path]"
    echo ""
    echo "Arguments:"
    echo "  tflite_file         Path to TFLite model file"
    echo "  platform            Target platform: MT6899 | MT6991 | MT8371"
    echo "  neuropilot_sdk_path Optional: Path to NeuroPilot SDK"
    echo ""
    echo "Example:"
    echo "  $0 ../model_prepare/model/helsinki_full.tflite MT8371"
    echo "  $0 ../model_prepare/model/helsinki_full.tflite MT6899 /path/to/neuron_sdk"
    exit 1
}

# 检查参数
if [ $# -lt 2 ]; then
    usage
fi

TFLITE_FILE="$1"
PLATFORM="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_PATH="${3:-$SCRIPT_DIR/../../../0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk}"

# 检查TFLite文件
if [ ! -f "$TFLITE_FILE" ]; then
    echo "Error: TFLite file not found: $TFLITE_FILE"
    exit 1
fi

# 检查SDK路径
NCC_TFLITE="$SDK_PATH/host/bin/ncc-tflite"
if [ ! -f "$NCC_TFLITE" ]; then
    echo "Error: ncc-tflite not found at: $NCC_TFLITE"
    echo "Please check NeuroPilot SDK path"
    exit 1
fi

# 根据平台设置编译参数
case $PLATFORM in
    MT6899)
        ARCH="mdla5.5,mvpu2.5"
        L1_SIZE="2048"
        NUM_MDLA="2"
        ;;
    MT6991)
        ARCH="mdla5.5,mvpu2.5"
        L1_SIZE="7168"
        NUM_MDLA="4"
        ;;
    MT8371)
        ARCH="mdla5.3,edma3.6"
        L1_SIZE="256"
        NUM_MDLA="1"
        ;;
    *)
        echo "Error: Unsupported platform: $PLATFORM"
        echo "Supported platforms: MT6899, MT6991, MT8371"
        exit 1
        ;;
esac

# 生成输出文件名
BASENAME=$(basename "$TFLITE_FILE" .tflite)
OUTPUT_DIR=$(dirname "$TFLITE_FILE")
DLA_FILE="${OUTPUT_DIR}/${BASENAME}_${PLATFORM}.dla"

echo "=========================================="
echo "Helsinki DLA Compilation"
echo "=========================================="
echo "Input: $TFLITE_FILE"
echo "Output: $DLA_FILE"
echo "Platform: $PLATFORM"
echo "Architecture: $ARCH"
echo "L1 Cache: ${L1_SIZE}KB"
echo "MDLA Cores: $NUM_MDLA"
echo ""

# 设置环境
export PATH="$SDK_PATH/host/bin:$PATH"

# 编译命令
echo "Running ncc-tflite..."
$NCC_TFLITE "$TFLITE_FILE" \
    --arch=$ARCH \
    --l1-size-kb=$L1_SIZE \
    --num-mdla=$NUM_MDLA \
    --relax-fp32 \
    --opt-accuracy \
    --opt-footprint \
    --fc-to-conv \
    -o "$DLA_FILE"

# 检查结果
if [ -f "$DLA_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "Compilation successful!"
    echo "Output: $DLA_FILE"
    echo "Size: $(du -h "$DLA_FILE" | cut -f1)"
    echo "=========================================="
else
    echo "Error: DLA file not generated"
    exit 1
fi
