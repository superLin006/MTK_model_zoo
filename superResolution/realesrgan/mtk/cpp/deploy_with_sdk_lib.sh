#!/bin/bash
# Real-ESRGAN Deploy Script with SDK Libraries
# Push binary and required MTK runtime libraries to Android device

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Real-ESRGAN Deploy Script"
echo "=========================================="

# 配置（优先使用环境变量，否则使用默认路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTK_NEUROPILOT_SDK="${MTK_NEUROPILOT_SDK:-$SCRIPT_DIR/../../../../0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk}"
TARGET_ABI="arm64-v8a"
TARGET_PLATFORM="mt8371"  # mt8371, mt6899, mt6991
DEVICE_DIR="/data/local/tmp/realesrgan"

# 检查设备连接
echo ""
echo "[1/5] Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo -e "${RED}[ERROR] No Android device found${NC}"
    echo "Please connect a device and enable USB debugging"
    exit 1
fi
echo -e "${GREEN}✓ Device connected${NC}"

# 检查编译输出
echo ""
echo "[2/5] Checking build output..."
BINARY="build/$TARGET_ABI/realesrgan_inference"
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}[ERROR] Binary not found: $BINARY${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi
echo -e "${GREEN}✓ Binary found: $BINARY${NC}"

# 创建设备目录
echo ""
echo "[3/5] Creating device directory..."
adb shell "mkdir -p $DEVICE_DIR"
echo -e "${GREEN}✓ Device directory created: $DEVICE_DIR${NC}"

# 推送可执行文件
echo ""
echo "[4/5] Pushing binary..."
adb push "$BINARY" "$DEVICE_DIR/"
adb shell "chmod 755 $DEVICE_DIR/realesrgan_inference"
echo -e "${GREEN}✓ Binary pushed${NC}"

# 推送MTK运行时库
echo ""
echo "[5/5] Pushing MTK runtime libraries..."

MTK_LIB_DIR="$MTK_NEUROPILOT_SDK/target/$TARGET_ABI"
REQUIRED_LIBS=(
    "libneuron_runtime.so"
    "libneuron_runtime.core.so"
    "libmdla.so.10.0"
    "libmdla_platform.so.10.0"
)

for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "$MTK_LIB_DIR/$lib" ]; then
        adb push "$MTK_LIB_DIR/$lib" "$DEVICE_DIR/"
        adb shell "chmod 644 $DEVICE_DIR/$lib"
        echo -e "  ${GREEN}✓ $lib${NC}"
    else
        echo -e "  ${YELLOW}⚠ Warning: $lib not found${NC}"
    fi
done

echo ""
echo -e "${GREEN}=========================================="
echo "Deploy Complete!"
echo "==========================================${NC}"
echo ""
echo "Device directory: $DEVICE_DIR"
echo ""
echo "Binary and libraries pushed successfully!"
echo ""
echo "Next steps:"
echo "  1. Push a test image:"
echo "     adb push <test_image> $DEVICE_DIR/"
echo ""
echo "  2. Push the DLA model:"
echo "     adb push <model.dla> $DEVICE_DIR/"
echo ""
echo "  3. Run inference:"
echo "     adb shell"
echo "     cd $DEVICE_DIR"
echo "     export LD_LIBRARY_PATH=.\$LD_LIBRARY_PATH"
echo "     ./realesrgan_inference <model.dla> <input.png> <output.png>"
echo ""
