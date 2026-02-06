#!/bin/bash
# Real-ESRGAN Quick Deploy and Test Script
# Deploy and run inference with a single command

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Real-ESRGAN Deploy & Test"
echo "=========================================="

# 配置
DEVICE_DIR="/data/local/tmp/realesrgan"
MODEL_PATH="../../../models/RealESRGAN_x4plus_339x510_MT8371.dla"
TEST_IMAGE="../../../test_data/input_510x339.png"

# 检查文件
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}[ERROR] Model not found: $MODEL_PATH${NC}"
    echo "Please complete the Python conversion first"
    exit 1
fi

if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${RED}[ERROR] Test image not found: $TEST_IMAGE${NC}"
    exit 1
fi

# 部署
echo ""
echo "[1/4] Deploying to device..."
bash deploy_with_sdk_lib.sh

# 推送模型和测试图像
echo ""
echo "[2/4] Pushing test data..."
adb push "$MODEL_PATH" "$DEVICE_DIR/model.dla"
adb push "$TEST_IMAGE" "$DEVICE_DIR/input.png"
echo -e "${GREEN}✓ Test data pushed${NC}"

# 运行推理
echo ""
echo "[3/4] Running inference..."
adb shell "cd $DEVICE_DIR && \
    export LD_LIBRARY_PATH=.\$LD_LIBRARY_PATH && \
    ./realesrgan_inference model.dla input.png output.png"

# 拉取结果
echo ""
echo "[4/4] Pulling output..."
adb pull "$DEVICE_DIR/output.png" "./output_test.png"
echo -e "${GREEN}✓ Output saved to: ./output_test.png${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo "Test Complete!"
echo "==========================================${NC}"
echo ""
echo "Output image: ./output_test.png"
echo "You can compare it with the input image:"
echo "  Input: $TEST_IMAGE"
echo "  Output: ./output_test.png"
