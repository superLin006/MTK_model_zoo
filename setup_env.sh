#!/bin/bash

# ============================================
# MTK Model Zoo - 环境设置脚本
# ============================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MTK Model Zoo - Environment Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查是否在正确的目录
if [ ! -f "README.md" ] || [ ! -d ".claude" ]; then
    echo -e "${RED}错误: 请在 MTK_model_zoo 根目录运行此脚本${NC}"
    exit 1
fi

PROJECT_ROOT=$(pwd)

# ============================================
# 1. 检查 MTK SDK
# ============================================
echo -e "${YELLOW}[1/5] 检查 MTK NeuroPilot SDK...${NC}"

SDK_DIR="$PROJECT_ROOT/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029"

if [ ! -d "$SDK_DIR" ]; then
    echo -e "${RED}✗ MTK SDK 未找到${NC}"
    echo ""
    echo "请下载 MTK NeuroPilot SDK 8.0.10 并解压到:"
    echo "  $SDK_DIR"
    echo ""
    echo "下载地址: https://neuropilot.mediatek.com/"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ MTK SDK 已安装${NC}"
fi

# ============================================
# 2. 检查 Python 环境
# ============================================
echo ""
echo -e "${YELLOW}[2/5] 检查 Python 环境...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 未安装${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"

# ============================================
# 3. 创建目录结构
# ============================================
echo ""
echo -e "${YELLOW}[3/5] 创建必要的目录结构...${NC}"

# 确保所有 .gitkeep 目录存在
while IFS= read -r gitkeep_file; do
    dir=$(dirname "$gitkeep_file")
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  ✓ Created: $dir"
    fi
done < <(find . -name ".gitkeep" 2>/dev/null)

echo -e "${GREEN}✓ 目录结构已准备${NC}"

# ============================================
# 4. 检查 Android NDK (可选)
# ============================================
echo ""
echo -e "${YELLOW}[4/5] 检查 Android NDK (用于C++编译)...${NC}"

if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo -e "${YELLOW}⚠ ANDROID_NDK_ROOT 未设置${NC}"
    echo "  如果需要编译 C++ 代码，请设置环境变量:"
    echo "  export ANDROID_NDK_ROOT=/path/to/android-ndk"
else
    echo -e "${GREEN}✓ Android NDK: $ANDROID_NDK_ROOT${NC}"
fi

# ============================================
# 5. 显示下一步操作
# ============================================
echo ""
echo -e "${YELLOW}[5/5] 环境检查完成！${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}下一步操作:${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "1. 下载模型权重文件"
echo "   - Whisper: 下载 base.pt 放到 whisper/mtk/models/"
echo "   - EDSR: 下载权重放到 superResolution/edsr/mtk/models/"
echo ""
echo "2. 准备测试数据"
echo "   - 音频文件放到 whisper/mtk/test_data/"
echo "   - 图像文件放到 superResolution/*/mtk/test_data/"
echo ""
echo "3. Python 端转换示例 (Whisper):"
echo "   cd whisper/mtk/python"
echo "   python step1_pt_to_torchscript.py"
echo "   python step2_torchscript_to_tflite.py"
echo "   python step3_tflite_to_dla.py --platform MT8371"
echo ""
echo "4. C++ 编译示例 (Whisper):"
echo "   cd whisper/mtk/cpp"
echo "   bash build_android.sh"
echo ""
echo "5. 查看文档:"
echo "   - 项目说明: README.md"
echo "   - Subagent 系统: .claude/subagents/README.md"
echo "   - 输出管理规范: .claude/standards/python_output_management.md"
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""
