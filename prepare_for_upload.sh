#!/bin/bash

# ============================================
# MTK Model Zoo - 准备上传脚本
# ============================================
# 清理嵌套的 git 仓库，准备上传到新仓库

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}准备上传到 GitHub${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

PROJECT_ROOT="/home/xh/projects/MTK"
cd "$PROJECT_ROOT" || exit 1

# ============================================
# 1. 检测嵌套的 git 仓库
# ============================================
echo -e "${YELLOW}[1/4] 检测嵌套的 git 仓库...${NC}"

NESTED_GITS=$(find . -name ".git" -type d | grep -v "^\./\.git$" || true)

if [ -n "$NESTED_GITS" ]; then
    echo "发现以下嵌套的 git 仓库:"
    echo "$NESTED_GITS"
    echo ""

    read -p "是否删除这些嵌套的 .git 目录? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        while IFS= read -r git_dir; do
            if [ -d "$git_dir" ]; then
                rm -rf "$git_dir"
                echo -e "${GREEN}✓ 已删除: $git_dir${NC}"
            fi
        done <<< "$NESTED_GITS"
    else
        echo -e "${YELLOW}⚠ 跳过删除，建议手动处理${NC}"
    fi
else
    echo -e "${GREEN}✓ 没有嵌套的 git 仓库${NC}"
fi

# ============================================
# 2. 检查 .gitignore
# ============================================
echo ""
echo -e "${YELLOW}[2/4] 检查 .gitignore...${NC}"

if [ ! -f ".gitignore" ]; then
    echo -e "${RED}✗ .gitignore 不存在！${NC}"
    exit 1
else
    echo -e "${GREEN}✓ .gitignore 已配置${NC}"
fi

# ============================================
# 3. 验证 .gitkeep 文件
# ============================================
echo ""
echo -e "${YELLOW}[3/4] 验证 .gitkeep 文件...${NC}"

GITKEEP_COUNT=$(find . -name ".gitkeep" | wc -l)
echo -e "${GREEN}✓ 找到 $GITKEEP_COUNT 个 .gitkeep 占位文件${NC}"

# ============================================
# 4. 模拟添加文件
# ============================================
echo ""
echo -e "${YELLOW}[4/4] 模拟 git add (预览将被添加的文件)...${NC}"
echo ""

# 重新初始化 git (如果已存在则跳过)
if [ ! -d ".git" ]; then
    git init
fi

echo "前50个将被添加的文件:"
git add -n . 2>&1 | head -50

echo ""
echo "统计信息:"
echo "  - 总计会添加: $(git add -n . 2>&1 | wc -l) 个文件"
echo "  - Python 文件: $(git add -n . 2>&1 | grep '\.py$' | wc -l)"
echo "  - C++ 文件: $(git add -n . 2>&1 | grep -E '\.(cpp|h|hpp)$' | wc -l)"
echo "  - 脚本文件: $(git add -n . 2>&1 | grep '\.sh$' | wc -l)"
echo "  - 文档文件: $(git add -n . 2>&1 | grep '\.md$' | wc -l)"
echo ""

# 检查是否有不应该添加的文件
echo "检查是否有模型文件或编译产物..."
SHOULD_IGNORE=$(git add -n . 2>&1 | grep -E "(\.pt|\.pth|\.dla|\.tflite|\.npy|/libs/|/obj/|__pycache__|\.pyc)" || true)

if [ -n "$SHOULD_IGNORE" ]; then
    echo -e "${RED}✗ 警告: 发现不应该添加的文件！${NC}"
    echo "$SHOULD_IGNORE"
    echo ""
    echo "请检查 .gitignore 配置"
    exit 1
else
    echo -e "${GREEN}✓ 没有发现不应该添加的文件${NC}"
fi

# ============================================
# 5. 提示下一步操作
# ============================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}准备完成！下一步操作:${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "1. 添加远程仓库:"
echo "   git remote add origin https://github.com/superLin006/MTK_model_zoo.git"
echo ""
echo "2. 添加所有文件:"
echo "   git add ."
echo ""
echo "3. 创建首次提交:"
echo "   git commit -m \"Initial commit: MTK Model Zoo\""
echo ""
echo "4. 推送到 GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo -e "${YELLOW}注意: 请确保已删除嵌套的 .git 目录！${NC}"
echo ""
