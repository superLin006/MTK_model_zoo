#!/bin/bash
set -ex

# SenseVoice 模型验证脚本
# 验证 TFLite 模型与 PyTorch 模型的一致性

echo "========================================"
echo "  SenseVoice TFLite 验证"
echo "========================================"
echo ""

# 确保设置 PYTORCH=0（导出模式）
if grep -q "PYTORCH = 1" config.py; then
    echo "❌ 错误: 请先将 config.py 中的 PYTORCH 设置为 0"
    echo "   PYTORCH=0 表示导出模式（用于 TFLite 转换和验证）"
    echo "   PYTORCH=1 表示原生模式（仅用于 PyTorch 推理对比）"
    exit 1
fi

echo "✅ PYTORCH=0 配置正确（导出模式）"
echo ""

# 步骤1: 运行 TFLite 推理并保存输出
echo "[步骤 1/3] 运行 TFLite 推理..."
python3 main.py --mode="CHECK_TFLITE" \
    --model_path="../models/sensevoice-small" \
    --audio_path="../audios/test_en.wav" \
    --tflite_file_path="model/sensevoice_complete.tflite"

echo ""
echo "✅ TFLite 推理完成"
echo "   输出文件:"
echo "   - output/tflite_logits.npy"
echo "   - output/tflite_features.npy"
echo ""

# 步骤2: 比较输出
echo "[步骤 2/3] 比较输出..."
python3 compare_outputs.py

echo ""
echo "✅ 验证完成"
echo ""

# 步骤3: 解码文本
echo "[步骤 3/3] 解码文本..."
python3 decode_text.py \
    --logits="output/tflite_logits.npy" \
    --tokens="../models/sensevoice-small/tokens.txt"

echo ""
echo "========================================"
echo "  验证总结"
echo "========================================"
echo "✅ TFLite 模型验证完成"
echo ""
echo "输出文件:"
echo "  - TFLite logits:  output/tflite_logits.npy"
echo "  - TFLite features: output/tflite_features.npy"
echo "  - 解码文本:        output/transcription.txt"
echo ""
