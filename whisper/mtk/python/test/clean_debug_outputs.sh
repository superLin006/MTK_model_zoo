#!/bin/bash
# 清理debug目录的中间输出文件

echo "🧹 清理 Whisper debug 输出..."
rm -f outputs/debug/*.npy
rm -f outputs/debug/*.bin

echo "✓ 清理完成"
echo "保留了baseline/torchscript/tflite/dla等关键输出"
