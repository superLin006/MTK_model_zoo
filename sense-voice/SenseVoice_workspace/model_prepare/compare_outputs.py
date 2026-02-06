#!/usr/bin/env python3
"""
SenseVoice 模型输出对比工具
比较 PyTorch、TorchScript 和 TFLite 模型的输出
"""

import numpy as np
import sys

def compare_outputs(pytorch_path, tflite_path, vocab_size=25055):
    """
    对比 PyTorch 和 TFLite 的输出

    Args:
        pytorch_path: PyTorch 输出的 numpy 文件路径
        tflite_path: TFLite 输出的 numpy 文件路径
        vocab_size: 词汇表大小
    """
    print("="*80)
    print("  SenseVoice 模型输出对比")
    print("="*80)
    print()

    # 加载数据
    print("加载输出数据...")
    pytorch_logits = np.load(pytorch_path)
    tflite_logits = np.load(tflite_path)

    print(f"PyTorch shape:  {pytorch_logits.shape}")
    print(f"TFLite shape:   {tflite_logits.shape}")
    print()

    # 检查形状
    if pytorch_logits.shape != tflite_logits.shape:
        print("⚠️  警告: 形状不匹配")
        print(f"   PyTorch: {pytorch_logits.shape}")
        print(f"   TFLite:  {tflite_logits.shape}")

        # 如果帧数不同，只比较重叠部分
        min_frames = min(pytorch_logits.shape[1], tflite_logits.shape[1])
        print(f"   只比较前 {min_frames} 帧")
        pytorch_logits = pytorch_logits[:, :min_frames, :]
        tflite_logits = tflite_logits[:, :min_frames, :]

    # 计算差异
    diff = np.abs(pytorch_logits - tflite_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print("差异统计:")
    print(f"  最大绝对误差: {max_diff:.6f}")
    print(f"  平均绝对误差: {mean_diff:.6f}")
    print(f"  相对误差:     {mean_diff / (np.abs(pytorch_logits).mean() + 1e-9) * 100:.2f}%")
    print()

    # Argmax 对比（预测的 token）
    pytorch_tokens = pytorch_logits.argmax(axis=-1)
    tflite_tokens = tflite_logits.argmax(axis=-1)

    token_match = (pytorch_tokens == tflite_tokens).sum()
    token_total = pytorch_tokens.size
    token_accuracy = token_match / token_total * 100

    print("Token 预测对比:")
    print(f"  匹配数:   {token_match} / {token_total}")
    print(f"  准确率:   {token_accuracy:.2f}%")
    print()

    # 判断是否通过
    if token_accuracy == 100.0:
        print("✅ 验证通过: Token 预测完全一致")
        result = "PASS"
    elif token_accuracy >= 99.9:
        print("✅ 验证通过: Token 预测基本一致 (误差 <0.1%)")
        result = "PASS"
    else:
        print(f"❌ 验证失败: Token 准确率 {token_accuracy:.2f}% < 99.9%")
        result = "FAIL"

    # 显示一些样本对比
    print()
    print("样本对比 (前5帧，各帧前5个token):")
    for i in range(min(5, pytorch_logits.shape[1])):
        print(f"  帧 {i}:")
        for j in range(min(5, vocab_size)):
            py_val = pytorch_logits[0, i, j]
            tf_val = tflite_logits[0, i, j]
            diff_val = abs(py_val - tf_val)
            status = "✓" if diff_val < 0.1 else "✗"
            print(f"    token[{j}]: PyTorch={py_val:8.4f}, TFLite={tf_val:8.4f}, diff={diff_val:8.4f} {status}")

    print()
    print("="*80)
    print(f"最终结果: {result}")
    print("="*80)

    return result == "PASS"


if __name__ == "__main__":
    import os

    # 检查文件是否存在
    output_dir = "output"

    pytorch_file = os.path.join(output_dir, "pytorch_logits.npy")
    tflite_file = os.path.join(output_dir, "tflite_logits.npy")

    if not os.path.exists(pytorch_file):
        print(f"❌ 错误: 找不到 PyTorch 输出文件: {pytorch_file}")
        print("   请先运行: python3 main.py --mode=PYTORCH")
        sys.exit(1)

    if not os.path.exists(tflite_file):
        print(f"❌ 错误: 找不到 TFLite 输出文件: {tflite_file}")
        print("   请先运行: python3 main.py --mode=CHECK_TFLITE")
        sys.exit(1)

    # 运行对比
    success = compare_outputs(pytorch_file, tflite_file)

    sys.exit(0 if success else 1)
