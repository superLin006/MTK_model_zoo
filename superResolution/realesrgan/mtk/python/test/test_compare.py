#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比PyTorch和TFLite模型输出
验证模型转换的正确性
"""

import sys
import os
import numpy as np
import torch
import cv2
import argparse

# 导入模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入MTK的TFLite
try:
    from mtk_converter import TFLiteExecutor
    MTK_TFLITE_AVAILABLE = True
except ImportError:
    MTK_TFLITE_AVAILABLE = False

from realesrgan_model import load_realesrgan_from_checkpoint


def preprocess(img_path, target_size=None):
    """Load and preprocess image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    if target_size is not None:
        h_target, w_target = target_size
        img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def run_pytorch_inference(model_path, img_tensor, device='cpu'):
    """Run PyTorch model inference."""
    print("\n--> 运行PyTorch推理...")
    model, scale = load_realesrgan_from_checkpoint(model_path, device)
    model.eval()

    img_torch = torch.from_numpy(img_tensor).to(device)

    with torch.no_grad():
        output = model(img_torch)

    output_np = output.cpu().numpy()
    print(f"    输出形状: {output_np.shape}")
    print(f"    输出范围: [{output_np.min():.4f}, {output_np.max():.4f}]")

    return output_np


def run_tflite_inference(tflite_path, img_tensor):
    """Run TFLite model inference using MTK TFLiteExecutor."""
    if not MTK_TFLITE_AVAILABLE:
        print("    ❌ TFLite不可用")
        return None

    print("\n--> 运行MTK TFLite推理...")

    # Load model
    executor = TFLiteExecutor(tflite_path)

    # Run inference
    output_list = executor.run([img_tensor.astype(np.float32)])
    output = output_list[0] if isinstance(output_list, list) else output_list

    print(f"    输出形状: {output.shape}")
    print(f"    输出范围: [{output.min():.4f}, {output.max():.4f}]")

    return output


def compare_outputs(pytorch_output, tflite_output):
    """Compare PyTorch and TFLite outputs."""
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)

    if pytorch_output is None or tflite_output is None:
        print("❌ 无法对比：缺少输出")
        return False

    # Ensure same shape
    if pytorch_output.shape != tflite_output.shape:
        print(f"❌ 形状不匹配: {pytorch_output.shape} vs {tflite_output.shape}")
        return False

    # Calculate metrics
    diff = np.abs(pytorch_output - tflite_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    mse = np.mean((pytorch_output - tflite_output) ** 2)

    # PSNR (peak signal-to-noise ratio)
    # 对于[0,1]范围的数据，max_value = 1.0
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    print(f"  最大差异:  {max_diff:.6f}")
    print(f"  平均差异:  {mean_diff:.6f}")
    print(f"  MSE:       {mse:.8f}")
    print(f"  PSNR:      {psnr:.2f} dB")

    # 判断是否匹配
    if max_diff < 1e-3:
        print(f"\n✓ 优秀匹配 (max_diff < 0.001)")
        return True
    elif max_diff < 1e-2:
        print(f"\n✓ 良好匹配 (max_diff < 0.01)")
        return True
    elif max_diff < 0.1:
        print(f"\n⚠ 可接受匹配 (max_diff < 0.1)")
        return True
    else:
        print(f"\n❌ 匹配较差 (max_diff >= 0.1)")
        return False


def save_comparison_images(pytorch_output, tflite_output, output_dir):
    """Save comparison images."""
    os.makedirs(output_dir, exist_ok=True)

    # Postprocess and save PyTorch output
    output_pt = pytorch_output.squeeze(0)
    output_pt = np.clip(output_pt, 0, 1)
    output_pt = (output_pt * 255.0).round().astype(np.uint8)
    output_pt = np.transpose(output_pt, (1, 2, 0))
    output_pt = cv2.cvtColor(output_pt, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, 'compare_pytorch.png'), output_pt)

    # Postprocess and save TFLite output
    if tflite_output is not None:
        output_tflite = tflite_output.squeeze(0)
        output_tflite = np.clip(output_tflite, 0, 1)
        output_tflite = (output_tflite * 255.0).round().astype(np.uint8)
        output_tflite = np.transpose(output_tflite, (1, 2, 0))
        output_tflite = cv2.cvtColor(output_tflite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'compare_tflite.png'), output_tflite)

    # Save difference image
    if tflite_output is not None:
        diff = np.abs(pytorch_output - tflite_output)
        diff = diff.squeeze(0)
        diff = (diff / diff.max() * 255.0).astype(np.uint8)
        diff = np.transpose(diff, (1, 2, 0))
        cv2.imwrite(os.path.join(output_dir, 'compare_diff.png'), diff)

    print(f"\n--> 对比图像保存到: {output_dir}")
    print(f"    - compare_pytorch.png: PyTorch输出")
    print(f"    - compare_tflite.png: TFLite输出")
    print(f"    - compare_diff.png: 差异图像")


def main():
    parser = argparse.ArgumentParser(
        description='对比PyTorch和TFLite模型输出',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本对比
  python test_compare.py --pytorch ../../models/RealESRNet_x4plus.pth --tflite ../../models/RealESRNet_x4plus_339x510.tflite --input ../../test_data/input_510x339.png

  # 指定输入尺寸
  python test_compare.py --pytorch ../../models/RealESRNet_x4plus.pth --tflite ../../models/RealESRNet_x4plus_339x510.tflite --input ../../test_data/input_510x339.png --input_size 339 510

  # 保存对比图像
  python test_compare.py --pytorch ../../models/RealESRNet_x4plus.pth --tflite ../../models/RealESRNet_x4plus_339x510.tflite --input ../../test_data/input_510x339.png --save_images
        """
    )

    parser.add_argument('--pytorch', type=str, required=True,
                       help='PyTorch .pth模型路径')
    parser.add_argument('--tflite', type=str, required=True,
                       help='TFLite模型路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--input_size', type=int, nargs=2, default=None,
                       metavar=('HEIGHT', 'WIDTH'),
                       help='输入尺寸 (height width)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='PyTorch推理设备')
    parser.add_argument('--save_images', action='store_true',
                       help='保存对比图像')

    args = parser.parse_args()

    print("=" * 70)
    print("Real-ESRGAN PyTorch vs TFLite 对比测试")
    print("=" * 70)
    print(f"PyTorch模型: {args.pytorch}")
    print(f"TFLite模型:  {args.tflite}")
    print(f"输入图像:    {args.input}")
    print(f"设备:        {args.device}")
    print("=" * 70)

    # Check files
    if not os.path.exists(args.pytorch):
        print(f"❌ PyTorch模型不存在: {args.pytorch}")
        sys.exit(1)
    if not os.path.exists(args.tflite):
        print(f"❌ TFLite模型不存在: {args.tflite}")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"❌ 输入图像不存在: {args.input}")
        sys.exit(1)

    # Preprocess
    print("\n--> 预处理输入图像...")
    input_size = tuple(args.input_size) if args.input_size else None
    img_tensor = preprocess(args.input, input_size)
    print(f"    输入形状: {img_tensor.shape}")

    # Run inference
    pytorch_output = run_pytorch_inference(args.pytorch, img_tensor, args.device)
    tflite_output = run_tflite_inference(args.tflite, img_tensor)

    # Compare
    success = compare_outputs(pytorch_output, tflite_output)

    # Save images if requested
    if args.save_images:
        save_comparison_images(pytorch_output, tflite_output, './output_compare')

    print("\n" + "=" * 70)
    if success:
        print("✓ 对比测试通过！")
    else:
        print("⚠ 对比测试未通过，请检查模型转换")
    print("=" * 70)


if __name__ == '__main__':
    main()
