#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Real-ESRGAN PyTorch model inference."""

import sys
import os
import numpy as np
from PIL import Image
import torch
import cv2
import argparse

# 添加父目录到路径以导入模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realesrgan_model import load_realesrgan_from_checkpoint


def preprocess(img_path, target_size=None):
    """Load and preprocess image for Real-ESRGAN.

    Args:
        img_path: Path to input image
        target_size: (height, width) to resize input. If None, use original size.

    Returns:
        img_tensor: Preprocessed image tensor (1, 3, H, W) in range [0, 1]
        original_size: Original image size (W, H)
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Store original size
    h_orig, w_orig = img.shape[:2]
    original_size = (w_orig, h_orig)
    print(f"原始图像尺寸: {w_orig}x{h_orig}")

    # Resize if specified
    if target_size is not None:
        h_target, w_target = target_size
        img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)
        print(f"调整到: {w_target}x{h_target}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] range (Real-ESRGAN expects [0, 1] input)
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension and convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    return img_tensor, original_size


def postprocess(output_tensor):
    """Convert model output tensor to image.

    Args:
        output_tensor: Model output (1, 3, H, W) in range [0, 1]

    Returns:
        output_img: Output image in BGR format (uint8)
    """
    # Remove batch dimension
    output = output_tensor.squeeze(0).cpu()

    # Clamp to [0, 1] and convert to numpy
    output = output.clamp(0, 1).numpy()

    # Convert to [0, 255] range and uint8
    output = (output * 255.0).round().astype(np.uint8)

    # CHW to HWC
    output = np.transpose(output, (1, 2, 0))

    # Convert RGB to BGR for OpenCV
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output


def test_pytorch_model(model_path, img_path, output_path, target_size=None, device='cpu'):
    """Test PyTorch Real-ESRGAN model inference.

    Args:
        model_path: Path to .pth model file
        img_path: Path to input image
        output_path: Path to save output image
        target_size: (height, width) to resize input. If None, use original size.
        device: Device to run inference on
    """
    print("=" * 70)
    print("Real-ESRGAN PyTorch模型推理测试")
    print("=" * 70)
    print(f"模型:  {model_path}")
    print(f"输入:  {img_path}")
    print(f"输出: {output_path}")
    print(f"设备: {device}")
    print("=" * 70)

    # Check files
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"错误: 输入图像不存在: {img_path}")
        sys.exit(1)

    # Load model
    device_obj = torch.device(device)
    model, scale = load_realesrgan_from_checkpoint(model_path, device_obj)

    # Preprocess input
    print("\n--> 预处理输入图像...")
    img_tensor, original_size = preprocess(img_path, target_size)
    img_tensor = img_tensor.to(device_obj)
    print(f"    输入张量形状: {img_tensor.shape}")
    print(f"    输入值范围: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

    # Run inference
    print("\n--> 运行推理...")
    with torch.no_grad():
        output_tensor = model(img_tensor)

    print(f"    输出张量形状: {output_tensor.shape}")
    print(f"    输出值范围: [{output_tensor.min():.2f}, {output_tensor.max():.2f}]")

    # Postprocess output
    print("\n--> 后处理输出...")
    output_img = postprocess(output_tensor)
    print(f"    输出图像形状: {output_img.shape}")

    # Save output
    print(f"\n--> 保存输出到 {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, output_img)

    # Print summary
    h_in, w_in = img_tensor.shape[2:]
    h_out, w_out = output_img.shape[:2]
    actual_scale = h_out // h_in

    print("\n" + "=" * 70)
    print("推理成功完成!")
    print("-" * 70)
    print(f"  输入尺寸:     {w_in}x{h_in}")
    print(f"  输出尺寸:    {w_out}x{h_out}")
    print(f"  预期倍数: x{scale}")
    print(f"  实际倍数:   x{actual_scale}")
    if actual_scale == scale:
        print(f"  ✓ 输出倍数匹配")
    else:
        print(f"  ⚠ 警告: 倍数不匹配!")
    print(f"  输出文件:    {output_path}")
    print("=" * 70)
    print("\n✓ PyTorch推理测试通过!")


def main():
    parser = argparse.ArgumentParser(
        description='Test Real-ESRGAN PyTorch model inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置测试
  python test_pytorch.py --model_path ../../models/RealESRNet_x4plus.pth --img_path ../../test_data/input_510x339.png

  # 使用自定义输入尺寸测试 (MTK NPU使用固定尺寸如510x339)
  python test_pytorch.py --model_path ../../models/RealESRNet_x4plus.pth --img_path ../../test_data/input_510x339.png --input_size 339 510

  # 测试动漫模型
  python test_pytorch.py --model_path ../../models/RealESRGAN_x4plus_anime_6B.pth --img_path ../../test_data/input_510x339.png
        """
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='PyTorch .pth模型文件路径'
    )
    parser.add_argument(
        '--img_path',
        type=str,
        default='../../test_data/input_510x339.png',
        help='输入图像路径'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output_pytorch.png',
        help='输出图像保存路径'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='目标输入尺寸 (height width)。如不指定将使用原始图像尺寸。MTK转换时使用固定尺寸如 339 510。'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='运行推理的设备'
    )

    args = parser.parse_args()

    # Convert input_size to tuple if provided
    target_size = tuple(args.input_size) if args.input_size else None

    # Run test
    test_pytorch_model(
        args.model_path,
        args.img_path,
        args.output_path,
        target_size=target_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
