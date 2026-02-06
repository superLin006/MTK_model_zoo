#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1: 导出TorchScript核心模型
Real-ESRGAN的模型输出在[0,1]范围，不需要额外的归一化处理
"""

import argparse
import os
import json
import torch
import time
from pathlib import Path

# 导入模型定义
from realesrgan_model import (
    load_realesrgan_from_checkpoint,
    RealESRGAN_x4plus,
    RealESRNet_x4plus,
    RealESRGAN_x4plus_anime_6B,
    RealESRGAN_x2plus
)


def export_torchscript(
    checkpoint_path: str,
    output_dir: str,
    scale: int = 4,
    input_size: tuple = (339, 510)
):
    """
    导出TorchScript核心模型

    Args:
        checkpoint_path: 原始.pth模型路径
        output_dir: 输出目录
        scale: 超分倍数
        input_size: 输入尺寸 (H, W)
    """
    os.makedirs(output_dir, exist_ok=True)

    h, w = input_size
    model_name = Path(checkpoint_path).stem

    print("="*70)
    print("步骤1: 导出TorchScript核心模型")
    print("="*70)
    print(f"  源模型:   {checkpoint_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  模型名称: {model_name}")
    print(f"  超分倍数: {scale}x")
    print(f"  输入尺寸: {h}x{w}")
    print("="*70)

    # 1. 加载完整模型
    print("\n[1/4] 加载Real-ESRGAN模型...")
    start = time.time()
    model, loaded_scale = load_realesrgan_from_checkpoint(checkpoint_path, 'cpu')
    model.eval()

    # 验证scale
    if scale != loaded_scale:
        print(f"  ⚠ 警告: 指定的scale({scale})与模型scale({loaded_scale})不匹配，使用模型scale")
        scale = loaded_scale

    print(f"  ✓ 完成 ({time.time() - start:.1f}s)")

    # 2. 验证模型
    print("\n[2/4] 验证模型...")
    start = time.time()
    dummy_input = torch.randn(1, 3, h, w)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  输入: {dummy_input.shape}")
    print(f"  输出: {output.shape}")
    print(f"  ✓ 验证通过 ({time.time() - start:.1f}s)")

    # 3. 转换为TorchScript
    print("\n[3/4] 转换为TorchScript...")
    start = time.time()

    pt_path = os.path.join(output_dir, f"{model_name}_core_{h}x{w}.pt")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(pt_path)

    pt_size_mb = os.path.getsize(pt_path) / 1024 / 1024
    print(f"  ✓ 保存: {os.path.basename(pt_path)}")
    print(f"  大小: {pt_size_mb:.1f} MB")
    print(f"  完成 ({time.time() - start:.1f}s)")

    # 4. 保存元数据
    print("\n[4/4] 保存元数据...")
    start = time.time()

    # 模型信息
    info = {
        'model_name': model_name,
        'scale': scale,
        'input_size': [h, w],
        'output_size': [h * scale, w * scale],
        'input_shape': [1, 3, h, w],
        'output_shape': [1, 3, h * scale, w * scale],
        'normalization': {
            'type': 'none',
            'description': 'Real-ESRGAN模型输入输出都在[0,1]范围，不需要额外的归一化'
        },
        'preprocessing': {
            'input_range': '[0, 1]',
            'description': '将图像从uint8 [0,255]转换为float32 [0,1]'
        },
        'postprocessing': {
            'output_range': '[0, 1]',
            'description': '将模型输出从[0,1]转换回uint8 [0,255]'
        },
        'files': {
            'pytorch_checkpoint': os.path.basename(checkpoint_path),
            'pytorch_core': os.path.basename(pt_path)
        }
    }

    info_path = os.path.join(output_dir, f"{model_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ 模型信息: {os.path.basename(info_path)}")
    print(f"  完成 ({time.time() - start:.1f}s)")

    print("\n" + "="*70)
    print("✓ TorchScript导出完成!")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  1. TorchScript: {os.path.basename(pt_path)} ({pt_size_mb:.1f} MB)")
    print(f"  2. 模型信息: {os.path.basename(info_path)}")

    print(f"\n下一步:")
    print(f"  python step2_torchscript_to_tflite.py --torchscript {pt_path}")

    return pt_path, info_path


def main():
    parser = argparse.ArgumentParser(description='步骤1: 导出TorchScript模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='原始.pth模型路径')
    parser.add_argument('--output_dir', type=str, default='../../models',
                       help='输出目录')
    parser.add_argument('--scale', type=int, default=None,
                       help='超分倍数 (如不指定将从模型自动推断)')
    parser.add_argument('--input_height', type=int, default=339,
                       help='输入高度')
    parser.add_argument('--input_width', type=int, default=510,
                       help='输入宽度')

    args = parser.parse_args()

    # 如果未指定scale，从checkpoint文件名推断
    if args.scale is None:
        filename = os.path.basename(args.checkpoint).lower()
        if 'x2' in filename:
            args.scale = 2
        else:
            args.scale = 4
        print(f"从文件名推断scale: x{args.scale}")

    export_torchscript(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        scale=args.scale,
        input_size=(args.input_height, args.input_width)
    )


if __name__ == '__main__':
    main()
