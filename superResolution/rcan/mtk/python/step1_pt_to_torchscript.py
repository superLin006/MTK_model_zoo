#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1: 导出TorchScript核心模型
分离出MeanShift，准备后续转换
"""

import argparse
import os
import json
import torch
import time
from pathlib import Path

from rcan_model import (
    load_rcan_from_checkpoint,
    create_core_model_from_full,
    get_meanshift_params
)


def export_torchscript(
    checkpoint_path: str,
    output_dir: str,
    scale: int = 4,
    input_size: tuple = (339, 510)
):
    """
    导出TorchScript核心模型
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
    print("\n[1/5] 加载完整RCAN模型...")
    start = time.time()
    full_model = load_rcan_from_checkpoint(checkpoint_path, scale)
    full_model.eval()
    print(f"  ✓ 完成 ({time.time() - start:.1f}s)")

    # 2. 创建核心模型
    print("\n[2/5] 创建核心推理模型（去除MeanShift）...")
    start = time.time()
    core_model = create_core_model_from_full(full_model, scale)
    core_model.eval()
    print(f"  ✓ 完成 ({time.time() - start:.1f}s)")

    # 清理内存
    del full_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. 验证
    print("\n[3/5] 验证核心模型...")
    start = time.time()
    dummy_input = torch.randn(1, 3, h, w)

    with torch.no_grad():
        output = core_model(dummy_input)

    print(f"  输入: {dummy_input.shape}")
    print(f"  输出: {output.shape}")
    print(f"  ✓ 验证通过 ({time.time() - start:.1f}s)")

    # 4. 转换为TorchScript
    print("\n[4/5] 转换为TorchScript...")
    start = time.time()

    pt_path = os.path.join(output_dir, f"{model_name}_core_{h}x{w}.pt")
    traced_model = torch.jit.trace(core_model, dummy_input)
    traced_model.save(pt_path)

    pt_size_mb = os.path.getsize(pt_path) / 1024 / 1024
    print(f"  ✓ 保存: {os.path.basename(pt_path)}")
    print(f"  大小: {pt_size_mb:.1f} MB")
    print(f"  完成 ({time.time() - start:.1f}s)")

    # 5. 保存元数据
    print("\n[5/5] 保存元数据...")
    start = time.time()

    # MeanShift参数
    meanshift_params = get_meanshift_params()
    params_path = os.path.join(output_dir, f"{model_name}_meanshift_params.json")
    with open(params_path, 'w') as f:
        json.dump(meanshift_params, f, indent=2)
    print(f"  ✓ MeanShift参数: {os.path.basename(params_path)}")

    # 模型信息
    info = {
        'model_name': model_name,
        'scale': scale,
        'input_size': [h, w],
        'output_size': [h * scale, w * scale],
        'input_shape': [1, 3, h, w],
        'output_shape': [1, 3, h * scale, w * scale],
        'files': {
            'pytorch_checkpoint': os.path.basename(checkpoint_path),
            'pytorch_core': os.path.basename(pt_path),
            'meanshift_params': os.path.basename(params_path)
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
    print(f"  2. MeanShift参数: {os.path.basename(params_path)}")
    print(f"  3. 模型信息: {os.path.basename(info_path)}")

    print(f"\n下一步:")
    print(f"  python step2_torchscript_to_tflite.py --torchscript {pt_path}")

    return pt_path, params_path, info_path


def main():
    parser = argparse.ArgumentParser(description='步骤1: 导出TorchScript模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='原始.pt模型路径')
    parser.add_argument('--output_dir', type=str, default='../models',
                       help='输出目录')
    parser.add_argument('--scale', type=int, default=4,
                       help='超分倍数')
    parser.add_argument('--input_height', type=int, default=339,
                       help='输入高度')
    parser.add_argument('--input_width', type=int, default=510,
                       help='输入宽度')

    args = parser.parse_args()

    export_torchscript(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        scale=args.scale,
        input_size=(args.input_height, args.input_width)
    )


if __name__ == '__main__':
    main()
