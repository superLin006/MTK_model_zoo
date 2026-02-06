#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2: 将TorchScript转换为TFLite
使用MTK Converter
"""

import argparse
import os
import json
import torch
import time
from pathlib import Path

try:
    import mtk_converter
    MTK_AVAILABLE = True
except ImportError:
    MTK_AVAILABLE = False
    print("❌ 错误: mtk_converter未安装")
    exit(1)


def convert_to_tflite(
    torchscript_path: str,
    output_dir: str = None,
    input_shape: list = None
):
    """
    将TorchScript转换为TFLite

    参数:
        torchscript_path: TorchScript .pt文件路径
        output_dir: 输出目录（默认与输入相同）
        input_shape: 输入形状 [1, 3, H, W]（从文件名自动推断）
    """
    if output_dir is None:
        output_dir = os.path.dirname(torchscript_path)

    print("="*70)
    print("步骤2: TorchScript -> TFLite")
    print("="*70)
    print(f"  输入: {torchscript_path}")
    print(f"  输出目录: {output_dir}")

    # 从文件名推断输入形状
    if input_shape is None:
        # 文件名格式: EDSR_x4_core_339x510.pt
        basename = os.path.basename(torchscript_path)
        try:
            # 提取 HxW
            size_part = basename.split('_')[-1].replace('.pt', '')  # 339x510
            h, w = map(int, size_part.split('x'))
            input_shape = [1, 3, h, w]
            print(f"  从文件名推断输入形状: {input_shape}")
        except:
            print("  ⚠ 无法从文件名推断输入形状，请使用--input_shape参数")
            return None

    print(f"  输入形状: {input_shape}")
    print("="*70)

    # 构建输出路径
    # EDSR_x4_core_339x510.pt -> EDSR_x4_339x510.tflite
    basename = os.path.basename(torchscript_path)
    basename = basename.replace('_core', '').replace('.pt', '.tflite')
    tflite_path = os.path.join(output_dir, basename)

    print(f"\n[1/2] 使用MTK Converter转换...")
    print("  这可能需要几秒到几分钟...")
    start = time.time()

    try:
        # 创建转换器
        converter = mtk_converter.PyTorchConverter.from_script_module_file(
            torchscript_path,
            input_shapes=[input_shape],
            input_types=[torch.float32],
        )

        # FP32精度
        converter.quantize = False

        # 转换
        print("  正在转换...")
        tflite_model = converter.convert_to_tflite()

        # 保存
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        tflite_size_mb = len(tflite_model) / 1024 / 1024
        elapsed = time.time() - start

        print(f"  ✓ 转换成功!")
        print(f"  输出: {os.path.basename(tflite_path)}")
        print(f"  大小: {tflite_size_mb:.1f} MB")
        print(f"  耗时: {elapsed:.1f}s")

    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 更新info.json
    print(f"\n[2/2] 更新模型信息...")
    info_path = torchscript_path.replace('_core_', '_').replace('.pt', '_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)

        info['files']['tflite'] = os.path.basename(tflite_path)
        info['data_type'] = 'float32'
        info['platform'] = 'MT8371 (MDLA 5.3)'

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"  ✓ 更新: {os.path.basename(info_path)}")
    else:
        print(f"  ⚠ 未找到info文件: {info_path}")

    print("\n" + "="*70)
    print("✓ TFLite转换完成!")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  TFLite: {os.path.basename(tflite_path)} ({tflite_size_mb:.1f} MB)")

    # 查找params文件
    params_path = torchscript_path.replace('_core_', '_').replace('.pt', '_meanshift_params.json')
    if os.path.exists(params_path):
        print(f"\n下一步:")
        print(f"  1. 测试TFLite: python test_tflite.py --tflite {tflite_path} --params {params_path} --input ../test_data/0853x4.png")
        print(f"  2. 转换为DLA:  python step3_tflite_to_dla.py --tflite {tflite_path} --platform MT8371")

    return tflite_path


def main():
    parser = argparse.ArgumentParser(
        description='步骤2: 将TorchScript转换为TFLite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step2_torchscript_to_tflite.py --torchscript ../models/EDSR_x4_core_256x256.pt
  python step2_torchscript_to_tflite.py --torchscript ../models/EDSR_x4_core_256x256.pt --input_shape 1 3 256 256
        """
    )

    parser.add_argument('--torchscript', type=str, required=True,
                       help='TorchScript .pt文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认与输入相同）')
    parser.add_argument('--input_shape', type=int, nargs=4, default=None,
                       help='输入形状 [B C H W]，例如: 1 3 339 510')

    args = parser.parse_args()

    if not MTK_AVAILABLE:
        return

    convert_to_tflite(
        torchscript_path=args.torchscript,
        output_dir=args.output_dir,
        input_shape=args.input_shape
    )


if __name__ == '__main__':
    main()
