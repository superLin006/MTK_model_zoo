#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2: 将Whisper TorchScript转换为TFLite
使用MTK Converter (直接转换，不经过ONNX)
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
    print("   请确保在MTK-whisper conda环境中运行")
    exit(1)


def convert_encoder_to_tflite(
    torchscript_path: str,
    output_dir: str,
    mel_frames: int = 3000
):
    """
    将Whisper Encoder转换为TFLite

    输入形状: [1, 80, 3000] (mel-spectrogram)
    输出形状: [1, 1500, 512] (encoder features)
    """
    print("="*70)
    print("步骤2.1: Encoder TorchScript -> TFLite")
    print("="*70)
    print(f"  输入: {torchscript_path}")
    print(f"  输出目录: {output_dir}")

    input_shape = [1, 80, mel_frames]
    print(f"  输入形状: {input_shape} (mel-spectrogram)")
    print("="*70)

    # 构建输出路径
    tflite_path = os.path.join(output_dir, f"encoder_base_80x{mel_frames}.tflite")

    print(f"\n使用MTK Converter转换...")
    start = time.time()

    try:
        # 创建转换器
        converter = mtk_converter.PyTorchConverter.from_script_module_file(
            torchscript_path,
            input_shapes=[input_shape],
            input_types=[torch.float32],
        )

        # FP32精度（不量化）
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

        return tflite_path

    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_decoder_to_tflite(
    torchscript_path: str,
    output_dir: str,
    max_seq_len: int = 448
):
    """
    将Whisper Decoder转换为TFLite

    输入1: token_embeddings [1, seq_len, 512]
    输入2: encoder_output [1, 1500, 512]
    输出: logits [1, seq_len, 51865]
    """
    print("\n" + "="*70)
    print("步骤2.2: Decoder TorchScript -> TFLite")
    print("="*70)
    print(f"  输入: {torchscript_path}")
    print(f"  输出目录: {output_dir}")

    # Decoder有两个输入
    input_shapes = [
        [1, max_seq_len, 512],  # token_embeddings
        [1, 1500, 512],         # encoder_output
    ]
    print(f"  输入1形状: {input_shapes[0]} (token_embeddings)")
    print(f"  输入2形状: {input_shapes[1]} (encoder_output)")
    print("="*70)

    # 构建输出路径
    tflite_path = os.path.join(output_dir, f"decoder_base_{max_seq_len}.tflite")

    print(f"\n使用MTK Converter转换...")
    start = time.time()

    try:
        # 创建转换器
        converter = mtk_converter.PyTorchConverter.from_script_module_file(
            torchscript_path,
            input_shapes=input_shapes,
            input_types=[torch.float32, torch.float32],
        )

        # FP32精度（不量化）
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

        return tflite_path

    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='步骤2: 将Whisper TorchScript转换为TFLite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step2_torchscript_to_tflite.py \\
      --encoder_pt ./models/encoder_base_3000.pt \\
      --decoder_pt ./models/decoder_base_448.pt \\
      --output_dir ./models
        """
    )

    parser.add_argument('--encoder_pt', type=str, required=True,
                       help='Encoder TorchScript文件路径')
    parser.add_argument('--decoder_pt', type=str, required=True,
                       help='Decoder TorchScript文件路径')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='输出目录')
    parser.add_argument('--mel_frames', type=int, default=3000,
                       help='Mel帧数（默认3000=30秒）')
    parser.add_argument('--max_seq_len', type=int, default=448,
                       help='Decoder最大序列长度')

    args = parser.parse_args()

    if not MTK_AVAILABLE:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 转换Encoder
    encoder_tflite = convert_encoder_to_tflite(
        args.encoder_pt,
        args.output_dir,
        args.mel_frames
    )

    if encoder_tflite is None:
        print("\n❌ Encoder转换失败，停止")
        return

    # 转换Decoder
    decoder_tflite = convert_decoder_to_tflite(
        args.decoder_pt,
        args.output_dir,
        args.max_seq_len
    )

    if decoder_tflite is None:
        print("\n❌ Decoder转换失败")
        return

    # 总结
    print("\n" + "="*70)
    print("✓ 所有TFLite转换完成!")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  Encoder TFLite: {os.path.basename(encoder_tflite)}")
    print(f"  Decoder TFLite: {os.path.basename(decoder_tflite)}")

    print(f"\n下一步:")
    print(f"  1. 测试TFLite: python test/test_tflite.py")
    print(f"  2. 转换为DLA:  python step3_tflite_to_dla.py")


if __name__ == '__main__':
    main()
