#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2: TorchScript → TFLite

分别转换 Encoder 和 Decoder

注意: Decoder 有 8 个输入, 需要正确列出所有 input_shapes 和 input_types

输出:
  models/moonshine_encoder.tflite
  models/moonshine_decoder.tflite
"""

import os
import sys
import time
import json
import torch
from pathlib import Path

try:
    import mtk_converter
except ImportError:
    print("ERROR: mtk_converter 未安装")
    sys.exit(1)

# ========== 路径配置 ==========
PYTHON_DIR = Path(__file__).parent
MODELS_DIR = PYTHON_DIR / "models"
TFLITE_DIR = PYTHON_DIR / "test" / "outputs" / "tflite"
TFLITE_DIR.mkdir(parents=True, exist_ok=True)

# ========== 固定形状参数 ==========
NUM_FRAMES = 2000    # 160000 // 80 (10s @ 16kHz)
T_ENC = 500          # encoder 输出帧数: ((2000//2)//2) = 500
MAX_DEC_LEN = 128    # decoder KV cache 最大长度 (10s音频token数更多)


def convert_encoder():
    """转换 Encoder TorchScript → TFLite"""
    print("\n" + "=" * 60)
    print("[Encoder] TorchScript → TFLite")
    print("=" * 60)

    encoder_pt = MODELS_DIR / "moonshine_encoder.pt"
    if not encoder_pt.exists():
        print(f"  ERROR: {encoder_pt} 不存在, 请先运行 step1_pt_to_torchscript.py")
        return None

    encoder_tflite = MODELS_DIR / "moonshine_encoder.tflite"

    print(f"  输入: {encoder_pt}")
    print(f"  输出: {encoder_tflite}")
    print(f"  输入 shape: [1, {NUM_FRAMES}, 80]")

    t0 = time.time()
    converter = mtk_converter.PyTorchConverter.from_script_module_file(
        str(encoder_pt),
        input_shapes=[[1, NUM_FRAMES, 80]],
        input_types=[torch.float32],
    )
    converter.quantize = False

    print("  转换中...")
    tflite_model = converter.convert_to_tflite()

    with open(encoder_tflite, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024 / 1024
    elapsed = time.time() - t0
    print(f"  完成! 大小: {size_mb:.1f} MB, 耗时: {elapsed:.1f}s")
    print(f"  输出: {encoder_tflite}")
    return encoder_tflite


def convert_decoder():
    """转换 Decoder TorchScript → TFLite"""
    print("\n" + "=" * 60)
    print("[Decoder] TorchScript → TFLite")
    print("=" * 60)

    decoder_pt = MODELS_DIR / "moonshine_decoder.pt"
    if not decoder_pt.exists():
        print(f"  ERROR: {decoder_pt} 不存在, 请先运行 step1_pt_to_torchscript.py")
        return None

    decoder_tflite = MODELS_DIR / "moonshine_decoder.tflite"

    print(f"  输入: {decoder_pt}")
    print(f"  输出: {decoder_tflite}")

    # Decoder 的 8 个输入
    # 0: decoder_embed:    [1, 1, 512]
    # 1: encoder_out:      [1, T_enc, 512]
    # 2: past_keys:        [10, 1, max_dec_len, 512]
    # 3: past_values:      [10, 1, max_dec_len, 512]
    # 4: cos_input:        [1, 1, 32]
    # 5: sin_input:        [1, 1, 32]
    # 6: attn_mask:        [1, 1, 1, max_dec_len+1]
    # 7: encoder_attn_mask:[1, 1, 1, T_enc]
    input_shapes = [
        [1, 1, 512],                            # decoder_embed
        [1, T_ENC, 512],                         # encoder_out
        [10, 1, MAX_DEC_LEN, 512],               # past_keys
        [10, 1, MAX_DEC_LEN, 512],               # past_values
        [1, 1, 32],                              # cos_input
        [1, 1, 32],                              # sin_input
        [1, 1, 1, MAX_DEC_LEN + 1],              # attn_mask
        [1, 1, 1, T_ENC],                        # encoder_attn_mask
    ]
    input_types = [torch.float32] * 8

    for i, (shape, dtype) in enumerate(zip(input_shapes, input_types)):
        print(f"  input[{i}]: {shape} ({dtype})")

    t0 = time.time()
    converter = mtk_converter.PyTorchConverter.from_script_module_file(
        str(decoder_pt),
        input_shapes=input_shapes,
        input_types=input_types,
    )
    converter.quantize = False

    print("  转换中...")
    tflite_model = converter.convert_to_tflite()

    with open(decoder_tflite, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024 / 1024
    elapsed = time.time() - t0
    print(f"  完成! 大小: {size_mb:.1f} MB, 耗时: {elapsed:.1f}s")
    print(f"  输出: {decoder_tflite}")
    return decoder_tflite


def main():
    print("=" * 70)
    print("步骤2: TorchScript → TFLite")
    print("=" * 70)

    encoder_tflite = convert_encoder()
    decoder_tflite = convert_decoder()

    print("\n" + "=" * 70)
    if encoder_tflite and decoder_tflite:
        enc_mb = os.path.getsize(encoder_tflite) / 1024 / 1024
        dec_mb = os.path.getsize(decoder_tflite) / 1024 / 1024
        print("步骤2 完成!")
        print(f"\n生成文件:")
        print(f"  1. {encoder_tflite} ({enc_mb:.1f} MB)")
        print(f"  2. {decoder_tflite} ({dec_mb:.1f} MB)")
        print(f"\n下一步:")
        print(f"  python step3_tflite_to_dla.py")
    else:
        print("步骤2 部分失败，请检查错误信息")
    print("=" * 70)


if __name__ == "__main__":
    main()
