#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2 (Streaming): TorchScript → TFLite (仅 Encoder Chunk)

与 offline 版本的区别:
  - 只转换 Chunk Encoder (Decoder 直接复用 offline 的 moonshine_decoder.tflite)
  - input_shapes=[[1, 160, 80]]  (chunk: 0.8s)
  - 输入: streaming/python/models/moonshine_encoder_chunk.pt
  - 输出: streaming/python/models/moonshine_encoder_chunk.tflite

Decoder 复用:
  /home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/python/models/moonshine_decoder.tflite
"""

import os
import sys
import time
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
OFFLINE_MODELS = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/python/models")

# ========== Chunk 参数 ==========
CHUNK_FRAMES = 160    # 0.8s chunk
CHUNK_T_ENC = 40      # ((160//2)//2) = 40


def convert_chunk_encoder():
    """转换 Chunk Encoder TorchScript → TFLite"""
    print("\n" + "=" * 60)
    print("[Chunk Encoder] TorchScript → TFLite")
    print(f"  input_shape: [1, {CHUNK_FRAMES}, 80]")
    print("=" * 60)

    encoder_pt = MODELS_DIR / "moonshine_encoder_chunk.pt"
    if not encoder_pt.exists():
        print(f"  ERROR: {encoder_pt} 不存在, 请先运行 step1_chunk_to_torchscript.py")
        return None

    encoder_tflite = MODELS_DIR / "moonshine_encoder_chunk.tflite"

    print(f"  输入: {encoder_pt}")
    print(f"  输出: {encoder_tflite}")

    t0 = time.time()
    converter = mtk_converter.PyTorchConverter.from_script_module_file(
        str(encoder_pt),
        input_shapes=[[1, CHUNK_FRAMES, 80]],
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


def main():
    print("=" * 70)
    print("步骤2 (Streaming): TorchScript → TFLite")
    print("=" * 70)

    encoder_tflite = convert_chunk_encoder()

    print("\n" + "=" * 70)
    if encoder_tflite:
        enc_mb = os.path.getsize(encoder_tflite) / 1024 / 1024
        decoder_tflite = OFFLINE_MODELS / "moonshine_decoder.tflite"
        dec_mb = os.path.getsize(decoder_tflite) / 1024 / 1024 if decoder_tflite.exists() else 0
        print("步骤2 (Streaming) 完成!")
        print(f"\n生成/复用文件:")
        print(f"  1. {encoder_tflite} ({enc_mb:.1f} MB)  [新生成]")
        print(f"  2. {decoder_tflite} ({dec_mb:.1f} MB)  [复用 offline]")
        print(f"\n下一步:")
        print(f"  python step3_chunk_to_dla.py")
    else:
        print("步骤2 (Streaming) 失败，请检查错误信息")
    print("=" * 70)


if __name__ == "__main__":
    main()
