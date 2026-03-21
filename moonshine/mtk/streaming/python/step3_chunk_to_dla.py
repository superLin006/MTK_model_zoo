#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤3 (Streaming): TFLite → DLA (仅 Encoder Chunk)

与 offline 版本的区别:
  - 只编译 Chunk Encoder (Decoder 直接复用 offline 的 moonshine_decoder.dla)
  - 输入: moonshine_encoder_chunk.tflite
  - 输出: moonshine_encoder_chunk.dla

目标平台: MT8371 (arch=mdla5.3,edma3.6, l1=256, mdla=1)

Decoder 复用:
  /home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/python/models/moonshine_decoder.dla
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# ========== 路径配置 ==========
NCC_TFLITE = "/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/host/bin/ncc-tflite"
SDK_LIB = "/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/host/lib"

PYTHON_DIR = Path(__file__).parent
MODELS_DIR = PYTHON_DIR / "models"
OFFLINE_MODELS = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/python/models")

# MT8371 编译参数 (与 offline 完全相同)
MT8371_ARGS = [
    "--arch=mdla5.3,edma3.6",
    "--l1-size-kb=256",
    "--num-mdla=1",
    "--relax-fp32",
    "--opt-accuracy",
    "--opt-footprint",
]


def compile_tflite_to_dla(tflite_path: Path, dla_path: Path, model_name: str) -> bool:
    """编译单个 TFLite → DLA"""
    print(f"\n{'='*60}")
    print(f"[{model_name}] TFLite → DLA")
    print(f"{'='*60}")
    print(f"  输入: {tflite_path}")
    print(f"  输出: {dla_path}")

    if not tflite_path.exists():
        print(f"  ERROR: TFLite 文件不存在: {tflite_path}")
        return False

    if not os.path.exists(NCC_TFLITE):
        print(f"  ERROR: ncc-tflite 不存在: {NCC_TFLITE}")
        return False

    # 构建命令
    cmd = [NCC_TFLITE, str(tflite_path)] + MT8371_ARGS + ["-o", str(dla_path)]
    print(f"\n  执行: {' '.join(cmd)}")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = SDK_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0

    # 打印编译输出
    if result.stdout:
        print(f"\n  编译输出:")
        for line in result.stdout.splitlines():
            if any(k in line for k in ["Error", "Warning", "error", "warning",
                                         "Compiling", "Optimizing", "Saving",
                                         "tensor", "operator", "success", "done"]):
                print(f"    {line}")

    if result.returncode != 0 or not dla_path.exists():
        print(f"\n  ERROR: 编译失败 (returncode={result.returncode})")
        if result.stderr:
            print(f"  stderr:\n{result.stderr[:2000]}")
        return False

    size_mb = os.path.getsize(dla_path) / 1024 / 1024
    print(f"\n  OK: 编译成功! 大小: {size_mb:.1f} MB, 耗时: {elapsed:.1f}s")
    return True


def main():
    print("=" * 70)
    print("步骤3 (Streaming): TFLite → DLA (MT8371)")
    print("=" * 70)
    print(f"  ncc-tflite: {NCC_TFLITE}")
    print(f"  平台参数: {' '.join(MT8371_ARGS)}")

    # Chunk Encoder
    encoder_tflite = MODELS_DIR / "moonshine_encoder_chunk.tflite"
    encoder_dla = MODELS_DIR / "moonshine_encoder_chunk.dla"

    enc_ok = compile_tflite_to_dla(encoder_tflite, encoder_dla, "Chunk Encoder")

    print("\n" + "=" * 70)
    print("步骤3 (Streaming) 汇总:")
    print("=" * 70)

    if enc_ok and encoder_dla.exists():
        size_mb = os.path.getsize(encoder_dla) / 1024 / 1024
        print(f"  OK: {encoder_dla.name} ({size_mb:.1f} MB)")
    else:
        print(f"  FAIL: Chunk Encoder DLA 生成失败")
        sys.exit(1)

    # 显示复用的 Decoder DLA
    decoder_dla = OFFLINE_MODELS / "moonshine_decoder.dla"
    if decoder_dla.exists():
        dec_mb = os.path.getsize(decoder_dla) / 1024 / 1024
        print(f"  OK: {decoder_dla} ({dec_mb:.1f} MB)  [复用 offline]")

    print("\n所有 DLA 文件就绪!")
    print("\n下一步: 开发 Streaming C++ 推理代码")


if __name__ == "__main__":
    main()
