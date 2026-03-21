#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤3: TFLite → DLA

分别编译 Encoder 和 Decoder

目标平台: MT8371 (arch=mdla5.3,edma3.6, l1=256, mdla=1)

输出:
  models/moonshine_encoder.dla
  models/moonshine_decoder.dla
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

# MT8371 编译参数
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
            # 只显示关键行
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
    print("步骤3: TFLite → DLA (MT8371)")
    print("=" * 70)
    print(f"  ncc-tflite: {NCC_TFLITE}")
    print(f"  平台参数: {' '.join(MT8371_ARGS)}")

    encoder_tflite = MODELS_DIR / "moonshine_encoder.tflite"
    decoder_tflite = MODELS_DIR / "moonshine_decoder.tflite"
    encoder_dla = MODELS_DIR / "moonshine_encoder.dla"
    decoder_dla = MODELS_DIR / "moonshine_decoder.dla"

    enc_ok = compile_tflite_to_dla(encoder_tflite, encoder_dla, "Encoder")
    dec_ok = compile_tflite_to_dla(decoder_tflite, decoder_dla, "Decoder")

    print("\n" + "=" * 70)
    print("步骤3 汇总:")
    print("=" * 70)

    all_ok = True
    for ok, name, path in [(enc_ok, "Encoder", encoder_dla), (dec_ok, "Decoder", decoder_dla)]:
        if ok and path.exists():
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  OK: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  FAIL: {name} DLA 生成失败")
            all_ok = False

    if all_ok:
        print("\n所有 DLA 文件生成成功!")
        print("\n下一步: 开发 C++ 推理代码")
    else:
        print("\n部分 DLA 文件生成失败, 请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
