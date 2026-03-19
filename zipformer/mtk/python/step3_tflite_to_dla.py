"""
step3_tflite_to_dla.py

将 TFLite 模型编译为 DLA 格式（MTK MT8371, MDLA 5.3）。

输出:
    models/encoder.dla
    models/decoder_npu.dla
    models/joiner.dla
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

SDK_ROOT   = Path(
    "/home/xh/projects/MTK_models_zoo/0_Toolkits/"
    "neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
)
NCC_TFLITE = SDK_ROOT / "host" / "bin" / "ncc-tflite"
SDK_LIB    = SDK_ROOT / "host" / "lib"

# 设置 LD_LIBRARY_PATH
env = os.environ.copy()
existing_ld = env.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = f"{SDK_LIB}:{existing_ld}"

# ncc-tflite 编译参数
NCC_ARCH         = "mdla5.3,edma3.6"
NCC_L1_SIZE_KB   = "256"    # KB
NCC_NUM_MDLA     = "1"


def compile_dla(name: str, tflite_path: Path, dla_path: Path) -> bool:
    """编译 TFLite → DLA"""
    print(f"\n{'='*50}")
    print(f"Compiling {name}: {tflite_path.name} → {dla_path.name}")

    cmd = [
        str(NCC_TFLITE),
        f"--arch={NCC_ARCH}",
        f"--l1-size-kb={NCC_L1_SIZE_KB}",
        f"--num-mdla={NCC_NUM_MDLA}",
        "--relax-fp32",
        "--opt-accuracy",
        "--opt-footprint",
        "--show-memory-summary",
        str(tflite_path),
        "-d", str(dla_path),
    ]
    print(f"  CMD: {' '.join(cmd)}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        elapsed = time.time() - t0

        if result.stdout:
            print("  STDOUT:")
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
        if result.stderr:
            print("  STDERR:")
            for line in result.stderr.strip().split("\n"):
                # Filter out INFO lines
                if "ERROR" in line or "WARNING" in line or "FAIL" in line or not line.startswith("I"):
                    print(f"    {line}")

        if result.returncode != 0:
            print(f"  FAILED (returncode={result.returncode}, {elapsed:.1f}s)")
            # Show full stderr on failure
            if result.stderr:
                print("  Full STDERR:")
                print(result.stderr[:2000])
            return False
        else:
            if dla_path.exists():
                size_mb = dla_path.stat().st_size / 1e6
                print(f"  SUCCESS: {dla_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
            else:
                print(f"  WARNING: returncode=0 but .dla not found")
            return True

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 300s")
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


def main():
    print("=" * 60)
    print("Step 3: TFLite → DLA")
    print("=" * 60)
    print(f"  ncc-tflite: {NCC_TFLITE}")
    print(f"  arch: {NCC_ARCH}")

    # Verify ncc-tflite works
    test_cmd = subprocess.run(
        [str(NCC_TFLITE), "--version"],
        capture_output=True, text=True, env=env
    )
    print(f"  ncc-tflite version: {test_cmd.stdout.strip() or test_cmd.stderr.strip()}")

    results = {}

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    ok = compile_dla(
        "Encoder",
        MODELS_DIR / "encoder.tflite",
        MODELS_DIR / "encoder.dla",
    )
    results["encoder"] = "PASS" if ok else "FAIL"

    # -------------------------------------------------------------------------
    # DecoderNPU
    # -------------------------------------------------------------------------
    ok = compile_dla(
        "DecoderNPU",
        MODELS_DIR / "decoder_npu.tflite",
        MODELS_DIR / "decoder_npu.dla",
    )
    results["decoder_npu"] = "PASS" if ok else "FAIL"

    # -------------------------------------------------------------------------
    # Joiner
    # -------------------------------------------------------------------------
    ok = compile_dla(
        "Joiner",
        MODELS_DIR / "joiner.tflite",
        MODELS_DIR / "joiner.dla",
    )
    results["joiner"] = "PASS" if ok else "FAIL"

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3 Summary")
    print("=" * 60)
    all_ok = True
    for name, status in results.items():
        print(f"  {name}: {status}")
        if status != "PASS":
            all_ok = False

    print("\nGenerated DLA files:")
    for p in sorted(MODELS_DIR.glob("*.dla")):
        print(f"  {p.name}: {p.stat().st_size/1e6:.2f} MB")

    if not all_ok:
        print("\n[WARN] Some DLA compilations failed.")
    else:
        print("\n[PASS] All DLA compilations successful!")


if __name__ == "__main__":
    main()
