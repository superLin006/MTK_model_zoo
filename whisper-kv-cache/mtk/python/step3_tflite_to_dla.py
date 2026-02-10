#!/usr/bin/env python3
"""
Step 3: Compile TFLite models to DLA format

This script compiles TFLite models to MTK DLA (Deep Learning Accelerator) format
for MT8371 NPU deployment.

Platform: MT8371 (MDLA 5.3)
- arch: mdla5.3,edma3.6
- l1: 256 KB
- mdla: 1 core

Output:
- models/encoder_base_80x3000_MT8371.dla
- models/decoder_base_448_MT8371.dla
"""

import os
import subprocess


def compile_tflite_to_dla(tflite_path, dla_path, model_name):
    """Compile TFLite to DLA using ncc-tflite"""
    print("\n" + "="*70)
    print(f"Compiling {model_name}: TFLite → DLA")
    print("="*70)

    # SDK paths
    sdk_root = "/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
    ncc_tflite = os.path.join(sdk_root, "host/bin/ncc-tflite")

    # Set library path for ncc-tflite
    env = os.environ.copy()
    lib_path = os.path.join(sdk_root, "host/lib")
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
    else:
        env["LD_LIBRARY_PATH"] = lib_path

    # Platform configuration for MT8371
    platform = "mt8371"
    arch = "mdla5.3,edma3.6"
    l1_size = "256"  # KB
    mdla_cores = "1"

    # Compilation flags
    compile_flags = [
        "--relax-fp32",      # Relax FP32 precision constraints
        "--opt-accuracy",    # Optimize for accuracy
        "--opt-footprint",   # Optimize memory footprint
    ]

    print(f"\nInput: {tflite_path}")
    print(f"Output: {dla_path}")
    print(f"Platform: {platform}")
    print(f"  Architecture: {arch}")
    print(f"  L1 cache: {l1_size} KB")
    print(f"  MDLA cores: {mdla_cores}")
    print(f"Flags: {' '.join(compile_flags)}")

    # Build command
    cmd = [
        ncc_tflite,
        tflite_path,
        f"--arch={arch}",
        f"--l1={l1_size}",
        f"--mdla={mdla_cores}",
        *compile_flags,
        "-o", dla_path
    ]

    print("\nRunning compilation...")
    print(f"Command: {' '.join(cmd)}")

    # Execute compilation
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )

        print("\n--- Compilation Output ---")
        print(result.stdout)

        if result.stderr:
            print("\n--- Warnings/Errors ---")
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Compilation failed!")
        print(f"Exit code: {e.returncode}")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")
        raise

    # Verify output
    if os.path.exists(dla_path):
        file_size_mb = os.path.getsize(dla_path) / 1024 / 1024
        print(f"\n✓ DLA model saved: {dla_path} ({file_size_mb:.2f} MB)")
        return True
    else:
        print(f"\n✗ DLA file not generated: {dla_path}")
        return False


def main():
    print("="*70)
    print("Whisper KV Cache: TFLite → DLA Compilation")
    print("Target: MT8371 (MDLA 5.3)")
    print("="*70)

    models_dir = "models"
    success = True

    # Compile encoder
    try:
        compile_tflite_to_dla(
            os.path.join(models_dir, "encoder_base_80x3000_MT8371.tflite"),
            os.path.join(models_dir, "encoder_base_80x3000_MT8371.dla"),
            "Encoder"
        )
    except Exception as e:
        print(f"\nEncoder compilation failed: {e}")
        success = False

    # Compile decoder
    try:
        compile_tflite_to_dla(
            os.path.join(models_dir, "decoder_base_448_MT8371.tflite"),
            os.path.join(models_dir, "decoder_base_448_MT8371.dla"),
            "Decoder with KV Cache"
        )
    except Exception as e:
        print(f"\nDecoder compilation failed: {e}")
        success = False

    # Summary
    print("\n" + "="*70)
    if success:
        print("DLA Compilation Complete!")
    else:
        print("DLA Compilation Failed (see errors above)")
    print("="*70)

    if success:
        print("\nGenerated DLA models:")
        for filename in ["encoder_base_80x3000_MT8371.dla", "decoder_base_448_MT8371.dla"]:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                print(f"  ✓ {filepath} ({size_mb:.2f} MB)")

        print("\nAll model files:")
        print(f"  models/encoder_base_80x3000_MT8371.dla - Encoder for NPU")
        print(f"  models/decoder_base_448_MT8371.dla - Decoder with KV cache for NPU")
        print(f"  models/token_embedding.npy - Token embedding weights for C++")
        print(f"  models/embedding_info.json - Metadata")

        print("\nNext steps:")
        print("  1. Copy DLA models and embedding weights to target device")
        print("  2. Implement C++ inference with KV cache management")
        print("  3. Test on MT8371 hardware")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
