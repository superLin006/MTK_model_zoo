"""
step2_torchscript_to_tflite.py

将 TorchScript 模型转换为 TFLite 格式 (float32, 非量化)。
使用 mtk_converter.PyTorchConverter。

输出:
    models/encoder.tflite
    models/decoder_npu.tflite
    models/joiner.tflite
"""

import sys
import time
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

SDK_PATH = Path(
    "/home/xh/projects/MTK_models_zoo/0_Toolkits/"
    "neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
)

# mtk_converter 应已安装在 conda env 中
import mtk_converter

# ---------------------------------------------------------------------------
# 固定输入形状
# ---------------------------------------------------------------------------
SEGMENT           = 103
BATCH_SIZE        = 1
LEFT_CONTEXT_LEN  = 128    # 32 * 4
DS_FACTORS        = (1, 2, 4, 8, 2)
ENCODER_DIM       = 256
ATTN_DIM          = 192
CONTEXT_SIZE      = 2
DECODER_DIM       = 512

def get_encoder_input_shapes():
    """返回 encoder 的所有输入形状列表 (NOTE: x_lens removed)"""
    shapes = []
    # x: [1, 103, 80]
    shapes.append([BATCH_SIZE, SEGMENT, 80])
    # x_lens removed — SEGMENT is fixed at 103

    # 5 sets of states
    for i in range(5):
        ds  = DS_FACTORS[i]
        lc  = LEFT_CONTEXT_LEN // ds

        # cached_len: [2, 1]  float32
        shapes.append([2, 1])
    for i in range(5):
        # cached_avg: [2, 1, 256]
        shapes.append([2, 1, ENCODER_DIM])
    for i in range(5):
        ds  = DS_FACTORS[i]
        lc  = LEFT_CONTEXT_LEN // ds
        # cached_key: [2, lc, 1, 192]
        shapes.append([2, lc, 1, ATTN_DIM])
    for i in range(5):
        ds  = DS_FACTORS[i]
        lc  = LEFT_CONTEXT_LEN // ds
        # cached_val: [2, lc, 1, 96]
        shapes.append([2, lc, 1, ATTN_DIM // 2])
    for i in range(5):
        ds  = DS_FACTORS[i]
        lc  = LEFT_CONTEXT_LEN // ds
        # cached_val2: [2, lc, 1, 96]
        shapes.append([2, lc, 1, ATTN_DIM // 2])
    for i in range(5):
        # cached_conv1: [2, 1, 256, 30]
        shapes.append([2, 1, ENCODER_DIM, 30])
    for i in range(5):
        # cached_conv2: [2, 1, 256, 30]
        shapes.append([2, 1, ENCODER_DIM, 30])

    return shapes

def get_encoder_input_types():
    """返回 encoder 的所有输入类型列表 (NOTE: x_lens removed)"""
    types = []
    # x: float32
    types.append(torch.float32)
    # x_lens removed
    # 35 states: all float32
    for _ in range(35):
        types.append(torch.float32)
    return types


# ===========================================================================
# 转换函数
# ===========================================================================
def convert_model(name: str, pt_path: Path, tflite_path: Path,
                  input_shapes: list, input_types: list = None):
    print(f"\n{'='*50}")
    print(f"Converting {name}: {pt_path.name} → {tflite_path.name}")
    print(f"  Input shapes: {input_shapes}")
    print(f"  Input types:  {[str(t) if t else 'float32' for t in (input_types or [])]}")

    t0 = time.time()
    try:
        converter = mtk_converter.PyTorchConverter.from_script_module_file(
            str(pt_path),
            input_shapes=input_shapes,
            input_types=input_types,
        )

        # float32, no quantization
        converter.quantize = False

        # Enable decomposition of ops that MDLA doesn't support
        converter.decompose_cumsum_ops         = True
        converter.decompose_gather_elements_ops = True

        print("  Running conversion...")
        tflite_model = converter.convert_to_tflite()

        with open(str(tflite_path), "wb") as f:
            f.write(tflite_model)

        size_mb = tflite_path.stat().st_size / 1e6
        elapsed = time.time() - t0
        print(f"  SUCCESS: {tflite_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===========================================================================
# main
# ===========================================================================
def main():
    print("=" * 60)
    print("Step 2: TorchScript → TFLite")
    print("=" * 60)

    results = {}

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    enc_shapes = get_encoder_input_shapes()
    enc_types  = get_encoder_input_types()

    print(f"\nEncoder input count: {len(enc_shapes)}")
    for i, (s, t) in enumerate(zip(enc_shapes, enc_types)):
        print(f"  [{i:2d}] shape={s}, type={t}")

    ok = convert_model(
        "Encoder",
        MODELS_DIR / "encoder.pt",
        MODELS_DIR / "encoder.tflite",
        enc_shapes,
        enc_types,
    )
    results["encoder"] = "PASS" if ok else "FAIL"

    # -------------------------------------------------------------------------
    # DecoderNPU
    # -------------------------------------------------------------------------
    ok = convert_model(
        "DecoderNPU",
        MODELS_DIR / "decoder_npu.pt",
        MODELS_DIR / "decoder_npu.tflite",
        input_shapes=[[BATCH_SIZE, CONTEXT_SIZE, DECODER_DIM]],
        input_types=[torch.float32],
    )
    results["decoder_npu"] = "PASS" if ok else "FAIL"

    # -------------------------------------------------------------------------
    # Joiner
    # -------------------------------------------------------------------------
    ok = convert_model(
        "Joiner",
        MODELS_DIR / "joiner.pt",
        MODELS_DIR / "joiner.tflite",
        input_shapes=[[BATCH_SIZE, ENCODER_DIM], [BATCH_SIZE, DECODER_DIM]],
        input_types=[torch.float32, torch.float32],
    )
    results["joiner"] = "PASS" if ok else "FAIL"

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2 Summary")
    print("=" * 60)
    all_ok = True
    for name, status in results.items():
        print(f"  {name}: {status}")
        if status != "PASS":
            all_ok = False

    print("\nGenerated TFLite files:")
    for p in sorted(MODELS_DIR.glob("*.tflite")):
        print(f"  {p.name}: {p.stat().st_size/1e6:.2f} MB")

    if not all_ok:
        print("\n[FAIL] Some conversions failed!")
        sys.exit(1)
    else:
        print("\n[PASS] All TFLite conversions successful!")


if __name__ == "__main__":
    main()
