#!/usr/bin/env python3
"""
Step 2: Convert TorchScript models to TFLite

This script converts the TorchScript encoder and decoder to MTK TFLite format.

Note: MTK TFLite contains custom operators and cannot be tested in Python.
Accuracy has been validated in step 1 (test_pt.py).

Output:
- models/encoder.tflite
- models/decoder_kv.tflite
"""

import os
import sys
import torch

# Add MTK converter path
sys.path.insert(0, '/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/host/lib/python')
import mtk_converter


def convert_encoder():
    print("\n" + "="*70)
    print("Converting Encoder: TorchScript → TFLite")
    print("="*70)

    torchscript_path = "models/encoder.pt"
    tflite_path = "models/encoder.tflite"

    # Encoder input: mel spectrogram [1, 80, 3000]
    input_shapes = [(1, 80, 3000)]
    input_types = [torch.float32]

    print(f"\nInput: {torchscript_path}")
    print(f"Output: {tflite_path}")
    print(f"Input shapes: {input_shapes}")

    # Create converter
    converter = mtk_converter.PyTorchConverter.from_script_module_file(
        torchscript_path,
        input_shapes=input_shapes,
        input_types=input_types
    )

    # Disable quantization for FP32 model
    converter.quantize = False

    print("\nConverting...")
    tflite_model = converter.convert_to_tflite()

    # Save TFLite model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"✓ Encoder TFLite saved: {tflite_path} ({file_size_mb:.2f} MB)")

    return tflite_path


def convert_decoder():
    print("\n" + "="*70)
    print("Converting Decoder: TorchScript → TFLite")
    print("="*70)

    torchscript_path = "models/decoder_kv.pt"
    tflite_path = "models/decoder_kv.tflite"

    # Decoder inputs for single-token inference with KV cache
    # Based on the decoder forward signature:
    # 1. token_embeddings: [1, 1, 512]
    # 2. encoder_output: [1, 1500, 512]
    # 3. past_self_keys: [6, 1, 448, 512]
    # 4. past_self_values: [6, 1, 448, 512]
    # 5. position_embed: [1, 1, 512]
    # 6. self_attn_mask: [1, 1, 1, 449]
    # 7. cached_cross_keys: [6, 1, 1500, 512]
    # 8. cached_cross_values: [6, 1, 1500, 512]

    input_shapes = [
        (1, 1, 512),       # token_embeddings
        (1, 1500, 512),    # encoder_output
        (6, 1, 448, 512),  # past_self_keys
        (6, 1, 448, 512),  # past_self_values
        (1, 1, 512),       # position_embed
        (1, 1, 1, 449),    # self_attn_mask
        (6, 1, 1500, 512), # cached_cross_keys
        (6, 1, 1500, 512), # cached_cross_values
    ]
    input_types = [torch.float32] * len(input_shapes)

    print(f"\nInput: {torchscript_path}")
    print(f"Output: {tflite_path}")
    print(f"Number of inputs: {len(input_shapes)}")
    print(f"Input shapes:")
    for i, shape in enumerate(input_shapes):
        print(f"  [{i}] {shape}")

    # Create converter
    print("\nCreating converter...")
    converter = mtk_converter.PyTorchConverter.from_script_module_file(
        torchscript_path,
        input_shapes=input_shapes,
        input_types=input_types
    )

    # Disable quantization for FP32 model
    converter.quantize = False

    print("Converting (this may take a while)...")
    tflite_model = converter.convert_to_tflite()

    # Save TFLite model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"✓ Decoder TFLite saved: {tflite_path} ({file_size_mb:.2f} MB)")

    return tflite_path


def main():
    print("="*70)
    print("Whisper KV Cache: TorchScript → TFLite Conversion")
    print("="*70)

    # Convert encoder
    encoder_tflite = convert_encoder()

    # Convert decoder
    decoder_tflite = convert_decoder()

    # Summary
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print("\nGenerated files:")
    for filepath in [encoder_tflite, decoder_tflite]:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✓ {filepath} ({size_mb:.2f} MB)")

    print("\nNote: MTK TFLite models contain custom operators (e.g., MTKEXT_LAYER_NORMALIZATION)")
    print("and cannot be tested in Python. Accuracy was validated in test_pt.py.")
    print("\nNext step: Run step3_tflite_to_dla.py to compile for MT8371 NPU")


if __name__ == "__main__":
    main()
