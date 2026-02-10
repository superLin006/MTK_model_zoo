#!/usr/bin/env python3
"""
Step 2: Convert TorchScript models to TFLite

This script converts the TorchScript encoder and decoder to MTK TFLite format.

Note: MTK TFLite contains custom operators and cannot be tested in Python.
Accuracy has been validated in step 1 (test_pt.py).

Output:
- {models_dir}/encoder_{model}_80x3000_MT8371.tflite
- {models_dir}/decoder_{model}_448_MT8371.tflite
"""

import os
import sys
import argparse
import torch

# Add MTK converter path
sys.path.insert(0, '/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/host/lib/python')
import mtk_converter


def convert_encoder(model_name, models_dir, n_mels):
    print("\n" + "="*70)
    print("Converting Encoder: TorchScript → TFLite")
    print("="*70)

    torchscript_path = os.path.join(models_dir, f"encoder_{model_name}_{n_mels}x3000_MT8371.pt")
    tflite_path = os.path.join(models_dir, f"encoder_{model_name}_{n_mels}x3000_MT8371.tflite")

    # Encoder input: mel spectrogram [1, n_mels, 3000]
    input_shapes = [(1, n_mels, 3000)]
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


def convert_decoder(model_name, models_dir, d_model, n_layers):
    print("\n" + "="*70)
    print("Converting Decoder: TorchScript → TFLite")
    print("="*70)

    torchscript_path = os.path.join(models_dir, f"decoder_{model_name}_448_MT8371.pt")
    tflite_path = os.path.join(models_dir, f"decoder_{model_name}_448_MT8371.tflite")

    # Decoder inputs for single-token inference with KV cache
    # 1. token_embeddings: [1, 1, d_model]
    # 2. encoder_output: [1, 1500, d_model]
    # 3. past_self_keys: [n_layers, 1, 448, d_model]
    # 4. past_self_values: [n_layers, 1, 448, d_model]
    # 5. position_embed: [1, 1, d_model]
    # 6. self_attn_mask: [1, 1, 1, 449]
    # 7. cached_cross_keys: [n_layers, 1, 1500, d_model]
    # 8. cached_cross_values: [n_layers, 1, 1500, d_model]

    input_shapes = [
        (1, 1, d_model),            # token_embeddings
        (1, 1500, d_model),         # encoder_output
        (n_layers, 1, 448, d_model),  # past_self_keys
        (n_layers, 1, 448, d_model),  # past_self_values
        (1, 1, d_model),            # position_embed
        (1, 1, 1, 449),             # self_attn_mask
        (n_layers, 1, 1500, d_model), # cached_cross_keys
        (n_layers, 1, 1500, d_model), # cached_cross_values
    ]
    input_types = [torch.float32] * len(input_shapes)

    print(f"\nModel: {model_name}  d_model={d_model}  n_layers={n_layers}")
    print(f"Input: {torchscript_path}")
    print(f"Output: {tflite_path}")
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
    parser = argparse.ArgumentParser(description="Convert Whisper TorchScript models to TFLite")
    parser.add_argument("--model", default="base", help="Model name (e.g. base, large-v3-turbo)")
    parser.add_argument("--n-mels", type=int, default=80, help="Mel spectrogram channels (default: 80 for base, 128 for large-v3-turbo)")
    parser.add_argument("--d-model", type=int, default=512, help="Model hidden dimension (default: 512 for base)")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of decoder layers (default: 6 for base)")
    parser.add_argument("--models-dir", default="models", help="Directory with TorchScript models (default: models)")
    args = parser.parse_args()

    print("="*70)
    print(f"Whisper {args.model}: TorchScript → TFLite Conversion")
    print(f"  n_mels={args.n_mels}  d_model={args.d_model}  n_layers={args.n_layers}")
    print("="*70)

    # Convert encoder
    encoder_tflite = convert_encoder(args.model, args.models_dir, args.n_mels)

    # Convert decoder
    decoder_tflite = convert_decoder(args.model, args.models_dir, args.d_model, args.n_layers)

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
    print(f"\nNext step: Run step3_tflite_to_dla.py --model {args.model} --models-dir {args.models_dir}")


if __name__ == "__main__":
    main()
