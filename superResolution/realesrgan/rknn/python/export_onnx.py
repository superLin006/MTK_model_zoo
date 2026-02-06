#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Real-ESRGAN PyTorch model to ONNX format."""

import sys
import os
import argparse
import torch
import onnx
from onnxsim import simplify

# Import model architecture
from realesrgan_arch import RRDBNet, RealESRNet_x4plus, RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B


def infer_model_type(model_path):
    """Infer model type from filename."""
    filename = os.path.basename(model_path).lower()

    if 'anime' in filename:
        print("Detected: RealESRGAN_x4plus_anime_6B")
        return 'anime', 4, 6
    elif 'esrnet' in filename:
        print("Detected: RealESRNet_x4plus (without GAN)")
        return 'esrnet', 4, 23
    elif 'x4' in filename or 'x4plus' in filename:
        print("Detected: RealESRGAN_x4plus")
        return 'esrgan', 4, 23
    elif 'x2' in filename:
        print("Detected: RealESRGAN_x2plus")
        return 'esrgan', 2, 23
    else:
        print("Default: RealESRNet_x4plus")
        return 'esrnet', 4, 23


def load_model(model_path, device='cpu'):
    """Load Real-ESRGAN model from checkpoint.

    Args:
        model_path: Path to .pth model file
        device: Device to load model on

    Returns:
        model: Loaded PyTorch model
        scale: Upscaling factor
    """
    print(f"Loading model from {model_path}...")

    # Infer model type
    model_type, scale, num_block = infer_model_type(model_path)

    # Create model
    if model_type == 'anime':
        model = RealESRGAN_x4plus_anime_6B()
    elif model_type == 'esrnet':
        model = RealESRNet_x4plus()
    else:  # esrgan
        model = RealESRGAN_x4plus()

    # Load state dict
    print(f"Loading state dict...")
    state_dict = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        # Check for common keys
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    # Remove 'module.' prefix if exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model = model.to(device)

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, scale


def export_onnx(model_path, output_path, input_size, opset_version=11, simplify_model=True):
    """Export Real-ESRGAN model to ONNX format.

    Args:
        model_path: Path to .pth model file
        output_path: Path to save ONNX model
        input_size: (height, width) for fixed input size
        opset_version: ONNX opset version
        simplify_model: Whether to simplify ONNX model
    """
    print("=" * 70)
    print("Real-ESRGAN PyTorch to ONNX Export")
    print("=" * 70)
    print(f"Model:  {model_path}")
    print(f"Output: {output_path}")
    print(f"Input size: {input_size[1]}x{input_size[0]} (WxH)")
    print(f"ONNX opset: {opset_version}")
    print("=" * 70)

    # Check model file
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    # Load model
    device = torch.device('cpu')
    model, scale = load_model(model_path, device)

    # Create dummy input
    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w, device=device)
    print(f"\n--> Creating dummy input: {dummy_input.shape}")

    # Test forward pass
    print(f"--> Testing forward pass...")
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"    Output shape: {dummy_output.shape}")
    print(f"    Expected scale: x{scale}")
    print(f"    Actual scale: x{dummy_output.shape[2] // dummy_input.shape[2]}")

    # Export to ONNX
    print(f"\n--> Exporting to ONNX...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # Fixed size for RKNN
        verbose=False
    )
    print(f"✓ ONNX model exported to: {output_path}")

    # Simplify ONNX model
    if simplify_model:
        print(f"\n--> Simplifying ONNX model...")
        try:
            onnx_model = onnx.load(output_path)
            print(f"    Original model: {len(onnx_model.graph.node)} nodes")

            model_simplified, check = simplify(onnx_model)

            if check:
                onnx.save(model_simplified, output_path)
                print(f"    Simplified model: {len(model_simplified.graph.node)} nodes")
                print(f"✓ Model simplified successfully")
            else:
                print(f"⚠ Warning: Simplification check failed, using original model")
        except Exception as e:
            print(f"⚠ Warning: Simplification failed: {e}")
            print(f"   Using original model")

    # Verify exported model
    print(f"\n--> Verifying ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")

        # Print model info
        print(f"\nModel information:")
        print(f"  Producer: {onnx_model.producer_name}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
        print(f"  Graph nodes: {len(onnx_model.graph.node)}")

        # Print input/output info
        for inp in onnx_model.graph.input:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            print(f"  Input '{inp.name}': {shape}")

        for out in onnx_model.graph.output:
            shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
            print(f"  Output '{out.name}': {shape}")

    except Exception as e:
        print(f"ERROR: ONNX model verification failed: {e}")
        sys.exit(1)

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("Export Completed Successfully!")
    print("-" * 70)
    print(f"  ONNX model saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Input shape: (1, 3, {h}, {w})")
    print(f"  Output shape: (1, 3, {h*scale}, {w*scale})")
    print(f"  Scale factor: x{scale}")
    print("=" * 70)
    print("\n✓ ONNX export PASSED!")
    print("\nNext steps:")
    print(f"  1. Test ONNX inference: python test_onnx.py --onnx_path {output_path}")
    print(f"  2. Convert to RKNN: python convert.py {output_path} rk3576 fp output.rknn")


def main():
    parser = argparse.ArgumentParser(
        description='Export Real-ESRGAN PyTorch model to ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with fixed input size 510x339 (recommended for RKNN)
  python export_onnx.py --model_path ../model/RealESRNet_x4plus.pth \\
                        --output_path ../model/RealESRNet_x4_510x339.onnx \\
                        --input_size 510 339

  # Export with input size 256x256
  python export_onnx.py --model_path ../model/RealESRNet_x4plus.pth \\
                        --output_path ../model/RealESRNet_x4_256x256.onnx \\
                        --input_size 256 256

  # Export anime model
  python export_onnx.py --model_path ../model/RealESRGAN_x4plus_anime_6B.pth \\
                        --output_path ../model/RealESRGAN_anime_510x339.onnx \\
                        --input_size 510 339
        """
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to PyTorch .pth model file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save ONNX model'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        required=True,
        metavar=('HEIGHT', 'WIDTH'),
        help='Fixed input size (height width). For RKNN, use fixed size like 510 339 or 256 256.'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=11,
        help='ONNX opset version (default: 11, recommended for RKNN)'
    )
    parser.add_argument(
        '--no_simplify',
        action='store_true',
        help='Skip ONNX model simplification'
    )

    args = parser.parse_args()

    # Export ONNX
    export_onnx(
        args.model_path,
        args.output_path,
        tuple(args.input_size),
        opset_version=args.opset_version,
        simplify_model=not args.no_simplify
    )


if __name__ == '__main__':
    main()
