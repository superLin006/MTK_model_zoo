#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Real-ESRGAN PyTorch model inference."""

import sys
import os
import numpy as np
from PIL import Image
import torch
import cv2
import argparse

# Import model architecture
from realesrgan_model import RRDBNet, RealESRNet_x4plus, RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B


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
    print(f"  Scale factor: x{scale}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, scale


def preprocess(img_path, target_size=None):
    """Load and preprocess image for Real-ESRGAN.

    Args:
        img_path: Path to input image
        target_size: (height, width) to resize input. If None, use original size.

    Returns:
        img_tensor: Preprocessed image tensor (1, 3, H, W) in range [0, 1]
        original_size: Original image size (W, H)
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Store original size
    h_orig, w_orig = img.shape[:2]
    original_size = (w_orig, h_orig)
    print(f"Original image size: {w_orig}x{h_orig}")

    # Resize if specified
    if target_size is not None:
        h_target, w_target = target_size
        img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)
        print(f"Resized to: {w_target}x{h_target}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] range (Real-ESRGAN expects [0, 1] input)
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension and convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    return img_tensor, original_size


def postprocess(output_tensor):
    """Convert model output tensor to image.

    Args:
        output_tensor: Model output (1, 3, H, W) in range [0, 1]

    Returns:
        output_img: Output image in BGR format (uint8)
    """
    # Remove batch dimension
    output = output_tensor.squeeze(0).cpu()

    # Clamp to [0, 1] and convert to numpy
    output = output.clamp(0, 1).numpy()

    # Convert to [0, 255] range and uint8
    output = (output * 255.0).round().astype(np.uint8)

    # CHW to HWC
    output = np.transpose(output, (1, 2, 0))

    # Convert RGB to BGR for OpenCV
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output


def test_pytorch_model(model_path, img_path, output_path, target_size=None, device='cpu'):
    """Test PyTorch Real-ESRGAN model inference.

    Args:
        model_path: Path to .pth model file
        img_path: Path to input image
        output_path: Path to save output image
        target_size: (height, width) to resize input. If None, use original size.
        device: Device to run inference on
    """
    print("=" * 70)
    print("Real-ESRGAN PyTorch Model Inference Test")
    print("=" * 70)
    print(f"Model:  {model_path}")
    print(f"Input:  {img_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print("=" * 70)

    # Check files
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"ERROR: Input image not found: {img_path}")
        sys.exit(1)

    # Load model
    device_obj = torch.device(device)
    model, scale = load_model(model_path, device_obj)

    # Preprocess input
    print("\n--> Preprocessing input image...")
    img_tensor, original_size = preprocess(img_path, target_size)
    img_tensor = img_tensor.to(device_obj)
    print(f"    Input tensor shape: {img_tensor.shape}")
    print(f"    Input value range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

    # Run inference
    print("\n--> Running inference...")
    with torch.no_grad():
        output_tensor = model(img_tensor)

    print(f"    Output tensor shape: {output_tensor.shape}")
    print(f"    Output value range: [{output_tensor.min():.2f}, {output_tensor.max():.2f}]")

    # Postprocess output
    print("\n--> Postprocessing output...")
    output_img = postprocess(output_tensor)
    print(f"    Output image shape: {output_img.shape}")

    # Save output
    print(f"\n--> Saving output to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, output_img)

    # Print summary
    h_in, w_in = img_tensor.shape[2:]
    h_out, w_out = output_img.shape[:2]
    actual_scale = h_out // h_in

    print("\n" + "=" * 70)
    print("Inference Completed Successfully!")
    print("-" * 70)
    print(f"  Input size:     {w_in}x{h_in}")
    print(f"  Output size:    {w_out}x{h_out}")
    print(f"  Expected scale: x{scale}")
    print(f"  Actual scale:   x{actual_scale}")
    if actual_scale == scale:
        print(f"  ✓ Output scale matches expected")
    else:
        print(f"  ⚠ Warning: Scale mismatch!")
    print(f"  Output file:    {output_path}")
    print("=" * 70)
    print("\n✓ PyTorch inference test PASSED!")


def main():
    parser = argparse.ArgumentParser(
        description='Test Real-ESRGAN PyTorch model inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings
  python test_pytorch.py --model_path ../model/RealESRNet_x4plus.pth --img_path ../model/input_256x256.png

  # Test with custom input size (for RKNN, use fixed size like 510x339)
  python test_pytorch.py --model_path ../model/RealESRNet_x4plus.pth --img_path ../model/input_256x256.png --input_size 510 339

  # Test anime model
  python test_pytorch.py --model_path ../model/RealESRGAN_x4plus_anime_6B.pth --img_path ../model/input_256x256.png
        """
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to PyTorch .pth model file'
    )
    parser.add_argument(
        '--img_path',
        type=str,
        default='../model/input_256x256.png',
        help='Path to input image'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output_pytorch.png',
        help='Path to save output image'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Target input size (height width). If not specified, will use original image size. For RKNN conversion, use fixed size like 510 339.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on'
    )

    args = parser.parse_args()

    # Convert input_size to tuple if provided
    target_size = tuple(args.input_size) if args.input_size else None

    # Run test
    test_pytorch_model(
        args.model_path,
        args.img_path,
        args.output_path,
        target_size=target_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
