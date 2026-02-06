#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Real-ESRGAN ONNX model inference."""

import sys
import os
import numpy as np
import cv2
import argparse
import onnxruntime as ort


def preprocess(img_path, target_size=None):
    """Load and preprocess image for Real-ESRGAN.

    Args:
        img_path: Path to input image
        target_size: (height, width) to resize input. If None, use original size.

    Returns:
        img_array: Preprocessed image array (1, 3, H, W) in range [0, 1]
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

    # Add batch dimension
    img_array = np.expand_dims(img, axis=0)

    return img_array, original_size


def postprocess(output_array):
    """Convert model output array to image.

    Args:
        output_array: Model output (1, 3, H, W) in range [0, 1]

    Returns:
        output_img: Output image in BGR format (uint8)
    """
    # Remove batch dimension
    output = output_array.squeeze(0)

    # Clamp to [0, 1]
    output = np.clip(output, 0, 1)

    # Convert to [0, 255] range and uint8
    output = (output * 255.0).round().astype(np.uint8)

    # CHW to HWC
    output = np.transpose(output, (1, 2, 0))

    # Convert RGB to BGR for OpenCV
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output


def test_onnx_model(onnx_path, img_path, output_path, target_size=None):
    """Test ONNX Real-ESRGAN model inference.

    Args:
        onnx_path: Path to ONNX model file
        img_path: Path to input image
        output_path: Path to save output image
        target_size: (height, width) to resize input. If None, use model's expected size.
    """
    print("=" * 70)
    print("Real-ESRGAN ONNX Model Inference Test")
    print("=" * 70)
    print(f"Model:  {onnx_path}")
    print(f"Input:  {img_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    # Check files
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"ERROR: Input image not found: {img_path}")
        sys.exit(1)

    # Load ONNX model
    print("\n--> Loading ONNX model...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        print(f"✓ ONNX model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model: {e}")
        sys.exit(1)

    # Get model input/output info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    print(f"\nModel information:")
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output name: {output_name}")
    print(f"  Output shape: {output_shape}")

    # Infer scale factor
    scale = output_shape[2] // input_shape[2]
    print(f"  Scale factor: x{scale}")

    # Determine target size
    if target_size is None:
        # Use model's expected input size
        target_size = (input_shape[2], input_shape[3])  # (H, W)
        print(f"  Using model's expected input size: {target_size[1]}x{target_size[0]}")

    # Preprocess input
    print("\n--> Preprocessing input image...")
    img_array, original_size = preprocess(img_path, target_size)
    print(f"    Input array shape: {img_array.shape}")
    print(f"    Input value range: [{img_array.min():.2f}, {img_array.max():.2f}]")

    # Verify input shape matches model
    if list(img_array.shape) != input_shape:
        print(f"ERROR: Input shape mismatch!")
        print(f"  Expected: {input_shape}")
        print(f"  Got: {list(img_array.shape)}")
        sys.exit(1)

    # Run inference
    print("\n--> Running ONNX inference...")
    try:
        output_array = session.run([output_name], {input_name: img_array})[0]
        print(f"    Output array shape: {output_array.shape}")
        print(f"    Output value range: [{output_array.min():.2f}, {output_array.max():.2f}]")
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        sys.exit(1)

    # Postprocess output
    print("\n--> Postprocessing output...")
    output_img = postprocess(output_array)
    print(f"    Output image shape: {output_img.shape}")

    # Save output
    print(f"\n--> Saving output to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, output_img)

    # Print summary
    h_in, w_in = img_array.shape[2:]
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
    print("\n✓ ONNX inference test PASSED!")
    print("\nNext steps:")
    print(f"  1. Convert to RKNN: python convert.py {onnx_path} rk3576 fp output.rknn")


def main():
    parser = argparse.ArgumentParser(
        description='Test Real-ESRGAN ONNX model inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ONNX model with automatic input size detection
  python test_onnx.py --onnx_path ../model/RealESRNet_x4_510x339.onnx \\
                      --img_path ../model/input_510x339.png

  # Test with specific output path
  python test_onnx.py --onnx_path ../model/RealESRNet_x4_510x339.onnx \\
                      --img_path ../model/input_510x339.png \\
                      --output_path ../model/output_onnx.png

  # Test anime model
  python test_onnx.py --onnx_path ../model/RealESRGAN_anime_510x339.onnx \\
                      --img_path ../model/input_510x339.png
        """
    )
    parser.add_argument(
        '--onnx_path',
        type=str,
        required=True,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--img_path',
        type=str,
        default='../model/input_510x339.png',
        help='Path to input image'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output_onnx.png',
        help='Path to save output image'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Target input size (height width). If not specified, will use model\'s expected input size.'
    )

    args = parser.parse_args()

    # Convert input_size to tuple if provided
    target_size = tuple(args.input_size) if args.input_size else None

    # Run test
    test_onnx_model(
        args.onnx_path,
        args.img_path,
        args.output_path,
        target_size=target_size
    )


if __name__ == '__main__':
    main()
