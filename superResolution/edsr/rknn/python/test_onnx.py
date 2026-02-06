#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test ONNX EDSR model inference."""

import sys
import os
import numpy as np
from PIL import Image
import onnxruntime as ort


def preprocess(img_path):
    """Load and preprocess image to match PyTorch format."""
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)  # HWC, 0-255
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, axis=0)  # NCHW
    return arr, img.size


def postprocess(arr):
    """Convert numpy array to image."""
    arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    arr = arr.squeeze(0)  # Remove batch dimension
    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    return Image.fromarray(arr, mode="RGB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_onnx.py <onnx_model_path> [image_path] [pytorch_output_path]")
        print("Example: python test_onnx.py ../model/edsr_x4.onnx ../model/test_input.png ../model/test_pytorch_output.png")
        sys.exit(1)

    model_path = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else "../model/test_input.png"
    pytorch_output_path = sys.argv[3] if len(sys.argv) > 3 else "../model/test_pytorch_output.png"

    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    print(f"Loading image from {img_path}...")
    inp, orig_size = preprocess(img_path)
    print(f"Input shape: {inp.shape}, Original size: {orig_size}")

    print("Running ONNX inference...")
    outputs = session.run([output_name], {input_name: inp})
    out = outputs[0]

    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.2f}, {out.max():.2f}]")

    # Save ONNX result
    out_img = postprocess(out)
    output_path = "../model/test_onnx_output.png"
    out_img.save(output_path)
    print(f"ONNX output saved to {output_path}")
    print(f"Output size: {out_img.size}")

    # Compare with PyTorch output if available
    if os.path.exists(pytorch_output_path):
        print(f"\nComparing with PyTorch output ({pytorch_output_path})...")
        pytorch_img = Image.open(pytorch_output_path)
        pytorch_arr = np.array(pytorch_img).astype(np.float32)
        onnx_arr = np.array(out_img).astype(np.float32)

        if pytorch_arr.shape == onnx_arr.shape:
            diff = np.abs(pytorch_arr - onnx_arr)
            mae = diff.mean()
            max_diff = diff.max()
            mse = (diff ** 2).mean()
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

            print(f"  MAE (Mean Absolute Error): {mae:.4f}")
            print(f"  Max difference: {max_diff:.4f}")
            print(f"  MSE (Mean Squared Error): {mse:.4f}")
            print(f"  PSNR: {psnr:.2f} dB")

            if mae < 1.0:
                print("  ✓ ONNX output matches PyTorch output (MAE < 1.0)")
            elif mae < 5.0:
                print("  ⚠ ONNX output is close to PyTorch output (MAE < 5.0)")
            else:
                print("  ✗ ONNX output differs significantly from PyTorch output")
        else:
            print(f"  Shape mismatch: PyTorch {pytorch_arr.shape} vs ONNX {onnx_arr.shape}")
    else:
        print(f"\nPyTorch output not found at {pytorch_output_path}, skipping comparison")

    print("\nONNX inference test PASSED!")


if __name__ == "__main__":
    main()
