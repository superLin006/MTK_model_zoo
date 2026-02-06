#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export PyTorch EDSR model to ONNX format (simplified version)."""

import sys
import os
import torch

# Import model definition from test_pytorch.py
from test_pytorch import EDSR


def export_onnx(pt_model_path, output_onnx_path, input_size=(256, 256)):
    """
    Export EDSR PyTorch model to ONNX (simplified).

    Args:
        pt_model_path: Path to .pt model file
        output_onnx_path: Path to save .onnx file
        input_size: Input image size (H, W), default 256x256
    """
    # Infer scale from model name
    scale = 4  # default
    if 'x2' in pt_model_path.lower():
        scale = 2
    elif 'x3' in pt_model_path.lower():
        scale = 3
    elif 'x4' in pt_model_path.lower():
        scale = 4

    print(f"Loading EDSR model (scale={scale})...")

    # Build and load model
    model = EDSR(scale=scale)
    state = torch.load(pt_model_path, map_location="cpu", weights_only=True)

    # Handle different state dict formats
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    # Remove 'module.' prefix if exists
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Create dummy input
    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w)

    # Export to ONNX with minimal options
    print(f"Exporting to {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )

    print(f"Done! Saved to {output_onnx_path}")
    return output_onnx_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <pt_model_path> [output_onnx_path] [input_height] [input_width]")
        print("Example: python export_onnx.py ../model/EDSR_x4.pt ../model/edsr_x4.onnx 256 256")
        sys.exit(1)

    pt_model_path = sys.argv[1]

    # Default output path
    if len(sys.argv) > 2:
        output_onnx_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(pt_model_path))[0]
        output_onnx_path = os.path.join(os.path.dirname(pt_model_path), f"{base_name}.onnx")

    # Input size
    input_h = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    input_w = int(sys.argv[4]) if len(sys.argv) > 4 else 256

    export_onnx(pt_model_path, output_onnx_path, (input_h, input_w))


if __name__ == "__main__":
    main()
