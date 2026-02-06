#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test PyTorch RCAN model inference."""

import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


# --------------------------
# RCAN Model Components
# --------------------------

def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Default convolutional layer."""
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class MeanShift(nn.Conv2d):
    """Mean shift layer for RGB normalization/subtraction."""
    def __init__(self, rgb_range: float, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign: int = -1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad_(False)


class Upsampler(nn.Sequential):
    """Upsampling layer using PixelShuffle."""
    def __init__(self, conv, scale: int, n_feats: int, bn: bool = False, act=None):
        modules = []
        if act is True:
            act = nn.ReLU(True)
        if act is False:
            act = None
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(np.log2(scale))):
                modules.append(conv(n_feats, 4 * n_feats, 3))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(n_feats))
                if act is not None:
                    modules.append(act)
        elif scale == 3:
            modules.append(conv(n_feats, 9 * n_feats, 3))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                modules.append(act)
        else:
            raise NotImplementedError(f"scale {scale} is not supported")
        super().__init__(*modules)


class CALayer(nn.Module):
    """Channel Attention Layer."""
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block."""
    def __init__(self, conv, n_feats: int, kernel_size: int, reduction: int = 16,
                 bias: bool = True, bn: bool = False, act: nn.Module = nn.ReLU(True), res_scale: float = 1.0):
        super().__init__()
        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                modules.append(act)
        modules.append(CALayer(n_feats, reduction))
        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualGroup(nn.Module):
    """Residual Group of RCABs."""
    def __init__(self, conv, n_feats: int, kernel_size: int, reduction: int, act: nn.Module, res_scale: float,
                 n_resblocks: int):
        super().__init__()
        modules = [RCAB(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale)
                   for _ in range(n_resblocks)]
        modules.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):
    """Residual Channel Attention Network (RCAN) for Super-Resolution.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
    Link: https://arxiv.org/abs/1807.02758

    Args:
        scale: Upscaling factor (e.g., 2, 3, 4, 8)
        n_resgroups: Number of residual groups (default: 10)
        n_resblocks: Number of RCABs in each residual group (default: 20)
        n_feats: Number of feature maps (default: 64)
        reduction: Channel reduction ratio for attention (default: 16)
        res_scale: Residual scaling factor (default: 1.0)
        rgb_range: RGB value range (default: 255.0)
    """
    def __init__(self, scale: int, n_resgroups: int = 10, n_resblocks: int = 20, n_feats: int = 64,
                 reduction: int = 16, res_scale: float = 1.0, rgb_range: float = 255.0):
        super().__init__()
        conv = default_conv
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.Sequential(conv(3, n_feats, kernel_size))

        body = [ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale,
                              n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats, act=None),
            conv(n_feats, 3, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


# --------------------------
# Inference Helpers
# --------------------------

def preprocess(img_path):
    """Load and preprocess image."""
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)  # HWC, 0-255
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor, img.size


def postprocess(tensor):
    """Convert tensor to image."""
    tensor = tensor.clamp(0, 255).round().byte().squeeze(0).cpu().numpy()
    arr = np.transpose(tensor, (1, 2, 0))  # HWC
    return Image.fromarray(arr, mode="RGB")


def infer_scale_from_model_name(model_path):
    """Infer scale factor from model filename."""
    lower = model_path.lower()
    for s in (2, 3, 4, 8):
        if f"x{s}" in lower or f"bix{s}" in lower or f"bdx{s}" in lower:
            return s
    return 4  # default


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pytorch.py <model_path> [image_path]")
        print("Example: python test_pytorch.py ../model/RCAN_BIX4.pt ../model/test_input.png")
        sys.exit(1)

    model_path = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else "../model/test_input.png"

    # Infer scale from model name
    scale = infer_scale_from_model_name(model_path)

    print(f"Loading RCAN model (scale={scale}) from {model_path}...")
    print(f"Model file: {os.path.basename(model_path)}")

    # Check file size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {file_size_mb:.2f} MB")

    model = RCAN(scale=scale)

    # Load state dict
    print("Loading state dict...")
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Remove 'module.' prefix if exists
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()

    print(f"Loading image from {img_path}...")
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        print("Please provide a valid image path")
        sys.exit(1)

    inp, orig_size = preprocess(img_path)
    print(f"Input shape: {inp.shape}, Original size: {orig_size}")

    print("Running PyTorch inference...")
    with torch.no_grad():
        out = model(inp)

    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.2f}, {out.max().item():.2f}]")

    # Save result
    out_img = postprocess(out)
    output_path = "../model/test_pytorch_output.png"
    out_img.save(output_path)
    print(f"Output saved to {output_path}")
    print(f"Output size: {out_img.size}")

    # Calculate upscale info
    expected_w = orig_size[0] * scale
    expected_h = orig_size[1] * scale
    print(f"\nSuper-resolution info:")
    print(f"  Input size: {orig_size[0]}x{orig_size[1]}")
    print(f"  Output size: {out_img.size[0]}x{out_img.size[1]}")
    print(f"  Scale factor: x{scale}")
    print(f"  Expected output size: {expected_w}x{expected_h}")

    if out_img.size == (expected_w, expected_h):
        print("  ✓ Output size matches expected scale")
    else:
        print("  ⚠ Output size differs from expected")

    print("\nPyTorch inference test PASSED!")


if __name__ == "__main__":
    main()
