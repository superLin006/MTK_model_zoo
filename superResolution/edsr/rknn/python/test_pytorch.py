#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test PyTorch EDSR model inference."""

import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Model components from MTKSuperResolution
def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range: float, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign: int = -1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad_(False)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats: int, kernel_size: int, bias: bool = True,
                 bn: bool = False, act=None, res_scale: float = 1.0):
        super().__init__()
        if act is None:
            act = nn.ReLU(True)
        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                modules.append(act)
        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
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


class EDSR(nn.Module):
    def __init__(self, scale: int, n_resblocks: int = 32, n_feats: int = 256,
                 res_scale: float = 0.1, rgb_range: float = 255.0):
        super().__init__()
        conv = default_conv
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(conv(3, n_feats, kernel_size))

        body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblocks)]
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats, act=None),
            conv(n_feats, 3, kernel_size)
        )

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pytorch.py <model_path> [image_path]")
        print("Example: python test_pytorch.py ../model/EDSR_x4.pt ../model/test_input.png")
        sys.exit(1)

    model_path = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else "../model/test_input.png"

    # Infer scale from model name
    scale = 4  # default
    if 'x2' in model_path.lower():
        scale = 2
    elif 'x3' in model_path.lower():
        scale = 3
    elif 'x4' in model_path.lower():
        scale = 4

    print(f"Loading EDSR model (scale={scale}) from {model_path}...")
    model = EDSR(scale=scale)

    # Load state dict
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
    inp, orig_size = preprocess(img_path)
    print(f"Input shape: {inp.shape}, Original size: {orig_size}")

    print("Running inference...")
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
    print("\nPyTorch inference test PASSED!")


if __name__ == "__main__":
    main()
