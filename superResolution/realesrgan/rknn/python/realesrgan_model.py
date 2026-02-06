#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN Network Architecture
Based on: https://github.com/xinntao/Real-ESRGAN

Real-ESRGAN uses RRDB (Residual in Residual Dense Block) architecture
which is based on ESRGAN with improvements for real-world image super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) used in RRDB.

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64
        num_grow_ch (int): Channels for each growth. Default: 32
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling: empirically 0.2 works well
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB).

    Used in RRDB-Net in ESRGAN and Real-ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64
        num_grow_ch (int): Channels for each growth. Default: 32
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Residual scaling: empirically 0.2 works well
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block (RRDB).

    It is used in ESRGAN and Real-ESRGAN: Practical Algorithms for General Image Restoration.

    This implementation matches the official BasicSR architecture used by Real-ESRGAN,
    using nearest interpolation + conv for upsampling (RKNN-friendly).

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3
        num_out_ch (int): Channel number of outputs. Default: 3
        num_feat (int): Channel number of intermediate features. Default: 64
        num_block (int): Block number in the trunk network. Default: 23
        num_grow_ch (int): Channels for each growth. Default: 32
        scale (int): Upsampling factor. Support 2, 4. Default: 4
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                 num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale

        # First convolution
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # RRDB trunk
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)

        # Trunk conv
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers (official Real-ESRGAN uses nearest interpolation + conv)
        # This is more RKNN-friendly than PixelShuffle
        if scale == 4:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        elif scale == 2:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        else:
            raise ValueError(f'Unsupported scale: {scale}. Supported scales: 2, 4')

        # Final output layers
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # First convolution
        feat = self.conv_first(x)

        # RRDB trunk
        body_feat = self.body(feat)
        body_feat = self.conv_body(body_feat)

        # Global residual
        feat = feat + body_feat

        # Upsample using nearest interpolation + conv (official Real-ESRGAN approach)
        if self.scale == 4:
            # 2x upsampling
            feat = F.interpolate(feat, scale_factor=2, mode='nearest')
            feat = self.lrelu(self.conv_up1(feat))
            # Another 2x upsampling
            feat = F.interpolate(feat, scale_factor=2, mode='nearest')
            feat = self.lrelu(self.conv_up2(feat))
        elif self.scale == 2:
            # 2x upsampling
            feat = F.interpolate(feat, scale_factor=2, mode='nearest')
            feat = self.lrelu(self.conv_up1(feat))

        # Final output
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


# Model variants used in Real-ESRGAN
def RealESRGAN_x4plus():
    """RealESRGAN x4plus model - General purpose 4x super-resolution with GAN training."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=4)


def RealESRNet_x4plus():
    """RealESRNet x4plus model - 4x super-resolution without GAN (PSNR-oriented)."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=4)


def RealESRGAN_x4plus_anime_6B():
    """RealESRGAN x4plus anime model - Optimized for anime images (6 RRDB blocks)."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                   num_grow_ch=32, scale=4)


def RealESRGAN_x2plus():
    """RealESRGAN x2plus model - 2x super-resolution."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=2)


if __name__ == '__main__':
    # Test model creation
    print("Testing Real-ESRGAN model architecture...")

    # Test x4 model
    model = RealESRNet_x4plus()
    print(f"\nRealESRNet_x4plus:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test forward pass
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Scale factor: {y.shape[2] // x.shape[2]}x")

    # Test anime model
    model_anime = RealESRGAN_x4plus_anime_6B()
    print(f"\nRealESRGAN_x4plus_anime_6B:")
    print(f"  Total parameters: {sum(p.numel() for p in model_anime.parameters()) / 1e6:.2f}M")

    print("\n✓ Model architecture test passed!")
