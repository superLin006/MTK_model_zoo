#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN模型定义和加载函数
用于MTK NPU转换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) used in RRDB."""

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
    """Residual in Residual Dense Block (RRDB)."""

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


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class RRDBNet(nn.Module):
    """Real-ESRGAN Network (RRDBNet).

    This is the core model without any pre/post-processing.
    The model expects input in [0, 1] range and outputs in [0, 1] range.
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

        # Upsampling layers (using nearest interpolation + conv for MTK compatibility)
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

        # Upsample using nearest interpolation + conv
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


def infer_model_type(model_path):
    """从模型文件名推断模型类型."""
    filename = model_path.lower()

    if 'anime' in filename:
        return 'anime', 4, 6
    elif 'esrnet' in filename:
        return 'esrnet', 4, 23
    elif 'x4' in filename or 'x4plus' in filename:
        return 'esrgan', 4, 23
    elif 'x2' in filename:
        return 'esrgan', 2, 23
    else:
        return 'esrnet', 4, 23


def load_realesrgan_from_checkpoint(checkpoint_path, device='cpu'):
    """从checkpoint加载Real-ESRGAN模型.

    Args:
        checkpoint_path: .pth模型文件路径
        device: 加载设备

    Returns:
        model: 加载的模型
        scale: 超分倍数
    """
    # 推断模型类型
    model_type, scale, num_block = infer_model_type(checkpoint_path)

    print(f"检测到模型类型: {model_type}, 倍数: x{scale}, 块数: {num_block}")

    # 创建模型
    if model_type == 'anime':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                       num_grow_ch=32, scale=4)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                       num_grow_ch=32, scale=scale)

    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 处理不同的checkpoint格式
    if isinstance(state_dict, dict):
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    # 移除'module.'前缀（如果存在）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ 模型加载成功 ({param_count:.2f}M 参数)")

    return model.to(device), scale


# 预定义的模型构建函数
def RealESRGAN_x4plus():
    """RealESRGAN x4plus 通用模型."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=4)


def RealESRNet_x4plus():
    """RealESRNet x4plus PSNR导向模型."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=4)


def RealESRGAN_x4plus_anime_6B():
    """RealESRGAN x4plus 动漫优化模型."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                   num_grow_ch=32, scale=4)


def RealESRGAN_x2plus():
    """RealESRGAN x2plus 2倍超分模型."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=2)


if __name__ == '__main__':
    # 测试模型创建
    print("测试Real-ESRGAN模型架构...")

    model = RealESRNet_x4plus()
    print(f"\nRealESRNet_x4plus:")
    print(f"  总参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"  输入形状:  {x.shape}")
    print(f"  输出形状: {y.shape}")
    print(f"  超分倍数: {y.shape[2] // x.shape[2]}x")

    print("\n✓ 模型架构测试通过!")
