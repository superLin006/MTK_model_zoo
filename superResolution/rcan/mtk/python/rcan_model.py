#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RCAN模型定义和转换工具
包含完整模型（带MeanShift）和核心模型（无MeanShift）
"""

import torch
import torch.nn as nn
import numpy as np


# ===========================
# 基础组件
# ===========================

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class MeanShift(nn.Conv2d):
    """
    MeanShift层：用于RGB均值归一化
    sign=-1: 减去均值 (sub_mean)
    sign=1: 加回均值 (add_mean)
    """
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class Upsampler(nn.Sequential):
    """上采样模块：支持2^n和3倍"""
    def __init__(self, conv, scale, n_feats, bn=False, act=None, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(np.log2(scale))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act is not None:
                    m.append(act)
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                m.append(act)
        else:
            raise NotImplementedError(f'scale {scale} is not supported')
        super(Upsampler, self).__init__(*m)


class CALayer(nn.Module):
    """通道注意力机制"""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block"""
    def __init__(self, conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
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

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualGroup(nn.Module):
    """残差组"""
    def __init__(self, conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules = [RCAB(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale)
                   for _ in range(n_resblocks)]
        modules.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# ===========================
# 完整RCAN模型
# ===========================

class RCAN(nn.Module):
    """
    完整RCAN模型（包含MeanShift前后处理）
    """
    def __init__(self, scale, n_resgroups=10, n_resblocks=20, n_feats=64,
                 reduction=16, res_scale=1.0, rgb_range=255.0):
        super(RCAN, self).__init__()
        conv = default_conv
        kernel_size = 3
        act = nn.ReLU(True)

        # MeanShift层
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # 头部
        self.head = nn.Sequential(conv(3, n_feats, kernel_size))

        # 主体：多个残差组
        body = [ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale,
                              n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        # 尾部
        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats, act=None),
            conv(n_feats, 3, kernel_size)
        )

    def forward(self, x):
        # 前处理：减去均值
        x = self.sub_mean(x)
        # 推理
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        # 后处理：加回均值
        x = self.add_mean(x)
        return x


# ===========================
# 核心RCAN模型（无MeanShift）
# ===========================

class RCANCore(nn.Module):
    """
    RCAN核心推理模型（不包含MeanShift）
    用于MTK NPU部署，前后处理在C++端实现
    """
    def __init__(self, scale, n_resgroups=10, n_resblocks=20, n_feats=64,
                 reduction=16, res_scale=1.0):
        super(RCANCore, self).__init__()
        conv = default_conv
        kernel_size = 3

        # 仅包含CNN推理部分
        self.head = nn.Sequential(conv(3, n_feats, kernel_size))

        body = [ResidualGroup(conv, n_feats, kernel_size, reduction, act=nn.ReLU(True),
                              res_scale=res_scale, n_resblocks=n_resblocks)
                for _ in range(n_resgroups)]
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats, act=None),
            conv(n_feats, 3, kernel_size)
        )

    def forward(self, x):
        # 输入已经减去均值
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        # 输出需要加回均值（在C++端完成）
        return x


# ===========================
# 模型加载和转换函数
# ===========================

def load_rcan_from_checkpoint(checkpoint_path, scale):
    """
    从.pt文件加载完整RCAN模型
    """
    # 加载checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # 处理不同的state_dict格式
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # 移除可能的'module.'前缀
    new_state = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state[new_key] = v

    # 创建模型（使用RCAN默认参数）
    model = RCAN(scale=scale)
    model.load_state_dict(new_state, strict=True)
    return model


def create_core_model_from_full(full_model, scale):
    """
    从完整模型创建核心模型（移除MeanShift层）
    通过复制权重实现
    """
    # 创建核心模型
    core_model = RCANCore(scale=scale)

    # 复制权重（排除MeanShift层）
    full_dict = full_model.state_dict()
    core_dict = core_model.state_dict()

    matched_keys = []
    for key in core_dict.keys():
        if key in full_dict:
            core_dict[key] = full_dict[key]
            matched_keys.append(key)
        else:
            print(f"  警告: 核心模型的key '{key}' 在完整模型中不存在")

    core_model.load_state_dict(core_dict)
    print(f"  复制了 {len(matched_keys)}/{len(core_dict)} 个权重")

    return core_model


def get_meanshift_params():
    """
    获取MeanShift参数
    这些参数需要在C++端用于前后处理
    """
    return {
        'rgb_mean': [0.4488, 0.4371, 0.4040],
        'rgb_std': [1.0, 1.0, 1.0],
        'rgb_range': 255.0
    }


def infer_scale_from_model_name(model_path):
    """从模型文件名推断scale"""
    model_path = model_path.lower()
    if 'bix2' in model_path or 'x2' in model_path:
        return 2
    elif 'bix3' in model_path or 'bdx3' in model_path or 'x3' in model_path:
        return 3
    elif 'bix4' in model_path or 'x4' in model_path:
        return 4
    elif 'bix8' in model_path or 'x8' in model_path:
        return 8
    return 4  # 默认


if __name__ == '__main__':
    # 测试模型创建
    print("测试RCAN模型创建...")
    model = RCAN(scale=4)
    print(f"完整模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    core_model = RCANCore(scale=4)
    print(f"核心模型参数量: {sum(p.numel() for p in core_model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")

    # 测试核心模型
    with torch.no_grad():
        y_core = core_model(x)
    print(f"核心模型输入: {x.shape}, 输出: {y_core.shape}")
