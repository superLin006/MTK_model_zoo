"""
EDSR (Enhanced Deep Residual Networks) Model Definition
For MTK NPU conversion
"""

import math
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """Default convolution layer"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    """RGB mean shift layer"""
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    """Upsampling module"""
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    """EDSR Model"""
    def __init__(self, n_resblocks=32, n_feats=256, scale=4, rgb_range=255,
                 res_scale=0.1, conv=default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # Define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # Define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # Define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                         'whose dimensions in the model are {} and '
                                         'whose dimensions in the checkpoint are {}.'
                                         .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                 .format(name))


class EDSRCore(nn.Module):
    """EDSR Core Model (without MeanShift for MTK conversion)"""
    def __init__(self, n_resblocks=32, n_feats=256, scale=4, res_scale=0.1, conv=default_conv):
        super(EDSRCore, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # Define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # Define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # Define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


def load_edsr_from_checkpoint(checkpoint_path, scale=4):
    """Load EDSR model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = EDSR(scale=scale)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    return model


def create_core_model_from_full(full_model, scale=4):
    """Create core model (without MeanShift) from full EDSR model"""
    core_model = EDSRCore(scale=scale)
    
    # Copy weights from full model to core model
    full_state = full_model.state_dict()
    core_state = core_model.state_dict()
    
    copied = 0
    for name in core_state.keys():
        if name in full_state:
            core_state[name].copy_(full_state[name])
            copied += 1
    
    print(f"  复制了 {copied}/{len(core_state)} 个权重")
    
    core_model.load_state_dict(core_state)
    return core_model


def get_meanshift_params():
    """Get MeanShift parameters for preprocessing/postprocessing"""
    return {
        'rgb_mean': [0.4488, 0.4371, 0.4040],
        'rgb_range': 255.0
    }
