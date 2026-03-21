#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moonshine Streaming Encoder - MTK NPU 优化版本

架构说明:
  - CPU 端完成: CMVN + AsinhCompression (因为 LOG 算子 MDLA 5.3 不支持)
  - NPU 输入: x_frames [1, num_frames, 80] (已完成 CMVN+Asinh 的帧特征)
  - NPU 内部:
      Linear(80→620) + SiLU
      → transpose → CausalConv1d(stride=2) + SiLU → CausalConv1d(stride=2)
      → transpose
      → 10x EncoderLayer (self-attn + MLP, sliding window mask)
      → LayerNorm
  - NPU 输出: encoder_hidden_states [1, T_enc, 620]

固定形状:
  - 输入: [1, 1171, 80] (对应 93680 samples / 80 = 1171 frames)
  - 输出: [1, 293, 620]

注意: Encoder 的 head_dim=64, q_proj 输出 512 (8*64), 不是 620
      Sliding window mask 预计算为 buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


def create_sliding_window_mask(seq_len: int, left: int, right: int) -> torch.Tensor:
    """
    预计算 sliding window attention mask.
    返回 [1, 1, seq_len, seq_len], 0=参与, -1e9=不参与

    Args:
        seq_len: 序列长度
        left: 向左看几帧 (不含当前帧, 即 dist < left 且 dist >= 0)
        right: 向右看几帧 (不含当前帧, 即 -dist < right 且 dist < 0)
    """
    mask = torch.full((seq_len, seq_len), -1e9)
    for q in range(seq_len):
        k_min = max(0, q - left + 1)
        k_max = min(seq_len, q + right + 1)
        mask[q, k_min:k_max] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


class MTKMoonshineLayerNorm(nn.Module):
    """
    MoonshineStreamingLayerNorm 等价实现
    LayerNorm(elementwise_affine=False) + gamma * (normed + unit_offset=1.0)
    等价于: normed * (gamma + 1.0)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln(x)
        gamma = self.gamma + 1.0
        return normed * gamma


class MTKMoonshineEncoderAttention(nn.Module):
    """
    Encoder Self-Attention (sliding window, no RoPE)
    head_dim=64, q_proj 输出 512 (8*64)
    """
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_dim = num_heads * head_dim  # 8*64=512
        self.scaling = head_dim ** -0.5
        self.hidden_size = hidden_size  # 620

        self.q_proj = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,     # [1, T, 620]
        attention_mask: torch.Tensor,    # [1, 1, T, T]
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        query = self.q_proj(hidden_states)  # [1, T, 512]
        key = self.k_proj(hidden_states)    # [1, T, 512]
        value = self.v_proj(hidden_states)  # [1, T, 512]

        # reshape to [B, heads, T, head_dim]
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query * self.scaling, key.transpose(-2, -1))
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.q_dim)
        return self.o_proj(attn_output)


class MTKMoonshineEncoderMLP(nn.Module):
    """Encoder FFN: Linear → GELU → Linear"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class MTKMoonshineEncoderLayer(nn.Module):
    """单个 Encoder Transformer Layer"""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        self.self_attn = MTKMoonshineEncoderAttention(hidden_size, num_heads, head_dim)
        self.mlp = MTKMoonshineEncoderMLP(hidden_size, intermediate_size)
        self.input_layernorm = MTKMoonshineLayerNorm(hidden_size)
        self.post_attention_layernorm = MTKMoonshineLayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [1, T, 620]
        attn_mask: torch.Tensor,       # [1, 1, T, T]
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MTKMoonshineEncoderNPU(nn.Module):
    """
    Moonshine Encoder NPU 部分

    输入 (CPU 已完成 CMVN + AsinhCompression):
        x_frames: [1, num_frames, 80]  frame_len=80

    内部流程:
        Linear(80→620) + SiLU
        → transpose → CausalConv1d(620→1240, k=5, s=2) + SiLU
        → CausalConv1d(1240→620, k=5, s=2)
        → transpose
        → 10x EncoderLayer (sliding window mask 预计算为 buffer)
        → LayerNorm

    输出:
        encoder_out: [1, T_enc, 620]

    固定形状: num_frames=1171 → T_enc=293
    """
    # 模型超参数
    HIDDEN_SIZE = 620
    NUM_HEADS = 8
    HEAD_DIM = 64
    INTERMEDIATE_SIZE = 2480
    NUM_LAYERS = 10
    FRAME_LEN = 80
    # sliding_windows per layer: [[left, right], ...]
    SLIDING_WINDOWS = [
        [16, 4], [16, 4], [16, 0], [16, 0], [16, 0],
        [16, 0], [16, 0], [16, 0], [16, 4], [16, 4]
    ]

    def __init__(self, num_frames: int = 1171):
        super().__init__()
        self.num_frames = num_frames

        # Embedder 中的 Linear + CausalConv1d
        self.linear = nn.Linear(self.FRAME_LEN, self.HIDDEN_SIZE, bias=False)

        # CausalConv1d 用 F.pad + Conv1d 实现
        # conv1: in=620, out=1240, k=5, s=2, left_pad=4
        self.conv1 = nn.Conv1d(
            self.HIDDEN_SIZE, self.HIDDEN_SIZE * 2,
            kernel_size=5, stride=2, bias=True
        )
        self.conv1_left_pad = 4  # (kernel_size-1)*dilation = (5-1)*1 = 4

        # conv2: in=1240, out=620, k=5, s=2, left_pad=4
        self.conv2 = nn.Conv1d(
            self.HIDDEN_SIZE * 2, self.HIDDEN_SIZE,
            kernel_size=5, stride=2, bias=True
        )
        self.conv2_left_pad = 4

        # Transformer layers
        self.layers = nn.ModuleList([
            MTKMoonshineEncoderLayer(
                self.HIDDEN_SIZE, self.NUM_HEADS, self.HEAD_DIM, self.INTERMEDIATE_SIZE
            )
            for _ in range(self.NUM_LAYERS)
        ])

        # Final LayerNorm
        self.final_norm = MTKMoonshineLayerNorm(self.HIDDEN_SIZE)

        # 预计算 T_enc (用于 sliding window mask)
        # num_frames → conv1(stride=2) → conv2(stride=2)
        # CausalConv1d with left_pad = k-1:
        #   output_len = (input_len + left_pad - k) // stride + 1
        #              = (input_len + (k-1) - k) // stride + 1
        #              = (input_len - 1) // stride + 1
        t_after_conv1 = (num_frames - 1) // 2 + 1
        t_enc = (t_after_conv1 - 1) // 2 + 1
        self.t_enc = t_enc

        # 预计算 sliding window mask
        self._register_sliding_window_masks(t_enc)

    def _register_sliding_window_masks(self, t_enc: int):
        """为每层预计算 sliding window mask"""
        for i, (left, right) in enumerate(self.SLIDING_WINDOWS):
            mask = create_sliding_window_mask(t_enc, left, right)  # [1, 1, T, T]
            self.register_buffer(f'attn_mask_{i}', mask)

    def forward(self, x_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_frames: [1, num_frames, 80] - CPU完成CMVN+Asinh后的帧特征

        Returns:
            encoder_out: [1, T_enc, 620]
        """
        # Linear(80→620) + SiLU
        hidden_states = F.silu(self.linear(x_frames))  # [1, num_frames, 620]

        # transpose for Conv1d: [1, 620, num_frames]
        hidden_states = hidden_states.transpose(1, 2)

        # CausalConv1d 1: stride=2
        hidden_states = F.pad(hidden_states, (self.conv1_left_pad, 0))
        hidden_states = self.conv1(hidden_states)  # [1, 1240, ~num_frames//2]
        hidden_states = F.silu(hidden_states)

        # CausalConv1d 2: stride=2
        hidden_states = F.pad(hidden_states, (self.conv2_left_pad, 0))
        hidden_states = self.conv2(hidden_states)  # [1, 620, T_enc]

        # transpose back: [1, T_enc, 620]
        hidden_states = hidden_states.transpose(1, 2)

        # 10x Transformer layers with precomputed sliding window masks
        for i, layer in enumerate(self.layers):
            attn_mask = getattr(self, f'attn_mask_{i}')
            hidden_states = layer(hidden_states, attn_mask)

        # Final LayerNorm
        hidden_states = self.final_norm(hidden_states)

        return hidden_states


def load_encoder_weights(mtk_encoder: MTKMoonshineEncoderNPU, hf_model):
    """
    从 HuggingFace 模型加载权重到 MTK 优化 Encoder

    Args:
        mtk_encoder: MTKMoonshineEncoderNPU 实例
        hf_model: MoonshineStreamingForConditionalGeneration 实例
    """
    hf_enc = hf_model.model.encoder
    embedder = hf_enc.embedder

    # Embedder Linear
    mtk_encoder.linear.weight.data.copy_(embedder.linear.weight.data)
    # (linear has no bias)

    # CausalConv1d weights
    mtk_encoder.conv1.weight.data.copy_(embedder.conv1.weight.data)
    mtk_encoder.conv1.bias.data.copy_(embedder.conv1.bias.data)
    mtk_encoder.conv2.weight.data.copy_(embedder.conv2.weight.data)
    mtk_encoder.conv2.bias.data.copy_(embedder.conv2.bias.data)

    # Transformer layers
    for i, (mtk_layer, hf_layer) in enumerate(zip(mtk_encoder.layers, hf_enc.layers)):
        # Self-attention weights (no bias for moonshine encoder)
        mtk_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        mtk_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        mtk_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        mtk_layer.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)

        # MLP
        mtk_layer.mlp.fc1.weight.data.copy_(hf_layer.mlp.fc1.weight.data)
        mtk_layer.mlp.fc1.bias.data.copy_(hf_layer.mlp.fc1.bias.data)
        mtk_layer.mlp.fc2.weight.data.copy_(hf_layer.mlp.fc2.weight.data)
        mtk_layer.mlp.fc2.bias.data.copy_(hf_layer.mlp.fc2.bias.data)

        # LayerNorm (gamma only, no bias in MoonshineStreamingLayerNorm)
        mtk_layer.input_layernorm.gamma.data.copy_(hf_layer.input_layernorm.gamma.data)
        mtk_layer.post_attention_layernorm.gamma.data.copy_(hf_layer.post_attention_layernorm.gamma.data)

    # Final LayerNorm
    mtk_encoder.final_norm.gamma.data.copy_(hf_enc.final_norm.gamma.data)

    print("Encoder weights loaded successfully!")
    return mtk_encoder


def preprocess_audio_cpu(input_values: np.ndarray, log_k: float, frame_len: int = 80) -> np.ndarray:
    """
    CPU 端完成 CMVN + AsinhCompression
    (因为 torch.asinh 需要 LOG, MDLA 5.3 不支持 LOG)

    Args:
        input_values: [1, T_audio] float32 raw waveform
        log_k: AsinhCompression 的 log_k 参数值 (标量)
        frame_len: 帧长, 默认 80 (5ms @ 16kHz)

    Returns:
        x_frames: [1, num_frames, 80] float32
    """
    # Reshape to frames
    T = input_values.shape[-1]
    num_frames = T // frame_len
    x = input_values[:, :num_frames * frame_len].reshape(1, num_frames, frame_len)

    # CMVN: 均值中心化 + RMS 归一化
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    rms = np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    x_normed = centered / rms

    # AsinhCompression: y = asinh(k * x), k = exp(log_k)
    k = np.exp(log_k)
    x_comp = np.arcsinh(k * x_normed)

    return x_comp.astype(np.float32)  # [1, num_frames, 80]


if __name__ == "__main__":
    # 快速测试
    print("Testing MTKMoonshineEncoderNPU...")
    model = MTKMoonshineEncoderNPU(num_frames=1171)
    model.eval()

    # dummy input
    x = torch.randn(1, 1171, 80)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 293, 620), f"Expected [1, 293, 620], got {out.shape}"
    print("OK: output shape matches [1, 293, 620]")
    print(f"t_enc computed: {model.t_enc}")
