#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moonshine Streaming Decoder - MTK NPU 优化版本

架构说明:
  - CPU 端完成:
      embed_tokens 查表 (GATHER 不支持)
      pos_emb(encoder_out 位置编码) 查表 + 加法
      encoder_proj(620→512) 线性变换 (可选移入 NPU)
      RoPE cos/sin 预计算查表
      causal mask 准备
      KV cache 更新
  - NPU 输入:
      decoder_embed:    [1, 1, 512]   (CPU 查表后的 embedding)
      encoder_out:      [1, T_enc, 512] (encoder 输出 + pos_emb + proj 后)
      past_keys:        [10, 1, max_dec_len, 512] (4D KV cache)
      past_values:      [10, 1, max_dec_len, 512]
      cos_input:        [1, 1, 32]  (当前位置的 RoPE cos)
      sin_input:        [1, 1, 32]  (当前位置的 RoPE sin)
      attn_mask:        [1, 1, 1, max_dec_len+1]
      encoder_attn_mask:[1, 1, 1, T_enc]
  - NPU 输出:
      logits:           [1, 1, 32768]
      new_keys:         [10, 1, 1, 512]
      new_values:       [10, 1, 1, 512]

固定形状:
  - T_enc = 293 (对应 93680 samples encoder 输出)
  - max_dec_len = 64
  - vocab_size = 32768

RoPE 说明:
  partial_rotary_factor=0.5, head_dim=64 → rot_dim=32
  只旋转前 32 维, 后 32 维 pass through
  cos/sin shape: [1, 1, 32] → 经 repeat_interleave(2) 扩展到 [1, 1, 32]
  注意: 这里传入 [1, 1, 32] 即 half-dim, forward 内部 repeat_interleave 到 32

Decoder MLP:
  fc1: Linear(512, 4096) → chunk(2, dim=-1) → x [2048], gate [2048]
  output: silu(gate) * x → fc2: Linear(2048, 512)
  (chunk 改为显式 split 避免 trace 问题)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转 hidden dims 的一半 (interleaved 格式)
    避免 strided_slice (MDLA 不支持 stride!=1)
    避免 5D tensor (MDLA rank 限制 ≤ 4)

    x shape: [B, H, 1, rot_dim=32]
    输入 B=1, seq=1 固定, 因此压缩到 4D 内操作

    等价于:
        x1 = x[..., 0::2]  → 偶数位置
        x2 = x[..., 1::2]  → 奇数位置
        return stack((-x2, x1), dim=-1).flatten(-2)

    新实现: 通过 squeeze+view+4D stack 避免 5D
    """
    # x: [1, 8, 1, 32]
    # squeeze seq dim → [1, 8, 32] (3D)
    B, H, T, D = x.shape[0], x.shape[1], x.shape[2], x.shape[-1]
    half = D // 2
    x3d = x.squeeze(2)            # [1, 8, 32]
    # view to [1, 8, 16, 2] → 4D, 最后两维是 (even, odd) 对
    x4d = x3d.view(B, H, half, 2)  # [1, 8, 16, 2]
    x_even = x4d[..., 0]           # [1, 8, 16] — 偶数位置
    x_odd = x4d[..., 1]            # [1, 8, 16] — 奇数位置
    # interleave: [-odd, even] → [1, 8, 16, 2] → flatten → [1, 8, 32]
    rotated = torch.stack([-x_odd, x_even], dim=-1)  # [1, 8, 16, 2] — 4D OK
    rotated_flat = rotated.view(B, H, D)             # [1, 8, 32]
    return rotated_flat.unsqueeze(2)                 # [1, 8, 1, 32]


def apply_rope_partial(
    q: torch.Tensor,  # [B, heads, 1, head_dim=64]
    k: torch.Tensor,  # [B, heads, 1, head_dim=64]
    cos: torch.Tensor,  # [1, 1, 32] - 已预展开 (interleaved format)
    sin: torch.Tensor,  # [1, 1, 32] - 已预展开
    rot_dim: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Partial RoPE: 只旋转前 rot_dim 维, 后面 pass through
    cos/sin 已由 CPU 端预展开为 interleaved 格式, shape [1, 1, 32]
    不使用 repeat_interleave (MDLA 不支持), CPU 端预计算

    对应 HF 的:
        cos[..., :16].repeat_interleave(2, dim=-1)  → [32]
    CPU 端等价:
        stack([cos_half, cos_half], dim=-1).flatten(-2)
    已在 precompute_rope_table_expanded() 中完成
    """
    # cos/sin: [1, 1, 32] → unsqueeze → [1, 1, 1, 32]
    cos_4d = cos.unsqueeze(1)  # [1, 1, 1, 32]
    sin_4d = sin.unsqueeze(1)  # [1, 1, 1, 32]

    # 分割旋转和非旋转部分
    q_rot = q[..., :rot_dim]
    q_pass = q[..., rot_dim:]
    k_rot = k[..., :rot_dim]
    k_pass = k[..., rot_dim:]

    # 应用 RoPE
    q_embed = (q_rot * cos_4d) + (rotate_half(q_rot) * sin_4d)
    k_embed = (k_rot * cos_4d) + (rotate_half(k_rot) * sin_4d)

    # 拼接
    q_out = torch.cat([q_embed, q_pass], dim=-1)
    k_out = torch.cat([k_embed, k_pass], dim=-1)
    return q_out, k_out


class MTKMoonshineDecoderSelfAttn(nn.Module):
    """
    Decoder Self-Attention with KV Cache
    head_dim=64, num_heads=8, hidden_size=512
    RoPE: partial (前32维旋转)

    输入:
        hidden_states: [1, 1, 512]
        past_key:      [1, max_dec_len, 512]  (当前层的 past K)
        past_value:    [1, max_dec_len, 512]  (当前层的 past V)
        cos:           [1, 1, 32]
        sin:           [1, 1, 32]
        attn_mask:     [1, 1, 1, max_dec_len+1]

    输出:
        attn_out:  [1, 1, 512]
        new_key:   [1, 1, 512]  (当前 token 的 K, 供 C++ 写入缓存)
        new_value: [1, 1, 512]  (当前 token 的 V)
    """
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, head_dim: int = 64, rot_dim: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rot_dim = rot_dim
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, 1, 512]
        past_key: torch.Tensor,       # [1, max_dec_len, 512]
        past_value: torch.Tensor,     # [1, max_dec_len, 512]
        cos: torch.Tensor,            # [1, 1, 32]
        sin: torch.Tensor,            # [1, 1, 32]
        attn_mask: torch.Tensor,      # [1, 1, 1, max_dec_len+1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = hidden_states.shape[0]

        # 计算 Q, K, V
        query = self.q_proj(hidden_states)  # [1, 1, 512]
        new_key = self.k_proj(hidden_states)    # [1, 1, 512]
        new_value = self.v_proj(hidden_states)  # [1, 1, 512]

        # Reshape to [B, heads, 1, head_dim]
        query = query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_cur = new_key.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE (只对 query 和 current key)
        query, key_cur = apply_rope_partial(query, key_cur, cos, sin, self.rot_dim)

        # 拼接历史 KV + 当前 KV
        # past_key: [1, max_dec_len, 512]
        # new_key:  [1, 1, 512]
        full_key = torch.cat([past_key, new_key], dim=1)    # [1, max_dec_len+1, 512]
        full_value = torch.cat([past_value, new_value], dim=1)

        kv_len = full_key.shape[1]
        full_key_4d = full_key.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        full_value_4d = full_value.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        # 注意: 拼接后的 full_key 是 raw K (未旋转 past + 未旋转 new_key)
        # 但 new_key (past 格式) 存的是旋转前的K, cross-attention 每次重新计算
        # 为简化: 存储旋转前的K, 每次全量重新旋转 (TorchScript 追踪兼容)
        # 实际上: past_key 中存储的是之前的 new_key (未旋转), 拼接后整体 view
        # 然而 full_key_4d 是从未旋转的 full_key 来的
        # 而 key_cur 是旋转后的...
        # 正确做法: 存储旋转后的K, 拼接后直接用于attention
        # 但这样 past_key 格式变为 [max_dec_len, 512] flat of rotated K
        # 我们采用: 存储未旋转K, 拼接后重新做rope (too slow for large cache)
        # 更好: 存储旋转后K, key_cur 替换 full_key 的最后一个位置
        # 最终决策: 存储旋转后 K/V (flat 512 per position)
        # full_key_4d 需要从旋转后重建
        # → 让 new_key 存储旋转后的K
        # 修改: new_key_for_cache 是旋转后的 key_cur

        # 重建: 将 key_cur (旋转后, [1, heads, 1, head_dim]) 展平
        new_key_rotated = key_cur.transpose(1, 2).contiguous().view(B, 1, self.hidden_size)  # [1, 1, 512]

        # 拼接 (past_key 存的也是旋转后K)
        full_key_rotated = torch.cat([past_key, new_key_rotated], dim=1)  # [1, max+1, 512]
        full_key_rotated_4d = full_key_rotated.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query * self.scaling, full_key_rotated_4d.transpose(-2, -1))
        attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        attn_output = torch.matmul(attn_weights, full_value_4d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_key_rotated, new_value


class MTKMoonshineDecoderCrossAttn(nn.Module):
    """
    Decoder Cross-Attention (encoder is fixed, no KV cache needed for cross-attn
    because encoder output doesn't change)
    但为了 NPU trace 简单, 每步重新计算 (encoder 固定, K/V 每步相同但 trace 需要固定输入)
    """
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,       # [1, 1, 512]
        encoder_hidden_states: torch.Tensor,  # [1, T_enc, 512]
        encoder_attn_mask: torch.Tensor,   # [1, 1, 1, T_enc]
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        T_enc = encoder_hidden_states.shape[1]

        query = self.q_proj(hidden_states)  # [1, 1, 512]
        key = self.k_proj(encoder_hidden_states)  # [1, T_enc, 512]
        value = self.v_proj(encoder_hidden_states)  # [1, T_enc, 512]

        query = query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query * self.scaling, key.transpose(-2, -1))
        attn_weights = attn_weights + encoder_attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.hidden_size)
        return self.o_proj(attn_output)


class MTKMoonshineDecoderMLP(nn.Module):
    """
    Decoder MLP: GLU 结构
    fc1: Linear(512, 4096) → split → x[2048], gate[2048]
    output: silu(gate) * x → fc2: Linear(2048, 512)
    """
    def __init__(self, hidden_size: int = 512, intermediate_size: int = 2048):
        super().__init__()
        # fc1 输出 intermediate_size*2 = 4096 (gate + up)
        self.fc1 = nn.Linear(hidden_size, intermediate_size * 2)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)  # [1, 1, 4096]
        half = x.shape[-1] // 2
        hidden, gate = x[..., :half], x[..., half:]  # 显式 split (trace 兼容)
        x = F.silu(gate) * hidden
        return self.fc2(x)


class MTKMoonshineDecoderLayer(nn.Module):
    """单个 Decoder Layer"""
    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        rot_dim: int = 32,
        intermediate_size: int = 2048,
    ):
        super().__init__()
        self.self_attn = MTKMoonshineDecoderSelfAttn(hidden_size, num_heads, head_dim, rot_dim)
        self.encoder_attn = MTKMoonshineDecoderCrossAttn(hidden_size, num_heads, head_dim)
        self.mlp = MTKMoonshineDecoderMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, bias=False)
        self.final_layernorm = nn.LayerNorm(hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_key, new_value = self.self_attn(
            hidden_states, past_key, past_value, cos, sin, attn_mask
        )
        hidden_states = residual + hidden_states

        # Cross-attention
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, encoder_hidden_states, encoder_attn_mask)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_key, new_value


class MTKMoonshineDecoderNPU(nn.Module):
    """
    Moonshine Decoder NPU 部分 (单步推理)

    注意: encoder proj (620→512) 和 pos_emb 加法由 CPU 完成后传入 encoder_out
         embed_tokens 查表由 CPU 完成后传入 decoder_embed

    输入:
        decoder_embed:    [1, 1, 512]      CPU 完成 embed_tokens 查表
        encoder_out:      [1, T_enc, 512]  CPU 完成 pos_emb 加法 + proj(620→512)
        past_keys:        [10, 1, max_dec_len, 512]
        past_values:      [10, 1, max_dec_len, 512]
        cos_input:        [1, 1, 32]  当前位置 RoPE cos (full rot_dim)
        sin_input:        [1, 1, 32]  当前位置 RoPE sin
        attn_mask:        [1, 1, 1, max_dec_len+1]
        encoder_attn_mask:[1, 1, 1, T_enc]

    输出:
        logits:     [1, 1, 32768]
        new_keys:   [10, 1, 1, 512]
        new_values: [10, 1, 1, 512]
    """
    NUM_LAYERS = 10
    HIDDEN_SIZE = 512
    NUM_HEADS = 8
    HEAD_DIM = 64
    ROT_DIM = 32
    INTERMEDIATE_SIZE = 2048
    VOCAB_SIZE = 32768

    def __init__(self, max_dec_len: int = 64):
        super().__init__()
        self.max_dec_len = max_dec_len

        self.layers = nn.ModuleList([
            MTKMoonshineDecoderLayer(
                self.HIDDEN_SIZE, self.NUM_HEADS, self.HEAD_DIM,
                self.ROT_DIM, self.INTERMEDIATE_SIZE
            )
            for _ in range(self.NUM_LAYERS)
        ])

        self.norm = nn.LayerNorm(self.HIDDEN_SIZE, bias=False)
        # proj_out (tied with embed_tokens, loaded from model.proj_out)
        self.proj_out = nn.Linear(self.HIDDEN_SIZE, self.VOCAB_SIZE, bias=False)

    def forward(
        self,
        decoder_embed: torch.Tensor,       # [1, 1, 512]
        encoder_out: torch.Tensor,         # [1, T_enc, 512]
        past_keys: torch.Tensor,           # [10, 1, max_dec_len, 512]
        past_values: torch.Tensor,         # [10, 1, max_dec_len, 512]
        cos_input: torch.Tensor,           # [1, 1, 32]
        sin_input: torch.Tensor,           # [1, 1, 32]
        attn_mask: torch.Tensor,           # [1, 1, 1, max_dec_len+1]
        encoder_attn_mask: torch.Tensor,   # [1, 1, 1, T_enc]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hidden_states = decoder_embed  # [1, 1, 512]

        new_keys_list = []
        new_values_list = []

        for i, layer in enumerate(self.layers):
            layer_past_key = past_keys[i]    # [1, max_dec_len, 512]
            layer_past_value = past_values[i]

            hidden_states, new_key, new_value = layer(
                hidden_states,
                encoder_out,
                layer_past_key,
                layer_past_value,
                cos_input,
                sin_input,
                attn_mask,
                encoder_attn_mask,
            )
            new_keys_list.append(new_key)    # [1, 1, 512]
            new_values_list.append(new_value)

        hidden_states = self.norm(hidden_states)
        logits = self.proj_out(hidden_states)  # [1, 1, 32768]

        new_keys = torch.stack(new_keys_list, dim=0)    # [10, 1, 1, 512]
        new_values = torch.stack(new_values_list, dim=0)

        return logits, new_keys, new_values


def load_decoder_weights(mtk_decoder: MTKMoonshineDecoderNPU, hf_model):
    """
    从 HuggingFace 模型加载权重到 MTK 优化 Decoder

    Args:
        mtk_decoder: MTKMoonshineDecoderNPU 实例
        hf_model: MoonshineStreamingForConditionalGeneration 实例
    """
    hf_dec = hf_model.model.decoder

    for i, (mtk_layer, hf_layer) in enumerate(zip(mtk_decoder.layers, hf_dec.layers)):
        # Self-attention
        mtk_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        mtk_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        mtk_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        mtk_layer.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)

        # Cross-attention (encoder_attn in HF)
        mtk_layer.encoder_attn.q_proj.weight.data.copy_(hf_layer.encoder_attn.q_proj.weight.data)
        mtk_layer.encoder_attn.k_proj.weight.data.copy_(hf_layer.encoder_attn.k_proj.weight.data)
        mtk_layer.encoder_attn.v_proj.weight.data.copy_(hf_layer.encoder_attn.v_proj.weight.data)
        mtk_layer.encoder_attn.o_proj.weight.data.copy_(hf_layer.encoder_attn.o_proj.weight.data)

        # MLP
        mtk_layer.mlp.fc1.weight.data.copy_(hf_layer.mlp.fc1.weight.data)
        mtk_layer.mlp.fc1.bias.data.copy_(hf_layer.mlp.fc1.bias.data)
        mtk_layer.mlp.fc2.weight.data.copy_(hf_layer.mlp.fc2.weight.data)
        mtk_layer.mlp.fc2.bias.data.copy_(hf_layer.mlp.fc2.bias.data)

        # LayerNorm
        mtk_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        mtk_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
        mtk_layer.final_layernorm.weight.data.copy_(hf_layer.final_layernorm.weight.data)

    # Final norm
    mtk_decoder.norm.weight.data.copy_(hf_dec.norm.weight.data)

    # proj_out (tied with embed_tokens)
    mtk_decoder.proj_out.weight.data.copy_(hf_model.proj_out.weight.data)

    print("Decoder weights loaded successfully!")
    return mtk_decoder


def precompute_rope_table(max_len: int = 128, partial_rotary_factor: float = 0.5,
                           head_dim: int = 64, rope_theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    预计算 RoPE cos/sin 查找表 (已预展开 interleaved 格式)

    等价于 HF 的:
        freqs = inv_freq @ positions    → [max_len, 16]
        emb = [freqs, freqs]            → [max_len, 32]
        cos = cos(emb)                  → [max_len, 32]
        cos_expanded = cos[:, :16].repeat_interleave(2, dim=-1)  → [max_len, 32] interleaved

    这里直接输出 cos_expanded 格式, 避免在 NPU 内使用 repeat_interleave

    Returns:
        cos_table: [max_len, 32]  interleaved: [c0,c0,c1,c1,...,c15,c15]
        sin_table: [max_len, 32]
    """
    rot_dim = int(head_dim * partial_rotary_factor)  # 32
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))  # [16]

    positions = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # [max_len, 16]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [max_len, 32]
    cos_raw = np.cos(emb).astype(np.float32)  # [max_len, 32]
    sin_raw = np.sin(emb).astype(np.float32)

    # 预展开: cos[:, :16].repeat_interleave(2) → interleaved format
    cos_half = cos_raw[:, :rot_dim // 2]  # [max_len, 16]
    sin_half = sin_raw[:, :rot_dim // 2]
    # stack + flatten = repeat_interleave(2)
    cos_table = np.stack([cos_half, cos_half], axis=-1).reshape(max_len, rot_dim)  # [max_len, 32]
    sin_table = np.stack([sin_half, sin_half], axis=-1).reshape(max_len, rot_dim)
    return cos_table, sin_table


def prepare_encoder_for_decoder_cpu(
    encoder_out_raw: np.ndarray,  # [1, T_enc, 620] from encoder
    pos_emb_weight: np.ndarray,   # [4096, 620] from decoder.pos_emb.weight
    proj_weight: np.ndarray,      # [512, 620] from decoder.proj.weight
) -> np.ndarray:
    """
    CPU 端完成 pos_emb 加法 + proj(620→512)
    等价于 decoder.forward() 中:
        encoder_hidden_states += pos_emb(arange(T_enc))
        encoder_hidden_states = proj(encoder_hidden_states)
    """
    T_enc = encoder_out_raw.shape[1]
    pos_embed = pos_emb_weight[:T_enc, :]  # [T_enc, 620]
    enc = encoder_out_raw + pos_embed[np.newaxis, :, :]  # [1, T_enc, 620]
    # proj: Linear(620→512), no bias
    enc_proj = enc @ proj_weight.T  # [1, T_enc, 512]
    return enc_proj.astype(np.float32)


if __name__ == "__main__":
    # 快速测试
    print("Testing MTKMoonshineDecoderNPU...")
    T_enc = 293
    max_dec_len = 64

    model = MTKMoonshineDecoderNPU(max_dec_len=max_dec_len)
    model.eval()

    # dummy inputs
    decoder_embed = torch.randn(1, 1, 512)
    encoder_out = torch.randn(1, T_enc, 512)
    past_keys = torch.zeros(10, 1, max_dec_len, 512)
    past_values = torch.zeros(10, 1, max_dec_len, 512)
    cos_input = torch.zeros(1, 1, 32)
    sin_input = torch.zeros(1, 1, 32)
    attn_mask = torch.full((1, 1, 1, max_dec_len + 1), -1e9)
    attn_mask[:, :, :, -1] = 0.0  # current token valid
    encoder_attn_mask = torch.zeros(1, 1, 1, T_enc)

    with torch.no_grad():
        logits, new_keys, new_values = model(
            decoder_embed, encoder_out, past_keys, past_values,
            cos_input, sin_input, attn_mask, encoder_attn_mask
        )

    print(f"logits: {logits.shape}")
    print(f"new_keys: {new_keys.shape}")
    print(f"new_values: {new_values.shape}")
    assert logits.shape == (1, 1, 32768)
    assert new_keys.shape == (10, 1, 1, 512)
    assert new_values.shape == (10, 1, 1, 512)
    print("OK: all output shapes correct")
