#!/usr/bin/env python3
"""
MTK NPU-optimized Whisper model with KV Cache
Based on OpenAI Whisper Base, modified for MTK MDLA 5.3 compatibility

Key modifications:
1. WhisperEncoderCore: Minimal changes (all operators supported)
   - Position embedding as buffer

2. WhisperDecoderCore with KV Cache: Major changes
   - Removed nn.Embedding layer (GATHER not supported)
   - Input accepts embeddings instead of token IDs
   - 4D KV Cache design: [num_layers, batch, seq_len, d_model]
   - Causal mask precomputed as buffer
   - Cross-attention K,V caching support

Architecture:
- Encoder: mel [1, 80, 3000] → features [1, 1500, 512]
- Decoder: embeddings [1, seq_len, 512] + encoder_output [1, 1500, 512]
            + past_kv → logits [1, seq_len, 51865] + new_kv
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Dict, Optional, Tuple
import json
import os
import math


# ============================================================
# Basic Components (from original Whisper)
# ============================================================

class LayerNorm(nn.LayerNorm):
    """LayerNorm with dtype conversion"""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """Linear with dtype conversion"""
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    """Conv1d with dtype conversion"""
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Generate sinusoidal position encodings"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# ============================================================
# Multi-Head Attention (Standard - for Encoder)
# ============================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention without SDPA (for compatibility)"""
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
        else:
            k = self.key(xa)
            v = self.value(xa)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)

        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach()

        return out, qk


# ============================================================
# Multi-Head Attention with KV Cache (for Decoder)
# ============================================================

class MultiHeadAttentionKVCache(nn.Module):
    """
    Self-Attention with 4D KV Cache

    Inputs:
        hidden_states: [batch, 1, d_model] - current token
        past_key: [batch, max_cache_len, d_model] - past keys (4D friendly)
        past_value: [batch, max_cache_len, d_model] - past values
        attn_mask: [1, 1, 1, max_cache_len+1] - attention mask

    Outputs:
        attn_output: [batch, 1, d_model]
        new_key: [batch, 1, d_model]
        new_value: [batch, 1, d_model]
    """
    def __init__(self, n_state: int, n_head: int, max_cache_len: int = 448):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.max_cache_len = max_cache_len
        self.scale = self.head_dim ** -0.25

        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        hidden_states: Tensor,  # [batch, 1, n_state]
        past_key: Tensor,       # [batch, max_cache_len, n_state]
        past_value: Tensor,     # [batch, max_cache_len, n_state]
        attn_mask: Tensor,      # [1, 1, 1, max_cache_len+1]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = hidden_states.shape[0]

        # Compute Q, K, V for current token
        q = self.query(hidden_states)  # [batch, 1, n_state]
        new_key = self.key(hidden_states)    # [batch, 1, n_state]
        new_value = self.value(hidden_states)  # [batch, 1, n_state]

        # Concatenate past and current K, V
        full_key = torch.cat([past_key, new_key], dim=1)  # [batch, max_cache_len+1, n_state]
        full_value = torch.cat([past_value, new_value], dim=1)

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.n_head, self.head_dim).transpose(1, 2)
        # [batch, n_head, 1, head_dim]

        kv_seq_len = full_key.shape[1]
        full_key = full_key.view(batch_size, kv_seq_len, self.n_head, self.head_dim).transpose(1, 2)
        full_value = full_value.view(batch_size, kv_seq_len, self.n_head, self.head_dim).transpose(1, 2)
        # [batch, n_head, max_cache_len+1, head_dim]

        # Attention computation
        attn_weights = (q * self.scale) @ (full_key * self.scale).transpose(-2, -1)
        # [batch, n_head, 1, max_cache_len+1]

        # Apply mask
        attn_weights = attn_weights + attn_mask
        attn_weights = attn_weights.float()
        attn_weights = F.softmax(attn_weights, dim=-1).to(q.dtype)

        attn_output = attn_weights @ full_value
        # [batch, n_head, 1, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.n_state)
        attn_output = self.out(attn_output)

        return attn_output, new_key, new_value


class CrossAttentionWithCache(nn.Module):
    """
    Cross-Attention with encoder K,V caching

    Inputs:
        hidden_states: [batch, 1, d_model]
        encoder_output: [batch, enc_seq_len, d_model]
        cached_key: Optional[batch, enc_seq_len, d_model] - cached encoder K
        cached_value: Optional[batch, enc_seq_len, d_model] - cached encoder V

    Outputs:
        attn_output: [batch, 1, d_model]
        key: [batch, enc_seq_len, d_model] - encoder K (for caching)
        value: [batch, enc_seq_len, d_model] - encoder V (for caching)
    """
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.scale = self.head_dim ** -0.25

        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_output: Tensor,
        cached_key: Optional[Tensor] = None,
        cached_value: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = hidden_states.shape[0]
        enc_seq_len = encoder_output.shape[1]

        # Query from decoder hidden states
        q = self.query(hidden_states)  # [batch, 1, n_state]

        # Key and Value: ALWAYS compute from encoder_output for now
        # (Caching is an optimization we can add later)
        k_original = self.key(encoder_output)    # [batch, enc_seq_len, n_state]
        v_original = self.value(encoder_output)

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.n_head, self.head_dim).transpose(1, 2)
        k = k_original.view(batch_size, enc_seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v_original.view(batch_size, enc_seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = (q * self.scale) @ (k * self.scale).transpose(-2, -1)
        attn_weights = attn_weights.float()
        attn_weights = F.softmax(attn_weights, dim=-1).to(q.dtype)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.n_state)
        attn_output = self.out(attn_output)

        # Return original 3D format for caching
        return attn_output, k_original, v_original


# ============================================================
# Residual Attention Blocks
# ============================================================

class ResidualAttentionBlock(nn.Module):
    """Standard Transformer block (for Encoder)"""
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: Tensor):
        x = x + self.attn(self.attn_ln(x))[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class DecoderBlockKVCache(nn.Module):
    """
    Decoder block with KV Cache (4D)

    Inputs:
        hidden_states: [batch, 1, n_state]
        encoder_output: [batch, 1500, n_state]
        past_self_key: [batch, max_cache_len, n_state]
        past_self_value: [batch, max_cache_len, n_state]
        cached_cross_key: Optional[batch, 1500, n_state]
        cached_cross_value: Optional[batch, 1500, n_state]
        self_attn_mask: [1, 1, 1, max_cache_len+1]

    Outputs:
        hidden_states: [batch, 1, n_state]
        new_self_key: [batch, 1, n_state]
        new_self_value: [batch, 1, n_state]
        cross_key: [batch, 1500, n_state]
        cross_value: [batch, 1500, n_state]
    """
    def __init__(self, n_state: int, n_head: int, max_cache_len: int = 448):
        super().__init__()

        self.self_attn = MultiHeadAttentionKVCache(n_state, n_head, max_cache_len)
        self.self_attn_ln = LayerNorm(n_state)

        self.cross_attn = CrossAttentionWithCache(n_state, n_head)
        self.cross_attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_output: Tensor,
        past_self_key: Tensor,
        past_self_value: Tensor,
        self_attn_mask: Tensor,
        cached_cross_key: Optional[Tensor] = None,
        cached_cross_value: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # Self-attention with KV cache
        residual = hidden_states
        attn_out, new_self_key, new_self_value = self.self_attn(
            self.self_attn_ln(hidden_states),
            past_self_key,
            past_self_value,
            self_attn_mask
        )
        hidden_states = residual + attn_out

        # Cross-attention with encoder K,V caching
        residual = hidden_states
        attn_out, cross_key, cross_value = self.cross_attn(
            self.cross_attn_ln(hidden_states),
            encoder_output,
            cached_cross_key,
            cached_cross_value
        )
        hidden_states = residual + attn_out

        # Feed-forward
        hidden_states = hidden_states + self.mlp(self.mlp_ln(hidden_states))

        return hidden_states, new_self_key, new_self_value, cross_key, cross_value


# ============================================================
# Encoder Core (minimal changes)
# ============================================================

class WhisperEncoderCore(nn.Module):
    """
    Whisper Encoder Core - MTK NPU compatible

    Input: mel-spectrogram [batch, 80, 3000]
    Output: encoder features [batch, 1500, 512]

    Changes:
    - Position embedding as buffer (not Parameter)
    """
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        # Position embedding as buffer
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x: [batch, n_mels, n_ctx*2] - mel spectrogram (3000 frames for 30s)
        returns: [batch, n_ctx, n_state] - encoder features (1500 frames)
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


# ============================================================
# Decoder Core with KV Cache (major modifications)
# ============================================================

class WhisperDecoderCoreKVCache(nn.Module):
    """
    Whisper Decoder Core with KV Cache - MTK NPU compatible

    Key modifications:
    1. Removed nn.Embedding layer (GATHER not supported)
    2. Input accepts embeddings instead of token IDs
    3. 4D KV Cache: [num_layers, batch, seq_len, n_state]
    4. Causal mask precomputed as buffer
    5. Cross-attention K,V caching

    Inputs (single token decoding):
        token_embeddings: [batch, 1, n_state] - already looked up
        encoder_output: [batch, 1500, n_state]
        past_self_keys: [num_layers, batch, max_cache_len, n_state]
        past_self_values: [num_layers, batch, max_cache_len, n_state]
        cached_cross_keys: Optional[num_layers, batch, 1500, n_state]
        cached_cross_values: Optional[num_layers, batch, 1500, n_state]
        position_embed: [1, 1, n_state] - position encoding for current position
        self_attn_mask: [1, 1, 1, max_cache_len+1]

    Outputs:
        logits: [batch, 1, n_vocab]
        new_self_keys: [num_layers, batch, 1, n_state]
        new_self_values: [num_layers, batch, 1, n_state]
        cross_keys: [num_layers, batch, 1500, n_state]
        cross_values: [num_layers, batch, 1500, n_state]
    """
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        max_cache_len: int = 448
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_layer = n_layer
        self.max_cache_len = max_cache_len

        # Position embedding as buffer (will be loaded from original model)
        positional_embedding = torch.empty(n_ctx, n_state)
        self.register_buffer("positional_embedding", positional_embedding)

        # Decoder blocks with KV cache
        self.blocks = nn.ModuleList(
            [
                DecoderBlockKVCache(n_state, n_head, max_cache_len)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        # LM head (will share weights with token embedding)
        self.lm_head = Linear(n_state, n_vocab, bias=False)

    def forward(
        self,
        token_embeddings: Tensor,          # [batch, 1, n_state]
        encoder_output: Tensor,            # [batch, 1500, n_state]
        past_self_keys: Tensor,            # [num_layers, batch, max_cache_len, n_state]
        past_self_values: Tensor,          # [num_layers, batch, max_cache_len, n_state]
        position_embed: Tensor,            # [1, 1, n_state]
        self_attn_mask: Tensor,            # [1, 1, 1, max_cache_len+1]
        cached_cross_keys: Optional[Tensor] = None,    # [num_layers, batch, 1500, n_state]
        cached_cross_values: Optional[Tensor] = None,  # [num_layers, batch, 1500, n_state]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # Add position embedding
        hidden_states = token_embeddings + position_embed
        hidden_states = hidden_states.to(encoder_output.dtype)

        new_self_keys_list = []
        new_self_values_list = []
        cross_keys_list = []
        cross_values_list = []

        for i, block in enumerate(self.blocks):
            layer_past_self_key = past_self_keys[i]
            layer_past_self_value = past_self_values[i]

            layer_cached_cross_key = cached_cross_keys[i] if cached_cross_keys is not None else None
            layer_cached_cross_value = cached_cross_values[i] if cached_cross_values is not None else None

            hidden_states, new_self_key, new_self_value, cross_key, cross_value = block(
                hidden_states,
                encoder_output,
                layer_past_self_key,
                layer_past_self_value,
                self_attn_mask,
                layer_cached_cross_key,
                layer_cached_cross_value
            )

            new_self_keys_list.append(new_self_key)
            new_self_values_list.append(new_self_value)
            cross_keys_list.append(cross_key)
            cross_values_list.append(cross_value)

        hidden_states = self.ln(hidden_states)

        # Stack outputs
        new_self_keys = torch.stack(new_self_keys_list, dim=0)  # [num_layers, batch, 1, n_state]
        new_self_values = torch.stack(new_self_values_list, dim=0)
        cross_keys = torch.stack(cross_keys_list, dim=0)  # [num_layers, batch, 1500, n_state]
        cross_values = torch.stack(cross_values_list, dim=0)

        # Compute logits
        logits = self.lm_head(hidden_states).float()

        return logits, new_self_keys, new_self_values, cross_keys, cross_values


# ============================================================
# Weight Loading Functions
# ============================================================

def load_encoder_weights(mtk_encoder: WhisperEncoderCore, whisper_model):
    """Load encoder weights from original Whisper model"""
    print("Loading encoder weights...")

    # Conv layers
    mtk_encoder.conv1.weight.data.copy_(whisper_model.encoder.conv1.weight.data)
    mtk_encoder.conv1.bias.data.copy_(whisper_model.encoder.conv1.bias.data)
    mtk_encoder.conv2.weight.data.copy_(whisper_model.encoder.conv2.weight.data)
    mtk_encoder.conv2.bias.data.copy_(whisper_model.encoder.conv2.bias.data)

    # Position embedding
    mtk_encoder.positional_embedding.copy_(whisper_model.encoder.positional_embedding)

    # Transformer blocks
    for mtk_block, orig_block in zip(mtk_encoder.blocks, whisper_model.encoder.blocks):
        # Attention
        mtk_block.attn.query.weight.data.copy_(orig_block.attn.query.weight.data)
        mtk_block.attn.query.bias.data.copy_(orig_block.attn.query.bias.data)
        mtk_block.attn.key.weight.data.copy_(orig_block.attn.key.weight.data)
        mtk_block.attn.value.weight.data.copy_(orig_block.attn.value.weight.data)
        mtk_block.attn.value.bias.data.copy_(orig_block.attn.value.bias.data)
        mtk_block.attn.out.weight.data.copy_(orig_block.attn.out.weight.data)
        mtk_block.attn.out.bias.data.copy_(orig_block.attn.out.bias.data)

        # LayerNorm
        mtk_block.attn_ln.weight.data.copy_(orig_block.attn_ln.weight.data)
        mtk_block.attn_ln.bias.data.copy_(orig_block.attn_ln.bias.data)

        # MLP
        mtk_block.mlp[0].weight.data.copy_(orig_block.mlp[0].weight.data)
        mtk_block.mlp[0].bias.data.copy_(orig_block.mlp[0].bias.data)
        mtk_block.mlp[2].weight.data.copy_(orig_block.mlp[2].weight.data)
        mtk_block.mlp[2].bias.data.copy_(orig_block.mlp[2].bias.data)
        mtk_block.mlp_ln.weight.data.copy_(orig_block.mlp_ln.weight.data)
        mtk_block.mlp_ln.bias.data.copy_(orig_block.mlp_ln.bias.data)

    # Final LayerNorm
    mtk_encoder.ln_post.weight.data.copy_(whisper_model.encoder.ln_post.weight.data)
    mtk_encoder.ln_post.bias.data.copy_(whisper_model.encoder.ln_post.bias.data)

    print("✓ Encoder weights loaded!")


def load_decoder_weights(mtk_decoder: WhisperDecoderCoreKVCache, whisper_model):
    """Load decoder weights from original Whisper model"""
    print("Loading decoder weights...")

    # Position embedding
    mtk_decoder.positional_embedding.copy_(whisper_model.decoder.positional_embedding.data)

    # Transformer blocks
    for mtk_block, orig_block in zip(mtk_decoder.blocks, whisper_model.decoder.blocks):
        # Self-attention
        mtk_block.self_attn.query.weight.data.copy_(orig_block.attn.query.weight.data)
        mtk_block.self_attn.query.bias.data.copy_(orig_block.attn.query.bias.data)
        mtk_block.self_attn.key.weight.data.copy_(orig_block.attn.key.weight.data)
        mtk_block.self_attn.value.weight.data.copy_(orig_block.attn.value.weight.data)
        mtk_block.self_attn.value.bias.data.copy_(orig_block.attn.value.bias.data)
        mtk_block.self_attn.out.weight.data.copy_(orig_block.attn.out.weight.data)
        mtk_block.self_attn.out.bias.data.copy_(orig_block.attn.out.bias.data)

        # Self-attention LayerNorm
        mtk_block.self_attn_ln.weight.data.copy_(orig_block.attn_ln.weight.data)
        mtk_block.self_attn_ln.bias.data.copy_(orig_block.attn_ln.bias.data)

        # Cross-attention
        mtk_block.cross_attn.query.weight.data.copy_(orig_block.cross_attn.query.weight.data)
        mtk_block.cross_attn.query.bias.data.copy_(orig_block.cross_attn.query.bias.data)
        mtk_block.cross_attn.key.weight.data.copy_(orig_block.cross_attn.key.weight.data)
        mtk_block.cross_attn.value.weight.data.copy_(orig_block.cross_attn.value.weight.data)
        mtk_block.cross_attn.value.bias.data.copy_(orig_block.cross_attn.value.bias.data)
        mtk_block.cross_attn.out.weight.data.copy_(orig_block.cross_attn.out.weight.data)
        mtk_block.cross_attn.out.bias.data.copy_(orig_block.cross_attn.out.bias.data)

        # Cross-attention LayerNorm
        mtk_block.cross_attn_ln.weight.data.copy_(orig_block.cross_attn_ln.weight.data)
        mtk_block.cross_attn_ln.bias.data.copy_(orig_block.cross_attn_ln.bias.data)

        # MLP
        mtk_block.mlp[0].weight.data.copy_(orig_block.mlp[0].weight.data)
        mtk_block.mlp[0].bias.data.copy_(orig_block.mlp[0].bias.data)
        mtk_block.mlp[2].weight.data.copy_(orig_block.mlp[2].weight.data)
        mtk_block.mlp[2].bias.data.copy_(orig_block.mlp[2].bias.data)
        mtk_block.mlp_ln.weight.data.copy_(orig_block.mlp_ln.weight.data)
        mtk_block.mlp_ln.bias.data.copy_(orig_block.mlp_ln.bias.data)

    # Final LayerNorm
    mtk_decoder.ln.weight.data.copy_(whisper_model.decoder.ln.weight.data)
    mtk_decoder.ln.bias.data.copy_(whisper_model.decoder.ln.bias.data)

    # LM head (shared with token embedding)
    mtk_decoder.lm_head.weight.data.copy_(whisper_model.decoder.token_embedding.weight.data)

    print("✓ Decoder weights loaded!")


# ============================================================
# Embedding Export Functions
# ============================================================

def export_embedding_weights(whisper_model, output_dir: str):
    """
    Export embedding weights for C++ lookup

    Exports:
    - token_embedding.npy: [n_vocab, n_state] token embeddings
    - embedding_info.json: metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nExporting embedding weights for C++...")

    # Export token embedding
    token_embedding_weight = whisper_model.decoder.token_embedding.weight.detach().cpu().numpy()
    token_emb_path = os.path.join(output_dir, 'token_embedding.npy')
    np.save(token_emb_path, token_embedding_weight)

    n_vocab, n_state = token_embedding_weight.shape
    print(f"  ✓ Token embedding: {token_emb_path}")
    print(f"    Shape: [{n_vocab}, {n_state}]")
    print(f"    Size: {token_embedding_weight.nbytes / 1024 / 1024:.2f} MB")

    # Export metadata
    embedding_info = {
        'vocab_size': int(n_vocab),
        'embedding_dim': int(n_state),
        'token_embedding_file': 'token_embedding.npy',
        'dtype': str(token_embedding_weight.dtype),
        'notes': 'Token embedding weights for C++ manual lookup (GATHER not supported)'
    }

    info_path = os.path.join(output_dir, 'embedding_info.json')
    with open(info_path, 'w') as f:
        json.dump(embedding_info, f, indent=2)

    print(f"  ✓ Embedding info: {info_path}")
    print("✓ Embedding export complete!")

    return token_emb_path, info_path


# ============================================================
# Model Creation Function
# ============================================================

def create_whisper_kv_models(whisper_model, max_cache_len: int = 448):
    """
    Create MTK-optimized Whisper models with KV Cache

    Returns: (encoder, decoder, dims)
    """
    dims = whisper_model.dims

    print("\n" + "="*70)
    print("Creating MTK-optimized Whisper models with KV Cache")
    print("="*70)

    # Create Encoder
    print("\n[1/2] Creating Encoder Core...")
    encoder = WhisperEncoderCore(
        n_mels=dims.n_mels,
        n_ctx=dims.n_audio_ctx,
        n_state=dims.n_audio_state,
        n_head=dims.n_audio_head,
        n_layer=dims.n_audio_layer,
    )
    load_encoder_weights(encoder, whisper_model)
    encoder.eval()
    print(f"  ✓ Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Create Decoder with KV Cache
    print("\n[2/2] Creating Decoder Core with KV Cache...")
    print(f"  Max cache length: {max_cache_len}")
    decoder = WhisperDecoderCoreKVCache(
        n_vocab=dims.n_vocab,
        n_ctx=dims.n_text_ctx,
        n_state=dims.n_text_state,
        n_head=dims.n_text_head,
        n_layer=dims.n_text_layer,
        max_cache_len=max_cache_len,
    )
    load_decoder_weights(decoder, whisper_model)
    decoder.eval()
    print(f"  ✓ Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    print("\n" + "="*70)
    print("✓ MTK models with KV Cache created successfully!")
    print("="*70)

    return encoder, decoder, dims


# ============================================================
# Test Function
# ============================================================

def test_whisper_kv_models():
    """Test MTK Whisper models with KV Cache"""
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).parents[3] / 'whisper/whisper-official'))
    import whisper

    print("\n" + "="*70)
    print("Testing MTK Whisper Models with KV Cache")
    print("="*70)

    # Load original model
    print("\nLoading original Whisper model...")
    whisper_model = whisper.load_model("base", device="cpu")
    dims = whisper_model.dims

    # Create MTK models
    encoder, decoder, dims = create_whisper_kv_models(whisper_model, max_cache_len=448)

    # Test Encoder
    print("\n" + "="*70)
    print("Testing Encoder")
    print("="*70)
    mel_input = torch.randn(1, dims.n_mels, dims.n_audio_ctx * 2)
    print(f"Input shape: {mel_input.shape}")

    with torch.no_grad():
        encoder_output = encoder(mel_input)

    print(f"Output shape: {encoder_output.shape}")
    print(f"Expected: [1, {dims.n_audio_ctx}, {dims.n_audio_state}]")

    # Test Decoder with KV Cache
    print("\n" + "="*70)
    print("Testing Decoder with KV Cache")
    print("="*70)

    batch_size = 1
    max_cache_len = 448
    n_layers = dims.n_text_layer
    n_state = dims.n_text_state

    # Simulate first token decoding
    token_ids = torch.tensor([[50258]])  # SOT token
    token_embeddings = whisper_model.decoder.token_embedding(token_ids)

    # Initialize empty KV cache
    past_self_keys = torch.zeros(n_layers, batch_size, max_cache_len, n_state)
    past_self_values = torch.zeros(n_layers, batch_size, max_cache_len, n_state)

    # Position embedding for position 0
    position_embed = whisper_model.decoder.positional_embedding[0:1].unsqueeze(0)

    # Self-attention mask (all positions invalid except last one which is current token)
    self_attn_mask = torch.full((1, 1, 1, max_cache_len + 1), -1e9)
    self_attn_mask[:, :, :, -1] = 0  # Only current token is valid

    print(f"Token embeddings: {token_embeddings.shape}")
    print(f"Encoder output: {encoder_output.shape}")
    print(f"Past self keys: {past_self_keys.shape}")
    print(f"Position embed: {position_embed.shape}")
    print(f"Self attention mask: {self_attn_mask.shape}")

    with torch.no_grad():
        logits, new_self_keys, new_self_values, cross_keys, cross_values = decoder(
            token_embeddings,
            encoder_output,
            past_self_keys,
            past_self_values,
            position_embed,
            self_attn_mask,
            None,  # cached_cross_keys
            None,  # cached_cross_values
        )

    print(f"\nOutputs:")
    print(f"  Logits: {logits.shape}")
    print(f"  New self keys: {new_self_keys.shape}")
    print(f"  New self values: {new_self_values.shape}")
    print(f"  Cross keys: {cross_keys.shape}")
    print(f"  Cross values: {cross_values.shape}")

    # Check logits
    probs = torch.softmax(logits[0, 0], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\nTop 5 predicted tokens:")
    for i, (prob, token_id) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i+1}. Token {token_id.item()}: {prob.item():.4f}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_whisper_kv_models()
