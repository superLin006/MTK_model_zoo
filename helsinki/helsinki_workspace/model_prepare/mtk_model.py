#!/usr/bin/env python3
"""
Helsinki Translation Model with KV Cache for MTK NPU - V2
修复 MT8371 不支持 5D tensor 的问题

解决方案:
- 将 KV Cache reshape 为 4D: [num_layers * num_heads, batch, seq, head_dim]
- 或者更简单: [num_layers, batch, seq, num_heads * head_dim]

这里采用第二种方案，将 KV 在 hidden_dim 维度合并:
- past_key: [num_layers, batch, max_cache_len, d_model]  (合并 heads 和 head_dim)
- past_value: [num_layers, batch, max_cache_len, d_model]

这样每层的 K/V 都是 [batch, seq, d_model] 的 3D tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import MarianMTModel, MarianTokenizer, MarianConfig
from typing import Optional, Tuple, List


# ============================================================
# Encoder (与 v1 相同)
# ============================================================

class MTKEncoderAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query * self.scaling, key.transpose(-2, -1))

        # Apply attention mask if provided (0 for valid, -1e9 for invalid)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class MTKEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, max_seq_len=512):
        super().__init__()
        self.self_attn = MTKEncoderAttention(d_model, num_heads, max_seq_len)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attn_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class MTKSinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal position embeddings compatible with MarianMT

    MarianMT uses a different format:
    - First half: sin values
    - Second half: cos values
    (Not interleaved like standard transformer)
    """
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        # Generate Marian-style position embeddings
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # First half: sin, Second half: cos (Marian style)
        pe[:, :half_dim] = torch.sin(position * emb)
        pe[:, half_dim:] = torch.cos(position * emb)

        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len, :]

    def get_position(self, position: int) -> torch.Tensor:
        return self.pe[:, position:position+1, :]

    def load_from_hf(self, hf_pos_embed):
        """Load position embeddings from HuggingFace model"""
        # MarianSinusoidalPositionalEmbedding stores weight as [max_position, d_model]
        with torch.no_grad():
            self.pe.copy_(hf_pos_embed.weight.unsqueeze(0))


class MTKEncoderNoEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_positions = MTKSinusoidalPositionalEmbedding(config.d_model, config.max_position_embeddings)
        self.layers = nn.ModuleList([
            MTKEncoderLayer(config.d_model, config.encoder_attention_heads,
                          config.encoder_ffn_dim, config.max_position_embeddings)
            for _ in range(config.encoder_layers)
        ])

    def forward(self, encoder_embeds: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = encoder_embeds.shape[1]
        hidden_states = encoder_embeds * self.embed_scale
        hidden_states = hidden_states + self.embed_positions(seq_len)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)

        return hidden_states


# ============================================================
# Decoder with KV Cache (V2 - 4D tensors)
# ============================================================

class MTKDecoderSelfAttentionKVCacheV2(nn.Module):
    """
    Self-Attention with KV Cache using 4D tensors

    KV Cache 格式: [batch, max_cache_len, d_model]
    内部会 reshape 为 [batch, heads, seq, head_dim]

    输入:
        hidden_states: [batch, 1, d_model]
        past_key: [batch, max_cache_len, d_model]  - 4D friendly
        past_value: [batch, max_cache_len, d_model]
        attn_mask: [1, 1, 1, max_cache_len+1] - attention mask (0 for valid, -inf for invalid)

    输出:
        attn_output: [batch, 1, d_model]
        new_key: [batch, 1, d_model]  - 当前token的K
        new_value: [batch, 1, d_model]  - 当前token的V
    """
    def __init__(self, d_model, num_heads, max_cache_len=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, 1, d_model]
        past_key: torch.Tensor,       # [batch, max_cache_len, d_model]
        past_value: torch.Tensor,     # [batch, max_cache_len, d_model]
        attn_mask: torch.Tensor,      # [1, 1, 1, max_cache_len+1] - 0 for valid, -1e9 for invalid
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]

        # 计算当前 token 的 Q, K, V
        query = self.q_proj(hidden_states)  # [batch, 1, d_model]
        new_key = self.k_proj(hidden_states)    # [batch, 1, d_model]
        new_value = self.v_proj(hidden_states)  # [batch, 1, d_model]

        # Reshape query: [batch, 1, d_model] -> [batch, heads, 1, head_dim]
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 拼接历史 KV 和当前 KV
        # past_key: [batch, max_cache_len, d_model]
        # new_key: [batch, 1, d_model]
        # concat: [batch, max_cache_len+1, d_model]
        full_key = torch.cat([past_key, new_key], dim=1)
        full_value = torch.cat([past_value, new_value], dim=1)

        # Reshape to attention format: [batch, heads, seq, head_dim]
        kv_seq_len = full_key.shape[1]  # max_cache_len + 1
        full_key = full_key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        full_value = full_value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        # query: [batch, heads, 1, head_dim]
        # full_key: [batch, heads, max_cache_len+1, head_dim]
        attn_weights = torch.matmul(query * self.scaling, full_key.transpose(-2, -1))
        # attn_weights: [batch, heads, 1, max_cache_len+1]

        # Apply attention mask: mask invalid cache positions
        # attn_mask: [1, 1, 1, max_cache_len+1], 0 for valid, -1e9 for invalid
        attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, full_value)
        # attn_output: [batch, heads, 1, head_dim]

        # Reshape back: [batch, 1, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output, new_key, new_value


class MTKDecoderCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attn_mask: Optional[torch.Tensor] = None,  # [1, 1, 1, src_seq_len]
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        src_seq_len = encoder_hidden_states.shape[1]

        query = self.q_proj(hidden_states)
        key = self.k_proj(encoder_hidden_states)
        value = self.v_proj(encoder_hidden_states)

        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, src_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, src_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query * self.scaling, key.transpose(-2, -1))

        # Apply encoder attention mask if provided
        if encoder_attn_mask is not None:
            attn_weights = attn_weights + encoder_attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        return self.out_proj(attn_output)


class MTKDecoderLayerKVCacheV2(nn.Module):
    """单层 Decoder with KV Cache"""
    def __init__(self, d_model, num_heads, ffn_dim, max_cache_len=64):
        super().__init__()
        self.self_attn = MTKDecoderSelfAttentionKVCacheV2(d_model, num_heads, max_cache_len)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = MTKDecoderCrossAttention(d_model, num_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.activation_fn = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        attn_mask: torch.Tensor,
        encoder_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self attention
        residual = hidden_states
        hidden_states, new_key, new_value = self.self_attn(
            hidden_states, past_key, past_value, attn_mask
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross attention
        residual = hidden_states
        hidden_states = self.encoder_attn(hidden_states, encoder_hidden_states, encoder_attn_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, new_key, new_value


class MTKDecoderKVCacheV2(nn.Module):
    """
    Decoder with KV Cache - V2 (4D tensors)

    输入:
        decoder_embed: [batch, 1, d_model]
        encoder_hidden_states: [batch, src_seq, d_model]
        past_key_0 ~ past_key_5: [batch, max_cache_len, d_model]  (6层，每层单独输入)
        past_value_0 ~ past_value_5: [batch, max_cache_len, d_model]
        cache_len: [1] int32

    输出:
        logits: [batch, 1, vocab_size]
        new_key_0 ~ new_key_5: [batch, 1, d_model]
        new_value_0 ~ new_value_5: [batch, 1, d_model]

    为了简化 NPU 接口，使用单个合并的 tensor:
        past_keys: [num_layers, batch, max_cache_len, d_model]  - 4D
        past_values: [num_layers, batch, max_cache_len, d_model]  - 4D
        new_keys: [num_layers, batch, 1, d_model]  - 4D
        new_values: [num_layers, batch, 1, d_model]  - 4D
    """
    def __init__(self, config, max_cache_len=64):
        super().__init__()
        self.config = config
        self.num_layers = config.decoder_layers
        self.d_model = config.d_model
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.max_cache_len = max_cache_len

        self.embed_positions = MTKSinusoidalPositionalEmbedding(
            config.d_model, config.max_position_embeddings
        )

        self.layers = nn.ModuleList([
            MTKDecoderLayerKVCacheV2(
                config.d_model, config.decoder_attention_heads,
                config.decoder_ffn_dim, max_cache_len
            )
            for _ in range(config.decoder_layers)
        ])

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        decoder_embed: torch.Tensor,          # [batch, 1, d_model]
        encoder_hidden_states: torch.Tensor,  # [batch, src_seq, d_model]
        past_keys: torch.Tensor,              # [num_layers, batch, max_cache_len, d_model]
        past_values: torch.Tensor,            # [num_layers, batch, max_cache_len, d_model]
        cache_len: torch.Tensor,              # [1] int32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 获取当前位置的位置编码
        position = cache_len[0].item() if not self.training else cache_len[0]
        # 使用 slice 而不是 item() 来保持 trace 兼容性
        pos_embed = self.embed_positions.pe[:, position:position+1, :]

        hidden_states = decoder_embed * self.embed_scale + pos_embed

        new_keys_list = []
        new_values_list = []

        for i, layer in enumerate(self.layers):
            layer_past_key = past_keys[i]    # [batch, max_cache_len, d_model]
            layer_past_value = past_values[i]

            hidden_states, new_key, new_value = layer(
                hidden_states,
                encoder_hidden_states,
                layer_past_key,
                layer_past_value,
                cache_len
            )

            new_keys_list.append(new_key)      # [batch, 1, d_model]
            new_values_list.append(new_value)

        # Stack: [num_layers, batch, 1, d_model]
        new_keys = torch.stack(new_keys_list, dim=0)
        new_values = torch.stack(new_values_list, dim=0)

        # LM Head
        logits = self.lm_head(hidden_states)

        return logits, new_keys, new_values


# ============================================================
# 为了避免 trace 时的动态索引问题，创建固定位置的版本
# ============================================================

class MTKDecoderKVCacheV2Fixed(nn.Module):
    """
    固定位置编码版本 - 位置通过输入 tensor 传入
    attention mask 也通过输入 tensor 传入
    encoder_attn_mask 用于 cross-attention，mask 掉 encoder 的 padding 位置
    """
    def __init__(self, config, max_cache_len=64):
        super().__init__()
        self.config = config
        self.num_layers = config.decoder_layers
        self.d_model = config.d_model
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.max_cache_len = max_cache_len

        # 预计算所有位置编码
        self.embed_positions = MTKSinusoidalPositionalEmbedding(
            config.d_model, config.max_position_embeddings
        )

        self.layers = nn.ModuleList([
            MTKDecoderLayerKVCacheV2(
                config.d_model, config.decoder_attention_heads,
                config.decoder_ffn_dim, max_cache_len
            )
            for _ in range(config.decoder_layers)
        ])

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # final_logits_bias - important for correct predictions!
        self.register_buffer('final_logits_bias', torch.zeros(1, config.vocab_size))

    def forward(
        self,
        decoder_embed: torch.Tensor,          # [batch, 1, d_model]
        encoder_hidden_states: torch.Tensor,  # [batch, src_seq, d_model]
        past_keys: torch.Tensor,              # [num_layers, batch, max_cache_len, d_model]
        past_values: torch.Tensor,            # [num_layers, batch, max_cache_len, d_model]
        position_embed: torch.Tensor,         # [1, 1, d_model] - 当前位置的位置编码
        attn_mask: torch.Tensor,              # [1, 1, 1, max_cache_len+1] - self-attention mask
        encoder_attn_mask: torch.Tensor,      # [1, 1, 1, src_seq] - encoder attention mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hidden_states = decoder_embed * self.embed_scale + position_embed

        new_keys_list = []
        new_values_list = []

        for i, layer in enumerate(self.layers):
            layer_past_key = past_keys[i]
            layer_past_value = past_values[i]

            hidden_states, new_key, new_value = layer(
                hidden_states,
                encoder_hidden_states,
                layer_past_key,
                layer_past_value,
                attn_mask,
                encoder_attn_mask
            )

            new_keys_list.append(new_key)
            new_values_list.append(new_value)

        new_keys = torch.stack(new_keys_list, dim=0)
        new_values = torch.stack(new_values_list, dim=0)

        logits = self.lm_head(hidden_states) + self.final_logits_bias

        return logits, new_keys, new_values


# ============================================================
# Weight Loading (与 v1 相同)
# ============================================================

def load_encoder_weights(mtk_encoder: MTKEncoderNoEmbed, hf_model: MarianMTModel):
    print("Loading encoder weights...")
    # Load position embeddings from original model
    mtk_encoder.embed_positions.load_from_hf(hf_model.model.encoder.embed_positions)

    for i, (mtk_layer, hf_layer) in enumerate(zip(mtk_encoder.layers, hf_model.model.encoder.layers)):
        mtk_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        mtk_layer.self_attn.q_proj.bias.data.copy_(hf_layer.self_attn.q_proj.bias.data)
        mtk_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        mtk_layer.self_attn.k_proj.bias.data.copy_(hf_layer.self_attn.k_proj.bias.data)
        mtk_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        mtk_layer.self_attn.v_proj.bias.data.copy_(hf_layer.self_attn.v_proj.bias.data)
        mtk_layer.self_attn.out_proj.weight.data.copy_(hf_layer.self_attn.out_proj.weight.data)
        mtk_layer.self_attn.out_proj.bias.data.copy_(hf_layer.self_attn.out_proj.bias.data)
        mtk_layer.self_attn_layer_norm.weight.data.copy_(hf_layer.self_attn_layer_norm.weight.data)
        mtk_layer.self_attn_layer_norm.bias.data.copy_(hf_layer.self_attn_layer_norm.bias.data)
        mtk_layer.fc1.weight.data.copy_(hf_layer.fc1.weight.data)
        mtk_layer.fc1.bias.data.copy_(hf_layer.fc1.bias.data)
        mtk_layer.fc2.weight.data.copy_(hf_layer.fc2.weight.data)
        mtk_layer.fc2.bias.data.copy_(hf_layer.fc2.bias.data)
        mtk_layer.final_layer_norm.weight.data.copy_(hf_layer.final_layer_norm.weight.data)
        mtk_layer.final_layer_norm.bias.data.copy_(hf_layer.final_layer_norm.bias.data)
    print("Encoder weights loaded!")


def load_decoder_weights(mtk_decoder, hf_model: MarianMTModel):
    print("Loading decoder weights...")
    # Load position embeddings from original model
    mtk_decoder.embed_positions.load_from_hf(hf_model.model.decoder.embed_positions)

    for i, (mtk_layer, hf_layer) in enumerate(zip(mtk_decoder.layers, hf_model.model.decoder.layers)):
        mtk_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        mtk_layer.self_attn.q_proj.bias.data.copy_(hf_layer.self_attn.q_proj.bias.data)
        mtk_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        mtk_layer.self_attn.k_proj.bias.data.copy_(hf_layer.self_attn.k_proj.bias.data)
        mtk_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        mtk_layer.self_attn.v_proj.bias.data.copy_(hf_layer.self_attn.v_proj.bias.data)
        mtk_layer.self_attn.out_proj.weight.data.copy_(hf_layer.self_attn.out_proj.weight.data)
        mtk_layer.self_attn.out_proj.bias.data.copy_(hf_layer.self_attn.out_proj.bias.data)
        mtk_layer.self_attn_layer_norm.weight.data.copy_(hf_layer.self_attn_layer_norm.weight.data)
        mtk_layer.self_attn_layer_norm.bias.data.copy_(hf_layer.self_attn_layer_norm.bias.data)

        mtk_layer.encoder_attn.q_proj.weight.data.copy_(hf_layer.encoder_attn.q_proj.weight.data)
        mtk_layer.encoder_attn.q_proj.bias.data.copy_(hf_layer.encoder_attn.q_proj.bias.data)
        mtk_layer.encoder_attn.k_proj.weight.data.copy_(hf_layer.encoder_attn.k_proj.weight.data)
        mtk_layer.encoder_attn.k_proj.bias.data.copy_(hf_layer.encoder_attn.k_proj.bias.data)
        mtk_layer.encoder_attn.v_proj.weight.data.copy_(hf_layer.encoder_attn.v_proj.weight.data)
        mtk_layer.encoder_attn.v_proj.bias.data.copy_(hf_layer.encoder_attn.v_proj.bias.data)
        mtk_layer.encoder_attn.out_proj.weight.data.copy_(hf_layer.encoder_attn.out_proj.weight.data)
        mtk_layer.encoder_attn.out_proj.bias.data.copy_(hf_layer.encoder_attn.out_proj.bias.data)
        mtk_layer.encoder_attn_layer_norm.weight.data.copy_(hf_layer.encoder_attn_layer_norm.weight.data)
        mtk_layer.encoder_attn_layer_norm.bias.data.copy_(hf_layer.encoder_attn_layer_norm.bias.data)

        mtk_layer.fc1.weight.data.copy_(hf_layer.fc1.weight.data)
        mtk_layer.fc1.bias.data.copy_(hf_layer.fc1.bias.data)
        mtk_layer.fc2.weight.data.copy_(hf_layer.fc2.weight.data)
        mtk_layer.fc2.bias.data.copy_(hf_layer.fc2.bias.data)
        mtk_layer.final_layer_norm.weight.data.copy_(hf_layer.final_layer_norm.weight.data)
        mtk_layer.final_layer_norm.bias.data.copy_(hf_layer.final_layer_norm.bias.data)

    mtk_decoder.lm_head.weight.data.copy_(hf_model.model.shared.weight.data)

    # Copy final_logits_bias - this is critical for correct predictions!
    mtk_decoder.final_logits_bias.copy_(hf_model.final_logits_bias)
    print("Decoder weights loaded!")


def create_encoder_decoder_kvcache_v2(model_path: str, max_cache_len: int = 64):
    """创建 V2 版本的模型"""
    print(f"Loading model from: {model_path}")

    hf_model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    config = hf_model.config

    print(f"Creating Encoder...")
    encoder = MTKEncoderNoEmbed(config)
    load_encoder_weights(encoder, hf_model)

    print(f"Creating Decoder V2 (Fixed position, 4D KV Cache)...")
    decoder = MTKDecoderKVCacheV2Fixed(config, max_cache_len)
    load_decoder_weights(decoder, hf_model)

    embedding_weights = hf_model.model.shared.weight.data.clone()

    # 导出位置编码表 (供 C++ 使用)
    position_embeddings = decoder.embed_positions.pe.clone()  # [1, max_pos, d_model]

    print(f"Models created!")
    print(f"  - Encoder params: {sum(p.numel() for p in encoder.parameters())}")
    print(f"  - Decoder params: {sum(p.numel() for p in decoder.parameters())}")

    return encoder, decoder, embedding_weights, position_embeddings, tokenizer, config


# ============================================================
# Test
# ============================================================

def create_attn_mask(cache_len: int, max_cache_len: int) -> torch.Tensor:
    """
    Create attention mask for KV cache (self-attention).

    For cache_len=0, only the current (last) position is valid.
    For cache_len=n, positions 0..n-1 in cache are valid, plus the current position.

    Returns: [1, 1, 1, max_cache_len+1] tensor with 0 for valid, -1e9 for invalid
    """
    mask = torch.full((1, 1, 1, max_cache_len + 1), -1e9)
    # Valid positions: [0, cache_len) in past cache, plus the last position (current)
    mask[:, :, :, :cache_len] = 0  # past valid positions
    mask[:, :, :, -1] = 0  # current position (always valid)
    return mask


def create_encoder_attn_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Create encoder attention mask for cross-attention.

    Input attention_mask: [batch, src_seq] with 1 for valid, 0 for padding
    Returns: [1, 1, 1, src_seq] tensor with 0 for valid, -1e9 for padding
    """
    # Convert from 1=valid/0=pad to 0=valid/-1e9=pad
    mask = (1.0 - attention_mask.float()) * -1e9
    return mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, src_seq]


def test_kvcache_v2():
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../models/Helsinki-NLP/opus-mt-en-zh"

    print("="*60)
    print("Testing KV Cache V2 Model")
    print("="*60)

    max_cache_len = 64
    src_seq_len = 64
    encoder, decoder, embedding_weights, position_embeddings, tokenizer, config = \
        create_encoder_decoder_kvcache_v2(model_path, max_cache_len)

    encoder.eval()
    decoder.eval()

    d_model = config.d_model
    num_layers = config.decoder_layers

    print(f"\n=== Encoder Test ===")
    encoder_embeds = torch.randn(1, src_seq_len, d_model)
    with torch.no_grad():
        encoder_output = encoder(encoder_embeds)
    print(f"Input: {encoder_embeds.shape}")
    print(f"Output: {encoder_output.shape}")

    print(f"\n=== Decoder Test (KV Cache V2) ===")
    # 初始化 KV Cache: [num_layers, batch, max_cache_len, d_model]
    past_keys = torch.zeros(num_layers, 1, max_cache_len, d_model)
    past_values = torch.zeros(num_layers, 1, max_cache_len, d_model)

    # Create encoder attention mask (assuming all positions valid for this test)
    encoder_attn_mask = torch.zeros(1, 1, 1, src_seq_len)  # All valid

    # 模拟生成 3 个 token
    for step in range(3):
        decoder_embed = torch.randn(1, 1, d_model)
        position_embed = position_embeddings[:, step:step+1, :]  # [1, 1, d_model]
        attn_mask = create_attn_mask(step, max_cache_len)  # [1, 1, 1, 65]

        with torch.no_grad():
            logits, new_keys, new_values = decoder(
                decoder_embed, encoder_output, past_keys, past_values,
                position_embed, attn_mask, encoder_attn_mask
            )

        print(f"Step {step}:")
        print(f"  past_keys shape: {past_keys.shape}")
        print(f"  logits: {logits.shape}")
        print(f"  new_keys: {new_keys.shape}")
        print(f"  new_values: {new_values.shape}")

        # 更新 KV Cache (在实际 C++ 中完成)
        past_keys[:, :, step:step+1, :] = new_keys
        past_values[:, :, step:step+1, :] = new_values

        next_token = logits[0, 0].argmax().item()
        print(f"  predicted: {next_token}")

    print("\n=== Test Passed! ===")

    return encoder, decoder, embedding_weights, position_embeddings, tokenizer, config


if __name__ == "__main__":
    test_kvcache_v2()
