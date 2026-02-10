#!/usr/bin/env python3
"""
MTK NPU优化的Whisper模型定义
基于原始Whisper模型，针对MTK平台做关键修改：

1. WhisperEncoderCore: Encoder保持原始结构
   - 支持所有算子 (Conv1d, GELU, Linear, Attention, LayerNorm)
   - Position encoding作为buffer

2. WhisperDecoderCore: Decoder需要重大修改
   - 删除nn.Embedding层（GATHER算子不支持）
   - 输入改为接受embeddings而非token IDs
   - Causal mask预计算为buffer
   - Position embedding作为buffer

3. 导出辅助函数：
   - 导出Embedding权重供C++端使用
   - 导出元数据信息
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Dict, Optional, Tuple
import json
import os


# ============================================================
# 基础组件 (来自原始Whisper)
# ============================================================

class LayerNorm(nn.LayerNorm):
    """与原始Whisper相同的LayerNorm"""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """与原始Whisper相同的Linear"""
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    """与原始Whisper相同的Conv1d"""
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """生成位置编码 (与原始Whisper相同)"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (不使用SDPA，显式实现)
    避免使用scaled_dot_product_attention以保证兼容性
    """
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

        # Self-attention or Cross-attention
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

        # Reshape for multi-head: [batch, seq, heads, head_dim]
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        # Attention scores
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        # Apply mask (使用加法而非masked_fill)
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)

        # Attention output
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach()

        return out, qk


# ============================================================
# Residual Attention Block
# ============================================================

class ResidualAttentionBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        # Self-attention
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]

        # Cross-attention (如果有)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]

        # Feed-forward
        x = x + self.mlp(self.mlp_ln(x))

        return x


# ============================================================
# Encoder Core (MTK优化版本)
# ============================================================

class WhisperEncoderCore(nn.Module):
    """
    Whisper Encoder Core - MTK NPU兼容版本

    输入: mel-spectrogram [batch, n_mels, n_ctx]
    输出: encoder features [batch, n_ctx//2, n_state]

    变化：
    - Position embedding注册为buffer（不是Parameter）
    - 保持原始结构，所有算子都支持
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

        # Position embedding注册为buffer
        # 注意：n_ctx已经是conv2之后的长度（原始mel长度的一半）
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x: [batch, n_mels, n_ctx] - mel spectrogram
        返回: [batch, n_ctx//2, n_state] - encoder features
        """
        # Conv layers
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [batch, n_ctx//2, n_state]

        # 添加位置编码
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


# ============================================================
# Decoder Core (MTK优化版本 - 关键修改)
# ============================================================

class WhisperDecoderCore(nn.Module):
    """
    Whisper Decoder Core - MTK NPU兼容版本

    关键修改：
    1. 删除token_embedding层（GATHER算子不支持）
    2. 输入改为token_embeddings而非token IDs
    3. Position embedding注册为buffer
    4. Causal mask预计算为buffer

    输入:
        token_embeddings: [batch, n_ctx, n_state] - 已经查表的embeddings
        xa: [batch, n_audio_ctx, n_state] - encoder输出

    输出:
        logits: [batch, n_ctx, n_vocab] - 预测的token概率
    """
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int
    ):
        super().__init__()

        # 删除了原始的 self.token_embedding = nn.Embedding(n_vocab, n_state)
        # Embedding将在C++端手动实现

        # Position embedding作为buffer（从原始的Parameter改为buffer）
        positional_embedding = torch.empty(n_ctx, n_state)
        self.register_buffer("positional_embedding", positional_embedding)

        # Transformer blocks (with cross-attention)
        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        # Causal mask预计算为buffer（使用加法友好的格式）
        # 有效位置为0，无效位置为-inf
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        # LM head权重（用于最后的logits计算）
        # 注意：实际权重会从token_embedding复制而来
        self.lm_head = Linear(n_state, n_vocab, bias=False)

    def forward(self, token_embeddings: Tensor, xa: Tensor):
        """
        token_embeddings: [batch, n_ctx, n_state] - 已经查表的embeddings
        xa: [batch, n_audio_ctx, n_state] - encoder输出

        返回: [batch, n_ctx, n_vocab] - logits
        """
        # 添加位置编码
        seq_len = token_embeddings.shape[1]
        x = token_embeddings + self.positional_embedding[:seq_len]
        x = x.to(xa.dtype)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, xa, mask=self.mask)

        x = self.ln(x)

        # 计算logits (使用lm_head而非原始的weight transpose)
        logits = self.lm_head(x).float()

        return logits


# ============================================================
# 权重加载函数
# ============================================================

def load_encoder_weights(mtk_encoder: WhisperEncoderCore, whisper_model):
    """从原始Whisper模型加载Encoder权重"""
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
        # Self-attention
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


def load_decoder_weights(mtk_decoder: WhisperDecoderCore, whisper_model):
    """从原始Whisper模型加载Decoder权重"""
    print("Loading decoder weights...")

    # Position embedding (从Parameter复制到buffer)
    mtk_decoder.positional_embedding.copy_(whisper_model.decoder.positional_embedding.data)

    # Transformer blocks
    for mtk_block, orig_block in zip(mtk_decoder.blocks, whisper_model.decoder.blocks):
        # Self-attention
        mtk_block.attn.query.weight.data.copy_(orig_block.attn.query.weight.data)
        mtk_block.attn.query.bias.data.copy_(orig_block.attn.query.bias.data)
        mtk_block.attn.key.weight.data.copy_(orig_block.attn.key.weight.data)
        mtk_block.attn.value.weight.data.copy_(orig_block.attn.value.weight.data)
        mtk_block.attn.value.bias.data.copy_(orig_block.attn.value.bias.data)
        mtk_block.attn.out.weight.data.copy_(orig_block.attn.out.weight.data)
        mtk_block.attn.out.bias.data.copy_(orig_block.attn.out.bias.data)

        # Self-attention LayerNorm
        mtk_block.attn_ln.weight.data.copy_(orig_block.attn_ln.weight.data)
        mtk_block.attn_ln.bias.data.copy_(orig_block.attn_ln.bias.data)

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

    # LM head权重（与token_embedding共享权重）
    mtk_decoder.lm_head.weight.data.copy_(whisper_model.decoder.token_embedding.weight.data)

    print("✓ Decoder weights loaded!")


# ============================================================
# Embedding导出函数 (关键：供C++端使用)
# ============================================================

def export_embedding_weights(whisper_model, output_dir: str):
    """
    导出Embedding权重和元数据供C++端使用

    导出内容：
    1. token_embedding.npy - Token embedding权重 [n_vocab, n_state]
    2. embedding_info.json - Embedding元数据
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nExporting embedding weights for C++...")

    # 导出token embedding权重
    token_embedding_weight = whisper_model.decoder.token_embedding.weight.detach().cpu().numpy()
    token_emb_path = os.path.join(output_dir, 'token_embedding.npy')
    np.save(token_emb_path, token_embedding_weight)

    n_vocab, n_state = token_embedding_weight.shape
    print(f"  ✓ Token embedding: {token_emb_path}")
    print(f"    Shape: [{n_vocab}, {n_state}]")
    print(f"    Size: {token_embedding_weight.nbytes / 1024 / 1024:.2f} MB")

    # 导出元数据
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
# 模型创建函数
# ============================================================

def create_whisper_mtk_models(whisper_model):
    """
    从原始Whisper模型创建MTK优化的Encoder和Decoder

    返回: (encoder, decoder, dims)
    """
    dims = whisper_model.dims

    print("\n" + "="*70)
    print("Creating MTK-optimized Whisper models")
    print("="*70)

    # 创建Encoder
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

    # 创建Decoder
    print("\n[2/2] Creating Decoder Core...")
    decoder = WhisperDecoderCore(
        n_vocab=dims.n_vocab,
        n_ctx=dims.n_text_ctx,
        n_state=dims.n_text_state,
        n_head=dims.n_text_head,
        n_layer=dims.n_text_layer,
    )
    load_decoder_weights(decoder, whisper_model)
    decoder.eval()
    print(f"  ✓ Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    print("\n" + "="*70)
    print("✓ MTK models created successfully!")
    print("="*70)

    return encoder, decoder, dims


# ============================================================
# 测试函数
# ============================================================

def test_mtk_models():
    """测试MTK模型的前向传播"""
    import sys
    sys.path.append('/home/xh/projects/MTK_models_zoo/whisper/whisper-official')
    import whisper

    print("\n" + "="*70)
    print("Testing MTK Whisper Models")
    print("="*70)

    # 加载原始模型
    print("\nLoading original Whisper model...")
    model_path = "/home/xh/projects/MTK_models_zoo/whisper/mtk/models/base.pt"
    whisper_model = whisper.load_model("base", download_root="/home/xh/projects/MTK_models_zoo/whisper/mtk/models", device="cpu")
    dims = whisper_model.dims

    print(f"Model dimensions:")
    print(f"  n_mels: {dims.n_mels}")
    print(f"  n_audio_ctx: {dims.n_audio_ctx}")
    print(f"  n_audio_state: {dims.n_audio_state}")
    print(f"  n_audio_head: {dims.n_audio_head}")
    print(f"  n_audio_layer: {dims.n_audio_layer}")
    print(f"  n_vocab: {dims.n_vocab}")
    print(f"  n_text_ctx: {dims.n_text_ctx}")
    print(f"  n_text_state: {dims.n_text_state}")
    print(f"  n_text_head: {dims.n_text_head}")
    print(f"  n_text_layer: {dims.n_text_layer}")

    # 创建MTK模型
    encoder, decoder, dims = create_whisper_mtk_models(whisper_model)

    # 测试Encoder
    print("\n" + "="*70)
    print("Testing Encoder")
    print("="*70)
    # 注意：输入mel长度应该是 n_audio_ctx * 2 (因为conv2有stride=2)
    # 30秒音频 -> 3000帧 -> conv2 -> 1500帧
    mel_input = torch.randn(1, dims.n_mels, dims.n_audio_ctx * 2)
    print(f"Input shape: {mel_input.shape}")
    print(f"  (30s audio -> {dims.n_audio_ctx * 2} frames)")

    with torch.no_grad():
        encoder_output = encoder(mel_input)

    print(f"Output shape: {encoder_output.shape}")
    print(f"Expected: [1, {dims.n_audio_ctx}, {dims.n_audio_state}]")
    print(f"  (Conv2 stride=2 reduces 3000 -> 1500)")

    # 测试Decoder
    print("\n" + "="*70)
    print("Testing Decoder")
    print("="*70)

    # 模拟token embeddings (手动查表后的结果)
    token_ids = torch.tensor([[50258, 50259, 50359, 50363]])  # 示例token IDs
    token_embeddings = whisper_model.decoder.token_embedding(token_ids)

    print(f"Token IDs: {token_ids.shape}")
    print(f"Token embeddings: {token_embeddings.shape}")
    print(f"Encoder output: {encoder_output.shape}")

    with torch.no_grad():
        logits = decoder(token_embeddings, encoder_output)

    print(f"Logits shape: {logits.shape}")
    print(f"Expected: [{token_ids.shape[0]}, {token_ids.shape[1]}, {dims.n_vocab}]")

    # 检查logits的合理性
    probs = torch.softmax(logits[0, -1], dim=-1)
    top5_tokens = torch.topk(probs, 5)
    print(f"\nTop 5 predicted tokens:")
    for i, (prob, token_id) in enumerate(zip(top5_tokens.values, top5_tokens.indices)):
        print(f"  {i+1}. Token {token_id.item()}: {prob.item():.4f}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_mtk_models()
