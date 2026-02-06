#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1: 导出Whisper TorchScript模型

任务：
1. 加载原始Whisper base.pt模型
2. 创建MTK优化的Encoder和Decoder
3. 分别导出为TorchScript格式
4. 导出Embedding权重供C++使用
5. 保存元数据

关键修改：
- Encoder: 保持原结构，Position embedding作为buffer
- Decoder: 删除Embedding层，输入改为embeddings
- 导出token_embedding.npy供C++手动查表
"""

import argparse
import os
import sys
import json
import torch
import time
import numpy as np
from pathlib import Path

# 添加whisper官方库路径
sys.path.append('/home/xh/projects/MTK/whisper/whisper-official')
import whisper

# 导入我们的MTK模型
from whisper_model import (
    create_whisper_mtk_models,
    export_embedding_weights
)


def export_torchscript(
    checkpoint_path: str,
    output_dir: str,
    encoder_input_frames: int = 3000,  # 30秒音频对应3000帧
    decoder_seq_len: int = 448,
):
    """
    导出Whisper TorchScript模型

    Args:
        checkpoint_path: 原始.pt模型路径（base.pt）
        output_dir: 输出目录
        encoder_input_frames: Encoder输入帧数（30s音频=3000帧）
        decoder_seq_len: Decoder序列长度
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("步骤1: Whisper Base - PyTorch → TorchScript")
    print("="*70)
    print(f"源模型:   {checkpoint_path}")
    print(f"输出目录: {output_dir}")
    print(f"Encoder输入: {encoder_input_frames} frames (30s audio)")
    print(f"Decoder序列: {decoder_seq_len} tokens")
    print("="*70)

    # =========================================================================
    # 1. 加载原始Whisper模型
    # =========================================================================
    print("\n[1/7] 加载原始Whisper模型...")
    start = time.time()

    # 从checkpoint路径提取模型目录
    model_dir = os.path.dirname(checkpoint_path)

    # 加载模型（强制CPU）
    whisper_model = whisper.load_model("base", download_root=model_dir, device="cpu")
    whisper_model.eval()
    dims = whisper_model.dims

    print(f"  ✓ 模型加载成功 ({time.time() - start:.1f}s)")
    print(f"  参数量: {sum(p.numel() for p in whisper_model.parameters()) / 1e6:.1f}M")
    print(f"  Encoder: {dims.n_audio_layer} layers, {dims.n_audio_state} dim")
    print(f"  Decoder: {dims.n_text_layer} layers, {dims.n_text_state} dim")

    # =========================================================================
    # 2. 创建MTK优化模型
    # =========================================================================
    print("\n[2/7] 创建MTK优化模型...")
    start = time.time()

    encoder, decoder, dims = create_whisper_mtk_models(whisper_model)

    print(f"  ✓ 完成 ({time.time() - start:.1f}s)")

    # 清理内存
    del whisper_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================================
    # 3. 验证Encoder
    # =========================================================================
    print("\n[3/7] 验证Encoder...")
    start = time.time()

    # Encoder输入: [batch, n_mels, frames]
    # 30s音频 -> 3000帧 -> Conv2(stride=2) -> 1500帧
    dummy_mel = torch.randn(1, dims.n_mels, encoder_input_frames)

    with torch.no_grad():
        encoder_output = encoder(dummy_mel)

    expected_frames = encoder_input_frames // 2  # Conv2 stride=2
    print(f"  输入: {list(dummy_mel.shape)}")
    print(f"  输出: {list(encoder_output.shape)}")
    print(f"  期望: [1, {expected_frames}, {dims.n_audio_state}]")

    assert encoder_output.shape == (1, expected_frames, dims.n_audio_state), \
        f"Encoder输出形状错误: {encoder_output.shape}"

    print(f"  ✓ 验证通过 ({time.time() - start:.1f}s)")

    # =========================================================================
    # 4. 验证Decoder
    # =========================================================================
    print("\n[4/7] 验证Decoder...")
    start = time.time()

    # Decoder输入: 已经查表的embeddings [batch, seq_len, n_state]
    dummy_embeddings = torch.randn(1, 4, dims.n_text_state)

    with torch.no_grad():
        logits = decoder(dummy_embeddings, encoder_output)

    print(f"  Embeddings输入: {list(dummy_embeddings.shape)}")
    print(f"  Encoder特征: {list(encoder_output.shape)}")
    print(f"  Logits输出: {list(logits.shape)}")
    print(f"  期望: [1, 4, {dims.n_vocab}]")

    assert logits.shape == (1, 4, dims.n_vocab), \
        f"Decoder输出形状错误: {logits.shape}"

    print(f"  ✓ 验证通过 ({time.time() - start:.1f}s)")

    # =========================================================================
    # 5. 导出TorchScript
    # =========================================================================
    print("\n[5/7] 导出TorchScript...")
    start = time.time()

    # 5.1 导出Encoder
    print("\n  [5.1] Encoder...")
    encoder_pt_path = os.path.join(output_dir, f"encoder_base_{encoder_input_frames}.pt")

    # Trace Encoder
    traced_encoder = torch.jit.trace(encoder, dummy_mel)
    traced_encoder.save(encoder_pt_path)

    encoder_size_mb = os.path.getsize(encoder_pt_path) / 1024 / 1024
    print(f"    ✓ {os.path.basename(encoder_pt_path)}")
    print(f"      大小: {encoder_size_mb:.1f} MB")

    # 5.2 导出Decoder
    print("\n  [5.2] Decoder...")
    decoder_pt_path = os.path.join(output_dir, f"decoder_base_{decoder_seq_len}.pt")

    # Decoder需要两个输入：embeddings和encoder_output
    # 使用最大序列长度trace
    dummy_embeddings_full = torch.randn(1, decoder_seq_len, dims.n_text_state)

    # Trace Decoder
    traced_decoder = torch.jit.trace(
        decoder,
        (dummy_embeddings_full, encoder_output)
    )
    traced_decoder.save(decoder_pt_path)

    decoder_size_mb = os.path.getsize(decoder_pt_path) / 1024 / 1024
    print(f"    ✓ {os.path.basename(decoder_pt_path)}")
    print(f"      大小: {decoder_size_mb:.1f} MB")

    print(f"\n  ✓ TorchScript导出完成 ({time.time() - start:.1f}s)")

    # =========================================================================
    # 6. 导出Embedding权重
    # =========================================================================
    print("\n[6/7] 导出Embedding权重...")
    start = time.time()

    # 重新加载原始模型以获取embedding权重
    whisper_model_for_emb = whisper.load_model(
        "base",
        download_root=model_dir,
        device="cpu"
    )

    token_emb_path, emb_info_path = export_embedding_weights(
        whisper_model_for_emb,
        output_dir
    )

    del whisper_model_for_emb

    print(f"  ✓ 完成 ({time.time() - start:.1f}s)")

    # =========================================================================
    # 7. 保存元数据
    # =========================================================================
    print("\n[7/7] 保存模型元数据...")
    start = time.time()

    metadata = {
        'model_name': 'whisper-base',
        'model_type': 'encoder-decoder',
        'audio_duration_sec': 30,
        'encoder': {
            'input_shape': [1, dims.n_mels, encoder_input_frames],
            'output_shape': [1, encoder_input_frames // 2, dims.n_audio_state],
            'n_mels': dims.n_mels,
            'n_ctx': dims.n_audio_ctx,  # conv后的长度
            'n_state': dims.n_audio_state,
            'n_head': dims.n_audio_head,
            'n_layer': dims.n_audio_layer,
            'file': os.path.basename(encoder_pt_path)
        },
        'decoder': {
            'input_embedding_shape': [1, decoder_seq_len, dims.n_text_state],
            'input_encoder_shape': [1, encoder_input_frames // 2, dims.n_audio_state],
            'output_shape': [1, decoder_seq_len, dims.n_vocab],
            'n_vocab': dims.n_vocab,
            'n_ctx': dims.n_text_ctx,
            'n_state': dims.n_text_state,
            'n_head': dims.n_text_head,
            'n_layer': dims.n_text_layer,
            'file': os.path.basename(decoder_pt_path)
        },
        'embedding': {
            'vocab_size': dims.n_vocab,
            'embedding_dim': dims.n_text_state,
            'token_embedding_file': os.path.basename(token_emb_path),
            'info_file': os.path.basename(emb_info_path)
        },
        'files': {
            'encoder_torchscript': os.path.basename(encoder_pt_path),
            'decoder_torchscript': os.path.basename(decoder_pt_path),
            'token_embedding': os.path.basename(token_emb_path),
            'embedding_info': os.path.basename(emb_info_path)
        },
        'notes': [
            'Encoder输入为mel-spectrogram，30秒音频对应3000帧',
            'Decoder输入为embeddings（已查表），不接受token IDs',
            'Token embedding需要在C++端手动实现（GATHER算子不支持）',
            'Conv2 stride=2将encoder输入从3000帧降采样到1500帧',
            '无KV cache实现（简化版本）'
        ]
    }

    metadata_path = os.path.join(output_dir, "whisper_base_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {os.path.basename(metadata_path)}")
    print(f"  完成 ({time.time() - start:.1f}s)")

    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "="*70)
    print("✓ 步骤1完成！")
    print("="*70)
    print("\n生成的文件:")
    print(f"  1. Encoder TorchScript:  {os.path.basename(encoder_pt_path)} ({encoder_size_mb:.1f} MB)")
    print(f"  2. Decoder TorchScript:  {os.path.basename(decoder_pt_path)} ({decoder_size_mb:.1f} MB)")
    print(f"  3. Token Embedding:      {os.path.basename(token_emb_path)}")
    print(f"  4. Embedding Info:       {os.path.basename(emb_info_path)}")
    print(f"  5. Model Metadata:       {os.path.basename(metadata_path)}")

    print("\n重要说明:")
    print("  • Encoder输入: mel-spectrogram [1, 80, 3000] (30秒音频)")
    print("  • Decoder输入: embeddings [1, seq_len, 512] (已查表)")
    print("  • Token embedding需要在C++端手动查表")
    print("  • 无KV cache（简化实现）")

    print("\n下一步:")
    print(f"  python step2_torchscript_to_tflite.py \\")
    print(f"    --encoder_pt {encoder_pt_path} \\")
    print(f"    --decoder_pt {decoder_pt_path} \\")
    print(f"    --output_dir {output_dir}")

    return {
        'encoder_pt': encoder_pt_path,
        'decoder_pt': decoder_pt_path,
        'token_embedding': token_emb_path,
        'embedding_info': emb_info_path,
        'metadata': metadata_path
    }


def main():
    parser = argparse.ArgumentParser(
        description='步骤1: Whisper PyTorch → TorchScript',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step1_pt_to_torchscript.py \\
    --checkpoint ../models/base.pt \\
    --output_dir ./models \\
    --encoder_frames 3000 \\
    --decoder_seq_len 448
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='原始Whisper .pt模型路径（例如: ../models/base.pt）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models',
        help='输出目录（默认: ./models）'
    )
    parser.add_argument(
        '--encoder_frames',
        type=int,
        default=3000,
        help='Encoder输入帧数，30秒音频=3000帧（默认: 3000）'
    )
    parser.add_argument(
        '--decoder_seq_len',
        type=int,
        default=448,
        help='Decoder最大序列长度（默认: 448）'
    )

    args = parser.parse_args()

    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 找不到checkpoint文件: {args.checkpoint}")
        sys.exit(1)

    # 执行转换
    export_torchscript(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        encoder_input_frames=args.encoder_frames,
        decoder_seq_len=args.decoder_seq_len
    )


if __name__ == '__main__':
    main()
