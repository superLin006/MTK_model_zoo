#!/usr/bin/env python3
"""
Step 1: Convert Whisper Base to TorchScript with KV Cache

This script:
1. Loads original OpenAI Whisper Base model
2. Creates MTK-optimized encoder and decoder with KV Cache
3. Transfers weights
4. Exports to TorchScript (.pt files)
5. Exports embedding weights (.npy)

Output:
- models/encoder.pt - TorchScript encoder
- models/decoder_kv.pt - TorchScript decoder with KV cache
- models/token_embedding.npy - Token embedding weights for C++
- models/embedding_info.json - Metadata
"""

import os
import sys
import torch
import numpy as np

# Add Whisper path
sys.path.append('/home/xh/projects/MTK_models_zoo/whisper/whisper-official')
import whisper

# Import MTK models
from whisper_kv_model import (
    create_whisper_kv_models,
    export_embedding_weights
)


def main():
    print("="*70)
    print("Whisper Base → TorchScript Conversion (with KV Cache)")
    print("="*70)

    # Configuration
    model_name = "base"
    models_dir = "models"
    max_cache_len = 448  # Maximum decoder cache length

    os.makedirs(models_dir, exist_ok=True)

    # Step 1: Load original Whisper model
    print("\n[Step 1/5] Loading original Whisper model...")
    whisper_model = whisper.load_model(
        model_name,
        download_root="/home/xh/projects/MTK_models_zoo/whisper/mtk/models",
        device="cpu"
    )
    whisper_model.eval()

    dims = whisper_model.dims
    print(f"Model: Whisper {model_name}")
    print(f"  Vocab size: {dims.n_vocab}")
    print(f"  Audio state: {dims.n_audio_state}")
    print(f"  Text state: {dims.n_text_state}")
    print(f"  Encoder layers: {dims.n_audio_layer}")
    print(f"  Decoder layers: {dims.n_text_layer}")

    # Step 2: Create MTK models
    print("\n[Step 2/5] Creating MTK-optimized models with KV Cache...")
    encoder, decoder, dims = create_whisper_kv_models(whisper_model, max_cache_len=max_cache_len)

    # Step 3: Export Encoder to TorchScript
    print("\n[Step 3/5] Exporting Encoder to TorchScript...")

    # Encoder input: mel spectrogram [1, 80, 3000] (30s audio)
    dummy_mel = torch.randn(1, dims.n_mels, dims.n_audio_ctx * 2)

    print(f"  Tracing encoder with input shape: {dummy_mel.shape}")
    with torch.no_grad():
        encoder_traced = torch.jit.trace(encoder, dummy_mel)

    encoder_path = os.path.join(models_dir, "encoder_base_80x3000_MT8371.pt")
    encoder_traced.save(encoder_path)

    file_size_mb = os.path.getsize(encoder_path) / 1024 / 1024
    print(f"  ✓ Encoder saved: {encoder_path} ({file_size_mb:.2f} MB)")

    # Step 4: Export Decoder to TorchScript
    print("\n[Step 4/5] Exporting Decoder with KV Cache to TorchScript...")

    # Decoder inputs for single token inference
    batch_size = 1
    n_layers = dims.n_text_layer
    n_state = dims.n_text_state
    enc_seq_len = dims.n_audio_ctx  # 1500

    # Test with dummy encoder output
    dummy_encoder_output = torch.randn(batch_size, enc_seq_len, n_state)

    # Dummy inputs for single token decoding
    dummy_token_embed = torch.randn(batch_size, 1, n_state)  # Single token
    dummy_past_self_keys = torch.zeros(n_layers, batch_size, max_cache_len, n_state)
    dummy_past_self_values = torch.zeros(n_layers, batch_size, max_cache_len, n_state)
    dummy_position_embed = torch.randn(1, 1, n_state)
    dummy_self_attn_mask = torch.zeros(1, 1, 1, max_cache_len + 1)
    dummy_cached_cross_keys = torch.randn(n_layers, batch_size, enc_seq_len, n_state)
    dummy_cached_cross_values = torch.randn(n_layers, batch_size, enc_seq_len, n_state)

    print(f"  Decoder inputs:")
    print(f"    token_embeddings: {dummy_token_embed.shape}")
    print(f"    encoder_output: {dummy_encoder_output.shape}")
    print(f"    past_self_keys: {dummy_past_self_keys.shape}")
    print(f"    past_self_values: {dummy_past_self_values.shape}")
    print(f"    position_embed: {dummy_position_embed.shape}")
    print(f"    self_attn_mask: {dummy_self_attn_mask.shape}")
    print(f"    cached_cross_keys: {dummy_cached_cross_keys.shape}")
    print(f"    cached_cross_values: {dummy_cached_cross_values.shape}")

    print(f"  Tracing decoder...")
    with torch.no_grad():
        decoder_traced = torch.jit.trace(
            decoder,
            (
                dummy_token_embed,
                dummy_encoder_output,
                dummy_past_self_keys,
                dummy_past_self_values,
                dummy_position_embed,
                dummy_self_attn_mask,
                dummy_cached_cross_keys,
                dummy_cached_cross_values,
            )
        )

    decoder_path = os.path.join(models_dir, "decoder_base_448_MT8371.pt")
    decoder_traced.save(decoder_path)

    file_size_mb = os.path.getsize(decoder_path) / 1024 / 1024
    print(f"  ✓ Decoder saved: {decoder_path} ({file_size_mb:.2f} MB)")

    # Step 5: Export embedding weights
    print("\n[Step 5/5] Exporting embedding weights...")
    export_embedding_weights(whisper_model, models_dir)

    # Verify exports
    print("\n" + "="*70)
    print("Verification")
    print("="*70)

    print("\nTesting TorchScript models...")

    # Test encoder
    with torch.no_grad():
        enc_out_original = encoder(dummy_mel)
        enc_out_traced = encoder_traced(dummy_mel)

    enc_diff = (enc_out_original - enc_out_traced).abs().max().item()
    print(f"  Encoder max diff: {enc_diff:.6e} (should be ~0)")

    # Test decoder
    with torch.no_grad():
        dec_out_original = decoder(
            dummy_token_embed,
            dummy_encoder_output,
            dummy_past_self_keys,
            dummy_past_self_values,
            dummy_position_embed,
            dummy_self_attn_mask,
            dummy_cached_cross_keys,
            dummy_cached_cross_values,
        )
        dec_out_traced = decoder_traced(
            dummy_token_embed,
            dummy_encoder_output,
            dummy_past_self_keys,
            dummy_past_self_values,
            dummy_position_embed,
            dummy_self_attn_mask,
            dummy_cached_cross_keys,
            dummy_cached_cross_values,
        )

    dec_diff = (dec_out_original[0] - dec_out_traced[0]).abs().max().item()
    print(f"  Decoder logits max diff: {dec_diff:.6e} (should be ~0)")

    # Summary
    print("\n" + "="*70)
    print("Export Complete!")
    print("="*70)
    print("\nGenerated files:")
    for filename in ["encoder_base_80x3000_MT8371.pt", "decoder_base_448_MT8371.pt", "token_embedding.npy", "embedding_info.json"]:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✓ {filepath} ({size_mb:.2f} MB)")

    print("\nNext step: Run test/test_pt.py to validate against baseline")


if __name__ == "__main__":
    main()
