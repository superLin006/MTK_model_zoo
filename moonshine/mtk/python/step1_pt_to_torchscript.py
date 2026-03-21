#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1: 导出 TorchScript 模型 (.pt)

将 Moonshine Streaming Small Encoder 和 Decoder 分别导出为 TorchScript
并验证数值正确性 (与 HuggingFace 原始模型对比)

输出:
  models/moonshine_encoder.pt  (~20MB)
  models/moonshine_decoder.pt  (~80MB)
  models/embed_tokens.npy      (vocab embedding 权重, 供 C++ 查表)
"""

import os
import sys
import time
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

# 将当前目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moonshine_encoder_model import MTKMoonshineEncoderNPU, load_encoder_weights, preprocess_audio_cpu
from moonshine_decoder_model import (
    MTKMoonshineDecoderNPU, load_decoder_weights,
    precompute_rope_table, prepare_encoder_for_decoder_cpu
)

# ========== 路径配置 ==========
MODEL_DIR = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/models/moonshine-streaming-small")
TEST_AUDIO = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/test_data/test_en.wav")
PYTHON_DIR = Path(__file__).parent
OUTPUT_MODELS_DIR = PYTHON_DIR / "models"
DEBUG_DIR = PYTHON_DIR / "test" / "outputs" / "debug"

OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ========== 固定形状参数 ==========
T_AUDIO_FIXED = 160000  # 10s @ 16kHz (恰好是80的倍数)
FRAME_LEN = 80          # 5ms @ 16kHz
NUM_FRAMES = T_AUDIO_FIXED // FRAME_LEN  # 2000
T_ENC = 500             # encoder 输出帧数: ((2000//2)//2) = 500
MAX_DEC_LEN = 128       # decoder KV cache 最大长度 (10s音频token数更多)
VOCAB_SIZE = 32768


def step1_export_encoder(hf_model):
    """导出 Encoder TorchScript"""
    print("\n" + "=" * 60)
    print("[Encoder] 导出 TorchScript")
    print("=" * 60)

    # 创建 MTK Encoder
    print(f"  创建 MTKMoonshineEncoderNPU (num_frames={NUM_FRAMES})...")
    mtk_encoder = MTKMoonshineEncoderNPU(num_frames=NUM_FRAMES)
    load_encoder_weights(mtk_encoder, hf_model)
    mtk_encoder.eval()

    # 准备 trace 输入: 用真实音频数据
    print("  准备 trace 输入...")
    audio, sr = sf.read(str(TEST_AUDIO))
    input_values = np.zeros(T_AUDIO_FIXED, dtype=np.float32)
    input_values[:min(len(audio), T_AUDIO_FIXED)] = audio[:min(len(audio), T_AUDIO_FIXED)]
    input_values = input_values[np.newaxis, :]  # [1, T_AUDIO_FIXED]

    log_k = hf_model.model.encoder.embedder.comp.log_k.item()
    x_frames = preprocess_audio_cpu(input_values, log_k)
    x_frames_t = torch.from_numpy(x_frames)
    print(f"  Encoder 输入 shape: {x_frames_t.shape}")

    # 验证 (与 HF 原始模型对比)
    print("  验证 MTK Encoder 输出...")
    with torch.no_grad():
        mtk_out = mtk_encoder(x_frames_t)
    print(f"  MTK Encoder 输出: {mtk_out.shape}")

    # HF 验证
    input_values_t = torch.from_numpy(input_values)
    with torch.no_grad():
        hf_enc_out = hf_model.model.encoder(input_values_t).last_hidden_state
    print(f"  HF Encoder 输出: {hf_enc_out.shape}")

    max_diff = (mtk_out - hf_enc_out).abs().max().item()
    mean_diff = (mtk_out - hf_enc_out).abs().mean().item()
    print(f"  数值对比: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_diff > 1e-3:
        print(f"  WARNING: max_diff={max_diff:.6f} 较大, 检查权重加载!")
    else:
        print(f"  OK: 数值匹配 (max_diff < 1e-3)")

    # 保存 debug
    np.save(DEBUG_DIR / "preprocessed_frames.npy", x_frames)
    np.save(DEBUG_DIR / "encoder_output.npy", mtk_out.numpy())
    print(f"  已保存 preprocessed_frames.npy, encoder_output.npy")

    # TorchScript trace
    print("  TorchScript trace...")
    t0 = time.time()
    with torch.no_grad():
        traced_encoder = torch.jit.trace(mtk_encoder, x_frames_t)
    print(f"  trace 完成 ({time.time()-t0:.1f}s)")

    # 保存
    encoder_pt_path = OUTPUT_MODELS_DIR / "moonshine_encoder.pt"
    traced_encoder.save(str(encoder_pt_path))
    size_mb = os.path.getsize(encoder_pt_path) / 1024 / 1024
    print(f"  已保存: {encoder_pt_path} ({size_mb:.1f} MB)")

    # 验证加载后结果一致
    print("  验证加载后的 .pt 文件...")
    loaded = torch.jit.load(str(encoder_pt_path))
    loaded.eval()
    with torch.no_grad():
        loaded_out = loaded(x_frames_t)
    max_diff2 = (loaded_out - mtk_out).abs().max().item()
    print(f"  加载验证 max_diff={max_diff2:.8f}")
    assert max_diff2 < 1e-6, f"加载后数值不一致: {max_diff2}"
    print(f"  OK: .pt 文件验证通过")

    return encoder_pt_path, x_frames


def step1_export_decoder(hf_model, x_frames: np.ndarray):
    """导出 Decoder TorchScript"""
    print("\n" + "=" * 60)
    print("[Decoder] 导出 TorchScript")
    print("=" * 60)

    # 创建 MTK Decoder
    print(f"  创建 MTKMoonshineDecoderNPU (max_dec_len={MAX_DEC_LEN})...")
    mtk_decoder = MTKMoonshineDecoderNPU(max_dec_len=MAX_DEC_LEN)
    load_decoder_weights(mtk_decoder, hf_model)
    mtk_decoder.eval()

    # 准备 trace 输入
    log_k = hf_model.model.encoder.embedder.comp.log_k.item()
    input_values = np.zeros(T_AUDIO_FIXED, dtype=np.float32)
    audio, sr = sf.read(str(TEST_AUDIO))
    input_values[:min(len(audio), T_AUDIO_FIXED)] = audio[:min(len(audio), T_AUDIO_FIXED)]
    input_values = input_values[np.newaxis, :]

    # Get encoder output
    input_values_t = torch.from_numpy(input_values)
    with torch.no_grad():
        encoder_out_raw = hf_model.model.encoder(input_values_t).last_hidden_state.numpy()

    pos_emb_weight = hf_model.model.decoder.pos_emb.weight.detach().numpy()
    proj_weight = hf_model.model.decoder.proj.weight.detach().numpy()
    embed_tokens_weight = hf_model.model.decoder.embed_tokens.weight.detach().numpy()

    # CPU: pos_emb + proj
    encoder_out_proj = prepare_encoder_for_decoder_cpu(encoder_out_raw, pos_emb_weight, proj_weight)
    cos_table, sin_table = precompute_rope_table(max_len=100)

    # 构造 trace 输入 (step=0, BOS token)
    BOS_TOKEN_ID = 1
    decoder_embed = embed_tokens_weight[BOS_TOKEN_ID:BOS_TOKEN_ID+1, :]
    decoder_embed_t = torch.from_numpy(decoder_embed[np.newaxis, :, :])  # [1, 1, 512]
    encoder_out_t = torch.from_numpy(encoder_out_proj)  # [1, T_enc, 512]
    past_keys_t = torch.zeros(10, 1, MAX_DEC_LEN, 512)
    past_values_t = torch.zeros(10, 1, MAX_DEC_LEN, 512)
    cos_t = torch.from_numpy(cos_table[0:1, :][np.newaxis, :, :])  # [1, 1, 32]
    sin_t = torch.from_numpy(sin_table[0:1, :][np.newaxis, :, :])  # [1, 1, 32]
    attn_mask = np.full((1, 1, 1, MAX_DEC_LEN + 1), -1e9, dtype=np.float32)
    attn_mask[:, :, :, -1] = 0.0
    attn_mask_t = torch.from_numpy(attn_mask)
    encoder_attn_mask_t = torch.zeros(1, 1, 1, T_ENC)

    print(f"  Decoder 输入 shapes:")
    print(f"    decoder_embed: {decoder_embed_t.shape}")
    print(f"    encoder_out:   {encoder_out_t.shape}")
    print(f"    past_keys:     {past_keys_t.shape}")
    print(f"    past_values:   {past_values_t.shape}")
    print(f"    cos_input:     {cos_t.shape}")
    print(f"    sin_input:     {sin_t.shape}")
    print(f"    attn_mask:     {attn_mask_t.shape}")
    print(f"    enc_attn_mask: {encoder_attn_mask_t.shape}")

    # 验证
    print("  验证 MTK Decoder 第一步输出...")
    with torch.no_grad():
        logits, new_keys, new_values = mtk_decoder(
            decoder_embed_t, encoder_out_t, past_keys_t, past_values_t,
            cos_t, sin_t, attn_mask_t, encoder_attn_mask_t
        )
    print(f"  logits: {logits.shape}, new_keys: {new_keys.shape}")

    # HF 第一步对比
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    from transformers import AutoModelForSpeechSeq2Seq
    with torch.no_grad():
        from transformers import MoonshineStreamingForConditionalGeneration
        hf_first_token = torch.tensor([[BOS_TOKEN_ID]])
        from transformers import AutoFeatureExtractor
        # 直接用原始 model.model.decoder 做单步推理
        from transformers.cache_utils import EncoderDecoderCache, DynamicCache
        enc_hidden_t = torch.from_numpy(encoder_out_raw)
        past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
        dec_out = hf_model.model.decoder(
            input_ids=hf_first_token,
            encoder_hidden_states=enc_hidden_t,
            past_key_values=past_kv,
            use_cache=True,
        )
        hf_logits = hf_model.proj_out(dec_out.last_hidden_state)

    max_diff = (logits - hf_logits).abs().max().item()
    mean_diff = (logits - hf_logits).abs().mean().item()
    print(f"  数值对比 (logits): max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}")

    if max_diff > 0.5:
        print(f"  WARNING: logits max_diff={max_diff:.4f} 较大!")
    else:
        print(f"  OK: logits 数值匹配 (max_diff < 0.5)")

    # 保存 embed_tokens
    embed_path = OUTPUT_MODELS_DIR / "embed_tokens.npy"
    np.save(embed_path, embed_tokens_weight)
    print(f"  已保存 embed_tokens.npy ({embed_tokens_weight.shape})")

    # TorchScript trace
    print("  TorchScript trace...")
    t0 = time.time()
    trace_inputs = (
        decoder_embed_t, encoder_out_t, past_keys_t, past_values_t,
        cos_t, sin_t, attn_mask_t, encoder_attn_mask_t
    )
    with torch.no_grad():
        traced_decoder = torch.jit.trace(mtk_decoder, trace_inputs)
    print(f"  trace 完成 ({time.time()-t0:.1f}s)")

    # 保存
    decoder_pt_path = OUTPUT_MODELS_DIR / "moonshine_decoder.pt"
    traced_decoder.save(str(decoder_pt_path))
    size_mb = os.path.getsize(decoder_pt_path) / 1024 / 1024
    print(f"  已保存: {decoder_pt_path} ({size_mb:.1f} MB)")

    # 验证加载
    print("  验证加载后的 .pt 文件...")
    loaded_dec = torch.jit.load(str(decoder_pt_path))
    loaded_dec.eval()
    with torch.no_grad():
        loaded_logits, _, _ = loaded_dec(*trace_inputs)
    max_diff2 = (loaded_logits - logits).abs().max().item()
    print(f"  加载验证 max_diff={max_diff2:.8f}")
    assert max_diff2 < 1e-5, f"加载后数值不一致: {max_diff2}"
    print(f"  OK: .pt 文件验证通过")

    return decoder_pt_path


def main():
    print("=" * 70)
    print("步骤1: 导出 TorchScript 模型")
    print("=" * 70)
    print(f"  模型路径: {MODEL_DIR}")
    print(f"  输出目录: {OUTPUT_MODELS_DIR}")
    print(f"  固定形状: T_audio={T_AUDIO_FIXED}, num_frames={NUM_FRAMES}, T_enc={T_ENC}")
    print(f"  max_dec_len={MAX_DEC_LEN}")

    # 加载 HF 模型
    print("\n加载 HuggingFace 模型...")
    t0 = time.time()
    from transformers import AutoModelForSpeechSeq2Seq
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(str(MODEL_DIR))
    hf_model.eval()
    print(f"  加载完成 ({time.time()-t0:.1f}s)")

    # 导出 Encoder
    encoder_pt_path, x_frames = step1_export_encoder(hf_model)

    # 导出 Decoder
    decoder_pt_path = step1_export_decoder(hf_model, x_frames)

    # 汇总
    print("\n" + "=" * 70)
    print("步骤1 完成!")
    print("=" * 70)
    encoder_mb = os.path.getsize(encoder_pt_path) / 1024 / 1024
    decoder_mb = os.path.getsize(decoder_pt_path) / 1024 / 1024
    embed_mb = os.path.getsize(OUTPUT_MODELS_DIR / "embed_tokens.npy") / 1024 / 1024
    print(f"\n生成文件:")
    print(f"  1. {encoder_pt_path} ({encoder_mb:.1f} MB)")
    print(f"  2. {decoder_pt_path} ({decoder_mb:.1f} MB)")
    print(f"  3. {OUTPUT_MODELS_DIR}/embed_tokens.npy ({embed_mb:.1f} MB)")
    print(f"\n下一步:")
    print(f"  python test/test_pt.py       (验证推理结果)")
    print(f"  python step2_torchscript_to_tflite.py  (转 TFLite)")

    # 保存 info
    info = {
        "encoder": {
            "input_shape": [1, NUM_FRAMES, 80],
            "output_shape": [1, T_ENC, 620],
            "file": str(encoder_pt_path),
            "size_mb": round(encoder_mb, 1),
        },
        "decoder": {
            "inputs": {
                "decoder_embed": [1, 1, 512],
                "encoder_out": [1, T_ENC, 512],
                "past_keys": [10, 1, MAX_DEC_LEN, 512],
                "past_values": [10, 1, MAX_DEC_LEN, 512],
                "cos_input": [1, 1, 32],
                "sin_input": [1, 1, 32],
                "attn_mask": [1, 1, 1, MAX_DEC_LEN + 1],
                "encoder_attn_mask": [1, 1, 1, T_ENC],
            },
            "outputs": {
                "logits": [1, 1, VOCAB_SIZE],
                "new_keys": [10, 1, 1, 512],
                "new_values": [10, 1, 1, 512],
            },
            "file": str(decoder_pt_path),
            "size_mb": round(decoder_mb, 1),
        }
    }
    info_path = OUTPUT_MODELS_DIR / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n模型信息: {info_path}")


if __name__ == "__main__":
    main()
