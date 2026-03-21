#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pt.py - 验证 MTK 优化模型的推理结果与 baseline 一致

验证标准: 转录文本必须与 baseline 完全一致
  baseline: "Mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."

步骤:
  1. 加载音频 → CPU 前处理 (CMVN + AsinhCompression)
  2. 加载 Encoder.pt → 推理得 encoder_output
  3. CPU 完成 encoder pos_emb 加法 + proj(620→512)
  4. 初始化 KV cache + RoPE table
  5. 循环 decoder 步骤 (加载 Decoder.pt, 每步推理)
  6. 解码 tokens 为文本
  7. 保存 debug 输出到 outputs/debug/
"""

import sys
import os
# 将 python 目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

# 路径配置
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baseline"
DEBUG_DIR = OUTPUT_DIR / "debug"
TORCHSCRIPT_DIR = OUTPUT_DIR / "torchscript"
MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_DIR = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/models/moonshine-streaming-small")
TEST_AUDIO = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/test_data/test_en.wav")

for d in [DEBUG_DIR, TORCHSCRIPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== 固定参数 ==========
FRAME_LEN = 80           # 5ms @ 16kHz
T_AUDIO_FIXED = 160000   # 固定音频长度: 10s @ 16kHz
NUM_FRAMES = T_AUDIO_FIXED // FRAME_LEN  # 2000
T_ENC = 500              # encoder 输出帧数: ((2000//2)//2) = 500
MAX_DEC_LEN = 128        # KV cache 最大长度 (10s音频token数更多)
VOCAB_SIZE = 32768
HIDDEN_DEC = 512
ENC_HIDDEN = 620
NUM_DEC_LAYERS = 10
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MAX_NEW_TOKENS = 120


def preprocess_audio_cpu(input_values: np.ndarray, log_k: float) -> np.ndarray:
    """CPU CMVN + AsinhCompression → [1, num_frames, 80]"""
    T = input_values.shape[-1]
    num_frames = T // FRAME_LEN
    x = input_values[:, :num_frames * FRAME_LEN].reshape(1, num_frames, FRAME_LEN)
    # CMVN
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    rms = np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    x_normed = centered / rms
    # AsinhCompression
    k = np.exp(log_k)
    x_comp = np.arcsinh(k * x_normed)
    return x_comp.astype(np.float32)


def precompute_rope_table(max_len: int = 128) -> tuple:
    """
    预计算 RoPE cos/sin 查找表 (已预展开 interleaved 格式, 避免 NPU 内 repeat_interleave)
    → (cos [max_len,32], sin [max_len,32])
    等价于 HF 的 cos[:, :16].repeat_interleave(2, dim=-1)
    """
    rot_dim = 32  # head_dim=64, partial=0.5
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))  # [16]
    positions = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # [max_len, 16]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [max_len, 32]
    cos_raw = np.cos(emb).astype(np.float32)
    sin_raw = np.sin(emb).astype(np.float32)
    # 预展开: cos[:, :16].repeat_interleave(2) → interleaved [max_len, 32]
    cos_half = cos_raw[:, :rot_dim // 2]  # [max_len, 16]
    sin_half = sin_raw[:, :rot_dim // 2]
    cos_table = np.stack([cos_half, cos_half], axis=-1).reshape(max_len, rot_dim)
    sin_table = np.stack([sin_half, sin_half], axis=-1).reshape(max_len, rot_dim)
    return cos_table, sin_table


def prepare_encoder_for_decoder(encoder_out_raw: np.ndarray,
                                  pos_emb_weight: np.ndarray,
                                  proj_weight: np.ndarray) -> np.ndarray:
    """CPU: encoder_out + pos_emb + proj(620→512)"""
    T = encoder_out_raw.shape[1]
    pos_embed = pos_emb_weight[:T, :]  # [T, 620]
    enc = encoder_out_raw + pos_embed[np.newaxis, :, :]  # [1, T, 620]
    enc_proj = enc @ proj_weight.T  # [1, T, 512]
    return enc_proj.astype(np.float32)


def run_inference():
    print("=" * 70)
    print("test_pt.py - Moonshine MTK 模型验证")
    print("=" * 70)

    # ========== 1. 加载音频 ==========
    print("\n[1/7] 加载音频...")
    audio, sr = sf.read(str(TEST_AUDIO))
    print(f"  音频: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}s")

    # Pad 到固定长度 T_AUDIO_FIXED (10s)
    input_values_padded = np.zeros(T_AUDIO_FIXED, dtype=np.float32)
    copy_len = min(len(audio), T_AUDIO_FIXED)
    input_values_padded[:copy_len] = audio[:copy_len].astype(np.float32)
    input_values = input_values_padded[np.newaxis, :]  # [1, T_AUDIO_FIXED]
    print(f"  Padded shape: {input_values.shape} (padded to {T_AUDIO_FIXED} = 10s)")

    # ========== 2. 加载 HF 模型参数 ==========
    print("\n[2/7] 加载模型参数...")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(str(MODEL_DIR))
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))

    log_k = hf_model.model.encoder.embedder.comp.log_k.item()
    embed_tokens_weight = hf_model.model.decoder.embed_tokens.weight.detach().numpy()  # [32768, 512]
    pos_emb_weight = hf_model.model.decoder.pos_emb.weight.detach().numpy()            # [4096, 620]
    proj_weight = hf_model.model.decoder.proj.weight.detach().numpy()                   # [512, 620]

    print(f"  log_k={log_k:.4f}, k={np.exp(log_k):.4f}")
    print(f"  embed_tokens: {embed_tokens_weight.shape}")
    print(f"  pos_emb: {pos_emb_weight.shape}")
    print(f"  proj: {proj_weight.shape}")

    # ========== 3. CPU 前处理 ==========
    print("\n[3/7] CPU 前处理 (CMVN + AsinhCompression)...")
    x_frames = preprocess_audio_cpu(input_values, log_k)
    print(f"  x_frames: {x_frames.shape}")

    # 保存 debug
    np.save(DEBUG_DIR / "preprocessed_frames.npy", x_frames)
    print(f"  已保存 preprocessed_frames.npy")

    # ========== 4. Encoder 推理 ==========
    print("\n[4/7] Encoder 推理...")
    encoder_pt_path = MODELS_DIR / "moonshine_encoder.pt"

    if not encoder_pt_path.exists():
        print(f"  ERROR: {encoder_pt_path} 不存在, 请先运行 step1_pt_to_torchscript.py")
        sys.exit(1)

    t0 = time.time()
    encoder_model = torch.jit.load(str(encoder_pt_path))
    encoder_model.eval()
    x_frames_t = torch.from_numpy(x_frames)
    with torch.no_grad():
        encoder_out_raw = encoder_model(x_frames_t).numpy()
    print(f"  encoder_out_raw: {encoder_out_raw.shape}")
    print(f"  encoder 推理时间: {(time.time()-t0)*1000:.1f}ms")

    np.save(DEBUG_DIR / "encoder_output.npy", encoder_out_raw)
    print(f"  已保存 encoder_output.npy")

    # ========== 5. 准备 Decoder 输入 ==========
    print("\n[5/7] 准备 Decoder 输入...")

    # CPU: encoder_out + pos_emb + proj
    encoder_out_proj = prepare_encoder_for_decoder(encoder_out_raw, pos_emb_weight, proj_weight)
    print(f"  encoder_out_proj (after pos_emb+proj): {encoder_out_proj.shape}")

    # 预计算 RoPE table
    cos_table, sin_table = precompute_rope_table(max_len=MAX_NEW_TOKENS + 10)
    print(f"  cos_table: {cos_table.shape}")

    # 初始化 KV cache (zeros)
    past_keys = np.zeros((NUM_DEC_LAYERS, 1, MAX_DEC_LEN, HIDDEN_DEC), dtype=np.float32)
    past_values = np.zeros((NUM_DEC_LAYERS, 1, MAX_DEC_LEN, HIDDEN_DEC), dtype=np.float32)

    # 初始化 encoder attention mask (全 0, 所有位置有效)
    encoder_attn_mask = np.zeros((1, 1, 1, T_ENC), dtype=np.float32)

    # 加载 Decoder
    decoder_pt_path = MODELS_DIR / "moonshine_decoder.pt"
    if not decoder_pt_path.exists():
        print(f"  ERROR: {decoder_pt_path} 不存在, 请先运行 step1_pt_to_torchscript.py")
        sys.exit(1)

    decoder_model = torch.jit.load(str(decoder_pt_path))
    decoder_model.eval()

    # ========== 6. Decoder 循环 ==========
    print("\n[6/7] Decoder 自回归推理...")

    # 准备固定 tensor
    encoder_out_t = torch.from_numpy(encoder_out_proj)
    encoder_attn_mask_t = torch.from_numpy(encoder_attn_mask)

    token_ids = [BOS_TOKEN_ID]
    generated_tokens = []
    first_logits = None
    t_dec_start = time.time()

    for step in range(MAX_NEW_TOKENS):
        current_token = token_ids[-1]

        # CPU embed_tokens 查表
        decoder_embed = embed_tokens_weight[current_token:current_token+1, :]  # [1, 512]
        decoder_embed_t = torch.from_numpy(
            decoder_embed[np.newaxis, :, :]  # [1, 1, 512]
        )

        # RoPE cos/sin for current position
        cos_cur = torch.from_numpy(cos_table[step:step+1, :][np.newaxis, :, :])  # [1, 1, 32]
        sin_cur = torch.from_numpy(sin_table[step:step+1, :][np.newaxis, :, :])  # [1, 1, 32]

        # Self-attention mask: [1, 1, 1, max_dec_len+1]
        attn_mask = np.full((1, 1, 1, MAX_DEC_LEN + 1), -1e9, dtype=np.float32)
        attn_mask[:, :, :, :step] = 0.0   # 历史 cache 有效 (0..step-1)
        attn_mask[:, :, :, -1] = 0.0       # 当前 token 有效
        attn_mask_t = torch.from_numpy(attn_mask)

        past_keys_t = torch.from_numpy(past_keys)
        past_values_t = torch.from_numpy(past_values)

        with torch.no_grad():
            logits, new_keys, new_values = decoder_model(
                decoder_embed_t,
                encoder_out_t,
                past_keys_t,
                past_values_t,
                cos_cur,
                sin_cur,
                attn_mask_t,
                encoder_attn_mask_t,
            )

        if step == 0:
            first_logits = logits.numpy().copy()
            np.save(DEBUG_DIR / "decoder_first_logits.npy", first_logits)
            print(f"  已保存 decoder_first_logits.npy")

        # 更新 KV cache (CPU 端写入)
        past_keys[:, :, step:step+1, :] = new_keys.numpy()
        past_values[:, :, step:step+1, :] = new_values.numpy()

        # Greedy decode
        next_token = int(logits[0, 0].argmax().item())
        token_ids.append(next_token)
        generated_tokens.append(next_token)

        if next_token == EOS_TOKEN_ID:
            print(f"  EOS at step {step}")
            break

    t_dec_end = time.time()
    print(f"  生成 {len(generated_tokens)} tokens, 耗时 {(t_dec_end-t_dec_start)*1000:.1f}ms")

    # ========== 7. 解码文本 ==========
    print("\n[7/7] 解码文本...")
    all_token_ids = token_ids  # [BOS, tok1, tok2, ..., EOS]
    decoded = processor.decode(all_token_ids, skip_special_tokens=True)
    print(f"\n  转录文本: \"{decoded}\"")

    # ========== 与 baseline 对比 ==========
    BASELINE_TEXT = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
    print(f"\n  baseline:   \"{BASELINE_TEXT}\"")

    if decoded.strip() == BASELINE_TEXT.strip():
        print("\n  [PASS] 转录文本与 baseline 完全一致!")
    else:
        print("\n  [FAIL] 转录文本与 baseline 不一致!")
        # 保存失败信息供调试
        print(f"\n  token_ids: {token_ids}")
        print(f"  generated_tokens: {generated_tokens}")

    # ========== 保存输出 ==========
    result = {
        "text": decoded,
        "tokens": all_token_ids,
        "match_baseline": decoded.strip() == BASELINE_TEXT.strip(),
        "num_tokens": len(generated_tokens),
        "dec_time_ms": round((t_dec_end - t_dec_start) * 1000, 1),
    }
    import json
    with open(TORCHSCRIPT_DIR / "test_pt_result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存到: {TORCHSCRIPT_DIR}/test_pt_result.json")

    return decoded.strip() == BASELINE_TEXT.strip()


if __name__ == "__main__":
    success = run_inference()
    sys.exit(0 if success else 1)
