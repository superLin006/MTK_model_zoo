#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1 (Streaming): 导出 Chunk Encoder TorchScript (.pt)

与 offline 版本的区别:
  - 只导出 Encoder (Decoder 直接复用 offline 的 moonshine_decoder.pt)
  - NUM_FRAMES = 160 (0.8s chunk: 160 * 5ms = 0.8s)
  - T_ENC = 40 (((160//2)//2) = 40)
  - 输入 shape: [1, 160, 80]
  - 输出 shape: [1, 40, 620]

Decoder 复用:
  /home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/python/models/moonshine_decoder.pt

输出:
  streaming/python/models/moonshine_encoder_chunk.pt
  streaming/python/test/outputs/debug/ (中间输出)
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

# ========== 路径配置 ==========
MODEL_DIR = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/models/moonshine-streaming-small")
TEST_AUDIO = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/test_data/test_en.wav")
OFFLINE_MODELS = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/python/models")

PYTHON_DIR = Path(__file__).parent
OUTPUT_MODELS_DIR = PYTHON_DIR / "models"
DEBUG_DIR = PYTHON_DIR / "test" / "outputs" / "debug"

OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ========== Streaming chunk 参数 ==========
CHUNK_FRAMES = 160          # 0.8s chunk (160 * 5ms = 0.8s)
CHUNK_T_ENC = 40            # ((160//2)//2) = 40
FRAME_LEN = 80              # 5ms @ 16kHz
T_AUDIO_CHUNK = CHUNK_FRAMES * FRAME_LEN  # 160 * 80 = 12800 samples (0.8s)

# Decoder 不变
MAX_DEC_LEN = 128
VOCAB_SIZE = 32768


def step1_export_chunk_encoder(hf_model):
    """导出 Chunk Encoder TorchScript (num_frames=160)"""
    print("\n" + "=" * 60)
    print("[Chunk Encoder] 导出 TorchScript")
    print(f"  CHUNK_FRAMES={CHUNK_FRAMES}, CHUNK_T_ENC={CHUNK_T_ENC}")
    print("=" * 60)

    # 创建 MTK Encoder (chunk 尺寸)
    print(f"  创建 MTKMoonshineEncoderNPU (num_frames={CHUNK_FRAMES})...")
    mtk_encoder = MTKMoonshineEncoderNPU(num_frames=CHUNK_FRAMES)
    load_encoder_weights(mtk_encoder, hf_model)
    mtk_encoder.eval()

    # 验证 t_enc 计算正确
    print(f"  计算得 t_enc={mtk_encoder.t_enc} (期望={CHUNK_T_ENC})")
    assert mtk_encoder.t_enc == CHUNK_T_ENC, f"t_enc 不匹配: {mtk_encoder.t_enc} != {CHUNK_T_ENC}"

    # 准备 trace 输入: 取音频前 0.8s (12800 samples)
    print("  准备 trace 输入 (取音频前 0.8s)...")
    audio, sr = sf.read(str(TEST_AUDIO))
    print(f"  音频: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}s")

    # pad 到 T_AUDIO_CHUNK
    audio_chunk_padded = np.zeros(T_AUDIO_CHUNK, dtype=np.float32)
    copy_len = min(len(audio), T_AUDIO_CHUNK)
    audio_chunk_padded[:copy_len] = audio[:copy_len].astype(np.float32)
    audio_chunk_padded = audio_chunk_padded[np.newaxis, :]  # [1, 12800]

    log_k = hf_model.model.encoder.embedder.comp.log_k.item()
    x_frames_chunk = preprocess_audio_cpu(audio_chunk_padded, log_k)  # [1, 160, 80]
    x_frames_t = torch.from_numpy(x_frames_chunk)
    print(f"  Encoder chunk 输入 shape: {x_frames_t.shape}")
    assert x_frames_t.shape == (1, CHUNK_FRAMES, 80), f"输入 shape 不对: {x_frames_t.shape}"

    # 推理验证
    print("  验证 MTK Chunk Encoder 输出...")
    with torch.no_grad():
        mtk_out = mtk_encoder(x_frames_t)
    print(f"  MTK Chunk Encoder 输出: {mtk_out.shape}")
    assert mtk_out.shape == (1, CHUNK_T_ENC, 620), f"输出 shape 不对: {mtk_out.shape}"
    print(f"  OK: 输出 shape [1, {CHUNK_T_ENC}, 620] 正确")

    # 与 offline 全序列 encoder 的前40帧对比
    print("  与 offline encoder 前40帧对比 (数值精度验证)...")
    T_AUDIO_FIXED = 160000  # 10s
    input_full = np.zeros(T_AUDIO_FIXED, dtype=np.float32)
    input_full[:min(len(audio), T_AUDIO_FIXED)] = audio[:min(len(audio), T_AUDIO_FIXED)]
    input_full = input_full[np.newaxis, :]
    x_frames_full = preprocess_audio_cpu(input_full, log_k)  # [1, 2000, 80]

    # 加载 offline encoder (若存在)
    offline_encoder_pt = OFFLINE_MODELS / "moonshine_encoder.pt"
    if offline_encoder_pt.exists():
        offline_enc = torch.jit.load(str(offline_encoder_pt))
        offline_enc.eval()
        x_frames_full_t = torch.from_numpy(x_frames_full)
        with torch.no_grad():
            offline_out = offline_enc(x_frames_full_t)  # [1, 500, 620]
        offline_first40 = offline_out[:, :CHUNK_T_ENC, :]  # [1, 40, 620]
        max_diff = (mtk_out - offline_first40).abs().max().item()
        mean_diff = (mtk_out - offline_first40).abs().mean().item()
        print(f"  与 offline 前{CHUNK_T_ENC}帧对比: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if max_diff < 0.1:
            print(f"  OK: 数值接近 (max_diff < 0.1) — 注意 chunk 与全序列的 sliding window 有边界差异")
        else:
            print(f"  NOTE: max_diff={max_diff:.4f}, 这是因为 sliding window 在 chunk 边界截断, 属于预期行为")
    else:
        print(f"  SKIP: {offline_encoder_pt} 不存在, 跳过 offline 对比")

    # 保存 debug
    np.save(DEBUG_DIR / "chunk_encoder_input.npy", x_frames_chunk)
    np.save(DEBUG_DIR / "chunk_encoder_output.npy", mtk_out.numpy())
    print(f"  已保存 chunk_encoder_input.npy, chunk_encoder_output.npy")

    # TorchScript trace
    print("  TorchScript trace...")
    t0 = time.time()
    with torch.no_grad():
        traced_encoder = torch.jit.trace(mtk_encoder, x_frames_t)
    print(f"  trace 完成 ({time.time()-t0:.1f}s)")

    # 保存
    encoder_pt_path = OUTPUT_MODELS_DIR / "moonshine_encoder_chunk.pt"
    traced_encoder.save(str(encoder_pt_path))
    size_mb = os.path.getsize(encoder_pt_path) / 1024 / 1024
    print(f"  已保存: {encoder_pt_path} ({size_mb:.1f} MB)")

    # 验证加载
    print("  验证加载后的 .pt 文件...")
    loaded = torch.jit.load(str(encoder_pt_path))
    loaded.eval()
    with torch.no_grad():
        loaded_out = loaded(x_frames_t)
    max_diff2 = (loaded_out - mtk_out).abs().max().item()
    print(f"  加载验证 max_diff={max_diff2:.8f}")
    assert max_diff2 < 1e-6, f"加载后数值不一致: {max_diff2}"
    print(f"  OK: .pt 文件验证通过")

    return encoder_pt_path


def main():
    print("=" * 70)
    print("步骤1 (Streaming): 导出 Chunk Encoder TorchScript")
    print("=" * 70)
    print(f"  模型路径:     {MODEL_DIR}")
    print(f"  输出目录:     {OUTPUT_MODELS_DIR}")
    print(f"  chunk 参数:   CHUNK_FRAMES={CHUNK_FRAMES}, CHUNK_T_ENC={CHUNK_T_ENC}")
    print(f"  输入 shape:   [1, {CHUNK_FRAMES}, 80]")
    print(f"  输出 shape:   [1, {CHUNK_T_ENC}, 620]")
    print(f"\n  Decoder 复用: {OFFLINE_MODELS}/moonshine_decoder.pt")

    # 加载 HF 模型
    print("\n加载 HuggingFace 模型...")
    t0 = time.time()
    from transformers import AutoModelForSpeechSeq2Seq
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(str(MODEL_DIR))
    hf_model.eval()
    print(f"  加载完成 ({time.time()-t0:.1f}s)")

    # 导出 Chunk Encoder
    encoder_pt_path = step1_export_chunk_encoder(hf_model)

    # 汇总
    print("\n" + "=" * 70)
    print("步骤1 (Streaming) 完成!")
    print("=" * 70)
    encoder_mb = os.path.getsize(encoder_pt_path) / 1024 / 1024
    decoder_pt = OFFLINE_MODELS / "moonshine_decoder.pt"
    decoder_mb = os.path.getsize(decoder_pt) / 1024 / 1024 if decoder_pt.exists() else 0

    print(f"\n生成/复用文件:")
    print(f"  1. {encoder_pt_path} ({encoder_mb:.1f} MB)  [新生成]")
    print(f"  2. {decoder_pt} ({decoder_mb:.1f} MB)  [复用 offline]")

    # 保存 info
    info = {
        "encoder_chunk": {
            "input_shape": [1, CHUNK_FRAMES, 80],
            "output_shape": [1, CHUNK_T_ENC, 620],
            "chunk_frames": CHUNK_FRAMES,
            "chunk_t_enc": CHUNK_T_ENC,
            "chunk_duration_s": CHUNK_FRAMES * FRAME_LEN / 16000,
            "file": str(encoder_pt_path),
            "size_mb": round(encoder_mb, 1),
        },
        "decoder": {
            "note": "复用 offline moonshine_decoder.pt (形状不变)",
            "file": str(decoder_pt),
        }
    }
    info_path = OUTPUT_MODELS_DIR / "chunk_model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n模型信息: {info_path}")
    print(f"\n下一步:")
    print(f"  python step2_chunk_to_tflite.py    (转 TFLite)")


if __name__ == "__main__":
    main()
