"""
test_pt.py

使用 TorchScript 模型验证推理结果是否与 baseline 一致。

Baseline:
- 识别文字: 对我做了介绍那么我想说的是大家如果对我的研究感兴趣呢
- Token IDs: [593, 520, 661, 618, 743, 744, 837, 525, 520, 509, 571, 519, 500, 538, 521, 1280, 1165, 593, 520, 519, 1259, 1260, 972, 1080, 2415, 584]
- 26 tokens, 6 chunks
"""

import sys
import time
import json
import numpy as np
import torch
import kaldifeat
import soundfile as sf
import scipy.signal
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR   = SCRIPT_DIR / "outputs"
DEBUG_DIR    = OUTPUT_DIR / "debug"

PROJECT_ROOT = SCRIPT_DIR.parent.parent   # zipformer-mtk/mtk/
MODELS_DIR   = SCRIPT_DIR.parent / "models"

CHECKPOINT   = PROJECT_ROOT / "models" / "checkpoint" / "pretrained.pt"
VOCAB_PATH   = PROJECT_ROOT / "test_data" / "vocab.txt"
AUDIO_PATH   = PROJECT_ROOT / "test_data" / "test.wav"

ENCODER_PT   = MODELS_DIR / "encoder.pt"
DECODER_PT   = MODELS_DIR / "decoder_npu.pt"
JOINER_PT    = MODELS_DIR / "joiner.pt"
EMB_WEIGHT   = MODELS_DIR / "decoder_embedding_weight.npy"

for d in [DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
BASELINE_TEXT     = "对我做了介绍那么我想说的是大家如果对我的研究感兴趣呢"
BASELINE_TOKENS   = [593, 520, 661, 618, 743, 744, 837, 525, 520, 509, 571, 519, 500, 538,
                     521, 1280, 1165, 593, 520, 519, 1259, 1260, 972, 1080, 2415, 584]

# ---------------------------------------------------------------------------
# 推理参数
# ---------------------------------------------------------------------------
CONTEXT_SIZE = 2
BLANK_ID     = 0
UNK_ID       = 2
SEGMENT      = 103
OFFSET       = 96


# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------
def read_vocab(path):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                vocab[parts[0]] = ""
            else:
                value, key = parts[0], parts[1]
                vocab[key] = value
    return vocab

def tokens_to_text(hyp, vocab):
    text = "".join(vocab.get(str(i), "") for i in hyp)
    return text.replace("\u2581", " ").strip()


# ---------------------------------------------------------------------------
# 音频 & Fbank
# ---------------------------------------------------------------------------
def load_audio(path, target_sr=16000):
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        n = int(round(len(data) / sr * target_sr))
        data = scipy.signal.resample(data, n)
        sr = target_sr
    return data.astype(np.float32), sr

def make_fbank(sr=16000):
    opts = kaldifeat.FbankOptions()
    opts.frame_opts.samp_freq  = float(sr)
    opts.mel_opts.num_bins     = 80
    opts.mel_opts.high_freq    = -400.0
    opts.frame_opts.dither     = 0.0
    opts.frame_opts.snip_edges = False
    return kaldifeat.OnlineFbank(opts)


# ---------------------------------------------------------------------------
# Embedding lookup (CPU端，模拟C++行为)
# ---------------------------------------------------------------------------
def embed_tokens(token_ids: list, emb_weight: np.ndarray) -> torch.Tensor:
    """
    从 embedding weight 中查找 token embeddings.
    Args:
        token_ids: list of int, length = context_size
        emb_weight: [vocab_size, decoder_dim] float32 numpy array
    Returns:
        embedded: [1, context_size, decoder_dim] float32 tensor
    """
    rows = []
    for tid in token_ids:
        safe_id = max(0, tid)
        rows.append(emb_weight[safe_id])
    embedded = np.stack(rows, axis=0)  # [context_size, decoder_dim]
    return torch.tensor(embedded, dtype=torch.float32).unsqueeze(0)  # [1, ctx, dim]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Zipformer TorchScript Verification Test")
    print("=" * 60)

    # 验证所有文件存在
    for p in [ENCODER_PT, DECODER_PT, JOINER_PT, EMB_WEIGHT, VOCAB_PATH, AUDIO_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # 加载词表
    vocab = read_vocab(VOCAB_PATH)
    print(f"Vocab size: {len(vocab)}")

    # 加载 TorchScript 模型
    print("Loading TorchScript models...")
    t0 = time.time()
    encoder = torch.jit.load(str(ENCODER_PT))
    decoder = torch.jit.load(str(DECODER_PT))
    joiner  = torch.jit.load(str(JOINER_PT))
    emb_weight = np.load(str(EMB_WEIGHT))  # [6254, 512]
    encoder.eval(); decoder.eval(); joiner.eval()
    print(f"  Loaded in {(time.time()-t0)*1000:.1f} ms")
    print(f"  emb_weight: {emb_weight.shape}")

    # 音频
    audio, sr = load_audio(AUDIO_PATH)
    duration  = len(audio) / sr
    print(f"Audio: {AUDIO_PATH.name}, {sr} Hz, {duration:.2f}s")

    # Fbank
    print("Extracting Fbank features...")
    t0    = time.time()
    fbank = make_fbank(sr)
    fbank.accept_waveform(sampling_rate=sr,
                          waveform=torch.tensor(audio, dtype=torch.float32))
    num_frames  = fbank.num_frames_ready
    all_frames  = torch.cat([fbank.get_frame(i) for i in range(num_frames)], dim=0)
    fbank_time  = time.time() - t0
    print(f"Fbank: {all_frames.shape}, {fbank_time*1000:.1f} ms")

    # tail padding
    tail_pad = torch.zeros(int(SEGMENT / 100.0 * sr), dtype=torch.float32)
    fbank.accept_waveform(sampling_rate=sr, waveform=tail_pad)
    padded_frames = fbank.num_frames_ready

    # 初始化 states（来自 MTK model）
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from zipformer_mtk_model import EncoderMTK, build_mtk_models
    enc_tmp, _, _ = build_mtk_models(str(CHECKPOINT))
    states = enc_tmp.get_init_state()
    del enc_tmp

    # 初始化 decoder 状态
    hyp    = [BLANK_ID] * CONTEXT_SIZE
    embedded = embed_tokens(hyp[-CONTEXT_SIZE:], emb_weight)  # [1, 2, 512]
    with torch.no_grad():
        decoder_out = decoder(embedded)  # [1, 512]

    # 流式 greedy search
    num_processed       = 0
    timestamp           = []
    frame_offset        = 0
    chunk_idx           = 0
    first_chunk_saved   = False
    joiner_sample_saved = False

    print(f"Running greedy search ({num_frames} frames, seg={SEGMENT}, offset={OFFSET}) ...")
    t_infer = time.time()

    with torch.no_grad():
        while num_processed + SEGMENT <= padded_frames:
            chunk_frames = [fbank.get_frame(num_processed + i) for i in range(SEGMENT)]
            chunk_np     = torch.cat(chunk_frames, dim=0).unsqueeze(0)  # [1, 103, 80]

            if not first_chunk_saved:
                np.save(str(DEBUG_DIR / "chunk0_fbank.npy"), chunk_np.numpy())
                first_chunk_saved = True

            # Encoder (TorchScript) — x_lens removed from interface
            result = encoder(chunk_np, *states)
            encoder_out = result[0]   # [1, T', 256]
            # 更新 states（result[1:] 是 35个新states）
            states = list(result[1:])

            if chunk_idx == 0:
                np.save(str(DEBUG_DIR / "chunk0_encoder_out.npy"), encoder_out.numpy())

            enc_squeezed = encoder_out.squeeze(0)  # [T', 256]
            T = enc_squeezed.size(0)

            for t in range(T):
                cur_enc    = enc_squeezed[t:t+1]  # [1, 256]
                joiner_out = joiner(cur_enc, decoder_out).squeeze(0)  # [vocab]
                y          = int(joiner_out.argmax(dim=0))

                if not joiner_sample_saved:
                    np.save(str(DEBUG_DIR / "sample_encoder_out.npy"), cur_enc.numpy())
                    np.save(str(DEBUG_DIR / "sample_decoder_out.npy"), decoder_out.numpy())
                    np.save(str(DEBUG_DIR / "sample_joiner_out.npy"),
                            joiner_out.unsqueeze(0).numpy())
                    joiner_sample_saved = True

                if y != BLANK_ID and y != UNK_ID:
                    timestamp.append(frame_offset + t)
                    hyp.append(y)
                    embedded    = embed_tokens(hyp[-CONTEXT_SIZE:], emb_weight)
                    decoder_out = decoder(embedded)

            frame_offset  += T
            num_processed += OFFSET
            chunk_idx     += 1
            print(f"  Chunk {chunk_idx}: processed {num_processed}/{padded_frames}, "
                  f"enc_out T'={T}, tokens so far: {len(hyp)-CONTEXT_SIZE}")

    infer_time = time.time() - t_infer

    # 后处理
    token_ids       = hyp[CONTEXT_SIZE:]
    text            = tokens_to_text(token_ids, vocab)
    frame_shift_s   = 10e-3 * 4
    real_timestamps = [round(frame_shift_s * t, 2) for t in timestamp]

    print(f"\nRecognized text:  {text}")
    print(f"Token IDs:        {token_ids}")
    print(f"Inference time:   {infer_time*1000:.1f} ms  ({chunk_idx} chunks)")

    # 保存 debug 输出
    np.save(str(DEBUG_DIR / "decoder_embedding_weight.npy"), emb_weight)
    np.save(str(DEBUG_DIR / "final_decoder_out.npy"), decoder_out.numpy())

    # ---------------------------------------------------------------------------
    # 验证结果
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    text_ok   = (text == BASELINE_TEXT)
    tokens_ok = (token_ids == BASELINE_TOKENS)

    print(f"  Text match:   {'✓ PASS' if text_ok   else '✗ FAIL'}")
    print(f"  Tokens match: {'✓ PASS' if tokens_ok else '✗ FAIL'}")

    if not text_ok:
        print(f"  Expected: {BASELINE_TEXT}")
        print(f"  Got:      {text}")
    if not tokens_ok:
        print(f"  Expected: {BASELINE_TOKENS}")
        print(f"  Got:      {token_ids}")

    if not (text_ok and tokens_ok):
        print("\n[FAIL] Results do not match baseline!")
        sys.exit(1)
    else:
        print("\n[PASS] TorchScript models produce correct results!")

    # 保存 debug 文件列表
    print(f"\nDebug files in {DEBUG_DIR}:")
    for fp in sorted(DEBUG_DIR.glob("*.npy")):
        arr = np.load(str(fp))
        print(f"  {fp.name}: shape={arr.shape} dtype={arr.dtype}")

    result_data = {
        "text":         text,
        "token_ids":    token_ids,
        "timestamps_s": real_timestamps,
        "num_chunks":   chunk_idx,
        "inference_ms": round(infer_time * 1000, 1),
        "text_match":   text_ok,
        "tokens_match": tokens_ok,
        "timestamp":    datetime.now().isoformat(),
    }

    result_path = OUTPUT_DIR / "test_pt_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")

    return result_data


if __name__ == "__main__":
    main()
