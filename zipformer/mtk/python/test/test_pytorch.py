"""
Zipformer PyTorch Baseline Test.

直接加载 pretrained.pt（csukuangfj/k2fsa-zipformer-bilingual-zh-en-t），
使用 pruned_transducer_stateless7_streaming/zipformer.py 构建模型推理。
不依赖 k2 / sentencepiece 等额外库。

输出保存到 outputs/baseline/
"""

import sys
import types
import contextlib
import warnings
import json
import time
import numpy as np
import torch
import torch.nn as nn
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
BASELINE_DIR = OUTPUT_DIR / "baseline"

PROJECT_ROOT = SCRIPT_DIR.parent.parent   # zipformer-mtk/mtk/
# 模型代码：使用旧版 pruned_transducer_stateless7_streaming
ICEFALL_EGS  = (PROJECT_ROOT.parent / "icefall" / "egs" /
                "librispeech" / "ASR" / "pruned_transducer_stateless7_streaming")
ICEFALL_ROOT = PROJECT_ROOT.parent / "icefall"

CHECKPOINT   = PROJECT_ROOT / "models" / "checkpoint" / "pretrained.pt"
VOCAB_PATH   = PROJECT_ROOT / "test_data" / "vocab.txt"
AUDIO_PATH   = PROJECT_ROOT / "test_data" / "test.wav"

for d in [DEBUG_DIR, BASELINE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# icefall.utils 最小 stub（只需 make_pad_mask, subsequent_chunk_mask, torch_autocast）
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _torch_autocast(device_type="cuda", **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.cuda.amp.autocast(enabled=False):
            yield

def _make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1
    max_len = max_len if max_len > 0 else int(lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(n, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def _subsequent_chunk_mask(size, chunk_size, num_left_chunks=-1, device=torch.device("cpu")):
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max(0, (i // chunk_size - num_left_chunks) * chunk_size)
        ending = min(size, (i // chunk_size + 1) * chunk_size)
        ret[i, start:ending] = True
    return ret

icefall_utils_stub = types.ModuleType("icefall.utils")
icefall_utils_stub.torch_autocast       = _torch_autocast
icefall_utils_stub.make_pad_mask        = _make_pad_mask
icefall_utils_stub.subsequent_chunk_mask = _subsequent_chunk_mask

icefall_stub = types.ModuleType("icefall")
icefall_stub.utils = icefall_utils_stub
sys.modules["icefall"]       = icefall_stub
sys.modules["icefall.utils"] = icefall_utils_stub
for _mod in ["icefall.checkpoint", "icefall.decode", "icefall.dist",
             "icefall.env", "icefall.lm_wrapper", "icefall.rnn_lm",
             "icefall.rnn_lm.model"]:
    sys.modules[_mod] = types.ModuleType(_mod)

# ---------------------------------------------------------------------------
# 导入模型文件（pruned_transducer_stateless7_streaming）
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ICEFALL_EGS))
sys.path.insert(0, str(ICEFALL_ROOT))

from zipformer import Zipformer   # noqa: E402
from decoder   import Decoder     # noqa: E402
from joiner    import Joiner      # noqa: E402


# ---------------------------------------------------------------------------
# 构建并加载模型
# 配置来自 HuggingFace README:
#   https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t
# ---------------------------------------------------------------------------
def build_and_load_model(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt   = torch.load(str(checkpoint_path), map_location="cpu")
    state  = ckpt["model"]

    # --- Encoder (Zipformer) ---
    encoder = Zipformer(
        num_features=80,
        output_downsampling_factor=2,
        encoder_dims=(256, 256, 256, 256, 256),
        attention_dim=(192, 192, 192, 192, 192),
        encoder_unmasked_dims=(192, 192, 192, 192, 192),
        zipformer_downsampling_factors=(1, 2, 4, 8, 2),
        nhead=(4, 4, 4, 4, 4),
        feedforward_dim=(768, 768, 768, 768, 768),
        num_encoder_layers=(2, 2, 2, 2, 2),
        cnn_module_kernels=(31, 31, 31, 31, 31),
        pos_dim=4,
        num_left_chunks=4,
        short_chunk_size=50,
        decode_chunk_size=32,  # decode_chunk_len=32 -> decode_chunk_size=32
    )

    # --- Decoder ---
    decoder = Decoder(
        vocab_size=6254,
        decoder_dim=512,
        blank_id=0,
        context_size=2,
    )

    # --- Joiner ---
    joiner = Joiner(
        encoder_dim=256,
        decoder_dim=512,
        joiner_dim=512,
        vocab_size=6254,
    )

    # Load weights (strict=False to skip training-only buffers like batch_count)
    enc_state = {k[len("encoder."):]: v
                 for k, v in state.items() if k.startswith("encoder.")}
    r = encoder.load_state_dict(enc_state, strict=False)
    if r.missing_keys:
        print(f"  Encoder missing keys ({len(r.missing_keys)}): {r.missing_keys[:3]}")

    dec_state = {k[len("decoder."):]: v
                 for k, v in state.items() if k.startswith("decoder.")}
    r = decoder.load_state_dict(dec_state, strict=False)
    if r.missing_keys:
        print(f"  Decoder missing keys: {r.missing_keys}")

    j_state = {k[len("joiner."):]: v
               for k, v in state.items() if k.startswith("joiner.")}
    r = joiner.load_state_dict(j_state, strict=False)
    if r.missing_keys:
        print(f"  Joiner missing keys: {r.missing_keys}")

    encoder.eval()
    decoder.eval()
    joiner.eval()
    print("  Model loaded OK.")
    return encoder, decoder, joiner


# ---------------------------------------------------------------------------
# 词表（vocab.txt 格式: "token id" 每行）
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
# 推理参数（与 ONNX baseline 对齐）
# decode_chunk_len=32 (at 50Hz) = 64 fbank frames (at 100Hz, after 2x subsampling)
# 但 ONNX baseline 用 segment=103, offset=96 -> 沿用保持一致
# ---------------------------------------------------------------------------
CONTEXT_SIZE  = 2
BLANK_ID      = 0
UNK_ID        = 2
SEGMENT       = 103   # fbank frames per chunk
OFFSET        = 96    # fbank frames stride


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Zipformer PyTorch Baseline Test (pretrained.pt)")
    print("=" * 60)

    for p in [CHECKPOINT, VOCAB_PATH, AUDIO_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # 词表
    vocab    = read_vocab(VOCAB_PATH)
    print(f"Vocab size: {len(vocab)}")

    # 音频
    audio, sr = load_audio(AUDIO_PATH)
    duration  = len(audio) / sr
    print(f"Audio: {AUDIO_PATH.name}, {sr} Hz, {duration:.2f}s")

    # Fbank
    print("Extracting Fbank features ...")
    t0    = time.time()
    fbank = make_fbank(sr)
    fbank.accept_waveform(sampling_rate=sr,
                          waveform=torch.tensor(audio, dtype=torch.float32))
    num_frames  = fbank.num_frames_ready
    all_frames  = torch.cat([fbank.get_frame(i) for i in range(num_frames)], dim=0)
    fbank_time  = time.time() - t0
    print(f"Fbank: {all_frames.shape}, {fbank_time*1000:.1f} ms")
    np.save(DEBUG_DIR / "fbank_features.npy", all_frames.numpy())

    # tail padding（对齐 ONNX baseline）
    tail_pad = torch.zeros(int(SEGMENT / 100.0 * sr), dtype=torch.float32)
    fbank.accept_waveform(sampling_rate=sr, waveform=tail_pad)
    padded_frames = fbank.num_frames_ready

    # 模型
    print("Loading model ...")
    encoder, decoder, joiner = build_and_load_model(CHECKPOINT)
    states = encoder.get_init_state()

    # 初始 decoder 状态
    hyp = [BLANK_ID] * CONTEXT_SIZE
    dec_in = torch.tensor([hyp], dtype=torch.int64)
    with torch.no_grad():
        decoder_out = decoder(dec_in, need_pad=False)  # [1, 1, 512]
        decoder_out = decoder_out.squeeze(1)            # [1, 512]

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
            x_lens       = torch.tensor([SEGMENT], dtype=torch.int64)

            if not first_chunk_saved:
                np.save(DEBUG_DIR / "chunk0_fbank.npy", chunk_np.numpy())
                first_chunk_saved = True

            # Encoder
            encoder_out, out_lens, states = encoder.streaming_forward(
                x=chunk_np, x_lens=x_lens, states=states,
            )
            # encoder_out: [1, T', encoder_dims[-1]] = [1, T', 256]

            if chunk_idx == 0:
                np.save(DEBUG_DIR / "chunk0_encoder_out.npy", encoder_out.numpy())

            enc_squeezed = encoder_out.squeeze(0)  # [T', 256]
            T = enc_squeezed.size(0)

            for t in range(T):
                cur_enc    = enc_squeezed[t:t+1]  # [1, 256]
                joiner_out = joiner(cur_enc, decoder_out).squeeze(0)  # [vocab]
                y          = int(joiner_out.argmax(dim=0))

                if not joiner_sample_saved:
                    np.save(DEBUG_DIR / "greedy_search_sample_encoder_out.npy", cur_enc.numpy())
                    np.save(DEBUG_DIR / "greedy_search_sample_decoder_out.npy", decoder_out.numpy())
                    np.save(DEBUG_DIR / "greedy_search_sample_joiner_out.npy",
                            joiner_out.unsqueeze(0).numpy())
                    joiner_sample_saved = True

                if y != BLANK_ID and y != UNK_ID:
                    timestamp.append(frame_offset + t)
                    hyp.append(y)
                    dec_in      = torch.tensor([hyp[-CONTEXT_SIZE:]], dtype=torch.int64)
                    decoder_out = decoder(dec_in, need_pad=False).squeeze(1)

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

    print(f"\nRecognized text: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Timestamps (s): {real_timestamps}")
    print(f"Inference time: {infer_time*1000:.1f} ms  ({chunk_idx} chunks)")
    print(f"RTF: {infer_time/duration:.3f}x")

    # debug 输出
    np.save(DEBUG_DIR / "final_decoder_out.npy", decoder_out.numpy())

    # baseline 结果
    result = {
        "text":              text,
        "token_ids":         token_ids,
        "timestamps_s":      real_timestamps,
        "audio_file":        AUDIO_PATH.name,
        "audio_duration_s":  round(duration, 3),
        "sample_rate":       sr,
        "num_fbank_frames":  int(all_frames.shape[0]),
        "num_chunks":        chunk_idx,
        "inference_time_ms": round(infer_time * 1000, 1),
        "rtf":               round(infer_time / duration, 4),
        "fbank_time_ms":     round(fbank_time * 1000, 1),
        "model": {
            "checkpoint":    CHECKPOINT.name,
            "type":          "pruned_transducer_stateless7_streaming Zipformer",
            "source":        "csukuangfj/k2fsa-zipformer-bilingual-zh-en-t",
            "encoder_dims":  "256x5",
            "vocab_size":    6254,
            "decode_chunk_size": 32,
            "num_left_chunks":   4,
        },
        "env": {"torch": torch.__version__},
        "timestamp": datetime.now().isoformat(),
    }

    result_path = BASELINE_DIR / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")

    debug_files = sorted(DEBUG_DIR.glob("*.npy"))
    print(f"\nDebug files ({len(debug_files)}):")
    for fp in debug_files:
        arr = np.load(fp)
        print(f"  {fp.name}: shape={arr.shape} dtype={arr.dtype}")

    print("\n" + "=" * 60)
    print("Baseline test DONE")
    print("=" * 60)
    return result


if __name__ == "__main__":
    main()
