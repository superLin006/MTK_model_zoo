"""
step1_pt_to_torchscript.py

将 MTK 优化版 Zipformer 模型导出为 TorchScript (.pt) 格式。

输出:
    models/encoder.pt
    models/decoder_npu.pt
    models/joiner.pt
    models/decoder_embedding_weight.npy
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # zipformer-mtk/mtk/
MODELS_DIR   = SCRIPT_DIR / "models"
CHECKPOINT   = PROJECT_ROOT / "models" / "checkpoint" / "pretrained.pt"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

from zipformer_mtk_model import build_mtk_models, EncoderMTK

# ---------------------------------------------------------------------------
# 固定推理参数
# ---------------------------------------------------------------------------
SEGMENT       = 103
BATCH_SIZE    = 1
CONTEXT_SIZE  = 2
DECODER_DIM   = 512
ENCODER_DIM   = 256
VOCAB_SIZE    = 6254

# ===========================================================================
# 1. 构建模型
# ===========================================================================
print("=" * 60)
print("Step 1: PT → TorchScript")
print("=" * 60)

print("\n[1/4] Building MTK models from checkpoint...")
t0 = time.time()
encoder_mtk, decoder_npu, joiner_mtk = build_mtk_models(str(CHECKPOINT))
print(f"  Built in {(time.time()-t0)*1000:.1f} ms")

# ===========================================================================
# 2. 导出 decoder_embedding_weight.npy
# ===========================================================================
print("\n[2/4] Exporting decoder embedding weight...")
# 需要重新加载 decoder（build_mtk_models 返回的是 DecoderNPU，没有 embedding）
from zipformer_mtk_model import (
    _THIS_DIR, _ICEFALL_EGS, _ICEFALL_ROOT,
    Zipformer, Decoder, Joiner
)
sys.path.insert(0, str(_ICEFALL_EGS))
sys.path.insert(0, str(_ICEFALL_ROOT))

ckpt  = torch.load(str(CHECKPOINT), map_location="cpu")
state = ckpt["model"]
decoder_orig = Decoder(vocab_size=6254, decoder_dim=512, blank_id=0, context_size=2)
dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
decoder_orig.load_state_dict(dec_state, strict=False)

emb_weight = decoder_orig.embedding.weight.detach().numpy()
emb_path   = MODELS_DIR / "decoder_embedding_weight.npy"
np.save(str(emb_path), emb_weight)
print(f"  Embedding weight shape: {emb_weight.shape}, dtype: {emb_weight.dtype}")
print(f"  Saved: {emb_path}")

# ===========================================================================
# 3. 导出 Encoder TorchScript
# ===========================================================================
print("\n[3/4] Exporting encoder.pt via torch.jit.trace...")

states = encoder_mtk.get_init_state()
x_dummy = torch.zeros(BATCH_SIZE, SEGMENT, 80)
# NOTE: x_lens removed from encoder input (SEGMENT=103 is fixed)

example_inputs = (x_dummy, *states)
print(f"  Input args: {len(example_inputs)} tensors")

t0 = time.time()
with torch.no_grad():
    encoder_ts = torch.jit.trace(encoder_mtk, example_inputs, strict=False)
print(f"  Traced in {(time.time()-t0)*1000:.1f} ms")

enc_path = MODELS_DIR / "encoder.pt"
torch.jit.save(encoder_ts, str(enc_path))
size_mb = enc_path.stat().st_size / 1e6
print(f"  Saved: {enc_path} ({size_mb:.1f} MB)")

# Verify load
ts_loaded = torch.jit.load(str(enc_path))
with torch.no_grad():
    result = ts_loaded(x_dummy, *states)
print(f"  Verify: encoder_out={result[0].shape}")

# ===========================================================================
# 4. 导出 DecoderNPU TorchScript
# ===========================================================================
print("\n[4/4] Exporting decoder_npu.pt via torch.jit.trace...")

embedded_dummy = torch.zeros(BATCH_SIZE, CONTEXT_SIZE, DECODER_DIM)
t0 = time.time()
with torch.no_grad():
    decoder_ts = torch.jit.trace(decoder_npu, embedded_dummy, strict=False)
print(f"  Traced in {(time.time()-t0)*1000:.1f} ms")

dec_path = MODELS_DIR / "decoder_npu.pt"
torch.jit.save(decoder_ts, str(dec_path))
size_mb = dec_path.stat().st_size / 1e6
print(f"  Saved: {dec_path} ({size_mb:.1f} MB)")

ts_loaded = torch.jit.load(str(dec_path))
with torch.no_grad():
    dec_out = ts_loaded(embedded_dummy)
print(f"  Verify: decoder_out={dec_out.shape}")

# ===========================================================================
# 5. 导出 Joiner TorchScript
# ===========================================================================
print("\n[5/5] Exporting joiner.pt via torch.jit.trace...")

enc_out_dummy = torch.zeros(BATCH_SIZE, ENCODER_DIM)
dec_out_dummy = torch.zeros(BATCH_SIZE, DECODER_DIM)
t0 = time.time()
with torch.no_grad():
    joiner_ts = torch.jit.trace(joiner_mtk, (enc_out_dummy, dec_out_dummy), strict=False)
print(f"  Traced in {(time.time()-t0)*1000:.1f} ms")

joi_path = MODELS_DIR / "joiner.pt"
torch.jit.save(joiner_ts, str(joi_path))
size_mb = joi_path.stat().st_size / 1e6
print(f"  Saved: {joi_path} ({size_mb:.1f} MB)")

ts_loaded = torch.jit.load(str(joi_path))
with torch.no_grad():
    logits = ts_loaded(enc_out_dummy, dec_out_dummy)
print(f"  Verify: logits={logits.shape}")

# ===========================================================================
# 完成
# ===========================================================================
print("\n" + "=" * 60)
print("Step 1 DONE")
print("=" * 60)
print("\nGenerated files:")
for p in sorted(MODELS_DIR.glob("*.pt")) + sorted(MODELS_DIR.glob("*.npy")):
    print(f"  {p.name}: {p.stat().st_size/1e6:.2f} MB")
