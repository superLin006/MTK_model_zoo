"""
Moonshine Streaming Small - PyTorch Baseline Test
使用 transformers 加载模型推理，保存输出用于后续格式对比
"""
import numpy as np
import json
import time
import torch
import soundfile as sf
from pathlib import Path

# 路径配置
MODEL_DIR = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/models/moonshine-streaming-small")
TEST_AUDIO = Path("/home/xh/projects/MTK_models_zoo/moonshine/mtk/offline/test_data/test_en.wav")
OUTPUT_DIR = Path(__file__).parent / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baseline"
DEBUG_DIR = OUTPUT_DIR / "debug"

for d in [BASELINE_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 加载音频
audio, sr = sf.read(str(TEST_AUDIO))
print(f"Audio: {len(audio)} samples, {sr}Hz, duration={len(audio)/sr:.2f}s")

# 加载模型
from transformers import MoonshineStreamingForConditionalGeneration, AutoProcessor

print("Loading model...")
processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
model = MoonshineStreamingForConditionalGeneration.from_pretrained(str(MODEL_DIR))
model.eval()
print("Model loaded.")

# 预处理
inputs = processor(audio, return_tensors="pt", sampling_rate=sr)
print(f"Input shape: {inputs['input_values'].shape}")

# 计算max_length
token_limit_factor = 6.5 / sr
seq_lens = inputs.attention_mask.sum(dim=-1) if 'attention_mask' in inputs else torch.tensor([len(audio)])
max_length = max(int((seq_lens * token_limit_factor).max().item()), 10)
print(f"max_length: {max_length}")

# 推理
t0 = time.time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=max_length)
t1 = time.time()

text = processor.decode(generated_ids[0], skip_special_tokens=True)
print(f"\nTranscription: {text}")
print(f"Inference time: {(t1-t0)*1000:.1f}ms")

# 保存debug中间输出
with torch.no_grad():
    # 保存预处理后的输入
    input_arr = inputs['input_values'].numpy()
    np.save(DEBUG_DIR / "preprocessed_audio.npy", input_arr)
    print(f"\nSaved preprocessed_audio.npy: {input_arr.shape}")

    # 保存encoder输出
    encoder_outputs = model.model.encoder(inputs['input_values'])
    encoder_out = encoder_outputs.last_hidden_state.numpy()
    np.save(DEBUG_DIR / "encoder_output.npy", encoder_out)
    print(f"Saved encoder_output.npy: {encoder_out.shape}")

# 保存baseline结果
result = {
    "text": text,
    "tokens": generated_ids[0].tolist(),
    "inference_time_ms": round((t1 - t0) * 1000, 1),
    "audio_duration_s": round(len(audio) / sr, 2),
    "input_shape": list(inputs['input_values'].shape),
    "encoder_output_shape": list(encoder_out.shape)
}

with open(BASELINE_DIR / "test_en.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

with open(BASELINE_DIR / "test_en.txt", "w") as f:
    f.write(text + "\n")

print(f"\nBaseline saved to: {BASELINE_DIR}")
print(f"Result: {result}")
