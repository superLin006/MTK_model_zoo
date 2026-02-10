"""
Whisper PyTorch Baseline Test with KV Cache Concept
用于生成baseline结果，供后续MTK NPU移植对比使用

KV Cache Concept:
- Whisper decoder uses self-attention and cross-attention
- In autoregressive decoding, the self-attention KV cache can be reused
- This avoids recomputing attention for previous tokens at each step
- Significant speedup for long sequences
- Modern Whisper implementations include KV cache optimization internally

Note: This test uses the standard Whisper API which includes KV cache optimization.
For MTK NPU deployment, we'll need to implement explicit KV cache management in C++.
"""

import os
import sys
import torch
import whisper
import time
import json
import numpy as np
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "python" / "test" / "outputs" / "baseline"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_whisper_with_kv_cache(audio_path, model_name="base", language=None):
    """
    运行Whisper baseline测试 (simulating KV cache behavior)

    Note: This uses the standard Whisper API which internally uses KV cache optimization.
    We track encoder/decoder timing separately to demonstrate where KV cache helps.

    Args:
        audio_path: 音频文件路径
        model_name: 模型名称 (tiny, base, small, medium, large)
        language: 语言代码 (en, zh等)，None表示自动检测

    Returns:
        dict: 包含识别结果、耗时等信息的字典
    """
    print(f"\n{'='*60}")
    print(f"Whisper PyTorch Baseline Test (with KV Cache)")
    print(f"{'='*60}")

    # 检查音频文件
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"\nAudio file: {audio_path}")
    print(f"Model: {model_name}")
    print(f"Language: {language if language else 'auto-detect'}")

    # 加载模型
    print(f"\nLoading model...")
    start_time = time.time()
    device = "cpu"
    print(f"Using device: {device}")

    model = whisper.load_model(model_name, device=device)
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f}s")

    # Load and preprocess audio
    print(f"\nProcessing audio...")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    # Encode audio (this is done once, not affected by KV cache)
    print(f"\nEncoding audio...")
    encoder_start = time.time()
    with torch.no_grad():
        encoder_output = model.encoder(mel.unsqueeze(0))
    encoder_time = time.time() - encoder_start
    print(f"Encoder time: {encoder_time:.2f}s")
    print(f"Encoder output shape: {encoder_output.shape}")

    # Transcribe using standard API (which uses KV cache internally in newer versions)
    print(f"\nDecoding (using standard transcribe API with internal KV cache)...")
    decoder_start = time.time()

    result = model.transcribe(
        audio_path,
        language=language,
        temperature=0.0,
        verbose=False
    )

    total_time = time.time() - start_time
    decoder_time = time.time() - decoder_start

    print(f"Total inference time: {total_time:.2f}s")
    print(f"Transcription: {result['text']}")

    # Count tokens
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        language=result['language'],
        task="transcribe"
    )
    tokens = tokenizer.encode(result['text'])

    # Prepare output
    output = {
        "audio_file": os.path.basename(audio_path),
        "model": model_name,
        "device": device,
        "language": result['language'],
        "encoder_time": encoder_time,
        "total_inference_time": total_time,
        "num_tokens": len(tokens),
        "text": result['text'],
        "segments": result['segments'],
        "model_load_time": model_load_time,
        "kv_cache_info": "Standard Whisper API uses internal KV cache optimization for decoder",
        "encoder_output_shape": list(encoder_output.shape),
        "audio_duration_s": 30.0
    }

    return output




def save_baseline_result(output, audio_basename):
    """保存baseline结果到JSON文件"""
    output_file = OUTPUT_DIR / f"baseline_{audio_basename}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nBaseline result saved to: {output_file}")

    # 同时保存纯文本结果
    text_file = OUTPUT_DIR / f"baseline_{audio_basename}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(output['text'])

    print(f"Baseline text saved to: {text_file}")


def main():
    """主函数：测试所有音频文件"""
    print(f"\nTest data directory: {TEST_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 查找所有音频文件
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3', '*.m4a']:
        audio_files.extend(TEST_DATA_DIR.glob(ext))

    if not audio_files:
        print("No audio files found in test_data directory!")
        return

    print(f"Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"  - {f.name}")

    # 测试每个音频文件
    all_results = {}

    for audio_file in audio_files:
        # 根据文件名判断语言
        audio_name = audio_file.stem
        if 'zh' in audio_name.lower() or 'chinese' in audio_name.lower():
            language = 'zh'
        elif 'en' in audio_name.lower() or 'english' in audio_name.lower() or 'jfk' in audio_name.lower():
            language = 'en'
        else:
            language = None  # 自动检测

        try:
            result = test_whisper_with_kv_cache(
                str(audio_file),
                model_name="base",
                language=language
            )

            # 保存结果
            save_baseline_result(result, audio_name)
            all_results[audio_name] = result

        except Exception as e:
            print(f"\nError processing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存汇总结果
    summary_file = OUTPUT_DIR / "baseline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"All baseline results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
