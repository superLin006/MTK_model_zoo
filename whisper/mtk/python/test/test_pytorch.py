"""
Whisper PyTorch Baseline Test
用于生成baseline结果，供后续MTK NPU移植对比使用
"""

import os
import sys
import torch
import whisper
import time
import json
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "python" / "test" / "outputs"

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)


def test_whisper_baseline(audio_path, model_name="base", language=None):
    """
    运行Whisper baseline测试

    Args:
        audio_path: 音频文件路径
        model_name: 模型名称 (tiny, base, small, medium, large)
        language: 语言代码 (en, zh等)，None表示自动检测

    Returns:
        dict: 包含识别结果、耗时等信息的字典
    """
    print(f"\n{'='*60}")
    print(f"Whisper PyTorch Baseline Test")
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
    # Force CPU for stability in WSL/non-CUDA environments
    device = "cpu"
    print(f"Using device: {device}")

    model = whisper.load_model(model_name, device=device)
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f}s")

    # 运行推理
    print(f"\nRunning inference...")
    start_time = time.time()

    result = model.transcribe(
        audio_path,
        language=language,
        temperature=0.0,  # 使用确定性解码
        verbose=False
    )

    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f}s")

    # 提取结果
    output = {
        "audio_file": os.path.basename(audio_path),
        "model": model_name,
        "device": device,
        "language_detected": result["language"],
        "language_specified": language,
        "text": result["text"],
        "segments": result["segments"],
        "model_load_time": model_load_time,
        "inference_time": inference_time,
        "total_time": model_load_time + inference_time
    }

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Detected Language: {result['language']}")
    print(f"Transcription: {result['text']}")
    print(f"\nSegments:")
    for i, segment in enumerate(result['segments'], 1):
        print(f"  [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    print(f"\nTiming:")
    print(f"  Model Load: {model_load_time:.2f}s")
    print(f"  Inference: {inference_time:.2f}s")
    print(f"  Total: {output['total_time']:.2f}s")

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
            result = test_whisper_baseline(
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
