#!/usr/bin/env python3
"""
SenseVoice Small ASR Model Conversion Pipeline
Complete model: CMVN + Encoder + CTC
"""
import torch
import os
import numpy as np
from config import *
import argparse
import sys

# Import our custom modules
from torch_model import SenseVoiceSmall
from model_utils import create_sensevoice_model, save_torchscript

# Try to import FunASR (for baseline comparison only)
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    print("Warning: FunASR not installed. Install with: pip install funasr")
    FUNASR_AVAILABLE = False

try:
    import mtk_converter
    MTK_CONVERTER_AVAILABLE = True
except ImportError:
    print("Warning: mtk_converter not available")
    MTK_CONVERTER_AVAILABLE = False


def create_prompt(language="en", text_norm="woitn", event1=1, event2=2):
    """
    创建 prompt 张量

    Args:
        language: 语言代码 (auto, zh, en, yue, ja, ko, nospeech)
        text_norm: 文本规范化模式 (withitn, woitn)
        event1: 事件类型 1 (HAPPY=1, SAD=2, ANGRY=3, NEUTRAL=4)
        event2: 事件类型 2 (Speech=2, Music=3, Applause=4)

    Returns:
        prompt: torch.Tensor [4] - [language_id, event1, event2, text_norm_id]
    """
    lid_dict = {
        "auto": 0, "zh": 3, "en": 4, "yue": 7,
        "ja": 11, "ko": 12, "nospeech": 13
    }
    textnorm_dict = {"withitn": 14, "woitn": 15}

    language_id = lid_dict.get(language, 4)
    text_norm_id = textnorm_dict.get(text_norm, 15)

    prompt = torch.tensor([language_id, event1, event2, text_norm_id], dtype=torch.int32)
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(description='SenseVoice Model Conversion')
    parser.add_argument('--mode', type=str, default=None,
                        choices=["SAVE_PT", "PYTORCH", "CHECK_TFLITE"],
                        help="activate different mode for porting and testing")
    parser.add_argument('--model_path', type=str,
                        default="../models/sensevoice-small",
                        help="Path to SenseVoice model directory")
    parser.add_argument('--audio_path', type=str,
                        default="../audios/test_en.wav",
                        help="Path to test audio file")
    parser.add_argument('--tflite_file_path', type=str, default=None,
                        help="TFLite file path for validation")
    parser.add_argument('--language', type=str, default="en",
                        choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                        help="Language for prompt (default: en)")
    parser.add_argument('--text_norm', type=str, default="woitn",
                        choices=["withitn", "woitn"],
                        help="Text normalization mode (default: woitn)")
    args = parser.parse_args()
    return args


def jit_model_save(model, save_filename, model_inputs):
    """Save model as TorchScript"""
    model = model.cpu()
    torch.jit.save(torch.jit.trace(model.eval(), model_inputs, strict=False), save_filename)
    print(f"Saved TorchScript model to: {save_filename}")


def save_file(x, filename):
    """Save numpy array to binary file"""
    x.tofile(filename)
    print(f"Saved {filename}")
    return


def load_and_preprocess_audio(audio_path, target_sr=16000):
    """
    Load audio file and convert to features
    Returns: mel-spectrogram features for SenseVoice
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("Error: librosa and soundfile required. Install with:")
        print("  pip install librosa soundfile")
        sys.exit(1)

    # Load audio
    print(f"Loading audio from: {audio_path}")
    audio, sr = sf.read(audio_path)

    # Resample if necessary
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    print(f"Audio shape: {audio.shape}, duration: {len(audio)/target_sr:.2f}s")
    return audio, target_sr


def extract_encoder_only(full_model):
    """
    Extract encoder-only part from SenseVoice model
    Similar to how we extracted text_model from CLIP
    """
    # SenseVoice structure: model.encoder
    if hasattr(full_model, 'encoder'):
        encoder = full_model.encoder
        print("✅ Extracted encoder from SenseVoice model")
        return encoder
    else:
        print("❌ Warning: Could not find encoder in model")
        return full_model


def save_model_complete(model, audio_features, language_id, event_id, event_type_id, text_norm_id, model_name="sensevoice_complete"):
    """
    Save complete SenseVoice model as TorchScript

    Args:
        model: Complete SenseVoice model (CMVN + Encoder + CTC)
        audio_features: Sample audio features [1, T, 560]
        language_id: Language ID scalar
        event_id: Event ID scalar
        event_type_id: Event type ID scalar
        text_norm_id: Text normalization ID scalar
        model_name: Output model name
    """
    if not os.path.exists('model'):
        os.mkdir('model')

    model_file = f'model/{model_name}.pt'
    print(f"Saving TorchScript to: {model_file}")

    if not os.path.isfile(model_file):
        model = model.cpu()
        model.eval()

        # Trace with sample inputs (5 inputs instead of 2)
        with torch.no_grad():
            save_torchscript(model, model_file, (audio_features, language_id, event_id, event_type_id, text_norm_id))
    else:
        print(f"{model_file} already exists.")

    return model_file


def inference_pytorch_custom(model, audio_path, language="en", text_norm="woitn"):
    """
    Run PyTorch inference with our custom model

    注意: 此功能已移至 test_converted_models.py
    test_converted_models.py 使用 FunASR 的特征提取，确保准确性

    这里只显示提示信息
    """
    print("\n" + "="*80)
    print("PYTORCH INFERENCE")
    print("="*80)
    print("\n⚠️  注意: PYTORCH 模式的推理已移至 test_converted_models.py")
    print("test_converted_models.py 使用 FunASR 提取特征，确保与原始模型一致")
    print(f"\n请运行:")
    print(f"  python3 test_converted_models.py --audio {audio_path} --language {language}")
    print("="*80 + "\n")

    # 不执行实际的推理，返回 None
    return None, None, None


def inference_pytorch_funasr(audio_path):
    """
    Run FunASR baseline inference (for comparison)

    Args:
        audio_path: Path to audio file

    Returns:
        result: FunASR transcription result
    """
    if not FUNASR_AVAILABLE:
        print("FunASR not available, skipping baseline inference")
        return None

    print("\n" + "="*80)
    print("FUNASR BASELINE INFERENCE")
    print("="*80 + "\n")

    # Load FunASR model
    model = AutoModel(model="../models/sensevoice-small")

    # Run inference
    result = model.generate(input=audio_path, batch_size_s=300)
    print(f"\n📝 Transcription Result:")
    print(f"Text: {result[0]['text']}")

    # Save result
    import json
    output_file = "output/funasr_transcription.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved result to: {output_file}")

    return result


def inference_tflite(tflite_file_path, audio_path):
    """
    Run TFLite inference and compare with PyTorch
    """
    if not MTK_CONVERTER_AVAILABLE:
        print("Error: mtk_converter not available")
        return

    if not os.path.exists('output'):
        os.mkdir('output')

    print("TFLITE: Start to inference...")

    # Load audio and extract features
    audio, sr = load_and_preprocess_audio(audio_path)

    # Load TFLite model
    if tflite_file_path.endswith('.tflite'):
        executor = mtk_converter.TFLiteExecutor(tflite_file_path)
    elif tflite_file_path.endswith('.mlir'):
        executor = mtk_converter.MlirExecutor(tflite_file_path)
    else:
        print(f"Error: Unknown file format: {tflite_file_path}")
        return

    print(f"Loaded TFLite model from: {tflite_file_path}")

    # TODO: Need to prepare input features in the correct format
    # This depends on SenseVoice's frontend configuration
    print("Note: TFLite inference requires properly formatted input features")


def main(args):
    """Main conversion pipeline"""

    print(f"\n{'='*80}")
    print(f"SenseVoice Model Conversion - Mode: {args.mode}")
    print(f"{'='*80}\n")

    if args.mode == "PYTORCH":
        # Load custom model
        print("Loading custom SenseVoice model...")
        model = create_sensevoice_model(args.model_path)
        print("✅ Model loaded successfully\n")

        # Run custom inference
        logits, features, prompt = inference_pytorch_custom(
            model, args.audio_path,
            language=args.language,
            text_norm=args.text_norm
        )

        # Optionally run FunASR baseline for comparison
        if FUNASR_AVAILABLE:
            print("\n" + "="*80)
            print("Running FunASR baseline for comparison...")
            print("="*80)
            inference_pytorch_funasr(args.audio_path)

    elif args.mode == "SAVE_PT":
        # Load custom model
        print("Loading custom SenseVoice model...")
        model = create_sensevoice_model(args.model_path)
        print("✅ Model loaded successfully\n")

        # Use FIXED shape for 10-second audio (166 frames)
        print("Using FIXED input shape for 10-second audio...")
        fixed_frames = 166  # 10s audio: (16000*10 - 400)/160 + 1 = 998 fbank frames -> (998-7)/6+1 = 166 LFR frames
        features = torch.randn(1, fixed_frames, 560)  # Dummy features for tracing

        # Create prompt parameters (4 separate scalar inputs)
        prompt = create_prompt(language=args.language, text_norm=args.text_norm)
        language_id, event_id, event_type_id, text_norm_id = prompt

        print(f"\nModel inputs (FIXED for 10s audio):")
        print(f"  - Features: {features.shape} [1, 166, 560]")
        print(f"  - Language ID: {language_id}")
        print(f"  - Event ID: {event_id}")
        print(f"  - Event Type ID: {event_type_id}")
        print(f"  - Text Norm ID: {text_norm_id}")

        # Test forward pass
        print("\nTesting forward pass...")
        model.eval()
        with torch.no_grad():
            logits = model(features, language_id, event_id, event_type_id, text_norm_id)
        print(f"Output shape: {logits.shape} (expected: [1, 170, 25055])")

        # Save to TorchScript
        print("\nSaving model to TorchScript...")
        model_file = save_model_complete(model, features, language_id, event_id, event_type_id, text_norm_id, "sensevoice_complete")
        print(f"✅ Model saved to: {model_file}")
        print(f"\n📌 Note: Model is traced with FIXED shape [1, 166, 560] for 10-second audio")

    elif args.mode == "CHECK_TFLITE":
        if args.tflite_file_path is None:
            print("Error: --tflite_file_path required for CHECK_TFLITE mode")
            sys.exit(1)

        inference_tflite(args.tflite_file_path, args.audio_path)

    else:
        print(f"Error: Invalid mode or missing dependencies")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Mode {args.mode} completed successfully")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    args = parse_args()

    # Validate PYTORCH flag for non-PYTORCH modes
    if args.mode != "PYTORCH":
        if PYTORCH != 0:
            print(f"❌ Error: For {args.mode} mode, please set PYTORCH=0 in config.py")
            sys.exit(1)

    main(args)
