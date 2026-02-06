#!/usr/bin/env python3
"""
测试TorchScript模型（.pt）

任务：
1. 加载encoder_base_3000.pt和decoder_base_448.pt
2. 加载token_embedding.npy（模拟C++端的手动查表）
3. 使用测试音频进行推理
4. 对比baseline结果
5. 保存输出

模拟C++端的工作流程：
- 手动实现token embedding lookup
- 实现简单的自回归解码循环
"""

import sys
import os
import json
import time
import numpy as np
import torch

# 添加whisper官方库路径
sys.path.append('/home/xh/projects/MTK/whisper/whisper-official')
import whisper
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from whisper.tokenizer import get_tokenizer


class WhisperTorchScriptInference:
    """使用TorchScript模型进行推理"""

    def __init__(self, encoder_pt_path, decoder_pt_path, token_embedding_path, device='cpu'):
        """
        Args:
            encoder_pt_path: Encoder TorchScript模型路径
            decoder_pt_path: Decoder TorchScript模型路径
            token_embedding_path: Token embedding权重路径 (.npy)
            device: 运行设备
        """
        self.device = device

        print("Loading TorchScript models...")

        # 加载Encoder
        print(f"  Loading encoder: {encoder_pt_path}")
        self.encoder = torch.jit.load(encoder_pt_path, map_location=device)
        self.encoder.eval()

        # 加载Decoder
        print(f"  Loading decoder: {decoder_pt_path}")
        self.decoder = torch.jit.load(decoder_pt_path, map_location=device)
        self.decoder.eval()

        # 加载Token Embedding权重（模拟C++端的行为）
        print(f"  Loading token embedding: {token_embedding_path}")
        self.token_embedding_weight = np.load(token_embedding_path)
        self.vocab_size, self.embedding_dim = self.token_embedding_weight.shape
        print(f"    Vocab size: {self.vocab_size}")
        print(f"    Embedding dim: {self.embedding_dim}")

        # 获取tokenizer
        self.tokenizer = get_tokenizer(multilingual=True)

        print("✓ Models loaded successfully!")

    def embed_tokens(self, token_ids):
        """
        手动实现token embedding lookup（模拟C++端）

        Args:
            token_ids: [batch, seq_len] 或 [seq_len]

        Returns:
            embeddings: [batch, seq_len, embedding_dim]
        """
        # 确保是2D
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        batch_size, seq_len = token_ids.shape

        # 手动查表（模拟C++的memcpy操作）
        embeddings = np.zeros((batch_size, seq_len, self.embedding_dim), dtype=np.float32)

        for b in range(batch_size):
            for i in range(seq_len):
                token_id = token_ids[b, i].item()
                if token_id < self.vocab_size:
                    embeddings[b, i] = self.token_embedding_weight[token_id]

        return torch.from_numpy(embeddings).to(self.device)

    def encode_audio(self, audio_path):
        """
        加载音频并编码为特征

        Args:
            audio_path: 音频文件路径

        Returns:
            encoder_output: [1, 1500, 512]
        """
        # 加载音频（30秒）
        audio = load_audio(audio_path)
        audio = pad_or_trim(audio, 30 * 16000)  # 30秒

        # 计算mel-spectrogram
        mel = log_mel_spectrogram(audio)  # [80, 3000]
        mel = mel.unsqueeze(0).to(self.device)  # [1, 80, 3000]

        # Encoder前向传播
        with torch.no_grad():
            encoder_output = self.encoder(mel)

        return encoder_output

    def decode_greedy(self, encoder_output, language='en', max_tokens=448):
        """
        贪婪解码（不使用KV cache）

        Args:
            encoder_output: [1, 1500, 512]
            language: 语言代码
            max_tokens: 最大token数

        Returns:
            tokens: 生成的token列表
        """
        # 初始tokens: [SOT, language, transcribe, no_timestamps]
        sot_token = self.tokenizer.sot
        language_token = self.tokenizer.special_tokens.get(f'<|{language}|>', self.tokenizer.language_token)
        transcribe_token = self.tokenizer.transcribe
        no_timestamps_token = self.tokenizer.no_timestamps

        tokens = [sot_token, language_token, transcribe_token, no_timestamps_token]

        # 自回归解码
        for _ in range(max_tokens - len(tokens)):
            # 当前tokens转为embeddings
            current_tokens = torch.tensor([tokens], device=self.device)
            token_embeddings = self.embed_tokens(current_tokens)

            # Decoder前向传播
            with torch.no_grad():
                logits = self.decoder(token_embeddings, encoder_output)

            # 获取最后一个位置的预测
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            next_token = torch.argmax(next_token_logits).item()

            # 添加到序列
            tokens.append(next_token)

            # 检查结束token
            if next_token == self.tokenizer.eot:
                break

        return tokens

    def transcribe(self, audio_path, language='en'):
        """
        完整转录流程

        Args:
            audio_path: 音频文件路径
            language: 语言代码

        Returns:
            result: 转录结果字典
        """
        print(f"\nTranscribing: {audio_path}")
        print(f"Language: {language}")

        # 编码音频
        print("  [1/2] Encoding audio...")
        start = time.time()
        encoder_output = self.encode_audio(audio_path)
        encode_time = time.time() - start
        print(f"    ✓ Encoder output: {encoder_output.shape} ({encode_time:.2f}s)")

        # 解码
        print("  [2/2] Decoding...")
        start = time.time()
        tokens = self.decode_greedy(encoder_output, language=language)
        decode_time = time.time() - start
        print(f"    ✓ Generated {len(tokens)} tokens ({decode_time:.2f}s)")

        # 解码为文本（去除特殊token）
        # 从tokens中移除开头的特殊token和结尾的EOT
        # 格式: [SOT, language, task, no_timestamps, ...actual_tokens..., EOT]
        text_tokens = tokens[4:]  # 跳过前4个特殊token
        if text_tokens and text_tokens[-1] == self.tokenizer.eot:
            text_tokens = text_tokens[:-1]  # 移除EOT

        text = self.tokenizer.decode(text_tokens).strip()

        result = {
            'audio_file': os.path.basename(audio_path),
            'language': language,
            'tokens': tokens,
            'text': text,
            'encode_time': encode_time,
            'decode_time': decode_time,
            'total_time': encode_time + decode_time,
            'num_tokens': len(tokens)
        }

        print(f"\n  Result: {text}")
        print(f"  Total time: {result['total_time']:.2f}s")

        return result


def compare_with_baseline(result, baseline_path):
    """
    对比baseline结果

    Args:
        result: 当前结果
        baseline_path: baseline结果路径

    Returns:
        comparison: 对比结果
    """
    if not os.path.exists(baseline_path):
        print(f"\n  Warning: Baseline not found: {baseline_path}")
        return None

    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    # 对比文本
    baseline_text = baseline.get('text', '').strip()
    result_text = result.get('text', '').strip()

    text_match = baseline_text == result_text

    comparison = {
        'text_match': text_match,
        'baseline_text': baseline_text,
        'result_text': result_text,
        'baseline_tokens': baseline.get('segments', [{}])[0].get('tokens', []) if baseline.get('segments') else [],
        'result_tokens': result.get('tokens', []),
    }

    return comparison


def main():
    """主测试流程"""
    print("="*70)
    print("测试 TorchScript 模型 (.pt)")
    print("="*70)

    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    test_data_dir = os.path.join(os.path.dirname(base_dir), 'test_data')
    output_dir = os.path.join(base_dir, 'test', 'outputs')
    baseline_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 模型路径
    encoder_pt = os.path.join(models_dir, 'encoder_base_3000.pt')
    decoder_pt = os.path.join(models_dir, 'decoder_base_448.pt')
    token_embedding = os.path.join(models_dir, 'token_embedding.npy')

    # 检查文件
    for path in [encoder_pt, decoder_pt, token_embedding]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    # 创建推理器
    print("\n" + "="*70)
    print("初始化推理器")
    print("="*70)
    inference = WhisperTorchScriptInference(
        encoder_pt_path=encoder_pt,
        decoder_pt_path=decoder_pt,
        token_embedding_path=token_embedding,
        device='cpu'
    )

    # 测试用例
    test_cases = [
        {
            'name': 'test_zh',
            'audio': os.path.join(test_data_dir, 'test_zh.wav'),
            'language': 'zh',
            'baseline': os.path.join(baseline_dir, 'baseline_test_zh.json')
        },
        {
            'name': 'test_en',
            'audio': os.path.join(test_data_dir, 'test_en.wav'),
            'language': 'en',
            'baseline': os.path.join(baseline_dir, 'baseline_test_en.json')
        },
        {
            'name': 'jfk',
            'audio': os.path.join(test_data_dir, 'jfk.flac'),
            'language': 'en',
            'baseline': os.path.join(baseline_dir, 'baseline_jfk.json')
        }
    ]

    # 运行测试
    results = {}

    for i, test_case in enumerate(test_cases):
        print("\n" + "="*70)
        print(f"测试 {i+1}/{len(test_cases)}: {test_case['name']}")
        print("="*70)

        # 检查音频文件
        if not os.path.exists(test_case['audio']):
            print(f"  Warning: Audio not found: {test_case['audio']}")
            continue

        # 转录
        result = inference.transcribe(
            audio_path=test_case['audio'],
            language=test_case['language']
        )

        # 对比baseline
        print("\n  Comparing with baseline...")
        comparison = compare_with_baseline(result, test_case['baseline'])

        if comparison:
            result['comparison'] = comparison
            match_str = "✓ MATCH" if comparison['text_match'] else "✗ MISMATCH"
            print(f"    Text match: {match_str}")

            if not comparison['text_match']:
                print(f"    Baseline: {comparison['baseline_text']}")
                print(f"    Result:   {comparison['result_text']}")

        # 保存结果
        output_path = os.path.join(output_dir, f'pt_{test_case["name"]}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved: {output_path}")

        results[test_case['name']] = result

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Text: {result['text']}")
        print(f"  Tokens: {result['num_tokens']}")
        print(f"  Time: {result['total_time']:.2f}s")

        if 'comparison' in result:
            match = result['comparison']['text_match']
            print(f"  Baseline match: {'✓ Yes' if match else '✗ No'}")

    # 保存汇总
    summary_path = os.path.join(output_dir, 'pt_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Summary saved: {summary_path}")

    print("\n" + "="*70)
    print("✓ TorchScript 测试完成！")
    print("="*70)


if __name__ == '__main__':
    main()
