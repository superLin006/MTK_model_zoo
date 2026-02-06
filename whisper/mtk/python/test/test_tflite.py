#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Whisper TFLite模型推理
使用MTK TFLite Runtime
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow未安装")
    sys.exit(1)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  whisper未安装，将跳过音频预处理")


class WhisperTFLiteInference:
    """Whisper TFLite推理类"""
    
    def __init__(self, encoder_path, decoder_path, embedding_path):
        """
        初始化
        
        Args:
            encoder_path: Encoder TFLite模型路径
            decoder_path: Decoder TFLite模型路径  
            embedding_path: Token embedding权重路径(.npy)
        """
        print("="*70)
        print("初始化Whisper TFLite模型")
        print("="*70)
        
        # 加载Encoder
        print(f"加载Encoder: {os.path.basename(encoder_path)}")
        self.encoder_interpreter = tf.lite.Interpreter(model_path=encoder_path)
        self.encoder_interpreter.allocate_tensors()
        
        self.encoder_input_details = self.encoder_interpreter.get_input_details()
        self.encoder_output_details = self.encoder_interpreter.get_output_details()
        
        print(f"  输入形状: {self.encoder_input_details[0]['shape']}")
        print(f"  输出形状: {self.encoder_output_details[0]['shape']}")
        
        # 加载Decoder
        print(f"\n加载Decoder: {os.path.basename(decoder_path)}")
        self.decoder_interpreter = tf.lite.Interpreter(model_path=decoder_path)
        self.decoder_interpreter.allocate_tensors()
        
        self.decoder_input_details = self.decoder_interpreter.get_input_details()
        self.decoder_output_details = self.decoder_interpreter.get_output_details()
        
        print(f"  输入1形状: {self.decoder_input_details[0]['shape']}")
        print(f"  输入2形状: {self.decoder_input_details[1]['shape']}")
        print(f"  输出形状: {self.decoder_output_details[0]['shape']}")
        
        # 加载Token Embedding权重
        print(f"\n加载Token Embedding: {os.path.basename(embedding_path)}")
        self.token_embedding = np.load(embedding_path)
        print(f"  形状: {self.token_embedding.shape}")
        print(f"  词汇表大小: {self.token_embedding.shape[0]}")
        print(f"  Embedding维度: {self.token_embedding.shape[1]}")
        
        # Whisper特殊token
        self.SOT = 50258  # Start of transcript
        self.EOT = 50257  # End of transcript
        self.TRANSCRIBE = 50359  # Transcribe task
        self.NO_TIMESTAMPS = 50363  # No timestamps
        
        # 语言token (简化版本)
        self.LANG_EN = 50259  # English
        self.LANG_ZH = 50260  # Chinese
        
        print("\n✓ 模型加载完成")
        print("="*70)
    
    def encode(self, mel):
        """
        Encoder推理
        
        Args:
            mel: mel-spectrogram [1, 80, 3000]
            
        Returns:
            encoder_output: [1, 1500, 512]
        """
        # 设置输入
        self.encoder_interpreter.set_tensor(
            self.encoder_input_details[0]['index'],
            mel.astype(np.float32)
        )
        
        # 推理
        self.encoder_interpreter.invoke()
        
        # 获取输出
        encoder_output = self.encoder_interpreter.get_tensor(
            self.encoder_output_details[0]['index']
        )
        
        return encoder_output
    
    def decode_step(self, token_embeddings, encoder_output):
        """
        Decoder单步推理
        
        Args:
            token_embeddings: [1, seq_len, 512]
            encoder_output: [1, 1500, 512]
            
        Returns:
            logits: [1, seq_len, 51865]
        """
        # 设置输入
        self.decoder_interpreter.set_tensor(
            self.decoder_input_details[0]['index'],
            token_embeddings.astype(np.float32)
        )
        self.decoder_interpreter.set_tensor(
            self.decoder_input_details[1]['index'],
            encoder_output.astype(np.float32)
        )
        
        # 推理
        self.decoder_interpreter.invoke()
        
        # 获取输出
        logits = self.decoder_interpreter.get_tensor(
            self.decoder_output_details[0]['index']
        )
        
        return logits
    
    def embed_tokens(self, token_ids):
        """
        Token ID转Embedding
        
        Args:
            token_ids: [seq_len] numpy array
            
        Returns:
            embeddings: [1, seq_len, 512]
        """
        embeddings = self.token_embedding[token_ids]
        return np.expand_dims(embeddings, 0)  # Add batch dim
    
    def decode_greedy(self, encoder_output, language='en', max_len=448):
        """
        贪婪解码（简化版本，无KV cache）
        
        Args:
            encoder_output: [1, 1500, 512]
            language: 'en' or 'zh'
            max_len: 最大序列长度
            
        Returns:
            token_ids: 解码的token序列
        """
        # 初始token序列
        if language == 'zh':
            tokens = [self.SOT, self.LANG_ZH, self.TRANSCRIBE, self.NO_TIMESTAMPS]
        else:
            tokens = [self.SOT, self.LANG_EN, self.TRANSCRIBE, self.NO_TIMESTAMPS]
        
        for i in range(max_len - len(tokens)):
            # 获取当前序列的embeddings
            token_embeddings = self.embed_tokens(np.array(tokens))
            
            # Pad到max_len（Decoder输入固定形状）
            if token_embeddings.shape[1] < max_len:
                pad_len = max_len - token_embeddings.shape[1]
                padding = np.zeros((1, pad_len, 512), dtype=np.float32)
                token_embeddings = np.concatenate([token_embeddings, padding], axis=1)
            
            # Decoder推理
            logits = self.decode_step(token_embeddings, encoder_output)
            
            # 只取当前位置的logits
            current_logits = logits[0, len(tokens)-1, :]
            
            # 贪婪采样
            next_token = np.argmax(current_logits)
            
            # 添加到序列
            tokens.append(int(next_token))
            
            # 如果是EOT，停止
            if next_token == self.EOT:
                break
        
        return tokens
    
    def transcribe(self, audio_path, language=None):
        """
        完整的语音识别流程
        
        Args:
            audio_path: 音频文件路径
            language: 'en', 'zh' 或 None (自动检测)
            
        Returns:
            text: 识别的文本
            tokens: token序列
        """
        print(f"\n处理音频: {os.path.basename(audio_path)}")
        
        # 1. 音频预处理 (使用whisper库)
        if WHISPER_AVAILABLE:
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).numpy()
        else:
            print("  ⚠️  跳过音频预处理（whisper库未安装）")
            # 使用dummy数据测试
            mel = np.random.randn(1, 80, 3000).astype(np.float32)
        
        print(f"  Mel shape: {mel.shape}")
        
        # 2. Encoder推理
        start = time.time()
        encoder_output = self.encode(mel)
        encoder_time = time.time() - start
        print(f"  Encoder推理: {encoder_time:.2f}s")
        print(f"  Encoder输出: {encoder_output.shape}")
        
        # 3. 语言检测（简化版本）
        if language is None:
            # 根据文件名猜测
            if 'zh' in audio_path or 'cn' in audio_path:
                language = 'zh'
            else:
                language = 'en'
        print(f"  语言: {language}")
        
        # 4. Decoder解码
        start = time.time()
        tokens = self.decode_greedy(encoder_output, language=language)
        decode_time = time.time() - start
        print(f"  Decoder解码: {decode_time:.2f}s")
        print(f"  生成tokens: {len(tokens)}")
        
        # 5. Token转文本
        if WHISPER_AVAILABLE:
            from whisper.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(multilingual=True)
            text = tokenizer.decode(tokens)
        else:
            text = f"[Tokens: {tokens}]"
        
        total_time = encoder_time + decode_time
        
        return {
            'text': text,
            'tokens': tokens,
            'language': language,
            'encoder_time': encoder_time,
            'decode_time': decode_time,
            'total_time': total_time
        }


def main():
    """主测试流程"""
    print("\n" + "="*70)
    print("Whisper TFLite模型推理测试")
    print("="*70)
    
    # 路径配置
    base_dir = Path(__file__).parent.parent
    encoder_path = base_dir / "models" / "encoder_base_80x3000.tflite"
    decoder_path = base_dir / "models" / "decoder_base_448.tflite"
    embedding_path = base_dir / "models" / "token_embedding.npy"
    
    # 检查文件
    for path in [encoder_path, decoder_path, embedding_path]:
        if not path.exists():
            print(f"❌ 文件不存在: {path}")
            return
    
    # 初始化推理器
    inference = WhisperTFLiteInference(
        str(encoder_path),
        str(decoder_path),
        str(embedding_path)
    )
    
    # 测试音频
    test_data_dir = base_dir.parent / "test_data"
    test_files = [
        ("test_en.wav", "en"),
        ("test_zh.wav", "zh"),
        ("jfk.flac", "en")
    ]
    
    # 输出目录
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    # 对每个测试文件进行推理
    for audio_file, lang in test_files:
        audio_path = test_data_dir / audio_file
        
        if not audio_path.exists():
            print(f"\n⚠️  音频文件不存在: {audio_path}")
            continue
        
        # 推理
        result = inference.transcribe(str(audio_path), language=lang)
        
        # 显示结果
        print(f"\n识别结果:")
        print(f"  文本: {result['text']}")
        print(f"  Tokens: {result['tokens'][:10]}... (共{len(result['tokens'])}个)")
        print(f"  总时间: {result['total_time']:.2f}s")
        
        # 保存结果
        output_file = output_dir / f"tflite_{audio_file.replace('.flac', '.wav').replace('.wav', '.json')}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  保存到: {output_file.name}")
        
        results.append({
            'audio': audio_file,
            'text': result['text'],
            'time': result['total_time']
        })
    
    # 生成总结报告
    summary_path = output_dir / "tflite_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✓ TFLite测试完成")
    print("="*70)
    print(f"\n结果总结:")
    for r in results:
        print(f"  {r['audio']}: {r['text'][:50]}... ({r['time']:.2f}s)")
    print(f"\n详细结果: {output_dir}")


if __name__ == '__main__':
    main()
