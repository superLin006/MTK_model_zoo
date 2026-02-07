#!/usr/bin/env python3
"""
生成vocab.txt文件（保持base64格式）

从官方的multilingual.tiktoken文件（格式：base64 rank）
转换为C++可读取的格式（格式：rank base64）
并添加special tokens
"""

import sys
from pathlib import Path

def generate_vocab():
    # 输入：官方tiktoken文件
    tiktoken_file = Path(__file__).parent.parent.parent / "whisper-official/whisper/assets/multilingual.tiktoken"

    # 输出：保持base64格式的vocab.txt
    output_file = Path(__file__).parent / "models/vocab.txt"

    if not tiktoken_file.exists():
        print(f"[ERROR] tiktoken文件不存在: {tiktoken_file}")
        return 1

    print(f"[INFO] 读取tiktoken文件: {tiktoken_file}")
    print(f"[INFO] 输出vocab文件: {output_file}")

    vocab_entries = []

    # 读取基础词表（50257个token）
    with open(tiktoken_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 格式: "base64_token rank"
            parts = line.split()
            if len(parts) != 2:
                print(f"[WARN] 跳过格式错误的行 {line_num}: {line}")
                continue

            base64_token, rank = parts
            rank = int(rank)

            # 保持base64格式（不解码！）
            vocab_entries.append((rank, base64_token))

    # 添加special tokens（50257-51864）
    # 参考 whisper/tokenizer.py
    special_tokens = [
        "<|endoftext|>",          # 50257
        "<|startoftranscript|>",  # 50258
    ]

    # 语言tokens (50259-50357, 99种语言)
    languages = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ]

    for lang in languages:
        special_tokens.append(f"<|{lang}|>")

    # 其他special tokens
    special_tokens.extend([
        "<|translate|>",      # 50358
        "<|transcribe|>",     # 50359
        "<|startoflm|>",      # 50360
        "<|startofprev|>",    # 50361
        "<|nospeech|>",       # 50362
        "<|notimestamps|>",   # 50363
    ])

    # 时间戳tokens (50364-51864)
    for i in range(1501):  # 0.00 到 30.00秒，每0.02秒一个
        timestamp = i * 0.02
        special_tokens.append(f"<|{timestamp:.2f}|>")

    # 添加special tokens到词表
    start_idx = 50257
    for i, token in enumerate(special_tokens):
        vocab_entries.append((start_idx + i, token))

    # 按rank排序
    vocab_entries.sort(key=lambda x: x[0])

    # 写入文件（格式：rank token）
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for rank, token in vocab_entries:
            f.write(f"{rank} {token}\n")

    print(f"[INFO] ✓ 成功生成 {len(vocab_entries)} 个词表条目")
    print(f"[INFO]   - 基础词表: 0-50256 ({50257} tokens)")
    print(f"[INFO]   - Special tokens: 50257-{vocab_entries[-1][0]} ({len(special_tokens)} tokens)")
    print(f"[INFO] ✓ 文件已保存: {output_file}")

    # 显示前10个和special tokens示例
    print("\n[INFO] 前10个token (base64编码):")
    for rank, token in vocab_entries[:10]:
        print(f"  {rank:5d} {token}")

    print("\n[INFO] Special tokens示例:")
    for rank, token in vocab_entries[50257:50270]:
        print(f"  {rank:5d} {token}")

    return 0

if __name__ == "__main__":
    sys.exit(generate_vocab())
