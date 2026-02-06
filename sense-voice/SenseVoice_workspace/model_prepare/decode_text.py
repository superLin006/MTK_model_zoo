#!/usr/bin/env python3
"""
SenseVoice CTC 解码工具
将 logits 转换为文本
"""

import numpy as np
import argparse
import sys


def load_vocab(vocab_file):
    """加载词汇表"""
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: token id 或 token id (带空格)
            parts = line.rsplit(' ', 1) if ' ' in line else line.rsplit('\t', 1)
            if len(parts) == 2:
                token, token_id = parts
                try:
                    vocab[int(token_id)] = token
                except ValueError:
                    continue
            else:
                vocab[len(vocab)] = line
    print(f"加载词汇表: {len(vocab)} tokens")
    return vocab


def ctc_decode(logits, vocab, blank_id=0):
    """
    CTC 解码

    Args:
        logits: [T, vocab_size] 或 [1, T, vocab_size]
        vocab: 词汇表字典 {id: token}
        blank_id: blank token id

    Returns:
        text: 解码后的文本
    """
    # 确保2D
    if len(logits.shape) == 3:
        logits = logits.squeeze(0)

    # Argmax 获取每帧的token
    tokens = logits.argmax(axis=-1)

    # CTC collapse (移除重复和blank)
    collapsed = []
    prev_token = None

    for token in tokens:
        if token == blank_id:
            prev_token = None
            continue
        if token != prev_token:
            collapsed.append(token)
            prev_token = token

    # 转换为文本
    text = "".join([vocab.get(t, f"<UNK_{t}>") for t in collapsed])

    return text, collapsed


def remove_special_tokens(text):
    """移除特殊token"""
    cleaned = text
    while cleaned.startswith("<|"):
        end_idx = cleaned.find("|>")
        if end_idx == -1:
            break
        cleaned = cleaned[end_idx + 2:]

    # 替换 ▁ 为空格
    cleaned = cleaned.replace("▁", " ").strip()

    return cleaned


def main():
    parser = argparse.ArgumentParser(description='SenseVoice CTC 解码')
    parser.add_argument('--logits', type=str, required=True,
                        help='Logits 文件路径 (.npy)')
    parser.add_argument('--tokens', type=str, required=True,
                        help='词汇表文件路径')
    parser.add_argument('--blank_id', type=int, default=0,
                        help='Blank token ID (默认: 0)')

    args = parser.parse_args()

    print("="*80)
    print("  SenseVoice CTC 解码")
    print("="*80)
    print()

    # 加载数据
    print(f"加载 logits: {args.logits}")
    logits = np.load(args.logits)
    print(f"  Shape: {logits.shape}")
    print()

    # 加载词汇表
    print(f"加载词汇表: {args.tokens}")
    vocab = load_vocab(args.tokens)
    print()

    # 解码
    print("CTC 解码中...")
    raw_text, tokens = ctc_decode(logits, vocab, args.blank_id)
    print(f"  Token 数量: {len(tokens)}")
    print(f"  原始文本: {raw_text}")
    print()

    # 后处理
    print("后处理...")
    final_text = remove_special_tokens(raw_text)
    print(f"  最终文本: {final_text}")
    print()

    # 保存结果
    output_file = "output/transcription.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)

    print("="*80)
    print(f"✅ 解码完成，已保存到: {output_file}")
    print("="*80)

    # 显示结果
    print()
    print("识别结果:")
    print(f"  \"{final_text}\"")
    print()


if __name__ == "__main__":
    main()
