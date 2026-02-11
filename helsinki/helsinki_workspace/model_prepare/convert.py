#!/usr/bin/env python3
"""
转换带 KV Cache 的 Helsinki 模型 - V2 (4D tensors)
"""

import argparse
import torch
import numpy as np
import os
import subprocess
from pathlib import Path

# 项目根目录（convert.py 在 .../helsinki/helsinki_workspace/model_prepare/ 下，向上3级到 MTK_models_zoo）
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]  # MTK_models_zoo/

from mtk_model import create_encoder_decoder_kvcache_v2

try:
    import mtk_converter
    MTK_AVAILABLE = True
except ImportError:
    MTK_AVAILABLE = False
    print("Warning: mtk_converter not available")


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Helsinki model with KV Cache V2')
    parser.add_argument('--model_path', type=str, default='../models/Helsinki-NLP/opus-mt-en-zh')
    parser.add_argument('--src_seq_len', type=int, default=64)
    parser.add_argument('--max_cache_len', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='model_kvcache')
    parser.add_argument('--skip_tflite', action='store_true')
    parser.add_argument('--skip_dla', action='store_true')
    parser.add_argument('--platform', type=str, default='MT8371')
    return parser.parse_args()


def save_torchscript(model, output_path, example_inputs):
    print(f"\nTracing: {output_path}")
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)
    traced.save(output_path)
    print(f"  Saved: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    return traced


def convert_tflite(pt_path, tflite_path, input_shapes, input_types):
    if not MTK_AVAILABLE:
        return False

    print(f"\nConverting to TFLite: {tflite_path}")
    try:
        converter = mtk_converter.PyTorchConverter.from_script_module_file(
            pt_path, input_shapes=input_shapes, input_types=input_types,
        )
        converter.quantize = False
        tflite_model = converter.convert_to_tflite()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  ✅ Saved: {os.path.getsize(tflite_path) / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compile_dla(tflite_path, dla_path, platform):
    sdk = os.environ.get(
        'MTK_NEURON_SDK',
        str(_PROJECT_ROOT / '0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk')
    )
    ncc = f"{sdk}/host/bin/ncc-tflite"

    cfg = {
        'MT8371': {'arch': 'mdla5.3,edma3.6', 'l1': '256', 'mdla': '1'},
        'MT6899': {'arch': 'mdla5.5,edma3.6', 'l1': '2048', 'mdla': '2'},
        'MT6991': {'arch': 'mdla5.5,edma3.6', 'l1': '7168', 'mdla': '4'},
    }[platform]

    print(f"\nCompiling DLA: {dla_path}")
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f"{sdk}/host/lib:" + env.get('LD_LIBRARY_PATH', '')

    cmd = [ncc, tflite_path,
           f'--arch={cfg["arch"]}', f'--l1-size-kb={cfg["l1"]}', f'--num-mdla={cfg["mdla"]}',
           '--relax-fp32', '--opt-accuracy', '--opt-footprint', '--fc-to-conv',
           '-o', dla_path]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(dla_path):
            print(f"  ✅ Saved: {os.path.getsize(dla_path) / 1024 / 1024:.1f} MB")
            return True
        else:
            print(f"  ❌ Failed:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    args = parse_args()

    print("="*60)
    print("Helsinki KV Cache V2 Conversion")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Create models
    encoder, decoder, embedding_weights, position_embeddings, tokenizer, config = \
        create_encoder_decoder_kvcache_v2(args.model_path, args.max_cache_len)

    encoder.eval()
    decoder.eval()

    d_model = config.d_model
    num_layers = config.decoder_layers
    vocab_size = config.vocab_size

    # ============================================================
    # Save embedding weights
    # ============================================================
    embed_path = os.path.join(args.output_dir, "embedding_weights.bin")
    print(f"\nSaving embedding: {embed_path}")
    embedding_weights.numpy().astype(np.float32).tofile(embed_path)
    print(f"  Shape: {embedding_weights.shape}")

    # Save meta
    with open(embed_path.replace('.bin', '_meta.txt'), 'w') as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"d_model={d_model}\n")

    # ============================================================
    # Save position embeddings (for C++ side)
    # ============================================================
    pos_path = os.path.join(args.output_dir, "position_embeddings.bin")
    print(f"\nSaving position embeddings: {pos_path}")
    pos_data = position_embeddings.squeeze(0).numpy()  # [max_pos, d_model]
    pos_data.astype(np.float32).tofile(pos_path)
    print(f"  Shape: {pos_data.shape}")

    # ============================================================
    # Save final_logits_bias (for C++ side)
    # Note: This is built into the TFLite model, but save separately for reference
    # ============================================================
    bias_path = os.path.join(args.output_dir, "final_logits_bias.bin")
    print(f"\nSaving final_logits_bias: {bias_path}")
    bias_data = decoder.final_logits_bias.squeeze(0).numpy()  # [vocab_size]
    bias_data.astype(np.float32).tofile(bias_path)
    print(f"  Shape: {bias_data.shape}")

    # ============================================================
    # Convert Encoder
    # ============================================================
    print("\n" + "="*60)
    print("Converting Encoder")
    print("="*60)

    enc_pt = os.path.join(args.output_dir, f"encoder_src{args.src_seq_len}.pt")
    enc_tflite = os.path.join(args.output_dir, f"encoder_src{args.src_seq_len}.tflite")
    enc_dla = os.path.join(args.output_dir, f"encoder_src{args.src_seq_len}_{args.platform}.dla")

    enc_input = torch.randn(1, args.src_seq_len, d_model)
    enc_attn_mask = torch.zeros(1, 1, args.src_seq_len, args.src_seq_len)  # [1,1,seq,seq] for encoder self-attn

    print(f"\nEncoder inputs:")
    print(f"  encoder_embeds: {enc_input.shape}")
    print(f"  encoder_attn_mask: {enc_attn_mask.shape}")

    save_torchscript(encoder, enc_pt, (enc_input, enc_attn_mask))

    if not args.skip_tflite:
        enc_shapes = [
            [1, args.src_seq_len, d_model],  # encoder_embeds
            [1, 1, args.src_seq_len, args.src_seq_len],  # encoder_attn_mask
        ]
        enc_types = [torch.float32, torch.float32]
        if convert_tflite(enc_pt, enc_tflite, enc_shapes, enc_types):
            if not args.skip_dla:
                compile_dla(enc_tflite, enc_dla, args.platform)

    # ============================================================
    # Convert Decoder
    # ============================================================
    print("\n" + "="*60)
    print("Converting Decoder (KV Cache V2)")
    print("="*60)

    dec_pt = os.path.join(args.output_dir, f"decoder_kv_src{args.src_seq_len}_cache{args.max_cache_len}.pt")
    dec_tflite = os.path.join(args.output_dir, f"decoder_kv_src{args.src_seq_len}_cache{args.max_cache_len}.tflite")
    dec_dla = os.path.join(args.output_dir, f"decoder_kv_src{args.src_seq_len}_cache{args.max_cache_len}_{args.platform}.dla")

    # Decoder inputs (all 4D or less)
    decoder_embed = torch.randn(1, 1, d_model)
    encoder_hidden = torch.randn(1, args.src_seq_len, d_model)
    past_keys = torch.zeros(num_layers, 1, args.max_cache_len, d_model)
    past_values = torch.zeros(num_layers, 1, args.max_cache_len, d_model)
    position_embed = torch.randn(1, 1, d_model)
    attn_mask = torch.zeros(1, 1, 1, args.max_cache_len + 1)  # self-attention mask
    encoder_attn_mask = torch.zeros(1, 1, 1, args.src_seq_len)  # cross-attention mask

    print(f"\nDecoder inputs:")
    print(f"  decoder_embed: {decoder_embed.shape}")
    print(f"  encoder_hidden: {encoder_hidden.shape}")
    print(f"  past_keys: {past_keys.shape}")
    print(f"  past_values: {past_values.shape}")
    print(f"  position_embed: {position_embed.shape}")
    print(f"  attn_mask: {attn_mask.shape}")
    print(f"  encoder_attn_mask: {encoder_attn_mask.shape}")

    save_torchscript(decoder, dec_pt,
                    (decoder_embed, encoder_hidden, past_keys, past_values,
                     position_embed, attn_mask, encoder_attn_mask))

    if not args.skip_tflite:
        dec_shapes = [
            [1, 1, d_model],                             # decoder_embed
            [1, args.src_seq_len, d_model],              # encoder_hidden
            [num_layers, 1, args.max_cache_len, d_model], # past_keys
            [num_layers, 1, args.max_cache_len, d_model], # past_values
            [1, 1, d_model],                             # position_embed
            [1, 1, 1, args.max_cache_len + 1],           # attn_mask
            [1, 1, 1, args.src_seq_len],                 # encoder_attn_mask
        ]
        dec_types = [torch.float32] * 7

        if convert_tflite(dec_pt, dec_tflite, dec_shapes, dec_types):
            if not args.skip_dla:
                compile_dla(dec_tflite, dec_dla, args.platform)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    files = [embed_path, pos_path, enc_pt, enc_tflite, enc_dla, dec_pt, dec_tflite, dec_dla]
    for f in files:
        if os.path.exists(f):
            print(f"  ✅ {os.path.basename(f)}: {os.path.getsize(f)/1024/1024:.1f} MB")
        else:
            print(f"  ❌ {os.path.basename(f)}")

    print(f"\nEncoder inputs:")
    print(f"  encoder_embeds     [1, {args.src_seq_len}, {d_model}]")
    print(f"  encoder_attn_mask  [1, 1, {args.src_seq_len}, {args.src_seq_len}]")
    print(f"Encoder output:")
    print(f"  encoder_hidden     [1, {args.src_seq_len}, {d_model}]")
    print(f"\nDecoder inputs:")
    print(f"  decoder_embed      [1, 1, {d_model}]")
    print(f"  encoder_hidden     [1, {args.src_seq_len}, {d_model}]")
    print(f"  past_keys          [{num_layers}, 1, {args.max_cache_len}, {d_model}]")
    print(f"  past_values        [{num_layers}, 1, {args.max_cache_len}, {d_model}]")
    print(f"  position_embed     [1, 1, {d_model}]")
    print(f"  attn_mask          [1, 1, 1, {args.max_cache_len + 1}]")
    print(f"  encoder_attn_mask  [1, 1, 1, {args.src_seq_len}]")
    print(f"\nDecoder outputs:")
    print(f"  logits           [1, 1, {vocab_size}]")
    print(f"  new_keys         [{num_layers}, 1, 1, {d_model}]")
    print(f"  new_values       [{num_layers}, 1, 1, {d_model}]")


if __name__ == "__main__":
    main()
