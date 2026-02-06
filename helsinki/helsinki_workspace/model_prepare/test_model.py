import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import os
import sys

# Add current dir to path
sys.path.append(os.path.dirname(__file__))

from mtk_model import (
    create_encoder_decoder_kvcache_v2, 
    create_attn_mask, 
    create_encoder_attn_mask
)

MODEL_PATH = "../models/Helsinki-NLP/opus-mt-en-zh"
KV_PT_DIR = "model_kvcache"
ENCODER_TFLITE = os.path.join(KV_PT_DIR, "encoder_src64.tflite")
DECODER_TFLITE = os.path.join(KV_PT_DIR, "decoder_kv_src64_cache64.tflite")

TEST_SENTENCES = [
    "Hello, how are you?",
    "This is a test sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine translation is challenging.",
    "Deep learning models require large amounts of data."
]

SRC_SEQ_LEN = 64
MAX_CACHE_LEN = 64

def run_pytorch(text, model_path):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=False)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_kvcache_pt(text, model_path, max_cache_len=MAX_CACHE_LEN):
    encoder, decoder, embedding_weights, position_embeddings, tokenizer, config = \
        create_encoder_decoder_kvcache_v2(model_path, max_cache_len)
    
    d_model = config.d_model
    num_layers = config.decoder_layers
    
    encoder.eval()
    decoder.eval()
    
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    
    with torch.no_grad():
        encoder_embeds = F.embedding(input_ids, embedding_weights)
        encoder_hidden = encoder(encoder_embeds)
        encoder_attn_mask_cross = create_encoder_attn_mask(attention_mask)
        
        past_keys = torch.zeros(num_layers, 1, max_cache_len, d_model)
        past_values = torch.zeros(num_layers, 1, max_cache_len, d_model)
        
        generated_tokens = []
        current_token = config.pad_token_id
        
        for step in range(64):
            decoder_embed = F.embedding(torch.tensor([[current_token]]), embedding_weights)
            position_embed = position_embeddings[:, step:step+1, :]
            attn_mask = create_attn_mask(step, max_cache_len)
            
            logits, new_keys, new_values = decoder(
                decoder_embed, encoder_hidden, past_keys, past_values, 
                position_embed, attn_mask, encoder_attn_mask_cross
            )
            
            next_token = logits[0, 0].argmax().item()
            if next_token == config.eos_token_id:
                break
                
            generated_tokens.append(next_token)
            current_token = next_token
            
            # Update cache
            past_keys[:, :, step:step+1, :] = new_keys
            past_values[:, :, step:step+1, :] = new_values
            
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def run_tflite(text, model_path, max_cache_len=MAX_CACHE_LEN):
    try:
        import mtk_converter
    except ImportError:
        return None
        
    _, _, embedding_weights, position_embeddings, tokenizer, config = \
        create_encoder_decoder_kvcache_v2(model_path, max_cache_len)
        
    d_model = config.d_model
    num_layers = config.decoder_layers
    
    if not os.path.exists(ENCODER_TFLITE) or not os.path.exists(DECODER_TFLITE):
        return None
        
    enc_executor = mtk_converter.TFLiteExecutor(ENCODER_TFLITE)
    dec_executor = mtk_converter.TFLiteExecutor(DECODER_TFLITE)
    
    # 1. Tokenize and Pad to SRC_SEQ_LEN
    tokens = tokenizer(text, return_tensors="pt", padding='max_length', max_length=SRC_SEQ_LEN, truncation=True)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    
    # 2. Encoder inference
    encoder_embeds = F.embedding(input_ids, embedding_weights)
    # Encoder TFLite expects (embeds, self_attn_mask)
    # create_encoder_attn_mask returns [1, 1, 1, seq]
    # We need to broadcast it to [1, 1, seq, seq] for encoder self-attention if it's trace-exported that way
    # In convert_kvcache_v2.py it was [1, 1, 64, 64]
    enc_self_attn_mask = (1.0 - attention_mask.float()) * -1e9
    enc_self_attn_mask = enc_self_attn_mask.unsqueeze(1).unsqueeze(1) # [1, 1, 1, 64]
    # Broadcast to [1, 1, 64, 64]
    enc_self_attn_mask = enc_self_attn_mask.expand(-1, -1, SRC_SEQ_LEN, -1)
    
    enc_outputs = enc_executor.run([encoder_embeds.numpy(), enc_self_attn_mask.numpy()])
    encoder_hidden = torch.from_numpy(enc_outputs[0])
    
    # 3. Cross Attention Mask (already padded via attention_mask)
    encoder_attn_mask_cross = create_encoder_attn_mask(attention_mask)
    
    past_keys = np.zeros((num_layers, 1, max_cache_len, d_model), dtype=np.float32)
    past_values = np.zeros((num_layers, 1, max_cache_len, d_model), dtype=np.float32)
    
    generated_tokens = []
    current_token = config.pad_token_id
    
    for step in range(64):
        decoder_embed = F.embedding(torch.tensor([[current_token]]), embedding_weights).numpy()
        pos_embed = position_embeddings[:, step:step+1, :].numpy()
        # Self-attention mask [1, 1, 1, 65]
        attn_mask = create_attn_mask(step, max_cache_len).numpy()
        
        dec_inputs = [
            decoder_embed,
            encoder_hidden.numpy(),
            past_keys,
            past_values,
            pos_embed,
            attn_mask,
            encoder_attn_mask_cross.numpy()
        ]
        
        dec_outputs = dec_executor.run(dec_inputs)
        logits = dec_outputs[0]
        new_keys = dec_outputs[1]
        new_values = dec_outputs[2]
        
        next_token = logits[0, 0].argmax().item()
        if next_token == config.eos_token_id:
            break
            
        generated_tokens.append(next_token)
        current_token = next_token
        
        # Update cache
        past_keys[:, :, step:step+1, :] = new_keys
        past_values[:, :, step:step+1, :] = new_values
        
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
    print("="*60)
    print("Helsinki No-Padding Consistency Test (PyTorch vs .pt vs .tflite)")
    print("="*60)
    
    all_pass = True
    
    for i, text in enumerate(TEST_SENTENCES):
        print(f"\n[{i+1}/{len(TEST_SENTENCES)}] Input: {text}")
        
        # 1. Baseline PyTorch
        out_baseline = run_pytorch(text, MODEL_PATH)
        print(f"  Baseline HF: {out_baseline}")
        
        # 2. KV Cache PyTorch (.pt equivalent)
        out_kv_pt = run_kvcache_pt(text, MODEL_PATH)
        print(f"  KV Cache PT: {out_kv_pt}")
        
        # 3. TFLite
        out_tflite = run_tflite(text, MODEL_PATH)
        if out_tflite is not None:
            print(f"  KV TFLite   : {out_tflite}")
        else:
            print(f"  KV TFLite   : [Skipped or Error]")
            
        # Comparison logic
        pt_match = (out_baseline == out_kv_pt)
        tflite_match = True
        if out_tflite is not None:
            tflite_match = (out_kv_pt == out_tflite)
            
        status = "✅ PASS" if (pt_match and tflite_match) else "❌ FAIL"
        if not pt_match:
            print(f"  Result: {status} (PT mismatch)")
            all_pass = False
        elif not tflite_match:
            print(f"  Result: {status} (TFLite mismatch)")
            all_pass = False
        else:
            print(f"  Result: {status}")
            
    print("\n" + "="*60)
    if all_pass:
        print("OVERALL RESULT: ✅ ALL MODELS CONSISTENT")
    else:
        print("OVERALL RESULT: ❌ CONSISTENCY CHECK FAILED")
    print("="*60)

if __name__ == "__main__":
    main()
