# Whisper Base Model with KV Cache - Baseline Results

**Date**: 2026-02-09
**Model**: OpenAI Whisper Base
**Platform**: CPU (baseline for MTK NPU deployment)
**Python Environment**: MTK-whisper-kv (Python 3.10)

---

## 1. Project Overview

This project implements Whisper speech recognition with **KV Cache optimization** for the MT8371 NPU platform. The KV cache technique significantly reduces computational cost during autoregressive decoding by caching key/value tensors from previous steps.

### KV Cache Benefits:
- **Reduced computation**: Avoids recomputing attention for all previous tokens at each step
- **Faster inference**: Only processes the newest token in decoder (after first pass)
- **Memory efficient**: Trades small memory increase for large computation savings
- **Especially beneficial for long sequences**: 30s audio generates ~50-200 tokens

---

## 2. Model Architecture

### Whisper Base Dimensions

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Encoder** | n_mels | 80 |
| | n_audio_ctx | 1500 |
| | n_audio_state | 512 |
| | n_audio_head | 8 |
| | n_audio_layer | 6 |
| **Decoder** | n_vocab | 51865 |
| | n_text_ctx | 448 |
| | n_text_state | 512 |
| | n_text_head | 8 |
| | n_text_layer | 6 |

### Model Components

#### Encoder
- **Input**: Mel-spectrogram [batch, 80, 3000] (30s audio at 100 frames/sec)
- **Conv1**: Conv1d(80, 512, kernel=3, padding=1) + GELU
- **Conv2**: Conv1d(512, 512, kernel=3, stride=2, padding=1) + GELU
  - Reduces time dimension: 3000 → 1500
- **Positional Encoding**: Sinusoidal [1500, 512]
- **Transformer**: 6 layers × (Self-Attention + MLP)
- **Output**: [batch, 1500, 512]

#### Decoder (with KV Cache)
- **Input 1**: Token IDs [batch, seq_len]
- **Input 2**: Encoder output [batch, 1500, 512]
- **Token Embedding**: [51865, 512]
- **Positional Encoding**: Learnable [448, 512]
- **Transformer**: 6 layers × (Self-Attention + Cross-Attention + MLP)
  - **Self-Attention**: Uses KV cache for previous tokens
  - **Cross-Attention**: Attends to encoder output (K,V cached once)
- **Output**: Logits [batch, seq_len, 51865]

---

## 3. Input/Output Specifications

### Encoder

```
Input:
  - mel_spectrogram: [1, 80, 3000], float32
    - 80 mel-frequency bins
    - 3000 frames (30s audio at 16kHz, 100 frames/sec)

Output:
  - encoder_features: [1, 1500, 512], float32
    - 1500 time steps (after conv stride=2)
    - 512 feature dimensions
```

### Decoder (per autoregressive step)

```
Input:
  - token_ids: [1, seq_len], int64
    - seq_len increases by 1 at each step
    - First step: SOT sequence [50258, 50259, 50359] (3-4 tokens)
    - Subsequent steps: append 1 new token

  - encoder_output: [1, 1500, 512], float32
    - Constant across all steps

  - kv_cache (optional): Dict of cached key/value tensors
    - For each of 6 decoder layers:
      - self_k: [1, num_heads, prev_seq_len, head_dim]
      - self_v: [1, num_heads, prev_seq_len, head_dim]
      - cross_k: [1, num_heads, 1500, head_dim] (cached once)
      - cross_v: [1, num_heads, 1500, head_dim] (cached once)

Output:
  - logits: [1, seq_len, 51865], float32
    - Token probability distribution
    - Use logits[:, -1, :] for next token prediction

  - updated_kv_cache: Updated cache for next step
```

### KV Cache Implementation Strategy

For MTK NPU deployment, we need to:

1. **First decoder pass** (SOT sequence):
   - Input: SOT tokens [50258, 50259, 50359]
   - Process full sequence
   - Save K,V tensors for all 6 layers

2. **Subsequent passes** (autoregressive):
   - Input: Only the newest token [1, 1]
   - Concatenate with cached K,V
   - Update cache with new K,V

3. **Cross-attention optimization**:
   - Encoder output K,V computed once
   - Reused across all decoder steps

---

## 4. Baseline Test Results

### Test Environment
- Device: CPU (Intel WSL2)
- Precision: FP32
- Temperature: 0.0 (greedy decoding)

### Test Audio Files

#### test_en.wav (English)
```
Duration: 5.44s
Language: en
Text: "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."

Performance:
  - Model load time: 0.44s
  - Encoder time: 0.33s
  - Total inference: 1.48s
  - Tokens generated: 22
  - Average logprob: -0.218
  - No speech prob: 0.007
```

#### test_zh.wav (Chinese)
```
Duration: ~3s
Language: zh
Text: "對我做了介紹我想說的是大家如果對我的研究感興趣"

Performance:
  - Model load time: 0.53s
  - Encoder time: 0.33s
  - Total inference: 1.81s
  - Tokens generated: 24
```

#### jfk.flac (English - JFK speech)
```
Duration: ~11s
Language: en
Text: "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."

Performance:
  - Model load time: 0.43s
  - Encoder time: 0.32s
  - Total inference: 1.47s
  - Tokens generated: 28
```

### Performance Summary

| Metric | test_en | test_zh | jfk |
|--------|---------|---------|-----|
| Inference Time | 1.48s | 1.81s | 1.47s |
| Encoder Time | 0.33s | 0.33s | 0.32s |
| Decoder Time | ~1.15s | ~1.48s | ~1.15s |
| Tokens | 22 | 24 | 28 |

**Key Observations**:
1. Encoder time is constant (~0.33s) regardless of output length
2. Decoder time scales with number of tokens
3. Average time per token: ~0.04-0.06s (with standard Whisper API)
4. With explicit KV cache, per-token time should be ~0.01-0.02s

---

## 5. KV Cache Expected Performance

### Without KV Cache (naive implementation)
- Each step processes **all previous tokens**
- Complexity: O(n²) where n = sequence length
- For 50 tokens: 1+2+3+...+50 = 1,275 token computations

### With KV Cache
- Each step processes **only the newest token**
- Complexity: O(n)
- For 50 tokens: 50 token computations
- **Theoretical speedup**: 25.5x for 50 tokens

### Practical Speedup
- Real speedup typically 2-5x due to:
  - Cache memory bandwidth costs
  - Cross-attention still processes full encoder output
  - Other overheads (softmax, layer norm, etc.)

---

## 6. MTK NPU Deployment Considerations

### Model Conversion Pipeline

```
PyTorch → TorchScript → TFLite → MTK DLA
                                     ↓
                                Encoder.dla
                                Decoder.dla (with KV cache I/O)
```

### Challenges

1. **Embedding Layer**:
   - `nn.Embedding` uses GATHER op (not supported on some NPUs)
   - **Solution**: Implement token lookup in C++ before NPU inference

2. **KV Cache Management**:
   - Cache must be maintained in C++ host code
   - Pass updated K,V tensors as additional inputs/outputs to NPU

3. **Dynamic Sequence Length**:
   - Decoder input size changes each step (1 token → 2 tokens → ...)
   - **Solution**: Use fixed max length with masking, or
   - **Better**: Run decoder with only last token + cache

4. **Cross-Attention K,V**:
   - Should be computed once from encoder output
   - Can be cached on host or passed to each decoder call

5. **Causal Masking**:
   - Self-attention mask must prevent attending to future tokens
   - Can be precomputed and passed as input

### Recommended C++ Architecture

```cpp
class WhisperKVCache {
    // Encoder (run once)
    rknn_context encoder_ctx;

    // Decoder (run autoregressively with cache)
    rknn_context decoder_ctx;

    // Token embedding (CPU lookup)
    float* token_embedding_table;  // [51865, 512]

    // KV cache buffers (per layer)
    struct LayerCache {
        std::vector<float> self_k;   // [num_heads, seq_len, head_dim]
        std::vector<float> self_v;
        std::vector<float> cross_k;  // [num_heads, 1500, head_dim]
        std::vector<float> cross_v;
    };
    LayerCache layer_caches[6];  // 6 decoder layers

    // Methods
    void encode(const float* mel_input);
    int decode_step(int token_id, int* next_token);
    void reset_cache();
};
```

---

## 7. Next Steps

### Phase 1: Model Export
- [ ] Export Encoder: PyTorch → TorchScript → TFLite → DLA
- [ ] Export Decoder: Remove embedding layer, add KV cache I/O
- [ ] Export token embedding weights as .npy

### Phase 2: C++ Implementation
- [ ] Implement audio preprocessing (FFT, mel-spectrogram)
- [ ] Implement token embedding lookup
- [ ] Implement KV cache management
- [ ] Integrate with MTK NPU runtime

### Phase 3: Optimization
- [ ] Profile encoder/decoder on NPU
- [ ] Optimize memory layout for cache
- [ ] Quantization (FP16 or INT8)
- [ ] Benchmark vs baseline

---

## 8. Reference Files

### Baseline Test Outputs
```
/home/xh/projects/MTK_models_zoo/whisper-kv-cache/mtk/python/test/outputs/baseline/
├── baseline_test_en.json      # English test results
├── baseline_test_en.txt        # English transcription
├── baseline_test_zh.json      # Chinese test results
├── baseline_test_zh.txt        # Chinese transcription
├── baseline_jfk.json          # JFK speech results
├── baseline_jfk.txt            # JFK transcription
└── baseline_summary.json      # All results summary
```

### Test Audio Files
```
/home/xh/projects/MTK_models_zoo/whisper-kv-cache/mtk/test_data/
├── test_en.wav               # English test audio (5.44s)
├── test_zh.wav               # Chinese test audio (~3s)
└── jfk.flac                  # JFK speech (~11s)
```

### Python Scripts
```
/home/xh/projects/MTK_models_zoo/whisper-kv-cache/mtk/python/test/
└── test_pytorch.py           # Baseline test with KV cache concept
```

---

## 9. Important Notes

1. **KV Cache is Critical**: Without KV cache, decoder inference would be extremely slow (quadratic complexity)

2. **Standard Whisper API**: The openai-whisper library includes KV cache optimization internally, which is why we see good performance

3. **MTK NPU Challenge**: We need to implement explicit KV cache management because:
   - NPU doesn't have built-in state management
   - Each inference call is stateless
   - Cache must be managed by host C++ code

4. **Memory Requirements**: KV cache adds memory overhead but dramatically reduces compute
   - Per layer: ~2.5 MB for full 448-token sequence
   - Total for 6 layers: ~15 MB
   - Well worth the compute savings

5. **Quantization Opportunity**: KV cache tensors can potentially be quantized (FP16 or even INT8) with minimal accuracy loss

---

## 10. Conclusion

The baseline test successfully validates:
- ✅ Whisper base model works correctly on CPU
- ✅ Audio files produce accurate transcriptions
- ✅ Standard API includes KV cache optimization
- ✅ Model dimensions and I/O shapes documented
- ✅ Performance metrics established for comparison

**Next Goal**: Export Encoder and Decoder models with proper KV cache I/O interfaces for MTK NPU deployment.
