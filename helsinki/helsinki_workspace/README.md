# Helsinki Translation Model - MTK NPU Conversion

Convert Helsinki-NLP/opus-mt translation model to MediaTek DLA format with KV Cache support.

## Directory Structure

```
helsinki_workspace/
├── model_prepare/              # Model conversion scripts
│   ├── mtk_model.py           # MTK-optimized PyTorch model with KV Cache
│   ├── convert.py             # Convert to TorchScript/TFLite/DLA
│   ├── test_model.py          # Test model consistency
│   ├── verify_kv_cache.py     # Verify KV cache flow
│   └── requirements.txt       # Python dependencies
├── compile/                    # DLA compilation scripts
│   └── compile_helsinki_fp.sh # Compile TFLite to DLA
├── models/                     # HuggingFace models (symlink)
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
conda create -n MTK-helsinki python=3.10
conda activate MTK-helsinki
pip install -r model_prepare/requirements.txt
```

### 2. Prepare Model

Download or link HuggingFace model:

```bash
# Option 1: Download
python -c "
from transformers import MarianMTModel, MarianTokenizer
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
model.save_pretrained('models/Helsinki-NLP/opus-mt-en-zh')
tokenizer.save_pretrained('models/Helsinki-NLP/opus-mt-en-zh')
"

# Option 2: Symlink existing
ln -s /path/to/existing/models models
```

### 3. Convert Model

```bash
cd model_prepare

# Convert to TorchScript, TFLite, and DLA
python convert.py \
    --model_path ../models/Helsinki-NLP/opus-mt-en-zh \
    --output_dir model_kvcache \
    --platform MT8371
```

### 4. Test Model

```bash
# Test converted model consistency
python test_model.py

# Verify KV cache flow
python verify_kv_cache.py
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Helsinki Model                        │
├─────────────────────────────────────────────────────────┤
│  Input Text → [Tokenizer] → Token IDs                   │
│                    ↓                                     │
│             [CPU Embedding]                              │
│                    ↓                                     │
│        [Encoder NPU] → Hidden States                    │
│                    ↓                                     │
│  ┌─────────────────────────────────────┐               │
│  │     Autoregressive Decoding         │               │
│  │  ┌───────────────────────────────┐  │               │
│  │  │ [CPU Embedding]               │  │               │
│  │  │       ↓                       │  │               │
│  │  │ [Decoder NPU + KV Cache 4D]   │  │               │
│  │  │       ↓                       │  │               │
│  │  │  Logits → Token ID            │  │               │
│  │  └───────────────────────────────┘  │               │
│  │         (repeat until EOS)          │               │
│  └─────────────────────────────────────┘               │
│                    ↓                                     │
│            [Detokenizer] → Output Text                  │
└─────────────────────────────────────────────────────────┘
```

## Model Specifications

| Component | Shape | Description |
|-----------|-------|-------------|
| Encoder Input | `[1, 64, 512]` | Embedded tokens |
| Encoder Mask | `[1, 1, 64, 64]` | Self-attention mask |
| Encoder Output | `[1, 64, 512]` | Hidden states |
| Decoder Embed | `[1, 1, 512]` | Current token embedding |
| KV Cache | `[6, 1, 64, 512]` | Past keys/values (4D) |
| Position Embed | `[1, 1, 512]` | Position encoding |
| Decoder Output | `[1, 1, 65001]` | Logits |
| New KV | `[6, 1, 1, 512]` | New keys/values |

### Configuration

```python
{
    "d_model": 512,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "num_heads": 8,
    "vocab_size": 65001,
    "max_position": 512,
    "src_seq_len": 64,
    "max_cache_len": 64
}
```

## Output Files

After conversion:

```
model_kvcache/
├── encoder_src64.pt                    # Encoder TorchScript
├── encoder_src64.tflite                # Encoder TFLite
├── encoder_src64_MT8371.dla            # Encoder DLA (~36 MB)
├── decoder_kv_src64_cache64.pt         # Decoder TorchScript
├── decoder_kv_src64_cache64.tflite     # Decoder TFLite
├── decoder_kv_src64_cache64_MT8371.dla # Decoder DLA (~112 MB)
├── embedding_weights.bin               # Embedding weights (~127 MB)
├── position_embeddings.bin             # Position embeddings (~1 MB)
├── source.spm                          # Source tokenizer
├── target.spm                          # Target tokenizer
└── vocab.txt                           # Vocabulary
```

## Supported Platforms

| Platform | SoC | MDLA | L1 Cache | Cores |
|----------|-----|------|----------|-------|
| MT8371 | Genio 700 | 5.3 | 256KB | 1 |
| MT6899 | Dimensity 1200 | 5.5 | 2048KB | 2 |
| MT6991 | Dimensity 9300 | 5.5 | 7168KB | 4 |

## Key Features

- **4D KV Cache**: Avoids MT8371's 5D tensor limitation
- **CPU Embedding**: GATHER op not supported on NPU
- **Attention Masks**: Proper handling of padding tokens
- **final_logits_bias**: Included in decoder for correct logits

## Performance (MT8371)

| Metric | Value |
|--------|-------|
| Model Load | ~210 ms |
| Encoder | ~4-6 ms |
| Decoder/token | ~10 ms |
| Total (10-15 tokens) | ~100-150 ms |

## References

- [Helsinki-NLP Models](https://huggingface.co/Helsinki-NLP)
- [MarianMT Documentation](https://huggingface.co/docs/transformers/model_doc/marian)
- MediaTek NeuroPilot SDK Documentation
