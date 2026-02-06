# Helsinki Translation - MTK NPU with KV Cache

English to Chinese translation model (MarianMT) optimized for MediaTek MT8371 NPU.

## Features

- **Encoder/Decoder Architecture**: Separate NPU models for encoder and decoder
- **4D KV Cache**: Avoids MT8371's 5D tensor limitation, improves inference performance
- **CPU Embedding**: GATHER op not supported on MT8371, embedding runs on CPU
- **Dynamic Library Loading**: Uses `dlopen` for Neuron Runtime, no link-time dependency

## Architecture

```
Input Text → [Tokenizer] → Token IDs
                              ↓
                       [CPU Embedding]
                              ↓
                 [Encoder NPU] → Hidden States
                              ↓
                 ┌─────────────────────────────┐
                 │  Autoregressive Decoding:   │
                 │  ┌───────────────────────┐  │
                 │  │ [CPU Embedding]       │  │
                 │  │       ↓               │  │
                 │  │ [Decoder NPU]         │  │
                 │  │  + KV Cache (4D)      │  │
                 │  │       ↓               │  │
                 │  │  Logits → Token ID    │  │
                 │  └───────────────────────┘  │
                 │         (repeat)            │
                 └─────────────────────────────┘
                              ↓
                       [Detokenizer] → Output Text
```

## Directory Structure

```
helsinki_mtk_cpp/
├── jni/
│   ├── Android.mk              # NDK build config
│   ├── Application.mk          # NDK app config
│   ├── src/
│   │   ├── helsinki/           # Core translation module
│   │   │   ├── helsinki.h
│   │   │   ├── helsinki.cc
│   │   │   └── main.cc
│   │   └── tokenizer/          # SentencePiece wrapper
│   │       ├── sp_tokenizer.h
│   │       └── sp_tokenizer.cc
│   └── third_party/
│       └── sentencepiece/      # Pre-built SentencePiece library
├── build.sh                    # Build script
├── deploy_and_test.sh          # Deploy and test on device
└── README.md
```

## Build

### Prerequisites

- Android NDK r25c or later
- Connected Android device with MT8371 SoC

### Build

```bash
# Set NDK path (optional, auto-detected)
export NDK_ROOT=/path/to/android-ndk

# Build
./build.sh
```

Output: `libs/arm64-v8a/helsinki_translate`

## Deploy and Test

```bash
# Deploy to device and run tests
./deploy_and_test.sh
```

Or manually:

```bash
adb shell
cd /data/local/tmp/helsinki
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/helsinki
./helsinki_translate . .
```

## Model Files

Required files (not included in repo):

```
encoder_src64_MT8371.dla           # Encoder DLA model (~36 MB)
decoder_kv_src64_cache64_MT8371.dla # Decoder DLA model (~112 MB)
embedding_weights.bin               # Embedding weights (~127 MB)
position_embeddings.bin             # Position embeddings (~1 MB)
embedding_weights_meta.txt          # Metadata
source.spm                          # Source tokenizer
target.spm                          # Target tokenizer
vocab.txt                           # Vocabulary
```

## Performance (MT8371)

| Metric | Value |
|--------|-------|
| Model Load | ~210 ms |
| Encoder | ~4-6 ms |
| Decoder per token | ~10 ms |
| Total (10-15 tokens) | ~100-150 ms |

## KV Cache Format

- Shape: `[num_layers, batch, max_cache_len, d_model]` = `[6, 1, 64, 512]`
- 4D format avoids MT8371's 5D tensor limitation

## Limitations

1. **MT8371 Platform**:
   - No GATHER op → CPU embedding
   - No 5D tensors → 4D KV cache
   - Small L1 cache (256KB)

2. **Input Length**:
   - Encoder: 64 tokens max
   - KV Cache: 64 tokens max

## License

Based on Helsinki-NLP/opus-mt-en-zh model.
