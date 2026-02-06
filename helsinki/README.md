# MTK-Helsinki

Helsinki-NLP translation model (MarianMT) ported to MediaTek NPU with KV Cache support.

## Overview

This project provides:
1. **Model Conversion** - Convert HuggingFace Helsinki model to MediaTek DLA format
2. **C++ Runtime** - Optimized inference on MTK NPU with KV Cache

## Project Structure

```
MTK-Helsinki/
├── helsinki_workspace/         # Model conversion workspace
│   ├── model_prepare/         # Python conversion scripts
│   │   ├── mtk_model.py      # MTK-optimized model with KV Cache
│   │   ├── convert.py        # Conversion script
│   │   └── test_model.py     # Model testing
│   └── compile/              # DLA compilation scripts
│
└── helsinki_mtk_cpp/          # C++ NPU runtime
    ├── jni/
    │   └── src/
    │       ├── helsinki/     # Core translation module
    │       └── tokenizer/    # SentencePiece wrapper
    ├── build.sh              # Build script
    └── deploy_and_test.sh    # Deploy and test
```

## Features

- **4D KV Cache**: Optimized for MT8371 (avoids 5D tensor limitation)
- **Separate Encoder/Decoder**: Efficient NPU utilization
- **CPU Embedding**: Handles unsupported GATHER op
- **Attention Masks**: Proper padding handling

## Quick Start

### 1. Convert Model

```bash
cd helsinki_workspace/model_prepare
pip install -r requirements.txt
python convert.py --model_path ../models/Helsinki-NLP/opus-mt-en-zh --platform MT8371
```

### 2. Build C++ Runtime

```bash
cd helsinki_mtk_cpp
./build.sh
```

### 3. Deploy and Test

```bash
./deploy_and_test.sh
```

## Performance (MT8371)

| Metric | Value |
|--------|-------|
| Model Load | ~210 ms |
| Encoder | ~4-6 ms |
| Decoder/token | ~10 ms |
| Translation (10-15 tokens) | ~100-150 ms |

## Supported Platforms

| Platform | SoC | MDLA | Status |
|----------|-----|------|--------|
| MT8371 | Genio 700 | 5.3 | ✅ Tested |
| MT6899 | Dimensity 1200 | 5.5 | Supported |
| MT6991 | Dimensity 9300 | 5.5 | Supported |

## Requirements

- Python 3.10+
- Android NDK r25c+
- MediaTek NeuroPilot SDK 8.0+
- PyTorch, Transformers

## License

Based on Helsinki-NLP/opus-mt models.

## References

- [Helsinki-NLP Models](https://huggingface.co/Helsinki-NLP)
- [MarianMT Documentation](https://huggingface.co/docs/transformers/model_doc/marian)
