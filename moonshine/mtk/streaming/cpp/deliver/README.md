# Moonshine MTK NPU Streaming ASR - Deliver Package

## Overview

Streaming ASR inference on MT8371 NPU using Moonshine model.
Uses chunk encoder (0.8s chunks) + offline decoder.

## Architecture

```
Audio stream (16kHz PCM)
  → chunk (0.8s = 12800 samples)
  → preprocess per-frame (CMVN + Asinh)
  → Chunk Encoder NPU: [1, 160, 80] → [1, 40, 620]
  → accumulate encoder output (500 frames = 10s)
  → Adapter projection: [500, 620] → [500, 512]
  → Decoder NPU (autoregressive): → tokens → text
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| CHUNK_FRAMES | 160 | Encoder input frames per call |
| CHUNK_T_ENC | 40 | Encoder output frames per call |
| STEP_FRAMES | 40 | Sliding step (no overlap) |
| MAX_ENC_FRAMES | 500 | Full decoder window (10s) |
| TRIGGER_ENC_FRAMES | 500 | Trigger decoder at this many frames |

## Files

```
deliver/
├── bin/moonshine_streaming_test   # Main executable
├── lib/libc++_shared.so           # C++ shared library
├── models/
│   ├── moonshine_encoder_chunk.dla  # Chunk encoder [1,160,80]→[1,40,620]
│   ├── moonshine_decoder.dla        # Decoder (reused from offline)
│   ├── embed_tokens.npy             # Token embeddings [32768, 512]
│   ├── proj_weight.npy              # Adapter projection [512, 620]
│   ├── log_k.npy                    # AsinhCompression parameter
│   └── vocab.txt                    # Vocabulary (id<TAB>piece)
└── test_data/
    ├── test_en.wav                  # Test audio (English, 16kHz)
    └── run_test.sh                  # Test runner script
```

## Usage

```bash
# Push to device:
adb push deliver/ /data/local/tmp/moonshine_streaming/

# On device:
cd /data/local/tmp/moonshine_streaming
sh test_data/run_test.sh
```

## Performance (MT8371, 5.855s audio)

| Component | Time | Notes |
|-----------|------|-------|
| Init | ~188 ms | One-time |
| Encoder total | ~360 ms | 27 calls × ~13 ms |
| Decoder total | ~436 ms | 3 calls |
| Total infer | ~1102 ms | |
| Audio duration | 5855 ms | |
| RTF | 0.188 | |

Encoder per call: ~13 ms (0.8s audio)
Decoder per token: ~16 ms

## Comparison with Offline

| | Offline | Streaming |
|--|---------|-----------|
| Encoder calls | 1 (full 10s) | 27 (0.8s chunks) |
| Encoder time | ~116 ms | ~360 ms total |
| Decoder calls | 1 | 3 |
| Total time | ~657 ms | ~1102 ms |
| Transcription | Mr. Quilter is the apostle of the middle classes... | Mr. Quilter is the Apostle of the Middle Class... |
