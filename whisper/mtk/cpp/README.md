# Whisper MTK NPU - C++ Inference Engine

C++ implementation of Whisper speech recognition optimized for MTK NPU (MT8371).

## Features

- ✅ Real-time speech recognition on MTK8371 NPU
- ✅ Support for English and Chinese audio
- ✅ ~700-800ms inference time for 6-second audio
- ✅ Autoregressive greedy decoding
- ✅ Mel spectrogram computation with FFTW3

## Project Structure

```
cpp/
├── jni/                      # Android NDK build system
│   ├── Android.mk           # Build configuration
│   ├── Application.mk       # App configuration
│   └── src/                 # All C++ source files
│       ├── main.cpp         # Test executable entry point
│       ├── whisper_inference.cpp  # Main inference engine
│       ├── whisper_inference.h
│       ├── neuron_executor.h
│       ├── utils/           # Utility functions
│       │   ├── audio_utils.cpp  # Audio preprocessing & mel spectrogram
│       │   └── audio_utils.h
│       └── mtk-npu/         # MTK NPU executor
│           ├── neuron_executor.cpp
│           └── neuron_executor.h
├── models/                   # Model resources
│   ├── vocab.txt            # Vocabulary (51865 tokens)
│   └── mel_80_filters.txt   # Mel filterbank (80 x 201)
├── build_android.sh         # Android build script
├── deploy_android.sh        # Deploy to device script
└── run_android_tests.sh     # Test runner script
```

**Note**: All C++ source files are now organized under `jni/src/` following Android NDK standard structure.

## Build Instructions

### Prerequisites

- Android NDK r25c
- MTK NeuroPilot SDK (libneuron_adapter.so)
- FFTW3 library (compiled for ARM64)
- ADB for device deployment

### Build for Android

```bash
./build_android.sh
```

This will compile the code using Android NDK and produce:
- `libs/arm64-v8a/whisper_test` - Executable for Android device

## Deployment

### Deploy to Device

```bash
./deploy_android.sh
```

This script will:
1. Push executable to `/data/local/tmp/whisper_mtk/`
2. Push models (encoder/decoder DLAs, embeddings, vocab, mel filters)
3. Push test audio files

### Run Tests

```bash
./run_android_tests.sh
```

Or manually:
```bash
adb shell
cd /data/local/tmp/whisper_mtk
./whisper_test ./models ./audio/test_en.wav en transcribe
```

## Key Implementation Details

### Mel Spectrogram Computation

The mel spectrogram implementation follows the RK Whisper reference:

1. **STFT Computation**: Uses FFTW3 to compute Short-Time Fourier Transform
2. **Transpose**: Transpose complex STFT from [num_frames x num_freqs] to [num_freqs x num_frames]
3. **Magnitude**: Compute power spectrum (real² + imag²), **discard last frame** (use `num_frames - 1`)
4. **Mel Filterbank**: Apply 80 mel filters via matrix multiplication [80 x 201] @ [201 x frames]
5. **Log Transform**: Apply clamp and log10 transform
6. **Padding**: Pad to 3000 frames (30 seconds of audio)

**Critical Detail**: The last STFT frame is discarded (`num_frames - 1`) to match Python Whisper behavior.

### Decoder

Autoregressive greedy decoding:
- Start with special tokens: [SOT, LANG, TASK, NO_TIMESTAMPS]
- Generate one token at a time
- Use token embeddings as decoder input (embedding lookup done in C++)
- Stop on EOT token or max iterations

## Test Results

### English Audio (test_en.wav)
```
Input: 5.855 seconds
Output: "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
Time: ~700-800ms
Status: ✅ Perfect match with Python reference
```

## Dependencies

- **MTK NeuroPilot SDK**: For NPU inference (`libneuron_adapter.so`)
- **FFTW3**: For Fast Fourier Transform
- **C++14**: Standard library

## Known Issues

- Chinese audio decoding has base64 decoder issues (English works perfectly)
- Mel spectrogram has minor numerical differences (~0.003) from Python in padding region (does not affect results)

## References

- Original Whisper: https://github.com/openai/whisper
- RK Whisper: `/home/xh/projects/rknn_model_zoo/examples/whisper/`
- MTK NeuroPilot: https://www.mediatek.com/innovations/artificial-intelligence
