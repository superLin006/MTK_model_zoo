# C++ 推理实现

Whisper KV Cache 在 MTK NPU (MT8371) 上的 C++ 推理实现。

当前编译配置：**large-v3-turbo**

## 快速开始

```bash
# 编译
bash build_android.sh

# 部署并测试
bash deploy_and_test.sh test_en.wav en        # 英文
bash deploy_and_test.sh test_zh.wav zh        # 中文
bash deploy_and_test.sh test_en.wav en debug  # 调试模式
```

支持语言：`en` `zh` `de` `es` `fr` `ja`（可在 `whisper_inference.cpp` 中继续扩展）

## 模型文件

部署脚本从 `python/models_large_turbo/` 读取以下文件并推送到设备：

```
models_large_turbo/
├── encoder_large-v3-turbo_128x3000_MT8371.dla   # 编码器 (1217 MB)
├── decoder_large-v3-turbo_448_MT8371.dla         # 解码器 KV Cache (327 MB)
├── token_embedding.npy                           # Token 嵌入 (51866 × 1280)
├── position_embedding.npy                        # 位置嵌入 (448 × 1280)
├── mel_128_filters.txt                           # Mel 滤波器 (25728 行)
└── vocab.txt                                     # 词汇表
```

## 切换模型

### 切换到 base 模型

**1. 修改 `jni/src/whisper_inference.h`：**

```cpp
// 模型变体（改这一行）
#define WHISPER_MODEL_VARIANT  WHISPER_MODEL_BASE   // 改为 BASE
```

**2. 修改 `jni/src/utils/audio_utils.h`：**

```cpp
#define N_MELS   80       // base=80, large-v3-turbo=128
#define VOCAB_NUM 51865   // base=51865, large-v3-turbo=51866
```

**3. 修改 `jni/src/whisper_inference.h` 中的模型配置：**

```cpp
int vocab_size_ = 51865;  // base=51865, large-v3-turbo=51866
int d_model_    = 512;    // base=512,   large-v3-turbo=1280
int num_layers_ = 6;      // base=6,     large-v3-turbo=4
```

**4. 修改 `deploy_and_test.sh`：**

```bash
MODELS_DIR=".../python/models"   # 改为 base 模型目录
```

**5. 重新编译：**

```bash
bash build_android.sh
```

### 模型参数速查

| 参数 | base | large-v3-turbo |
|------|------|----------------|
| `WHISPER_MODEL_VARIANT` | `WHISPER_MODEL_BASE` | `WHISPER_MODEL_LARGE_V3_TURBO` |
| `N_MELS` | 80 | 128 |
| `VOCAB_NUM` | 51865 | 51866 |
| `d_model_` | 512 | 1280 |
| `num_layers_` | 6 | 4 |
| mel filter 文件 | `mel_80_filters.txt` | `mel_128_filters.txt` |
| initial tokens [2] | 50359 `<\|transcribe\|>` | 50360 `<\|startoflm\|>` |
| initial tokens [3] | 50363 `<\|notimestamps\|>` | 50364 `<\|0.00\|>` |

## 调试控制

通过环境变量运行时切换，无需重新编译：

```bash
export WHISPER_DEBUG=0   # 生产模式（默认）
export WHISPER_DEBUG=1   # 调试模式（打印 mel、encoder output、每步 logits 等）
```

## 性能（large-v3-turbo，MT8371）

实测结果（多语言）：

| 音频 | 时长 | 耗时 | RTF |
|------|------|------|-----|
| 英文 5.8s | 5.8s | 8.6s | 1.47x |
| 中文 5.8s | 5.8s | 8.9s | 1.52x |
| 中文 18s | 18s | 11.4s | 0.63x |
| 英文 28.8s | 28.8s | 11.7s | 0.41x |
| 德文 5.8s | 5.8s | 4.9s | 0.84x |
| 日文 4.7s | 4.7s | 4.9s | 1.03x |

> RTF < 1.0 = 快于实时；短音频因模型加载有固定开销，RTF 偏高。

## 关键实现说明

### 解码起始序列（initial tokens）

large-v3-turbo 与 base 的解码起始序列不同：

```
base:           [SOT=50258, lang, transcribe=50359, no_timestamps=50363]
large-v3-turbo: [SOT=50258, lang, sot_lm=50360,    timestamp_begin=50364]
```

这由 `WHISPER_MODEL_VARIANT` 宏在编译时选择（`whisper_inference.cpp` 中的 `#if` 分支）。

### KV Cache 解码流程

```
Phase 1：逐步喂入 4 个 initial tokens，初始化 self-attention KV cache
          第一步同时初始化 cross-attention cache（encoder 输出的投影）
Phase 2：自回归生成，每步取 logits argmax（范围 0..EOT），遇到 EOT 停止
```

### 内存占用（large-v3-turbo）

```
Encoder output:   7.5 MB  (1×1500×1280×float32)
Decoder KV cache: 76 MB   (4层 self + 4层 cross，各 1500/448 × 1280)
```
