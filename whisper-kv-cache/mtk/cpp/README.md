# C++ 推理实现

Whisper KV Cache 在 MTK NPU (MT8371) 上的 C++ 推理实现。

## 快速开始

```bash
# 编译
bash build_android.sh

# 部署并测试
bash deploy_and_test.sh test_en.wav en       # 英文，正常模式
bash deploy_and_test.sh test_zh.wav zh       # 中文，正常模式
bash deploy_and_test.sh test_en.wav en debug # 调试模式
```

仅部署不运行（只推送文件到设备）：

```bash
bash deploy_and_test.sh
```

## 模型文件

部署脚本从 `mtk/models/` 读取以下文件：

```
models/
├── encoder_base_80x3000_MT8371.dla   # 编码器
├── decoder_base_448_MT8371.dla       # 解码器（KV Cache）
├── token_embedding.npy               # Token 嵌入（51865 × 512）
├── position_embedding.npy            # 位置嵌入
├── mel_80_filters.txt                # Mel 滤波器
└── vocab.txt                         # 词汇表
```

## 调试控制

通过环境变量运行时切换，无需重新编译：

```bash
export WHISPER_DEBUG=0   # 生产模式（默认）
export WHISPER_DEBUG=1   # 调试模式
```

## 性能示例

```
Audio Duration:      28.82s
Audio Preprocessing: 146.28 ms
Encoder Inference:    88.45 ms
Decoder Inference:  3395.22 ms (77 tokens, 44.16 ms/token)
Total Time:         3629.95 ms

RTF:        0.126x
Throughput: 7.95x realtime
```

## 切换其他模型规格

当前硬编码为 Base 模型参数，修改 `jni/src/whisper_inference.h`：

```cpp
int d_model_ = 512;    // tiny=384, base=512, small=768, medium=1024, large=1280
int num_layers_ = 6;   // tiny=4,   base=6,   small=12,  medium=24,   large=32
```

修改后重新生成 DLA 文件并重新编译。
