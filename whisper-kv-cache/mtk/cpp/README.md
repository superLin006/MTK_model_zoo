# Whisper KV Cache - MTK NPU C++ 实现

这是 Whisper 语音识别模型的 KV Cache 优化版本，专门针对联发科 (MTK) NPU 平台进行了优化。

## 项目结构

```
whisper-kv-cache/mtk/
├── models/                      # 模型文件（生产环境使用）
│   ├── encoder.dla             # 编码器 DLA 模型
│   ├── decoder_kv.dla          # 解码器 KV Cache DLA 模型
│   ├── token_embedding.npy     # Token 嵌入
│   ├── position_embedding.npy  # 位置嵌入
│   ├── mel_80_filters.txt      # Mel 频谱滤波器
│   └── vocab.txt               # 词汇表
├── test_data/                   # 测试音频文件
├── python/                      # Python 导出脚本（开发用）
└── cpp/                         # C++ 实现（生产用）
    ├── jni/                    # Android NDK 代码
    │   ├── src/                # 源代码
    │   │   ├── whisper_inference.cpp
    │   │   ├── whisper_inference.h
    │   │   └── utils/          # 工具函数
    │   └── libs/               # 编译产物
    ├── build_android.sh        # Android 编译脚本
    └── deploy_and_test.sh      # 部署和测试脚本
```

## 快速开始

### 1. 编译

```bash
cd /home/xh/projects/MTK/whisper-kv-cache/mtk/cpp
bash build_android.sh
```

### 2. 部署到设备

只部署，不运行测试：

```bash
bash deploy_and_test.sh
```

### 3. 部署并测试

**正常模式（生产环境，无 DEBUG 信息）：**

```bash
bash deploy_and_test.sh test_en.wav en
bash deploy_and_test.sh test_zh.wav zh
bash deploy_and_test.sh test_part1.wav zh
```

**调试模式（开发环境，显示所有 DEBUG 信息）：**

```bash
bash deploy_and_test.sh test_en.wav en debug
```

### 4. 手动测试（在设备上）

如果你想手动运行测试：

```bash
# 正常模式
adb shell "cd /data/local/tmp/whisper_kv_test && export LD_LIBRARY_PATH=. && export WHISPER_DEBUG=0 && ./whisper_kv_test . test_en.wav en"

# 调试模式
adb shell "cd /data/local/tmp/whisper_kv_test && export LD_LIBRARY_PATH=. && export WHISPER_DEBUG=1 && ./whisper_kv_test . test_en.wav en"
```

## 性能指标说明

运行测试后，程序会输出详细的性能报告，包括：

### 时间指标

- **Audio Duration**: 音频时长
- **Audio Preprocessing**: 音频预处理耗时（包含 STFT、mel 频谱计算等）
- **Encoder Inference**: 编码器推理耗时（MTK NPU 执行）
- **Decoder Inference**: 解码器推理总耗时（自回归解码，每个 token 一次推理）
- **Total Time**: 总推理时间

### 性能指标

- **RTF (Real-Time Factor)**: 实时率 = 推理时间 / 音频时长
  - RTF < 1.0: 比实时快（例如 RTF=0.12 表示比实时快 8.3 倍）
  - RTF = 1.0: 恰好实时
  - RTF > 1.0: 比实时慢

- **Throughput**: 吞吐量 = 1.0 / RTF（表示比实时快多少倍）

- **Tokens/Second**: 每秒生成的 token 数量

### 内存指标

- **Encoder Memory**: 编码器内存占用
- **Decoder KV Cache Memory**: 解码器 KV Cache 内存占用
- **Total Memory**: 总内存占用

### 示例输出

```
========================================
  Performance Test Report
========================================
Audio Duration:      28.82 seconds
Audio Preprocessing:  146.28 ms
Encoder Inference:    88.45 ms
Decoder Inference:    3395.22 ms (77 tokens, 44.16 ms/token)
Total Time:          3629.95 ms

Performance Metrics:
  RTF (Real-Time Factor): 0.126x
  Throughput:            7.95x realtime
  Tokens/Second:         21.21 tokens/s

Memory Usage:
  Encoder Memory:       2.93 MB
  Decoder KV Cache:     45.68 MB
  Total Memory:         48.61 MB
========================================
```

## 调试信息控制

### 设计理念

- **开发阶段**: DEBUG 信息非常有用，帮助定位问题
- **生产/验证阶段**: DEBUG 信息冗余，会淹没性能报告

### 实现方式

通过 **`WHISPER_DEBUG` 环境变量**在运行时控制调试输出，无需重新编译：

- `WHISPER_DEBUG=0`: 禁用 DEBUG 信息（生产模式）
- `WHISPER_DEBUG=1`: 启用 DEBUG 信息（开发模式）

### 代码实现

所有 DEBUG 输出都使用条件判断：

```cpp
// whisper_inference.cpp
if (debug_mode_) {
    std::cout << "[DEBUG] ..." << std::endl;
}

// audio_utils.cpp
if (is_debug_enabled()) {
    std::cout << "[DEBUG] ..." << std::endl;
}
```

这样做的好处：
- ✅ 运行时控制，无需重新编译
- ✅ 零性能开销（编译器会优化掉未执行的分支）
- ✅ 适合生产环境部署
- ✅ 便于问题排查

### 使用示例

```bash
# 生产环境（干净的输出，只看性能指标）
bash deploy_and_test.sh test_en.wav en

# 开发调试（查看所有内部信息）
bash deploy_and_test.sh test_en.wav en debug
```

## 支持的 Whisper 模型

目前代码支持 Whisper Base 模型。如果需要使用其他模型（tiny/small/medium/large），需要修改 `whisper_inference.h` 中的模型参数：

### 模型参数对照表

| 模型 | d_model | n_layers | 说明 |
|------|---------|----------|------|
| tiny | 384 | 4 | 最小最快 |
| base | 512 | 6 | 当前使用 |
| small | 768 | 12 | 中等精度 |
| medium | 1024 | 24 | 高精度 |
| large | 1280 | 32 | 最高精度 |

### 修改步骤

1. 编辑 `jni/src/whisper_inference.h`，找到以下行：

```cpp
int d_model_ = 512;        // 修改为对应模型的值
int num_layers_ = 6;       // 修改为对应模型的值
```

2. 使用 Python 导出脚本重新生成对应模型的 DLA 文件：

```bash
cd ../python
python export_models.py --model small  # 或 tiny/medium/large
```

3. 更新 models 目录（DLA 文件会自动生成到正确位置）

4. 重新编译和部署：

```bash
cd ../cpp
bash build_android.sh
bash deploy_and_test.sh test_en.wav en
```

**注意：** `max_cache_len_` 固定为 448，所有模型都相同。

## 模型文件管理

### 目录说明

- **`mtk/models/`**: 生产环境使用的模型文件
  - 包含 DLA 模型、嵌入、mel 滤波器、词汇表
  - 这是 `deploy_and_test.sh` 实际推送到设备的文件

- **`mtk/python/models/`**: Python 导出脚本生成的中间文件
  - 包含 `.pt`, `.tflite`, `.dla` 等多种格式
  - 开发阶段使用，不应直接部署

### 最佳实践

1. 使用 Python 脚本导出新模型后，复制必要文件到 `mtk/models/`：

```bash
cd /home/xh/projects/MTK/whisper-kv-cache/mtk
cp python/models/encoder.dla models/
cp python/models/decoder_kv.dla models/
cp python/models/token_embedding.npy models/
cp python/models/position_embedding.npy models/
```

2. `mtk/models/` 中的文件是最终部署版本，确保它们始终是最新的

3. 不要直接编辑 `mtk/models/` 中的文件，始终从 Python 导出后复制

## 测试音频文件

项目包含以下测试音频：

- `test_en.wav`: 英文测试音频（~28秒）
- `test_zh.wav`: 中文测试音频（~28秒）
- `jfk.flac`: JFK 演讲（短音频）
- `test_part1.wav`: 长音频测试（第一部分，~28秒）
- `test_part2.wav`: 长音频测试（第二部分，~28秒）

**注意：** Whisper 模型对超过 30 秒的音频支持有限，建议使用 `ffmpeg` 或其他工具将长音频切分为 30 秒以内的片段。

### 切分音频示例

```bash
# 切分音频为 30 秒的片段
ffmpeg -i long_audio.wav -f segment -segment_time 30 -c copy output_%03d.wav
```

## 常见问题

### Q: 为什么使用环境变量控制 DEBUG 而不是编译宏？

A: 有几个原因：
1. **运行时切换**: 无需重新编译即可切换调试模式
2. **同一个二进制**: 开发和生产可以使用同一个可执行文件
3. **零开销**: 现代编译器会优化掉未执行的条件分支
4. **便于自动化**: CI/CD 流程中容易控制

### Q: 如何测试更大的模型（small/medium/large）？

A: 需要三个步骤：
1. 修改 `whisper_inference.h` 中的 `d_model_` 和 `num_layers_` 参数
2. 使用 Python 导出脚本重新生成对应模型的 DLA 文件
3. 复制新的 DLA 文件到 `mtk/models/`，然后重新编译和部署

### Q: RTF 是什么意思？

A: RTF (Real-Time Factor) 是推理速度的指标：
- RTF = 0.1 表示处理 1 秒音频只需要 0.1 秒（快 10 倍）
- RTF = 1.0 表示实时处理（处理 1 秒音频需要 1 秒）
- RTF = 2.0 表示比实时慢 2 倍（处理 1 秒音频需要 2 秒）

### Q: KV Cache 有什么好处？

A: KV Cache 优化了 Transformer 解码器的自回归推理：
- **无 KV Cache**: 每次生成 token 都需要重新计算所有历史 token 的 K/V 值
- **有 KV Cache**: 缓存历史 token 的 K/V 值，只计算当前 token
- **性能提升**: 在长序列生成时可以获得 5-10 倍的加速

### Q: 为什么不把所有文件都放在 python/models/?

A: 职责分离：
- `python/models/`: Python 开发环境的输出目录，包含多种中间格式（.pt, .tflite, .dla）
- `mtk/models/`: 生产环境的干净目录，只包含部署所需的最终文件
- 这样可以避免意外部署开发文件，保持生产环境的清洁

## 技术细节

### MTK Neuron API

项目使用 MTK Neuron Runtime API 来加载和执行 DLA 模型：

- `NeuronModel_restoreFromCompiledNetwork()`: 加载 DLA 模型
- `NeuronExecution_create()`: 创建执行实例
- `NeuronExecution_setInput()`: 设置输入数据
- `NeuronExecution_setOutput()`: 设置输出缓冲区
- `NeuronExecution_compute()`: 执行推理

### 音频预处理流程

1. **加载音频**: 使用 libsndfile 加载 WAV/FLAC 文件
2. **重采样**: 转换为 16kHz 单声道
3. **STFT**: 短时傅里叶变换
4. **Mel 频谱**: 80 个 mel 频率通道
5. **归一化**: Log mel 频谱归一化

### 自回归解码

1. 使用特殊 token 初始化（`<|startoftranscript|>`, `<|zh|>`, `<|notimestamps|>`）
2. 循环生成 token 直到遇到结束标记或达到最大长度
3. 每次迭代更新 KV Cache
4. 使用贪心解码策略（argmax）

## 参考资料

- [Whisper 论文](https://arxiv.org/abs/2212.04356)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [MTK NeuroPilot SDK 文档](https://github.com/MediaTek-NeuroPilot)

## License

本项目基于 OpenAI Whisper，遵循 MIT License。
