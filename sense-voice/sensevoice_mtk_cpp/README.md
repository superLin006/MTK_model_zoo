# SenseVoice MTK NPU C++ 推理

SenseVoice Small 模型在 MediaTek NPU 上的 C++ 推理实现。

---

## 📋 简介

本项目提供完整的 SenseVoice 语音识别 C++ 推理管道，专为 MTK NPU 平台优化。

### 特性

- ✅ **端到端推理**: WAV 音频输入 → 文本输出
- ✅ **高性能**: RTF < 0.04 (实时率 < 4%)
- ✅ **多语言支持**: 中文、英文、粤语、日语、韩语
- ✅ **特征提取**: kaldi-native-fbank (与训练一致)
- ✅ **CTC 解码**: Greedy search 解码
- ✅ **自动处理**: Padding/Truncation 适配固定输入

---

## 📁 目录结构

```
sensevoice_mtk_cpp/
├── jni/                             # NDK 构建系统
│   ├── Android.mk                   # NDK 构建配置
│   ├── Application.mk               # NDK 应用配置
│   ├── src/                         # 源代码
│   │   ├── sensevoice/              # SenseVoice 核心代码
│   │   │   ├── include/
│   │   │   │   ├── sensevoice.h         # 主接口
│   │   │   │   ├── sensevoice_config.h  # 配置结构
│   │   │   │   ├── sensevoice_model.h   # 模型封装
│   │   │   │   ├── audio_frontend.h     # 音频前端
│   │   │   │   └── tokenizer.h          # 分词器
│   │   │   └── src/
│   │   │       ├── sensevoice.cpp
│   │   │       ├── sensevoice_model.cpp
│   │   │       ├── audio_frontend.cpp
│   │   │       ├── tokenizer.cpp
│   │   │       └── main.cpp             # 可执行程序入口
│   │   ├── executor/                  # NPU 执行器
│   │   │   ├── Executor.h
│   │   │   ├── ExecutorFactory.h/cpp
│   │   │   ├── NeuronExecutor.h/cpp
│   │   │   └── NeuronUsdkExecutor.h/cpp
│   │   ├── neuron/                    # NeuroPilot API
│   │   │   ├── NeuronRuntimeLibrary.h/cpp
│   │   │   └── api/                    # Neuron API 头文件
│   │   ├── common/                    # 公共工具
│   │   │   ├── Log.h                   # 日志系统
│   │   │   ├── Macros.h
│   │   │   └── SharedLib.h
│   │   ├── trace/                     # 性能分析
│   │   │   ├── Trace.h/cpp
│   │   │   ├── ScopeProfiler.h/cpp
│   │   │   └── Stopwatch.h/cpp
│   │   └── utils/                     # 工具函数
│   │       ├── Utils.h/cpp
│   │       ├── MemAllocator.h/cpp
│   │       └── DumpWorker.h/cpp
│   └── third_party/
│       └── easyloggingpp/             # 日志库
│           ├── include/easyloggingpp/easylogging++.h
│           ├── easylogging++.cc
│           └── Android.mk
├── build.sh                          # 构建脚本
└── deploy_and_test.sh                # 部署测试脚本
```

---

## 🚀 快速开始

### 前置要求

- Android NDK r25c
- kaldi-native-fbank 预编译库
- MediaTek NeuroPilot SDK (设备上)
- 已编译的 DLA 模型文件

### 构建步骤

#### 1. 安装 kaldi-native-fbank

```bash
# 克隆并编译
git clone https://github.com/csukuangfj/kaldi-native-fbank.git
cd kaldi-native-fbank
mkdir build-android && cd build-android
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_STL=c++_shared \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 复制到项目目录
mkdir -p /home/xh/projects/MTK/1_third_party/kaldi_native_fbank/Android
cp -r install/* /home/xh/projects/MTK/1_third_party/kaldi_native_fbank/Android/
```

#### 2. 构建可执行程序

```bash
cd /home/xh/projects/MTK/sense-voice/sensevoice_mtk_cpp
./build.sh
```

构建输出:
```
libs/arm64-v8a/
├── sensevoice_main      # 主程序
└── libc++_shared.so     # C++ 运行时
```

#### 3. 部署到设备

```bash
./deploy_and_test.sh --test <audio_file>
```

或手动部署:

```bash
# 创建设备目录
adb shell "mkdir -p /data/local/tmp/sensevoice"

# 推送文件
adb push libs/arm64-v8a/sensevoice_main /data/local/tmp/sensevoice/
adb push libs/arm64-v8a/libc++_shared.so /data/local/tmp/sensevoice/
adb push <path/to>/sensevoice_MT8371.dla /data/local/tmp/sensevoice/
adb push <path/to>/tokens.txt /data/local/tmp/sensevoice/
adb push <audio_file> /data/local/tmp/sensevoice/

# 运行
adb shell "cd /data/local/tmp/sensevoice && \
           export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/data/local/tmp/sensevoice && \
           ./sensevoice_main sensevoice_MT8371.dla tokens.txt <audio_file>"
```

---

## 📖 使用方法

### 命令行参数

```bash
./sensevoice_main <model.dla> <tokens.txt> <audio.wav> [language] [text_norm]
```

### 参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|-------|--------|
| model.dla | DLA 模型文件路径 | - | 必填 |
| tokens.txt | 词汇表文件 | - | 必填 |
| audio.wav | 音频文件 (16kHz mono WAV) | - | 必填 |
| language | 语言提示 | auto, zh, en, yue, ja, ko | auto |
| text_norm | 文本规范化 | with_itn, without_itn | without_itn |

### 示例

```bash
# 自动检测语言
./sensevoice_main sensevoice_MT8371.dla tokens.txt test.wav

# 指定中文
./sensevoice_main sensevoice_MT8371.dla tokens.txt test.wav zh

# 指定英文 + 文本规范化
./sensevoice_main sensevoice_MT8371.dla tokens.txt test.wav en with_itn
```

---

## 🏗️ 代码架构

### 核心模块

#### 1. SenseVoice (主接口)

```cpp
namespace sensevoice {

class SenseVoice {
public:
    // 初始化
    bool Initialize(const std::string& model_path,
                    const std::string& tokens_path);

    // 识别音频文件
    RecognitionResult RecognizeFile(const std::string& audio_path,
                                    Language language = Language::Auto,
                                    TextNorm text_norm = TextNorm::WithoutITN);

    // 识别音频样本
    RecognitionResult Recognize(const std::vector<float>& samples,
                                Language language = Language::Auto,
                                TextNorm text_norm = TextNorm::WithoutITN);
};

}  // namespace sensevoice
```

#### 2. AudioFrontend (音频前端)

- WAV 文件加载
- Fbank 特征提取 (kaldi-native-fbank)
- LFR (Low Frame Rate) 变换
- CMVN (Mean & Variance Normalization)

```cpp
// 特征提取流程
Raw Audio (16kHz) → Fbank (80-dim) → LFR (560-dim) → CMVN
```

#### 3. Tokenizer (分词器)

- CTC Greedy Search 解码
- Token ID → 文本转换
- 特殊 token 过滤 (`<|zh|>`, `<|en|>`, etc.)

#### 4. SenseVoiceModel (模型封装)

- NeuronUsdk 执行器管理
- 输入输出 tensor 管理
- Padding/Truncation 处理

---

## 📊 性能指标

### MT8371 测试结果

| 音频 | 时长 | 语言 | RTF | 识别结果 |
|------|------|------|-----|---------|
| test_zh.wav | 5.6s | 中文 | 0.036 | ✅ 完全正确 |
| audio5.wav | 9.3s | 英文 | 0.024 | ✅ 完全正确 |
| test_en.wav | 5.9s | 英文 | 0.028 | ✅ 完全正确 |

**性能说明**:
- **推理时间**: ~200ms (10秒音频)
- **RTF**: < 0.04 (处理时间 < 音频时长的 4%)
- **内存占用**: ~450MB
- **APU 频率**: 自动调节 (30000)

---

## ⚙️ 配置

### 模型配置 (sensevoice_config.h)

```cpp
// 音频配置
struct AudioConfig {
    int sample_rate = 16000;      // 采样率
    int num_mel_bins = 80;        // Fbank 维度
    int frame_length = 25;        // 帧长 (ms)
    int frame_shift = 10;         // 帧移 (ms)
};

// 模型配置
struct ModelConfig {
    std::string model_path;
    int vocab_size = 25055;       // 词汇表大小
    int input_feat_dim = 560;     // LFR 后特征维度 (80 * 7)
    int encoder_out_dim = 512;    // 编码器输出维度
    int num_heads = 4;            // 注意力头数
};
```

### 编译配置 (Application.mk)

```makefile
APP_ABI := arm64-v8a
APP_STL := c++_shared
APP_CPPFLAGS := -std=c++17 -fexceptions -frtti
APP_PLATFORM := android-29
```

---

## ⚠️ 注意事项

### 1. 音频长度限制

- 模型固定输入: **166 帧 = ~10 秒音频**
- 短音频: 自动 padding 到 166 帧
- 长音频: 截断前 166 帧 (后续内容丢失)

**解决方案**: 使用滑动窗口分段处理长音频

### 2. 特征提取

- ✅ **必须使用**: kaldi-native-fbank
- ❌ **不要使用**: librosa (与训练时特征有差异)

### 3. Prompt Embedding

- 语言、事件等参数在模型编译时已固定
- 运行时参数不生效 (避免 GATHER 操作)
- 默认配置: auto 语言 + Speech 事件

### 4. 内存对齐

- 输入特征维度必须是 560 (80 * 7)
- 确保 float32 数据类型
- 注意字节对齐

---

## 🐛 调试

### 启用调试日志

```cpp
// 在代码中设置日志级别
#define ELPP_DEBUG
```

### 常见问题

**Q: 编译时找不到 kaldi-native-fbank 头文件**
A: 确保 `KALDI_FBANK_PATH` 正确指向 include 目录

**Q: 运行时出现 "Couldn't find the shape info"**
A: 检查 `NeuronUsdkExecutor.cpp` 中是否添加了 SenseVoice 配置

**Q: 识别结果为空**
A: 检查:
1. 音频是否为 16kHz mono
2. DLA 模型是否匹配当前平台
3. tokens.txt 是否正确

**Q: 输出全是 inf**
A: 检查输入特征是否正确，LFR 变换后的维度应该是 560

---

## 🔧 依赖库

### 必需

- **kaldi-native-fbank**: 特征提取
- **NeuroPilot SDK**: NPU 运行时 (设备自带)
- **easyloggingpp**: 日志库
- **libc++_shared**: C++ 运行时

### 可选

- **APUWareUtilsLib**: APU 电源管理 (提升性能)

---

## 📚 参考资料

- [kaldi-native-fbank GitHub](https://github.com/csukuangfj/kaldi-native-fbank)
- [FunASR SenseVoice](https://github.com/alibaba-damo-academy/FunASR)
- MediaTek NeuroPilot SDK 文档

---

## 📄 许可证

MIT License

---

**测试状态**: ✅ MT8371 通过
**部署就绪**: ✅ 是
**最后更新**: 2026-01-12
