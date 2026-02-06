# MTK SenseVoice

SenseVoice Small 语音识别模型在 MediaTek NPU (MTK NeuroPilot) 上的完整部署方案。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-MTK%20NPU-orange.svg)](https://www.mediatek.com/products/smartphones)
[![Status](https://img.shields.io/badge/status-ready--to--deploy-success.svg)]()

---

## 📋 项目简介

本项目提供了从 FunASR SenseVoice Small 模型到 MTK NPU 部署的完整工具链：

- ✅ **模型转换**: PyTorch → TorchScript → TFLite → DLA
- ✅ **C++ 推理**: 完整的端到端推理管道
- ✅ **多平台支持**: MT6899 / MT6991 / MT8371
- ✅ **高性能**: RTF < 0.04 (实时率，< 4% 处理时间)

### 支持的语言

- 🇨🇳 中文 (zh)
- 🇬🇧 英文 (en)
- 🇭🇰 粤语 (yue)
- 🇯🇵 日语 (ja)
- 🇰🇷 韩语 (ko)

---

## 📁 项目结构

```
MTK-sense-voice/
├── SenseVoice_workspace/          # 模型转换工作区
│   ├── models/                    # 原始模型
│   ├── audios/                    # 测试音频
│   ├── model_prepare/             # PyTorch → TFLite 转换
│   └── compile/                   # TFLite → DLA 编译
│
└── sensevoice_mtk_cpp/            # C++ 推理代码
    ├── jni/                       # NDK 构建系统
    │   ├── src/
    │   │   ├── sensevoice/        # SenseVoice 核心代码
    │   │   ├── executor/          # NPU 执行器
    │   │   ├── neuron/            # NeuroPilot API
    │   │   ├── common/            # 公共工具
    │   │   ├── trace/             # 性能分析
    │   │   └── utils/             # 工具函数
    │   ├── third_party/
    │   │   └── easyloggingpp/     # 日志库
    │   ├── Android.mk             # NDK 构建配置
    │   └── Application.mk         # NDK 应用配置
    ├── build.sh                   # 构建脚本
    └── deploy_and_test.sh         # 部署测试脚本
```

---

## 🚀 快速开始

### 方式一：直接使用已编译的 DLA 模型

如果你已经有 DLA 模型文件，可以直接使用 C++ 推理代码：

```bash
cd sensevoice_mtk_cpp
./build.sh
./deploy_and_test.sh --test <audio_file>
```

### 方式二：完整的模型转换流程

#### 1. 环境准备

**Python 环境**:

```bash
# 创建 Python 环境
conda create -n MTK-sensevoice python=3.10
conda activate MTK-sensevoice

# 安装 Python 依赖
cd SenseVoice_workspace/model_prepare
pip install torch torchvision torchaudio
pip install funasr modelscope
pip install librosa
```

**MTK NeuroPilot SDK** (必需):

```bash
# 下载地址 (需要 MTK 账号)
# https://vendor.mediatek.com/

# 推荐版本: neuropilot-sdk-basic-8.0.10 或更高
# 安装路径示例
export NEUROPILOT_SDK="/home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"

# 验证安装
ls "$NEUROPILOT_SDK/host/bin/ncc-tflite"
```

**Android NDK** (必需):

```bash
# 下载 Android NDK r25c
# https://developer.android.com/ndk/downloads

# 设置环境变量 (可选，build.sh 会自动查找)
export ANDROID_NDK="/home/xh/Android/Ndk/android-ndk-r25c"
```

#### 2. 下载模型

```bash
cd SenseVoice_workspace/models
modelscope download --model iic/SenseVoiceSmall --local_dir sensevoice-small
```

#### 3. 模型转换

```bash
cd SenseVoice_workspace/model_prepare

# Step 1: 保存为 TorchScript (固定166帧 = 10秒音频)
python3 main.py --mode=SAVE_PT

# Step 2: 转换为 TFLite
python3 pt2tflite.py -i model/sensevoice_complete.pt \
                     -o model/sensevoice_complete.tflite \
                     --float 1

# Step 3: 验证转换结果
python3 test_converted_models.py --audio ../audios/test_en.wav
```

#### 4. 编译 DLA

```bash
cd SenseVoice_workspace/compile

# 设置 SDK 路径 (如果之前没有设置环境变量)
NEUROPILOT_SDK="/home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"

# 选择目标平台: MT6899 / MT6991 / MT8371
./compile_sensevoice_fp.sh \
    ../model_prepare/model/sensevoice_complete.tflite \
    MT8371 \
    "$NEUROPILOT_SDK"
```

#### 5. 构建 C++ 推理程序

```bash
cd ../../sensevoice_mtk_cpp
./build.sh
```

---

## 📊 模型规格

### 输入输出规格

| 项目 | Shape | 类型 | 说明 |
|------|-------|------|------|
| 输入1 (特征) | `[1, 166, 560]` | float32 | Fbank + LFR 特征 |
| 输入2 (语言) | `[1]` | float32 | 语言 ID |
| 输入3 (事件1) | `[1]` | float32 | 事件 ID |
| 输入4 (事件2) | `[1]` | float32 | 事件类型 ID |
| 输入5 (文本规范) | `[1]` | float32 | 文本规范化 ID |
| 输出 | `[1, 170, 25055]` | float32 | CTC logits |

### 音频处理参数

- **采样率**: 16 kHz mono
- **固定长度**: 10秒 (166帧)
- **特征维度**: 80 (Fbank) → 560 (LFR: 80×7)
- **短音频**: 自动 padding 到 166 帧
- **长音频**: 截断前 166 帧 (约 10 秒)

---

## ✅ 测试结果

### MT8371 平台测试

| 音频 | 时长 | 语言 | 识别结果 | RTF |
|------|------|------|---------|-----|
| test_zh.wav | 5.6s | 中文 | ✅ "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢" | 0.036 |
| audio5.wav | 9.3s | 英文 | ✅ "the media tech deep learning accelerator mdla is a powerful and efficient..." | 0.024 |
| test_en.wav | 5.9s | 英文 | ✅ "mister quilter is the apostle of the middle classes..." | 0.028 |

**性能指标**:
- 推理速度: ~200ms (10秒音频)
- RTF (实时率): < 0.04
- 内存占用: ~450MB

---

## 🔧 支持平台

| 平台 | SoC | MDLA版本 | L1缓存 | 核心数 | 状态 |
|------|-----|---------|--------|--------|------|
| MT6899 | Dimensity 1200/1100 | MDLA5.5 | 2048KB | 2 | ✅ |
| MT6991 | Dimensity 9300/9200 | MDLA5.5 | 7168KB | 4 | ✅ |
| MT8371 | Genio 700 | MDLA5.3 + EDMA3.6 | 256KB | 1 | ✅ |

---

## 📖 使用文档

### C++ 推理使用

```bash
# 基本用法
./sensevoice_main <model.dla> <tokens.txt> <audio.wav> [language] [text_norm]

# 示例
./sensevoice_main sensevoice_MT8371.dla tokens.txt test.wav
./sensevoice_main sensevoice_MT8371.dla tokens.txt test.wav zh
./sensevoice_main sensevoice_MT8371.dla tokens.txt test.wav auto with_itn
```

### 参数说明

- **language**: `auto` / `zh` / `en` / `yue` / `ja` / `ko`
- **text_norm**: `with_itn` (文本规范化) / `without_itn` (原始输出)

---

## 🛠️ 技术栈

### 模型转换
- **PyTorch**: 深度学习框架
- **FunASR**: 模型下载和特征提取
- **TFLite**: 中间格式
- **NeuroPilot Compiler**: DLA 编译器

### C++ 推理
- **NDK**: Android NDK r25c
- **kaldi-native-fbank**: 特征提取
- **NeuroPilot SDK**: NPU 运行时
- **easyloggingpp**: 日志库

---

## ⚠️ 注意事项

### 1. 音频长度限制
- 模型固定为 10 秒音频 (166 帧)
- 超过 10 秒会被截断，丢失后半部分
- 建议使用滑动窗口处理长音频

### 2. 特征提取
- ✅ **使用 kaldi-native-fbank** (与训练一致)
- ❌ **不要使用 librosa** (会有差异)

### 3. Prompt Embedding
- 语言、事件等参数在模型编译时已固定
- 运行时参数不生效（避免 GATHER 操作）
- 默认配置: auto 语言 + Speech 事件

---

## 🎯 常见问题

**Q: 为什么固定 10 秒音频？**
A: DLA 编译需要固定 shape 以优化性能。可通过修改 `model_prepare/main.py` 中的 `fixed_frames` 调整。

**Q: 如何处理长音频？**
A: 使用滑动窗口分段处理，每段 10 秒，步长 8-9 秒保留上下文。

**Q: 推理速度慢怎么办？**
A: 检查是否启用了 APU 电源管理，确保 NPU 频率正常。

**Q: 不同平台可以通用 DLA 文件吗？**
A: 不可以，每个平台需要单独编译。

**Q: 编译 DLA 时提示 `ncc-tflite: command not found` 怎么办？**
A: 需要设置正确的 NeuroPilot SDK 路径：

```bash
# 1. 确认 SDK 已安装
ls /home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/host/bin/ncc-tflite

# 2. 编译时传入正确的 SDK 路径
./compile_sensevoice_fp.sh \
    ../model_prepare/model/sensevoice_complete.tflite \
    MT8371 \
    /home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk
```

**Q: 如何获取 NeuroPilot SDK？**
A: 需要访问 MediaTek 官方网站 (https://vendor.mediatek.com/) 并注册账号，下载对应版本的 SDK。推荐使用 `neuropilot-sdk-basic-8.0.10` 或更高版本。

---

## 📚 参考资料

- [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)
- [SenseVoice ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall)
- MediaTek NeuroPilot SDK 文档

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**最后更新**: 2026-01-12
