# MTK Model Zoo

MTK NPU 算法模型移植工作区，包含多个深度学习模型在 MTK NeuroPilot SDK 上的实现。

## 📁 项目结构

```
MTK/
├── .claude/                    # Claude Code 配置和文档
│   ├── subagents/             # Subagent 自动化模板
│   ├── standards/             # 代码规范文档
│   └── doc/                   # 知识库文档
│
├── whisper/                   # Whisper 语音识别模型
│   └── mtk/
│       ├── python/            # Python端转换（.pt → .tflite → .dla）
│       └── cpp/               # C++ Android推理实现
│
├── superResolution/           # 超分辨率模型集合
│   ├── edsr/                  # EDSR 模型
│   ├── rcan/                  # RCAN 模型
│   └── realesrgan/            # RealESRGAN 模型
│
├── sense-voice/               # SenseVoice 语音识别
│
├── helsinki/                  # Helsinki Transformer
│
└── 0_Toolkits/               # MTK SDK 和工具（不上传）
```

## 🎯 支持的模型

### ✅ 已验证
- **Whisper** (语音识别) - 基于OpenAI Whisper base模型
- **EDSR** (超分辨率) - Enhanced Deep Super-Resolution

### 🔄 开发中
- **RCAN** (超分辨率)
- **RealESRGAN** (超分辨率)
- **SenseVoice** (语音识别)
- **Helsinki** (Transformer)

## 🛠️ 技术栈

- **平台**: MTK NeuroPilot SDK 8.0.10
- **目标芯片**: MT8371, MT6899, MT6991
- **深度学习框架**: PyTorch → TorchScript → TFLite → DLA
- **推理引擎**: MTK Neuron Runtime
- **开发环境**: Python 3.10, Android NDK

## 📦 快速开始

### 1. 环境准备

**注意**: 本仓库不包含 MTK SDK 和模型权重文件，需要单独下载。

```bash
# 1. Clone 仓库
git clone https://github.com/superLin006/MTK_model_zoo.git
cd MTK_model_zoo

# 2. 下载 MTK NeuroPilot SDK
# 将 SDK 解压到 0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/

# 3. 下载模型权重
# 放置到对应项目的 models/ 目录
# 例如: whisper/mtk/models/base.pt
```

### 2. Python 端模型转换

以 Whisper 为例：

```bash
cd whisper/mtk/python

# Step 1: PyTorch → TorchScript
python step1_pt_to_torchscript.py

# Step 2: TorchScript → TFLite
python step2_torchscript_to_tflite.py

# Step 3: TFLite → DLA
python step3_tflite_to_dla.py --platform MT8371
```

转换后的模型保存在 `python/models/` 目录。

### 3. C++ Android 推理

```bash
cd whisper/mtk/cpp

# 编译 (需要 Android NDK)
bash build_android.sh

# 部署到设备
bash deploy_android.sh

# 运行测试
bash run_android_tests.sh
```

## 📚 文档

### Claude Code Subagent 系统

本项目使用 Claude Code 的 Subagent 系统实现算法移植的自动化：

- **project-initializer**: 项目初始化和环境配置
- **operator-analyst**: 算子兼容性分析
- **python-converter**: Python端模型转换
- **cpp-implementer**: C++ 推理实现
- **android-deployer**: Android 部署和测试

详见：`.claude/subagents/README.md`

### 标准和规范

- **Python 输出管理**: `.claude/standards/python_output_management.md`
- **MTK 算子支持列表**: `.claude/doc/mtk_mdla_operators.md`
- **最佳实践知识库**: `.claude/doc/mtk_npu_knowledge_base.md`

## 🔧 开发工作流

```
1. 算子分析 (operator-analyst)
   ↓
2. Python 转换 (python-converter)
   - .pt → TorchScript → TFLite → DLA
   ↓
3. C++ 实现 (cpp-implementer)
   - 预处理、模型加载、推理、后处理
   ↓
4. Android 部署 (android-deployer)
   - 编译、部署、测试
```

## 📊 性能基准

| 模型 | 平台 | 推理时间 | 精度 |
|------|------|----------|------|
| Whisper Base | MT8371 | ~700-800ms | 95%+ |
| EDSR x2 | MT8371 | TBD | TBD |

## ⚠️ 重要说明

### 不上传的内容
本仓库通过 `.gitignore` 排除了以下内容：

- ❌ MTK SDK (太大，需单独下载)
- ❌ 模型权重文件 (.pt, .pth, .tflite, .dla)
- ❌ 测试数据 (音频、图像文件)
- ❌ 编译产物 (libs/, obj/, __pycache__)
- ❌ 中间输出 (test/outputs/)

### 保留的内容
- ✅ 源代码 (.py, .cpp, .h)
- ✅ 配置文件 (.json, .yaml, Android.mk, CMakeLists.txt)
- ✅ 构建脚本 (.sh)
- ✅ 文档 (.md)
- ✅ Claude Code 配置 (.claude/)

### 目录占位
使用 `.gitkeep` 文件保留空目录结构：
- `models/` - 模型权重目录（需自行下载）
- `test_data/` - 测试数据目录（需自行准备）
- `test/outputs/` - 测试输出目录（自动生成）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目代码使用 MIT 许可证。

**注意**: 各模型的原始权重文件需遵守其各自的许可证：
- Whisper: MIT License (OpenAI)
- EDSR: Proprietary
- 其他模型请查看各自的官方仓库

## 🔗 相关资源

- [MTK NeuroPilot SDK 文档](https://neuropilot.mediatek.com/)
- [Whisper 官方仓库](https://github.com/openai/whisper)
- [EDSR 官方仓库](https://github.com/sanghyun-son/EDSR-PyTorch)

---

**创建日期**: 2026-02-06
**维护者**: superLin006
