# MTK Model Zoo

MTK NPU 算法模型移植工作区，包含多个深度学习模型在 MTK NeuroPilot SDK 上的实现。

## 项目结构

```
MTK_models_zoo/
├── 0_Toolkits/                 # MTK SDK 和工具（不上传 GitHub）
│   └── neuropilot-sdk-basic-8.0.10-build20251029/
│       └── neuron_sdk/
│
├── 1_third_party/              # 第三方库（OpenCV 等）
│
├── whisper/                    # Whisper 语音识别模型
│   └── mtk/
│       ├── python/             # Python 端转换（.pt → .tflite → .dla）
│       │   ├── step1_pt_to_torchscript.py
│       │   ├── step2_torchscript_to_tflite.py
│       │   ├── step3_tflite_to_dla.py
│       │   ├── whisper_kv_model.py
│       │   ├── prepare_calib_data.py
│       │   ├── models/         # 模型权重（不上传）
│       │   ├── models_large_turbo/
│       │   └── test/
│       ├── cpp/                # C++ Android 推理实现
│       │   ├── build_android.sh
│       │   └── deploy_and_test.sh
│       └── test_data/          # 测试音频（不上传）
│
├── superResolution/            # 超分辨率模型集合
│   ├── edsr/
│   │   ├── mtk/                # MTK NPU 实现（输入固定 510x339）
│   │   └── rknn/               # RKNN 实现（参考）
│   ├── rcan/
│   │   ├── mtk/
│   │   └── rknn/
│   └── realesrgan/
│       ├── mtk/                # MTK NPU 实现（输入固定 510x339）
│       └── rknn/               # RKNN 实现（参考）
│
├── sense-voice/                # SenseVoice 语音识别
│   ├── SenseVoice_workspace/   # Python 端转换
│   └── sensevoice_mtk_cpp/     # C++ Android 推理实现
│
├── helsinki/                   # Helsinki NLP Transformer
│   ├── helsinki_workspace/     # Python 端转换
│   └── helsinki_mtk_cpp/       # C++ Android 推理实现
│
└── .claude/                    # Claude Code 配置和知识库
    ├── subagents/
    ├── standards/
    └── doc/
```

## 支持的模型

| 模型 | 类型 | 说明 |
|------|------|------|
| **Whisper** | 语音识别 | OpenAI Whisper large-v3-turbo，含 KV Cache，支持多语言 |
| **RealESRGAN** | 超分辨率 | x4 超分，输入 510x339，输出 2040x1356 |
| **EDSR** | 超分辨率 | x4 超分，输入 510x339，输出 2040x1356 |
| **RCAN** | 超分辨率 | x4 超分 |
| **SenseVoice** | 语音识别 | 多语言语音识别与情感分析 |
| **Helsinki** | NLP Transformer | 神经机器翻译 |

## 技术栈

- **平台**: MTK NeuroPilot SDK 8.0.10
- **目标芯片**: MT8371 / MT6899 / MT6991
- **转换链**: PyTorch → TorchScript → TFLite → DLA
- **推理引擎**: MTK Neuron Runtime
- **开发环境**: Python 3.10，Android NDK r25c

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/superLin006/MTK_model_zoo.git
cd MTK_model_zoo
```

### 2. 准备工具链

将以下内容放置到对应目录（不包含在仓库中，需自行下载）：

```
0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/   # MTK NeuroPilot SDK
```

配置环境变量（所有脚本会自动 fallback 到项目内相对路径，也可手动指定）：

```bash
export ANDROID_NDK=/path/to/android-ndk-r25c
# MTK SDK 默认从 0_Toolkits/ 自动查找，也可手动指定：
export MTK_NEURON_SDK=/path/to/neuron_sdk
```

### 3. Python 端模型转换

以 RealESRGAN 为例：

```bash
cd superResolution/realesrgan/mtk/python

# Step 1: PyTorch → TorchScript
python step1_pt_to_torchscript.py --checkpoint ../models/RealESRGAN_x4plus.pth

# Step 2: TorchScript → TFLite
python step2_torchscript_to_tflite.py --torchscript ../models/RealESRGAN_x4plus_core_510x339.pt

# Step 3: TFLite → DLA
python step3_tflite_to_dla.py --tflite ../models/RealESRGAN_x4plus_510x339.tflite --platform MT8371
```

### 4. 推理测试

```bash
# Python 端快速验证
cd superResolution/realesrgan/mtk/python/test
python test_pytorch.py \
  --model_path ../../models/RealESRGAN_x4plus.pth \
  --img_path ../../test_data/input_510x339.png \
  --output_path ../../test_data/output/result.png \
  --input_size 339 510
```

### 5. C++ Android 推理

```bash
cd superResolution/realesrgan/mtk/cpp

# 编译（需要设置 ANDROID_NDK）
bash build.sh

# 部署到设备并测试
bash deploy_with_sdk_lib.sh
```

## 路径约定

本项目所有脚本均使用**相对路径**，无需修改任何硬编码路径即可在任意机器上运行。

| 工具 | 查找方式 |
|------|----------|
| MTK NeuroPilot SDK | 自动从 `0_Toolkits/` 相对路径查找，或读取 `MTK_NEURON_SDK` 环境变量 |
| MTK Converter (Python) | 自动从 `0_Toolkits/` 相对路径查找，或读取 `MTK_CONVERTER_PATH` 环境变量 |
| Android NDK | 读取 `ANDROID_NDK` 环境变量 |
| rknn_model_zoo (外部) | 读取 `RKNN_MODEL_ZOO` 环境变量 |

## 不上传的内容

通过 `.gitignore` 排除：

- MTK SDK（体积过大，需单独下载）
- 模型权重文件（`.pt`, `.pth`, `.tflite`, `.dla`）
- 测试数据（音频、图像）
- 编译产物（`libs/`, `obj/`, `__pycache__`）
- 测试输出（`test/outputs/`）

## 相关资源

- [MTK NeuroPilot SDK](https://neuropilot.mediatek.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## 许可证

本项目代码使用 MIT 许可证。各模型原始权重遵守其各自的许可证。

---

**维护者**: superLin006
