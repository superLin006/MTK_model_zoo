# Real-ESRGAN MTK NPU 超分辨率

Real-ESRGAN超分辨率模型在MTK NPU平台上的完整实现，支持MT8371/MT6899/MT6991等芯片。

## 项目简介

本项目实现了Real-ESRGAN在MTK NPU上的端到端部署，包括：
- Python端：PyTorch模型转换为DLA格式
- C++端：Android设备上的NPU推理实现
- 已在MT8371设备上测试验证通过

**测试结果**：
- 输入：510×339 → 输出：2040×1356 (4倍超分)
- 推理时间：~4.6秒/帧 (MT8371)
- 输出质量：正常，无错误

## 快速开始

### 1. Python端：模型转换

将PyTorch模型转换为MTK NPU可用的DLA格式。

```bash
# 激活环境
conda activate MTK-superResolution

# Step 1: PyTorch → TorchScript
cd python
python step1_pt_to_torchscript.py \
    --checkpoint ../models/RealESRGAN_x4plus.pth \
    --input_height 339 \
    --input_width 510

# Step 2: TorchScript → TFLite
python step2_torchscript_to_tflite.py \
    --torchscript ../models/RealESRGAN_x4plus_core_339x510.pt

# Step 3: TFLite → DLA
python step3_tflite_to_dla.py \
    --tflite ../models/RealESRGAN_x4plus_339x510.tflite \
    --platform MT8371
```

**输出**：`models/RealESRGAN_x4plus_510x339_MT8371.dla`

### 2. C++端：编译和部署

在Android设备上编译和运行推理。

```bash
cd cpp

# 编译
./build.sh

# 部署到设备
adb push build/arm64-v8a/arm64-v8a/realesrgan_inference /data/local/tmp/realesrgan/
adb shell chmod 755 /data/local/tmp/realesrgan/realesrgan_inference

# 推送MTK运行时库
MTK_SDK="/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
adb push $MTK_SDK/mt8371/libneuron_runtime.8.so /data/local/tmp/realesrgan/libneuron_runtime.so
adb push $MTK_SDK/mt8371/libc++.so /data/local/tmp/realesrgan/
adb push $MTK_SDK/mt8371/libcutils.so /data/local/tmp/realesrgan/
adb push $MTK_SDK/mt8371/libbase.so /data/local/tmp/realesrgan/

# 推送模型和测试图像
adb push ../models/RealESRGAN_x4plus_510x339_MT8371.dla /data/local/tmp/realesrgan/model.dla
adb push ../test_data/input_510x339.png /data/local/tmp/realesrgan/input.png

# 运行推理
adb shell "cd /data/local/tmp/realesrgan && \
    export LD_LIBRARY_PATH=.\$LD_LIBRARY_PATH && \
    ./realesrgan_inference model.dla input.png output.png"

# 拉取结果
adb pull /data/local/tmp/realesrgan/output.png ./
```

## 项目结构

```
mtk/
├── python/                    # Python端：模型转换
│   ├── realesrgan_model.py   # Real-ESRGAN模型定义
│   ├── step1_pt_to_torchscript.py
│   ├── step2_torchscript_to_tflite.py
│   ├── step3_tflite_to_dla.py
│   ├── requirements.txt
│   └── test/                 # 测试脚本
│       ├── test_pytorch.py
│       ├── test_tflite.py
│       └── test_compare.py
│
├── cpp/                      # C++端：Android推理
│   ├── jni/
│   │   ├── Android.mk
│   │   ├── Application.mk
│   │   └── src/
│   │       ├── main.cpp
│   │       ├── realesrgan.h
│   │       ├── realesrgan.cpp
│   │       └── mtk_npu/
│   │           ├── neuron_executor.h
│   │           └── neuron_executor.cpp
│   ├── third_party/stb/      # STB图像库
│   ├── build.sh
│   ├── deploy_with_sdk_lib.sh
│   └── deploy_and_test.sh
│
├── models/                   # 模型文件
│   ├── RealESRGAN_x4plus.pth              # PyTorch模型
│   ├── RealESRGAN_x4plus_510x339.tflite  # TFLite模型
│   └── RealESRGAN_x4plus_510x339_MT8371.dla  # DLA模型
│
└── test_data/               # 测试数据
    ├── input_510x339.png
    └── output/
```

## 环境要求

### Python端
- Python 3.10
- PyTorch 1.13.1
- MTK Converter 8.16.0
- TensorFlow 2.13.0

### C++端
- Android NDK r25c
- MTK NeuroPilot SDK 8.0.10
- Android设备：MT8371/MT6899/MT6991

## Real-ESRGAN vs EDSR

| 特性 | EDSR | Real-ESRGAN |
|------|------|-------------|
| 归一化 | MeanShift (减均值) | 简单除以255 |
| 架构 | ResBlock | RRDB |
| 参数量 | ~1.5M | ~16.7M |
| 输入尺寸 | 256×256 | 510×339 |
| 输出尺寸 | 1024×1024 | 2040×1356 |
| 质量 | 良好 | 更好 |
| 速度 | 更快 | 较慢 |

**Real-ESRGAN特点**：
- 输入输出都在[0,1]范围，不需要MeanShift
- 前处理：`pixel / 255.0`
- 后处理：`clip(pixel, 0, 1) * 255`

## 技术要点

### 前处理（C++）
```cpp
// Real-ESRGAN：简单归一化到[0,1]
float scale = 1.0f / 255.0f;
output[idx] = (float)input[idx] * scale;
```

### 后处理（C++）
```cpp
// Real-ESRGAN：clip并转换回[0,255]
float val = input[idx];
val = (val < 0.0f) ? 0.0f : (val > 1.0f) ? 1.0f : val;
output[idx] = (unsigned char)(val * 255.0f + 0.5f);
```

## 性能数据

**MT8371测试结果**：
- 初始化：~460 ms
- 推理：~4669 ms
- 输出：2040×1356 RGB PNG
- 内存：~35 MB

**优化建议**：
- 使用INT8量化（当前FP32）
- 使用anime_6B变体（参数更少）
- 升级到MT6899/MT6991（性能更好）

## 依赖资源

- MTK SDK：`/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk`
- Android NDK：`/home/xh/Android/Ndk/android-ndk-r25c`
- Real-ESRGAN官方：https://github.com/xinntao/Real-ESRGAN

## 已完成

- ✅ Python端模型转换（PyTorch → TorchScript → TFLite → DLA）
- ✅ C++端推理实现
- ✅ Android NDK交叉编译
- ✅ MT8371设备测试验证
- ✅ 输出质量验证

## 参考

- EDSR MTK实现：`../edsr/mtk/`
- MTK NPU知识库：`~/.mtk_npu_knowledge_base.md`
- MTK支持算子：`~/.mtk_mdla_operators.md`

## 许可

本项目基于Real-ESRGAN和MTK SDK开发，仅供学习研究使用。
