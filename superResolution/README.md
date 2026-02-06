# SuperResolution

超分辨率模型实现，支持EDSR、RCAN、Real-ESRGAN，适配MTK NPU和RKNN平台。

## 项目简介

本项目实现了多个主流超分辨率模型在边缘设备上的部署，包括：
- **EDSR**: 增强型深度残差网络
- **RCAN**: 残差通道注意力网络
- **Real-ESRGAN**: 真实场景超分辨率（支持GAN训练）

## 项目结构

```
SuperResolution/
├── edsr/                          # EDSR超分辨率
│   ├── mtk/                       # MTK NPU实现
│   │   ├── python/                # Python模型转换
│   │   ├── cpp/                   # C++推理实现
│   │   └── models/                # DLA模型文件（使用Git LFS）
│   └── rknn/                      # RKNN实现
│       ├── python/                # Python模型转换
│       └── cpp/                   # C++推理实现
│
├── rcan/                         # RCAN超分辨率
│   ├── mtk/                       # MTK NPU实现
│   │   ├── python/                # Python模型转换
│   │   ├── cpp/                   # C++推理实现
│   │   └── models/                # DLA模型文件（使用Git LFS）
│   └── rknn/                      # RKNN实现
│       ├── python/                # Python模型转换
│       └── cpp/                   # C++推理实现
│
├── realesrgan/                   # Real-ESRGAN超分辨率 ✨
│   ├── mtk/                       # MTK NPU实现
│   │   ├── python/                # Python模型转换
│   │   ├── cpp/                   # C++推理实现
│   │   ├── models/                # DLA模型文件（使用Git LFS）
│   │   └── test_data/             # 测试数据
│   └── rknn/                      # RKNN实现
│       ├── python/                # Python模型转换
│       ├── cpp/                   # C++推理实现
│       ├── dataset/               # 校准数据集
│       └── model/                 # 测试图像
│
└── data/                          # 数据和预训练模型
    ├── models/                    # 预训练模型
    │   ├── edsr/                  # EDSR模型文件
    │   ├── rcan/                  # RCAN模型文件
    │   └── realesrgan/            # Real-ESRGAN模型文件 ✨
    └── test_images/               # 测试图像
```

## 支持的平台

- **MTK NPU**: MT8371, MT6899, MT6991, MT8189 (MDLA 5.3/5.5)
- **RKNN**: RK3588, RK3566, RK3568, RK3576

## 模型说明

### EDSR (Enhanced Deep Residual Networks)
- **输入**: 256×256 RGB
- **输出**: 1024×1024 RGB (4x超分)
- **架构**: ResBlock
- **参数量**: ~1.5M
- **特点**: 速度快，资源占用小
- **论文**: [EDSR: Enhanced Deep Residual Networks](https://arxiv.org/abs/1707.02921)

### RCAN (Residual Channel Attention Networks)
- **输入**: 510×339 RGB
- **输出**: 2040×1356 RGB (4x超分)
- **架构**: RCAB (Residual Channel Attention Block)
- **参数量**: ~15.6M
- **特点**: 使用通道注意力机制
- **论文**: [RCAN: Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

### Real-ESRGAN (Real-World Blind Super-Resolution) ✨
- **输入**: 510×339 RGB
- **输出**: 2040×1356 RGB (4x超分)
- **架构**: RRDB (Residual in Residual Dense Block)
- **参数量**: ~16.7M
- **特点**:
  - 适用于真实场景图像
  - 不需要MeanShift归一化
  - 输出质量更好
- **论文**: [Real-ESRGAN: Training with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)

## 快速开始

### MTK平台

#### EDSR
```bash
cd edsr/mtk/cpp
./build.sh
./deploy_with_sdk_lib.sh --test
```

#### RCAN
```bash
cd rcan/mtk/cpp
./build.sh
./deploy_with_sdk_lib.sh --test
```

#### Real-ESRGAN ✨
```bash
cd realesrgan/mtk/cpp
./build.sh
./deploy_with_sdk_lib.sh --test
```

### RKNN平台

#### EDSR
```bash
cd edsr/rknn/python
python convert.py
```

#### RCAN
```bash
cd rcan/rknn/python
python convert.py
```

#### Real-ESRGAN ✨
```bash
cd realesrgan/rknn/python
python convert.py
```

## 性能对比

### MTK MT8371平台

| 模型 | 输入 | 输出 | 推理时间 | FPS | 参数量 | 质量 |
|------|------|------|----------|-----|--------|------|
| EDSR | 256×256 | 1024×1024 | ~4600ms | 0.22 | 1.5M | 良好 |
| RCAN | 510×339 | 2040×1356 | ~4000ms | 0.25 | 15.6M | 优秀 |
| **Real-ESRGAN** | 510×339 | 2040×1356 | ~4669ms | 0.21 | 16.7M | **最佳** |

### MTK MT8189平台

| 模型 | 输入 | 输出 | 推理时间 | FPS |
|------|------|------|----------|-----|
| EDSR | 256×256 | 1024×1024 | ~7000ms | 0.14 |
| RCAN | 510×339 | 2040×1356 | ~4000ms | 0.25 |

### 性能分析

**EDSR**:
- 优势：速度最快，参数量最小
- 适用：实时应用、资源受限场景

**RCAN**:
- 优势：使用注意力机制，质量较好
- 适用：平衡性能和质量

**Real-ESRGAN**:
- 优势：输出质量最佳，适合真实场景
- 适用：离线处理、质量优先场景
- 特点：无需MeanShift，简化了前/后处理

## 模型对比

### 技术特点对比

| 特性 | EDSR | RCAN | Real-ESRGAN |
|------|------|------|-------------|
| **归一化** | MeanShift | MeanShift | 无 (/255) |
| **架构** | ResBlock | RCAB | RRDB |
| **注意力** | 无 | 通道注意力 | 无 |
| **参数量** | 1.5M | 15.6M | 16.7M |
| **训练** | PSNR | PSNR | GAN+PSNR |
| **场景** | 通用 | 通用 | 真实场景 |
| **速度** | 快 | 中 | 慢 |
| **质量** | 良好 | 优秀 | 最佳 |

### 前处理对比

```python
# EDSR & RCAN: MeanShift归一化
output = (input / 255.0) - rgb_mean  # 减去均值

# Real-ESRGAN: 简单归一化
output = input / 255.0  # 直接除以255
```

### 后处理对比

```python
# EDSR & RCAN: MeanShift反归一化
output = (model_output + rgb_mean) * 255.0  # 加回均值

# Real-ESRGAN: 简单反归一化+clip
output = np.clip(model_output, 0, 1) * 255.0  # clip后转换
```

## 依赖

### Python端
```bash
# 基础依赖
pip install torch torchvision numpy pillow opencv-python

# MTK平台
pip install mtk-converter tensorflow

# RKNN平台
pip install rknn-toolkit2
```

### C++端
- **编译**: Android NDK r25c+
- **MTK**: MTK NeuroPilot SDK 8.0.10
- **RKNN**: RKNN Toolkit2

### 硬件要求
- **开发机**: Ubuntu 20.04+, Python 3.10
- **MTK设备**: Android 8.1+, MT8371/MT6899/MT6991
- **RKNN设备**: RK3588/RK3566/RK3568

## 模型文件

预训练模型位置：`data/models/`

### EDSR
- `EDSR_x2.pt` - 2倍超分
- `EDSR_x3.pt` - 3倍超分
- `EDSR_x4.pt` - 4倍超分

### RCAN
- `RCAN_BIX2.pt` - 2倍超分
- `RCAN_BIX3.pt` - 3倍超分
- `RCAN_BIX4.pt` - 4倍超分
- `RCAN_BIX8.pt` - 8倍超分

### Real-ESRGAN ✨
- `RealESRGAN_x4plus.pth` - 4倍超分（通用场景）
- `RealESRGAN_x2plus.pth` - 2倍超分
- `RealESRGAN_x4plus_anime_6B.pth` - 动漫优化版

详细说明见各模型的 `model_info.txt` 文件。

## 测试结果

所有模型均在真实设备上测试验证：

### MTK MT8371测试结果 ✨

**Real-ESRGAN测试**:
- 输入：510×339 PNG
- 输出：2040×1356 PNG (4倍超分)
- 推理时间：4669ms
- 输出质量：正常，无错误
- 测试图像：见 `realesrgan/mtk/test_data/`

### RKNN RK3588测试结果 ✨

**Real-ESRGAN测试**:
- 支持FP16和INT8量化
- 测试图像：见 `realesrgan/rknn/model/`
- 输出对比：FP16 vs INT8质量对比

## 转换流程

所有模型都遵循相同的转换流程：

```
PyTorch (.pt/.pth)
    ↓
TorchScript (.pt)
    ↓
TFLite (.tflite)
    ↓
DLA (.dla) / RKNN (.rknn)
```

详细转换命令见各模型目录下的文档。

## 参考资料

### 论文
- [EDSR](https://arxiv.org/abs/1707.02921): Enhanced Deep Residual Networks for Single Image Super-Resolution
- [RCAN](https://arxiv.org/abs/1807.02758): Residual Channel Attention Networks for Image Super-Resolution
- [Real-ESRGAN](https://arxiv.org/abs/2107.10833): Training Real-World Blind Super-Resolution with Pure Synthetic Data

### 官方实现
- [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch)
- [RCAN](https://github.com/YapengTian/RCAN-IconVideocluster)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

### 工具
- [MTK NeuroPilot SDK](https://github.com/MediaTek-NeuronPilot)
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)

## 项目特色

✨ **完整实现**: Python转换 + C++推理 + 设备测试
✨ **多平台支持**: MTK NPU + RKNN
✨ **多模型支持**: EDSR + RCAN + Real-ESRGAN
✨ **详细文档**: 完整的使用说明和API文档
✨ **真实测试**: 所有模型在真实设备上验证

## 更新日志

### 2026-01-29
- ✨ 添加Real-ESRGAN完整实现
- ✨ 支持MTK NPU和RKNN双平台
- ✨ 完整的测试数据和输出结果
- 📝 更新项目文档

## License

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交GitHub Issue。
