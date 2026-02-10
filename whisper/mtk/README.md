# Whisper on MTK NPU

Whisper语音识别模型移植到MTK NPU平台的项目

## 项目信息

- **算法名称**: Whisper
- **算法类型**: 语音识别 (Automatic Speech Recognition)
- **模型版本**: base (71.83M parameters)
- **官方仓库**: https://github.com/openai/whisper
- **目标平台**: MT8371
- **音频输入**: 30s固定长度

## 目录结构

```
whisper/mtk/
├── python/
│   ├── models/          # 存放转换后的模型 (.pt/.tflite/.dla)
│   └── test/            # 测试脚本和结果
│       ├── test_pytorch.py      # PyTorch baseline测试
│       └── outputs/             # 测试输出结果
├── cpp/                 # C++实现（预留）
├── models/              # 原始PyTorch模型
│   └── base.pt         # Whisper base模型 (139MB)
└── test_data/           # 测试音频数据
    ├── jfk.flac        # 英文测试音频 (11s)
    ├── test_en.wav     # 英文测试音频 (5.86s)
    └── test_zh.wav     # 中文测试音频 (5.61s)
```

## 环境配置

### Conda环境: MTK-whisper
```bash
# 环境从MTK-clip-encoder克隆
conda create --name MTK-whisper --clone MTK-clip-encoder

# 激活环境
conda activate MTK-whisper

# 已安装的关键包
- Python 3.10.19
- openai-whisper 20250625
- torch 1.13.1
- numpy 1.24.3
```

## Baseline测试结果

### 测试配置
- **设备**: CPU (FP32)
- **模型**: base
- **温度**: 0.0 (确定性解码)

### 测试结果

#### 1. 中文音频 (test_zh.wav)
- **时长**: 5.61s
- **识别结果**: `對我做了介紹我想說的是大家如果對我的研究感興趣`
- **推理时间**: 1.22s
- **语言检测**: zh

#### 2. 英文音频 (test_en.wav)
- **时长**: 5.86s
- **识别结果**: `Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.`
- **推理时间**: 0.69s
- **语言检测**: en

#### 3. JFK演讲 (jfk.flac)
- **时长**: 11.00s
- **识别结果**: `And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.`
- **推理时间**: 0.71s
- **语言检测**: en

### 性能指标
| 音频文件 | 时长 | 模型加载 | 推理时间 | 总时间 |
|---------|------|---------|---------|--------|
| test_zh.wav | 5.61s | 0.62s | 1.22s | 1.84s |
| test_en.wav | 5.86s | 0.45s | 0.69s | 1.14s |
| jfk.flac | 11.00s | 0.36s | 0.71s | 1.07s |

## 快速开始

### 运行PyTorch Baseline测试
```bash
cd /home/xh/projects/MTK_models_zoo/whisper/mtk/python/test
conda activate MTK-whisper
python test_pytorch.py
```

### 测试单个音频文件
```python
from test_pytorch import test_whisper_baseline

result = test_whisper_baseline(
    audio_path="/path/to/audio.wav",
    model_name="base",
    language="en"  # 可选: en, zh, 或None自动检测
)
```

## 下一步计划

1. ✓ 项目结构搭建
2. ✓ Conda环境配置
3. ✓ 下载Whisper base模型
4. ✓ 准备测试数据
5. ✓ PyTorch baseline测试
6. ⬜ 模型分析与拆解
7. ⬜ ONNX导出
8. ⬜ TFLite转换
9. ⬜ MTK DLA转换
10. ⬜ NPU推理实现
11. ⬜ 性能对比与优化

## 参考资料

- 官方仓库: https://github.com/openai/whisper
- 参考实现: /home/xh/projects/rknn_model_zoo/examples/whisper
- 目录结构参考: /home/xh/projects/MTK_models_zoo/superResolution/edsr/mtk

## 备注

- Baseline结果保存在 `python/test/outputs/` 目录
- 所有测试使用CPU运行，确保在WSL环境下的稳定性
- 中文测试使用繁体中文音频
- 模型文件较大(139MB)，首次加载需要下载
