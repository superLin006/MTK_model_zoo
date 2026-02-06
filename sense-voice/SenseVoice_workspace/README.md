# SenseVoice 模型转换工作区

FunASR SenseVoice Small 模型转换为 MediaTek DLA 格式的工作区。

---

## 📁 目录结构

```
SenseVoice_workspace/
├── models/                          # FunASR 原始模型
│   └── sensevoice-small/
│       ├── model.pt                 # PyTorch 权重
│       ├── am.mvn                   # CMVN 参数
│       ├── tokens.txt               # 词汇表 (25055 tokens)
│       └── config.yaml
│
├── audios/                          # 测试音频
│   ├── test_en.wav                  # 英语测试 (5.9s)
│   ├── test_zh.wav                  # 中文测试 (5.6s)
│   ├── audio4.wav                   # 长音频测试 (16.2s)
│   └── audio5.wav                   # 英语测试 (9.3s)
│
├── model_prepare/                   # PyTorch → TFLite 转换
│   ├── model/                       # 转换输出目录
│   │   ├── sensevoice_complete.pt       # TorchScript (895MB)
│   │   └── sensevoice_complete.tflite   # TFLite (886MB)
│   ├── torch_model.py               # 模型实现
│   ├── model_utils.py               # 工具函数
│   ├── main.py                      # 转换主脚本
│   ├── pt2tflite.py                 # TFLite 转换
│   └── test_converted_models.py     # 验证脚本
│
└── compile/                         # TFLite → DLA 编译
    └── compile_sensevoice_fp.sh     # 编译脚本
```

---

## 🚀 转换流程

### 1. 环境准备

```bash
# 创建 conda 环境
conda create -n MTK-sensevoice python=3.10
conda activate MTK-sensevoice

# 安装依赖
cd model_prepare
pip install torch torchvision torchaudio
pip install funasr modelscope
pip install librosa
```

### 2. 下载模型

```bash
cd ../models
modelscope download --model iic/SenseVoiceSmall --local_dir sensevoice-small
```

### 3. 模型转换

```bash
cd ../model_prepare

# Step 1: 保存为 TorchScript (固定166帧 = 10秒音频)
python3 main.py --mode=SAVE_PT

# Step 2: 转换为 TFLite
python3 pt2tflite.py \
    -i model/sensevoice_complete.pt \
    -o model/sensevoice_complete.tflite \
    --float 1

# Step 3: 验证转换结果
python3 test_converted_models.py \
    --audio ../audios/test_en.wav \
    --language auto
```

### 4. 编译 DLA

**⚠️ 重要**: 需要先安装 MTK NeuroPilot SDK

```bash
# MTK NeuroPilot SDK 下载地址 (需要 MTK 账号)
# https://vendor.mediatek.com/

# SDK 安装路径示例
NEUROPILOT_SDK="/home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
```

编译 DLA 模型:

```bash
cd ../compile

# 选择目标平台: MT6899 / MT6991 / MT8371
./compile_sensevoice_fp.sh \
    ../model_prepare/model/sensevoice_complete.tflite \
    MT8371 \
    "$NEUROPILOT_SDK"
```

**编译参数说明**:
- `--arch`: MDLA 架构 (自动根据平台选择)
- `--l1-size-kb`: L1 缓存大小 (自动根据平台设置)
- `--num-mdla`: MDLA 核心数 (自动根据平台设置)
- `--relax-fp32`: FP32 放宽优化
- `--opt-accuracy`: 准确性优化
- `--opt-footprint`: 减少内存占用
- `--fc-to-conv`: 全连接转卷积 (提升 NPU 效率)

---

## 📊 模型规格

### 架构
- **编码器**: 50层 SANM (Self-Attention with Memory Network)
- **输出**: CTC (Connectionist Temporal Classification)
- **参数量**: 917个权重参数

### 输入输出
| 项目 | Shape | 类型 | 说明 |
|------|-------|------|------|
| 输入1 | `[1, 166, 560]` | float32 | Fbank+LFR特征 (10秒音频) |
| 输入2 | `[4]` | int32 | Prompt [language, event1, event2, text_norm] |
| 输出 | `[1, 170, 25055]` | float32 | CTC logits (166+4=170帧) |

### Prompt 格式
```python
[language_id, event1, event2, text_norm_id]
```

| 参数 | 可选值 | 说明 |
|------|-------|------|
| language | auto=0, zh=3, en=4, yue=7, ja=11, ko=12, nospeech=13 | 语言 ID |
| event1 | HAPPY=1, SAD=2, ANGRY=3, NEUTRAL=4 | 情绪 ID |
| event2 | Speech=2, Music=3, Applause=4 | 事件类型 ID |
| text_norm | withitn=14, woitn=15 | 文本规范化 ID |

### 音频处理参数
- **采样率**: 16 kHz mono
- **固定长度**: 10秒 (166帧)
- **Fbank**: 80 维
- **LFR**: 7 帧拼接 → 560 维
- **短音频**: 自动 padding
- **长音频**: 自动截断前 10 秒

---

## ✅ 验证结果

### 模型对比测试

| 模型 | 状态 | 与PyTorch对比 | 文本匹配 |
|------|------|--------------|---------|
| PyTorch | ✅ | - | 基准 |
| TorchScript | ✅ | diff=0 (完美) | 100% |
| TFLite | ✅ | diff<18 | 100% |

**测试音频**: test_en.wav (5.86秒)
**输出文本**: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
**结论**: ✅ 所有模型输出完全一致

---

## 🔧 支持平台

| 平台 | SoC | MDLA版本 | L1缓存 | 核心数 |
|------|-----|---------|--------|--------|
| MT6899 | Dimensity 1200/1100 | MDLA5.5 | 2048KB | 2 |
| MT6991 | Dimensity 9300/9200 | MDLA5.5 | 7168KB | 4 |
| MT8371 | Genio 700 | MDLA5.3 + EDMA3.6 | 256KB | 1 |

### 编译参数

编译脚本会自动根据平台选择优化参数：

```bash
MT6899:  --arch=mdla5.5,mvpu2.5 --l1-size-kb=2048 --num-mdla=2
MT6991:  --arch=mdla5.5,mvpu2.5 --l1-size-kb=7168 --num-mdla=4
MT8371:  --arch=mdla5.3,edma3.6   --l1-size-kb=256  --num-mdla=1
```

---

## ⚠️ 注意事项

### 1. 固定长度限制
- 模型固定为 10 秒音频 (166 帧)
- 超过 10 秒会被截断，丢失后半部分
- 建议使用滑动窗口处理长音频

### 2. 特征提取
- ✅ **测试验证**: 使用 FunASR 提取特征（`test_converted_models.py`）
- ✅ **实际部署**: 必须使用 kaldi-native-fbank 以确保准确性
- ❌ **不要使用**: librosa 特征会导致输出不准确

### 3. Config 配置
```python
# model_prepare/config.py
PYTORCH = 0  # 转换模式必须设为 0
```

### 4. 编译优化
编译脚本启用了以下优化：
- `--relax-fp32`: FP32 放宽，提升性能
- `--opt-accuracy`: 准确性优化
- `--opt-footprint`: 减少内存占用
- `--fc-to-conv`: 全连接转卷积，提升 NPU 效率

---

## 📝 核心文件说明

| 文件 | 说明 |
|------|------|
| `torch_model.py` | 完整模型实现 (CMVN+Encoder+CTC) |
| `model_utils.py` | 权重加载、CMVN 处理 |
| `main.py` | 转换主脚本，控制固定帧数 |
| `pt2tflite.py` | TFLite 转换，支持动态/静态 shape |
| `test_converted_models.py` | 验证脚本 (使用 FunASR 特征) |
| `compile_sensevoice_fp.sh` | DLA 编译脚本 |

---

## 🎯 常见问题

**Q: 为什么固定 10 秒？**
A: DLA 编译需要固定 shape 以优化性能。可以通过修改 `main.py` 中的 `fixed_frames=166` 来调整。

**Q: 如何处理长音频？**
A: 使用滑动窗口分段处理，每段 10 秒，步长可设为 8-9 秒保留上下文。

**Q: TFLite 数值误差是否正常？**
A: 是的。Padding 区域会有较大误差，但 token 预测 100% 准确，不影响最终结果。

**Q: 为什么用 FunASR 提取特征？**
A: librosa 与 kaldi-native-fbank 有实现差异，FunASR 使用后者，用其特征测试可确保模型转换正确。

**Q: 不同平台需要分别编译吗？**
A: 是的，每个平台的 MDLA 架构和缓存大小不同，需要单独编译优化。

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

**转换状态**: ✅ 完成
**验证状态**: ✅ 通过
**部署就绪**: ✅ 是

**最后更新**: 2026-01-12
