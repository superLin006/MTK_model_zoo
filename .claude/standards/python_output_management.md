# Python端输出文件管理规范

> 统一管理测试输出和中间文件，保持项目整洁、易于调试

---

## 📁 标准目录结构

```
{project}/mtk/python/
├── models/                    # 模型文件
│   ├── encoder_*.pt
│   ├── encoder_*.tflite
│   ├── encoder_*.dla
│   └── *.npy                  # 权重文件（如embedding）
│
├── test/                      # 测试脚本
│   ├── test_pt.py
│   ├── test_tflite.py
│   ├── test_dla.py
│   └── outputs/               # ← 所有输出集中在这里
│       ├── baseline/          # 原始PyTorch输出（ground truth）
│       ├── torchscript/       # TorchScript输出
│       ├── tflite/            # TFLite输出
│       ├── dla/               # DLA输出
│       └── debug/             # 中间输出（给C++对比用）
│
├── step1_*.py                 # 转换脚本
├── step2_*.py
└── step3_*.py
```

---

## 📂 各子目录说明

### 1. `test/outputs/baseline/`
**用途**：存放原始PyTorch模型的输出，作为所有转换的对比基准

**包含**：
```
baseline/
├── test_en.json         # 完整输出（tokens, text, metadata）
├── test_en.txt          # 纯文本输出（方便查看）
├── test_zh.json
├── test_zh.txt
└── summary.json         # 所有测试用例的汇总
```

### 2. `test/outputs/torchscript/`
**用途**：TorchScript模型的测试输出

**包含**：
```
torchscript/
├── test_en.json
├── test_zh.json
└── diff_vs_baseline.txt # 精度对比报告
```

### 3. `test/outputs/tflite/`
**用途**：TFLite模型的测试输出

**包含**：
```
tflite/
├── test_en.json
├── test_zh.json
└── diff_vs_baseline.txt # 精度对比报告
```

### 4. `test/outputs/dla/`
**用途**：DLA模型的测试输出

**包含**：
```
dla/
├── test_en.json
├── test_zh.json
├── diff_vs_baseline.txt # 精度对比报告
└── performance.json     # 性能数据（推理时间）
```

### 5. `test/outputs/debug/` ⭐ **重要**
**用途**：保存中间输出，供C++实现时逐层对比

**包含**：
```
debug/
├── preprocessed_input.npy    # 预处理后的输入（如mel频谱图）
├── encoder_output.npy        # encoder输出
├── decoder_logits.npy        # decoder logits
├── embedding_output.npy      # embedding查询结果
└── *.npy                     # 任何需要C++对比的中间结果
```

**格式**：统一使用 `.npy` 格式（numpy和C++都能读取）

---

## 📝 文件命名规范

### 测试输出文件
```
{stage}_{test_case}.{ext}
```

**示例**：
- `baseline_test_en.json` - PyTorch baseline（英文）
- `tflite_test_zh.json` - TFLite测试（中文）
- `dla_jfk.json` - DLA测试（JFK音频）

### 中间调试文件
```
{component}.npy
```

**示例**：
- `encoder_output.npy` - encoder输出
- `preprocessed_mel.npy` - 预处理mel频谱图
- `decoder_logits.npy` - decoder logits

### 对比报告
```
diff_vs_{reference}.txt
```

**示例**：
- `diff_vs_baseline.txt` - 与baseline对比
- `diff_vs_pt.txt` - 与PyTorch对比

---

## 💻 代码实现模板

### 1. 路径配置（test/test_config.py）

```python
from pathlib import Path

# 目录路径
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR = TEST_DIR / "outputs"

# 各阶段输出目录
BASELINE_DIR = OUTPUT_DIR / "baseline"
TORCHSCRIPT_DIR = OUTPUT_DIR / "torchscript"
TFLITE_DIR = OUTPUT_DIR / "tflite"
DLA_DIR = OUTPUT_DIR / "dla"
DEBUG_DIR = OUTPUT_DIR / "debug"

# 确保目录存在
for d in [BASELINE_DIR, TORCHSCRIPT_DIR, TFLITE_DIR, DLA_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

### 2. 工具函数（test/test_utils.py）

```python
import json
import numpy as np
from test_config import OUTPUT_DIR, DEBUG_DIR

def save_output(stage, test_name, data, format="json"):
    """
    保存测试输出

    Args:
        stage: "baseline" | "torchscript" | "tflite" | "dla"
        test_name: "test_en" | "test_zh" | "jfk"
        data: 输出数据（dict或str）
        format: "json" | "txt"
    """
    stage_dir = OUTPUT_DIR / stage

    if format == "json":
        file = stage_dir / f"{test_name}.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:  # txt
        file = stage_dir / f"{test_name}.txt"
        with open(file, "w", encoding="utf-8") as f:
            f.write(data)

    print(f"✓ Saved: {file}")
    return file

def save_debug(name, data):
    """
    保存中间输出（给C++对比用）

    Args:
        name: 描述性名称（如 "encoder_output"）
        data: numpy数组
    """
    file = DEBUG_DIR / f"{name}.npy"
    np.save(file, data)
    print(f"[DEBUG] Saved {name}: shape={data.shape}, dtype={data.dtype}")
    print(f"        → {file}")
    return file
```

### 3. 使用示例

```python
# test/test_pt.py
from test_utils import save_output, save_debug

def test_pytorch(audio_file):
    # 预处理
    mel = preprocess(audio)
    save_debug("preprocessed_mel", mel)  # ← C++可以对比

    # Encoder
    encoder_out = model.encoder(mel)
    save_debug("encoder_output", encoder_out.numpy())  # ← C++可以对比

    # Decoder
    decoder_out = model.decoder(encoder_out, tokens)
    save_debug("decoder_logits", decoder_out.numpy())  # ← C++可以对比

    # 保存最终结果
    result = {
        "audio": audio_file,
        "tokens": tokens.tolist(),
        "text": decoded_text
    }
    save_output("baseline", "test_en", result, format="json")
    save_output("baseline", "test_en", decoded_text, format="txt")
```

---

## 🔧 .gitignore 配置

```gitignore
# 只ignore debug目录（可重新生成）
mtk/python/test/outputs/debug/*.npy
mtk/python/test/outputs/debug/*.bin

# 保留其他输出（作为验证基准）
!mtk/python/test/outputs/baseline/
!mtk/python/test/outputs/tflite/
!mtk/python/test/outputs/dla/
```

---

## 🧹 清理脚本

```bash
#!/bin/bash
# clean_debug_outputs.sh

echo "清理debug目录..."
rm -rf mtk/python/test/outputs/debug/*.npy
rm -rf mtk/python/test/outputs/debug/*.bin

echo "✓ 清理完成"
echo "保留了baseline/tflite/dla等关键输出"
```

---

## ✅ 优势

1. **结构清晰**：每个阶段独立存放，一目了然
2. **易于对比**：baseline作为ground truth，其他阶段与之对比
3. **调试友好**：debug目录专门存放C++需要的中间输出
4. **不混乱**：不会与代码文件混在一起
5. **可追溯**：保留完整的测试输出历史

---

## 🔄 迁移现有项目

如果现有项目输出文件混在一起，执行：

```bash
cd {project}/mtk/python/test/outputs

# 创建子目录
mkdir -p baseline torchscript tflite dla debug

# 移动文件（示例）
mv baseline_*.json baseline_*.txt baseline/
mv pt_*.json torchscript/
mv tflite_*.json tflite/
mv dla_*.json dla/

echo "✓ 迁移完成"
```

---

**版本**：v1.0
**创建日期**：2025-02-05
**适用于**：所有MTK NPU算法移植项目
