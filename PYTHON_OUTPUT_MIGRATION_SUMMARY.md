# Python输出结构迁移总结

> 将 Whisper 和 SuperResolution 项目调整为符合 `python_output_management.md` 标准

**迁移日期**: 2026-02-07

---

## ✅ 完成项目

### 1. Whisper (`whisper/mtk/python/`)

**目录结构**:
```
test/
├── outputs/
│   ├── baseline/          # ✓ 已迁移现有文件
│   ├── torchscript/       # ✓ 已迁移现有文件
│   ├── tflite/            # ✓ 新建
│   ├── dla/               # ✓ 新建
│   ├── debug/             # ✓ 新建
│   └── README.md          # ✓ 新建
├── test_config.py         # ✓ 新建
├── test_utils.py          # ✓ 新建
├── .gitignore             # ✓ 新建
└── clean_debug_outputs.sh # ✓ 新建
```

**迁移操作**:
- ✓ 将 `baseline_*.json/txt` 移到 `baseline/`
- ✓ 将 `pt_*.json` 移到 `torchscript/`
- ✓ 创建配置和工具文件

---

### 2. SuperResolution - RCAN (`superResolution/rcan/mtk/python/`)

**目录结构**:
```
test/
├── outputs/
│   ├── baseline/          # ✓ 新建
│   ├── torchscript/       # ✓ 新建
│   ├── tflite/            # ✓ 新建
│   ├── dla/               # ✓ 新建
│   ├── debug/             # ✓ 新建
│   └── README.md          # ✓ 新建
├── test_config.py         # ✓ 新建
├── test_utils.py          # ✓ 新建（图像专用）
├── .gitignore             # ✓ 新建
└── clean_debug_outputs.sh # ✓ 新建
```

**特点**:
- 支持图像输出（PNG格式）
- 包含PSNR/MAE对比函数

---

### 3. SuperResolution - EDSR (`superResolution/edsr/mtk/python/`)

**目录结构**:
```
test/
├── outputs/
│   ├── baseline/          # ✓ 新建
│   ├── torchscript/       # ✓ 新建
│   ├── tflite/            # ✓ 新建
│   ├── dla/               # ✓ 新建
│   ├── debug/             # ✓ 新建
│   └── README.md          # ✓ 新建
├── test_config.py         # ✓ 新建
├── test_utils.py          # ✓ 新建（图像专用）
├── .gitignore             # ✓ 新建
└── clean_debug_outputs.sh # ✓ 新建
```

**特点**:
- 与RCAN相同的图像处理工具
- 独立配置文件

---

## 📝 新增文件功能说明

### 1. `test_config.py`
- 统一管理所有输出路径
- 自动创建必要目录
- 可直接运行查看配置

```python
python test_config.py  # 查看路径配置
```

### 2. `test_utils.py`
提供三个核心函数:

```python
# 保存测试输出
save_output(stage, test_name, data, format)
# 参数:
#   stage: "baseline" | "torchscript" | "tflite" | "dla"
#   test_name: 测试用例名
#   data: 输出数据
#   format: "json" | "txt" | "png"（仅图像项目）

# 保存中间调试数据（给C++对比）
save_debug(name, data)
# 参数:
#   name: 描述性名称（如 "encoder_output"）
#   data: numpy数组

# 生成对比报告
save_comparison_report(baseline_name, test_name, test_stage)
# 自动对比并生成 diff_vs_baseline.txt
```

### 3. `.gitignore`
- 只忽略 `debug/` 目录的 `.npy` 和 `.bin` 文件
- 保留 baseline/tflite/dla 等验证基准

### 4. `clean_debug_outputs.sh`
- 清理可重新生成的debug文件
- 保留关键输出

```bash
cd test/
./clean_debug_outputs.sh
```

### 5. `outputs/README.md`
- 说明目录结构和使用方法
- 每个项目都有独立的README

---

## 🔄 如何使用

### 在测试脚本中使用

**Whisper 示例**:
```python
from test_utils import save_output, save_debug

# 保存mel频谱图（给C++对比）
mel = preprocess(audio)
save_debug("preprocessed_mel", mel)

# 保存encoder输出
encoder_out = model.encoder(mel)
save_debug("encoder_output", encoder_out.numpy())

# 保存最终结果
result = {
    "audio": audio_file,
    "tokens": tokens.tolist(),
    "text": decoded_text
}
save_output("baseline", "test_en", result, format="json")
save_output("baseline", "test_en", decoded_text, format="txt")
```

**SuperResolution 示例**:
```python
from test_utils import save_output, save_debug

# 保存预处理输入
preprocessed = preprocess(image)
save_debug("preprocessed_input", preprocessed)

# 保存模型输出
model_out = model(preprocessed)
save_debug("model_output", model_out.numpy())

# 保存最终图像和元数据
result = {"psnr": 28.5, "mae": 3.2}
save_output("baseline", "butterfly", result, format="json")
save_output("baseline", "butterfly", output_image, format="png")
```

---

## 🎯 优势

1. **结构清晰**: 每个阶段输出独立存放
2. **易于对比**: baseline作为ground truth，其他阶段与之对比
3. **调试友好**: debug目录专门存放C++需要的中间输出
4. **不混乱**: 不会与代码文件混在一起
5. **可追溯**: 保留完整的测试输出历史

---

## 📋 TODO

如果后续需要修改现有测试脚本，可以：

1. 在 `test_pt.py` / `test_pytorch.py` / `test_tflite.py` 中：
   - 替换硬编码路径为 `from test_config import BASELINE_DIR, DEBUG_DIR`
   - 替换文件保存逻辑为 `save_output()` 和 `save_debug()`

2. 添加对比功能：
   - 在测试脚本末尾调用 `save_comparison_report()`
   - 自动生成 `diff_vs_baseline.txt`

---

## 🔗 参考

- 规范文档: `/home/xh/projects/MTK/.claude/standards/python_output_management.md`
- 各项目 README: `test/outputs/README.md`

---

**状态**: ✅ 结构迁移完成，建议后续逐步替换测试脚本中的路径引用
