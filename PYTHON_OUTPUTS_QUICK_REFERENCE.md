# Python输出管理 - 快速参考

## 📦 已完成项目

- ✅ **Whisper** (`whisper/mtk/python/`)
- ✅ **RCAN** (`superResolution/rcan/mtk/python/`)
- ✅ **EDSR** (`superResolution/edsr/mtk/python/`)

---

## 🎯 核心API

### 1. 保存测试输出

```python
from test_utils import save_output

# JSON格式
save_output("baseline", "test_en", {"text": "Hello"}, format="json")

# 纯文本格式
save_output("baseline", "test_en", "Hello world", format="txt")

# 图像格式（仅superResolution）
save_output("baseline", "butterfly", output_image, format="png")
```

**参数**:
- `stage`: `"baseline"` | `"torchscript"` | `"tflite"` | `"dla"`
- `test_name`: 测试用例名（如 `"test_en"`, `"butterfly"`)
- `data`: 输出数据
- `format`: `"json"` | `"txt"` | `"png"`

---

### 2. 保存中间调试数据

```python
from test_utils import save_debug

# 保存numpy数组（给C++对比用）
save_debug("encoder_output", encoder_out.numpy())
save_debug("preprocessed_mel", mel_spectrogram)
```

**输出**:
```
[DEBUG] Saved encoder_output: shape=(1, 1500, 512), dtype=float32
        → /path/to/test/outputs/debug/encoder_output.npy
```

---

### 3. 生成对比报告

```python
from test_utils import save_comparison_report

# 对比 tflite 输出与 baseline
save_comparison_report("test_en", "test_en", "tflite")
```

**生成**:
- `outputs/tflite/diff_vs_baseline.txt`

---

## 📁 目录结构

```
{project}/mtk/python/test/
├── outputs/
│   ├── baseline/          # PyTorch ground truth
│   ├── torchscript/       # TorchScript输出
│   ├── tflite/            # TFLite输出
│   ├── dla/               # DLA输出
│   ├── debug/             # 中间输出（给C++对比）
│   └── README.md          # 使用说明
├── test_config.py         # 路径配置
├── test_utils.py          # 工具函数
├── .gitignore             # 忽略debug文件
└── clean_debug_outputs.sh # 清理脚本
```

---

## 🔧 常用命令

### 查看配置
```bash
cd {project}/mtk/python/test/
python test_config.py
```

### 清理debug输出
```bash
cd {project}/mtk/python/test/
./clean_debug_outputs.sh
```

### 查看目录结构
```bash
tree -L 3 test/outputs/
```

---

## 💡 使用示例

### Whisper完整流程

```python
from test_utils import save_output, save_debug

def test_whisper_baseline():
    # 1. 预处理
    mel = preprocess(audio)
    save_debug("preprocessed_mel", mel)  # 给C++对比

    # 2. Encoder
    encoder_out = encoder(mel)
    save_debug("encoder_output", encoder_out.numpy())

    # 3. Decoder
    decoder_out = decoder(encoder_out, tokens)
    save_debug("decoder_logits", decoder_out.numpy())

    # 4. 保存最终结果
    result = {
        "audio": "test_en.wav",
        "tokens": tokens.tolist(),
        "text": decoded_text,
        "duration": duration
    }
    save_output("baseline", "test_en", result, format="json")
    save_output("baseline", "test_en", decoded_text, format="txt")
```

### SuperResolution完整流程

```python
from test_utils import save_output, save_debug

def test_rcan_baseline():
    # 1. 预处理
    preprocessed = preprocess(lr_image)
    save_debug("preprocessed_input", preprocessed)

    # 2. 模型推理
    sr_output = model(preprocessed)
    save_debug("model_output", sr_output.numpy())

    # 3. 后处理
    sr_image = postprocess(sr_output)

    # 4. 保存结果
    result = {
        "image": "butterfly.png",
        "input_shape": lr_image.shape,
        "output_shape": sr_image.shape
    }
    save_output("baseline", "butterfly", result, format="json")
    save_output("baseline", "butterfly", sr_image, format="png")
```

---

## ⚠️ 注意事项

1. **debug目录被gitignore**: debug中的.npy文件不会被提交到git
2. **baseline是ground truth**: 其他阶段都应该与baseline对比
3. **命名规范**: 使用描述性名称，如 `"preprocessed_mel"` 而不是 `"output1"`
4. **数据格式**: debug数据统一使用numpy的.npy格式

---

## 📚 参考文档

- 完整规范: `/home/xh/projects/MTK/.claude/standards/python_output_management.md`
- 迁移总结: `/home/xh/projects/MTK/PYTHON_OUTPUT_MIGRATION_SUMMARY.md`
- 各项目README: `test/outputs/README.md`
