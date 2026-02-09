# Python端输出文件管理规范 v2.0

---

## 标准目录结构

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

## 各子目录内容

### `baseline/` — ground truth
```
baseline/
├── test_en.json         # 完整输出（tokens, text, metadata）
├── test_en.txt          # 纯文本输出
└── summary.json         # 所有测试用例汇总
```

### `torchscript/` / `tflite/` / `dla/` — 各阶段输出
```
{stage}/
├── test_en.json
└── diff_vs_baseline.txt # 精度对比报告
```

### `debug/` — C++对比用中间输出
```
debug/
├── preprocessed_input.npy    # 预处理后的输入
├── encoder_output.npy        # encoder输出
├── decoder_logits.npy        # decoder logits
└── *.npy                     # 任何需要C++对比的中间结果
```

**格式**：统一使用 `.npy`（numpy和C++都能读取）

---

## 文件命名规范

- 测试输出：`{test_case}.{json|txt}`（放在对应stage子目录下）
- 中间调试：`{component}.npy`（如 `encoder_output.npy`）
- 对比报告：`diff_vs_{reference}.txt`

---

## 保存规范

**不要创建独立的 test_config.py / test_utils.py**。每个项目的输出格式不同，直接在 test_pt.py 中保存即可。

### 保存方式

```python
# 在 test_pt.py 中直接使用：
import numpy as np, json
from pathlib import Path

output_dir = Path(__file__).parent / "outputs"
debug_dir = output_dir / "debug"
baseline_dir = output_dir / "baseline"
for d in [debug_dir, baseline_dir]:
    d.mkdir(parents=True, exist_ok=True)

# 保存中间输出（给C++对比用）
np.save(debug_dir / "preprocessed_mel.npy", mel_data)
np.save(debug_dir / "encoder_output.npy", encoder_out.numpy())

# 保存最终结果
with open(baseline_dir / "test_en.json", "w", encoding="utf-8") as f:
    json.dump({"text": decoded_text, "tokens": tokens}, f, ensure_ascii=False, indent=2)
```

---

## .gitignore 配置

```gitignore
# debug目录可重新生成
mtk/python/test/outputs/debug/*.npy
mtk/python/test/outputs/debug/*.bin

# 保留其他输出作为验证基准
!mtk/python/test/outputs/baseline/
!mtk/python/test/outputs/tflite/
!mtk/python/test/outputs/dla/
```

---

**版本**: v2.0
**改动**: 精简冗余说明，删除emoji装饰，保留核心目录结构和代码模板
