# Whisper 测试输出目录

> 符合 `/home/xh/projects/MTK/.claude/standards/python_output_management.md` 规范

## 📁 目录结构

```
outputs/
├── baseline/          # 原始PyTorch输出（ground truth）
│   ├── test_en.json   # 英文测试
│   ├── test_zh.json   # 中文测试
│   ├── jfk.json       # JFK音频测试
│   └── summary.json   # 测试汇总
│
├── torchscript/       # TorchScript输出
│   ├── test_en.json
│   ├── test_zh.json
│   └── diff_vs_baseline.txt  # 对比报告
│
├── tflite/            # TFLite输出
│   ├── test_en.json
│   ├── test_zh.json
│   └── diff_vs_baseline.txt
│
├── dla/               # DLA输出
│   ├── test_en.json
│   ├── test_zh.json
│   ├── diff_vs_baseline.txt
│   └── performance.json      # 性能数据
│
└── debug/             # 中间输出（给C++对比用）
    ├── preprocessed_mel.npy     # mel频谱图
    ├── encoder_output.npy       # encoder输出
    ├── decoder_logits.npy       # decoder logits
    └── *.npy                    # 其他中间结果
```

## 📝 文件命名规范

- **测试输出**: `{test_case}.json` 或 `{test_case}.txt`
- **中间调试文件**: `{component}.npy`
- **对比报告**: `diff_vs_baseline.txt`

## 🔧 使用工具函数

```python
from test_utils import save_output, save_debug

# 保存测试输出
save_output("baseline", "test_en", result, format="json")

# 保存中间调试数据
save_debug("preprocessed_mel", mel_spectrogram)

# 生成对比报告
save_comparison_report("test_en", "test_en", "tflite")
```

## 🧹 清理

运行清理脚本删除可重新生成的debug文件：

```bash
./clean_debug_outputs.sh
```
