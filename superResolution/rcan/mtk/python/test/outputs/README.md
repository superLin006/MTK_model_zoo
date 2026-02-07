# RCAN 测试输出目录

> 符合 `/home/xh/projects/MTK/.claude/standards/python_output_management.md` 规范

## 📁 目录结构

```
outputs/
├── baseline/          # 原始PyTorch输出（ground truth）
│   ├── butterfly.json # 测试用例1
│   ├── butterfly.png  # 超分辨率输出图像
│   ├── baby.json      # 测试用例2
│   └── baby.png
│
├── torchscript/       # TorchScript输出
│   ├── butterfly.json
│   ├── butterfly.png
│   └── diff_vs_baseline.txt  # 对比报告
│
├── tflite/            # TFLite输出
│   ├── butterfly.json
│   ├── butterfly.png
│   └── diff_vs_baseline.txt
│
├── dla/               # DLA输出
│   ├── butterfly.json
│   ├── butterfly.png
│   ├── diff_vs_baseline.txt
│   └── performance.json      # 性能数据
│
└── debug/             # 中间输出（给C++对比用）
    ├── preprocessed_input.npy   # 预处理输入
    ├── model_output.npy         # 模型原始输出
    └── *.npy                    # 其他中间结果
```

## 📝 文件命名规范

- **测试输出**: `{test_case}.json` 和 `{test_case}.png`
- **中间调试文件**: `{component}.npy`
- **对比报告**: `diff_vs_baseline.txt`

## 🔧 使用工具函数

```python
from test_utils import save_output, save_debug

# 保存测试输出
save_output("baseline", "butterfly", result, format="json")
save_output("baseline", "butterfly", output_image, format="png")

# 保存中间调试数据
save_debug("preprocessed_input", preprocessed)

# 生成对比报告
save_comparison_report("butterfly", "butterfly", "tflite")
```

## 🧹 清理

运行清理脚本删除可重新生成的debug文件：

```bash
./clean_debug_outputs.sh
```
