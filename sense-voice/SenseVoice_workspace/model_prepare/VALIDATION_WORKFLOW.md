# SenseVoice 模型验证完整流程

## 📝 配置文件说明

### `config.py` 中的 `PYTORCH` 参数

```python
PYTORCH = 0  # 0=导出模式, 1=原生模式
```

**作用说明：**

| 模式 | PYTORCH 值 | 用途 | 模型行为 |
|------|-----------|------|----------|
| 导出模式 | 0 | SAVE_PT | 移除不兼容输出，导出 TorchScript |
| 导出模式 | 0 | CHECK_TFLITE | 验证 TFLite 模型 |
| 原生模式 | 1 | PYTORCH | 保存完整 PyTorch 输出（基准） |

## 🔄 完整验证流程

### 步骤 1: 导出 TorchScript 模型

```bash
cd /home/xh/projects/MTK/sense-voice/SenseVoice_workspace/model_prepare

# 确保 PYTORCH=0
cat config.py  # 应该显示: PYTORCH = 0

# 导出模型
python3 main.py --mode="SAVE_PT" \
    --model_path="../models/sensevoice-small" \
    --audio_path="../audios/test_en.wav"
```

**输出：**
- `model/sensevoice_complete.pt`

---

### 步骤 2: 转换为 TFLite

```bash
python3 pt2tflite.py \
    -i model/sensevoice_complete.pt \
    -o model/sensevoice_complete.tflite \
    --float 1
```

**输出：**
- `model/sensevoice_complete.tflite`

---

### 步骤 3: 运行 PyTorch 基准测试

```bash
# 修改 config.py: PYTORCH = 1
sed -i 's/PYTORCH = 0/PYTORCH = 1/' config.py

# 运行 PyTorch 推理
python3 main.py --mode="PYTORCH" \
    --model_path="../models/sensevoice-small" \
    --audio_path="../audios/test_en.wav"
```

**输出：**
- `output/pytorch_logits.npy` - PyTorch 基准输出
- `output/pytorch_features.npy` - 特征
- `output/pytorch_prompt.npy` - Prompt

---

### 步骤 4: 验证 TFLite 模型

```bash
# 恢复 config.py: PYTORCH = 0
sed -i 's/PYTORCH = 1/PYTORCH = 0/' config.py

# 运行 TFLite 推理
python3 main.py --mode="CHECK_TFLITE" \
    --model_path="../models/sensevoice-small" \
    --audio_path="../audios/test_en.wav" \
    --tflite_file_path="model/sensevoice_complete.tflite"
```

**输出：**
- `output/tflite_logits.npy` - TFLite 输出
- `output/tflite_features.npy` - 特征

---

### 步骤 5: 对比输出

```bash
# 比较输出差异
python3 compare_outputs.py
```

**验证标准：**
- ✅ Token 准确率 ≥ 99.9%
- ✅ 最大绝对误差 < 1.0
- ✅ 平均绝对误差 < 0.1

---

### 步骤 6: 解码文本（可选）

```bash
# 解码 TFLite 输出
python3 decode_text.py \
    --logits="output/tflite_logits.npy" \
    --tokens="../models/sensevoice-small/tokens.txt"
```

**输出：**
- `output/transcription.txt`

---

## 🚀 一键验证脚本

```bash
# 完整验证流程（自动化）
bash 3_check_tflite.sh
```

---

## 📊 预期输出示例

```
========================================
  SenseVoice 模型输出对比
========================================

加载输出数据...
PyTorch shape:  (1, 102, 25055)
TFLite shape:   (1, 170, 25055)

差异统计:
  最大绝对误差: 0.023415
  平均绝对误差: 0.000587
  相对误差:     0.01%

Token 预测对比:
  匹配数:   15300 / 15300
  准确率:   100.00%

✅ 验证通过: Token 预测完全一致

最终结果: PASS
========================================
```

---

## ⚠️ 常见问题

### Q1: 为什么需要 PYTORCH=0 和 PYTORCH=1 两种模式？

**A:** TFLite 转换时需要简化模型输出（移除不兼容的层），而 PyTorch 基准测试需要完整输出。

### Q2: 忘记切换 PYTORCH 模式会怎样？

**A:** 代码会自动检查并报错：
```
AssertionError: Except for Pytorch inference mode, please modify PYTORCH @config.py to 0 first.
```

### Q3: 验证失败怎么办？

**A:** 检查以下几点：
1. 是否使用了相同的音频文件
2. PYTORCH 模式是否正确
3. LFR 参数是否一致（LFR_M=7, LFR_N=6）
4. CMVN 归一化是否一致

---

## 📁 输出文件结构

```
output/
├── pytorch_logits.npy      # PyTorch 基准输出
├── pytorch_features.npy     # PyTorch 特征
├── pytorch_prompt.npy       # PyTorch Prompt
├── tflite_logits.npy        # TFLite 输出
├── tflite_features.npy      # TFLite 特征
└── transcription.txt        # 解码文本
```

---

## ✅ 验证检查清单

- [ ] PyTorch 推理成功
- [ ] TorchScript 导出成功
- [ ] TFLite 转换成功
- [ ] PyTorch 基准输出已保存
- [ ] TFLite 输出已保存
- [ ] 输出对比通过（准确率 ≥ 99.9%）
- [ ] DLA 编译成功
- [ ] C++ 推理验证通过

---

## 🔗 相关脚本

- `0_run.sh` - 初始 PyTorch 测试
- `1_save_pt.sh` - 导出 TorchScript
- `2_pt2tflite.sh` - 转换为 TFLite
- `3_check_tflite.sh` - 验证 TFLite
- `compare_outputs.py` - 对比工具
- `decode_text.py` - 解码工具
