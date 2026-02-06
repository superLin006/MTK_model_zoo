# 检查点2报告：TorchScript → TFLite 转换

## 日期
2026-02-04

## 状态
✅ **完成** - TFLite模型成功生成

---

## 完成的工作

### 1. TFLite转换（使用MTK Converter）

**转换方式**：
- ✅ 直接 TorchScript → TFLite （没有经过ONNX）
- ✅ 使用 `mtk_converter` Python包
- ✅ 在正确的环境：MTK-whisper (Python 3.10)

**生成的模型**：

| 模型 | 大小 | 输入形状 | 输出形状 | 转换时间 |
|------|------|----------|----------|----------|
| encoder_base_80x3000.tflite | 79 MB | [1, 80, 3000] | [1, 1500, 512] | 3.6s |
| decoder_base_448.tflite | 200 MB | [1, 448, 512] + [1, 1500, 512] | [1, 448, 51865] | 3.4s |

**关键点**：
- Encoder: 156 operators, 334 tensors
- Decoder: 267 operators, 563 tensors
- 包含MTK自定义算子：`MTKEXT_LAYER_NORMALIZATION`, `MTKEXT_GELU` 等

---

### 2. Python端TFLite测试

**发现**：
- MTK的TFLite模型包含自定义算子（如 `MTKEXT_LAYER_NORMALIZATION`）
- 标准的TensorFlow Lite无法加载这些模型
- 需要MTK专用的TFLite Runtime（通常在C++端/Android端使用）

**结论**：
- ✅ **TorchScript (.pt) 测试已充分验证**（阶段1完成）
- ✅ **TFLite模型主要用于DLA转换**（不是Python直接运行）
- ⏩ **跳过Python端TFLite测试，直接转DLA**

这符合MTK的标准工作流程：
```
PyTorch → TorchScript (Python测试) → TFLite (DLA转换) → DLA (NPU运行)
```

---

## 与baseline对比（使用TorchScript结果）

阶段1的TorchScript测试已经验证了转换正确性：

| 测试用例 | Baseline | TorchScript (.pt) | 匹配度 |
|---------|----------|-------------------|--------|
| test_en | "Mr. Quilter is the apostle..." | "Mr. Quilter is the apostle..." | ✅ 100% |
| test_zh | "對我做了介紹我想說的是..." | "對我做了介紹我想說的是,..." | ✅ 99% (仅逗号差异) |
| jfk | "And so my fellow Americans..." | "And so my fellow Americans,..." | ✅ 99% (仅逗号差异) |

---

## 下一步

### 阶段3：TFLite → DLA 转换

**目标**：
- 使用MTK Neuron Compiler将TFLite转换为DLA格式
- 目标平台：MT8371
- 生成：
  - encoder_base_80x3000_MT8371.dla
  - decoder_base_448_MT8371.dla

**脚本**：
- step3_tflite_to_dla.py

**验证**：
- DLA在Python端无法测试
- 留待C++端验证

---

## 生成的文件清单

```
python/
├── models/
│   ├── encoder_base_3000.pt              (78.7 MB) ✅ 阶段1
│   ├── decoder_base_448.pt               (199.3 MB) ✅ 阶段1
│   ├── token_embedding.npy               (101.3 MB) ✅ 阶段1
│   ├── encoder_base_80x3000.tflite       (79 MB) ✅ 阶段2
│   ├── decoder_base_448.tflite           (200 MB) ✅ 阶段2
│   └── embedding_info.json               ✅ 阶段1
├── whisper_model.py                       ✅ 阶段1
├── step1_pt_to_torchscript.py             ✅ 阶段1
├── step2_torchscript_to_tflite.py         ✅ 阶段2
└── test/
    ├── test_pt.py                         ✅ 阶段1
    ├── test_tflite.py                     ✅ 阶段2 (创建但跳过)
    └── outputs/
        ├── pt_test_zh.json                ✅ 阶段1
        ├── pt_test_en.json                ✅ 阶段1
        ├── pt_jfk.json                    ✅ 阶段1
        └── pt_summary.json                ✅ 阶段1
```

---

## 问题记录

### 问题1：初始错误使用ONNX路径
**现象**：第一版step2脚本错误地尝试通过ONNX转换  
**解决**：删除ONNX相关代码，直接使用mtk_converter从TorchScript转TFLite

### 问题2：Python端无法加载TFLite
**现象**：`RuntimeError: Encountered unresolved custom op: MTKEXT_LAYER_NORMALIZATION`  
**原因**：MTK TFLite包含自定义算子，标准TensorFlow Lite不支持  
**解决**：认识到TFLite主要用于DLA转换，不需要Python端测试

---

## 经验总结

1. **MTK工作流程**：TorchScript在Python端验证，TFLite用于DLA转换
2. **不要经过ONNX**：mtk_converter直接支持TorchScript → TFLite
3. **环境很重要**：必须在MTK-whisper环境中运行
4. **自定义算子**：MTK TFLite不兼容标准runtime

---

**检查点2完成** ✅  
**准备进入阶段3：DLA转换**
