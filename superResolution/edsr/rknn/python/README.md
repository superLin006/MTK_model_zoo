# EDSR Python 脚本说明

## 主要脚本

### 1. 模型导出和转换

#### `export_onnx.py`
从PyTorch模型导出ONNX格式
```bash
python export_onnx.py
```
- 输入: `../model/EDSR_x4.pt` (PyTorch模型)
- 输出: `../model/edsr_x4.onnx`

#### `convert.py`
将ONNX模型转换为RKNN格式（FP16）
```bash
python convert.py
```
- 输入: `../model/edsr_x4.onnx`
- 输出: `../model/edsr_x4_fp.rknn`

### 2. 量化相关

#### `prepare_quantization_dataset.py`
准备INT8量化所需的校准数据集
```bash
python prepare_quantization_dataset.py
```
- 输出: `../dataset/calibration/` 目录下的校准图像
- 建议: 添加自己的真实图像（10-50张，256x256 RGB）

#### `convert_quantized.py`
将ONNX转换为INT8量化RKNN模型
```bash
python convert_quantized.py
```
- 输入: `../model/edsr_x4.onnx` + 校准数据集
- 输出: `../model/edsr_x4_int8.rknn`
- 注意: 需要先运行 `prepare_quantization_dataset.py`

### 3. 测试和验证

#### `test_pytorch.py`
测试PyTorch原始模型
```bash
python test_pytorch.py
```

#### `test_onnx.py`
测试ONNX模型
```bash
python test_onnx.py
```

#### `accuracy_analysis.py`
运行RKNN精度分析（对比ONNX vs RKNN各层输出）
```bash
python accuracy_analysis.py <onnx_path> <image_path> [platform]
```

## 完整工作流程

### FP16模型转换
```bash
# 1. 导出ONNX
python export_onnx.py

# 2. 转换为RKNN FP16
python convert.py

# 3. 测试（可选）
python test_onnx.py
```

### INT8量化模型转换
```bash
# 1. 准备校准数据
python prepare_quantization_dataset.py
# （建议添加更多真实图像到 dataset/calibration/）

# 2. 执行量化转换
python convert_quantized.py

# 3. 部署到设备测试
cd ..
./push_and_run.sh
```

## 文件组织

```
edsr/
├── python/              # Python脚本
│   ├── export_onnx.py
│   ├── convert.py
│   ├── convert_quantized.py
│   └── ...
├── model/               # 模型文件
│   ├── EDSR_x4.pt      # PyTorch模型
│   ├── edsr_x4.onnx    # ONNX模型
│   ├── edsr_x4_fp.rknn # FP16 RKNN模型
│   └── edsr_x4_int8.rknn # INT8 RKNN模型
├── dataset/             # 数据集
│   └── calibration/    # 量化校准图像
└── cpp/                 # C++推理代码
```

## 注意事项

1. **FP16 vs INT8**
   - FP16: 精度高，模型大，速度较慢
   - INT8: 精度略低，模型小，速度快
   - 当前FP16模型有色彩失真问题，建议尝试INT8量化

2. **校准数据集**
   - 量化效果很大程度取决于校准数据的质量
   - 应包含多样化的场景：人脸、风景、文字、建筑等
   - 建议10-50张，覆盖实际应用场景

3. **测试环境**
   - Python测试: 使用simulator（模拟环境）
   - 最终测试: 必须在真实设备（RK3576）上进行
