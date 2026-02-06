# RCAN Python转换脚本

本目录包含RCAN模型从PyTorch到RKNN的完整转换脚本。

## 脚本说明

### test_pytorch.py

测试PyTorch RCAN模型推理。

**使用方法**:
```bash
python test_pytorch.py <model_path> [image_path]

# 示例:
python test_pytorch.py ../model/rcan_bix4_pytorch.pt ../model/input_256x256.png
```

**输出**:
- `../model/test_pytorch_output.png` - 超分辨率结果图像

**功能**:
- 加载PyTorch RCAN模型
- 从文件名自动推断scale倍数 (支持x2, x3, x4, x8)
- 执行推理并保存结果
- 验证输出尺寸是否正确

### export_onnx.py

将PyTorch模型导出为ONNX格式。

**使用方法**:
```bash
python export_onnx.py <pt_model_path> [output_onnx_path] [height] [width]

# 示例 (256x256输入):
python export_onnx.py ../model/rcan_bix4_pytorch.pt ../model/rcan_x4.onnx 256 256
```

**参数**:
- `<pt_model_path>`: PyTorch模型文件路径
- `[output_onnx_path]`: 输出ONNX文件路径 (可选)
- `[height]`: 输入图像高度 (默认256)
- `[width]`: 输入图像宽度 (默认256)

**重要说明**:
- RKNN需要固定输入尺寸，不能使用动态形状
- ONNX opset版本: 11
- 自动执行ONNX简化

**使用的算子**: Conv, Add, Mul, Relu, Sigmoid, GlobalAveragePool, DepthToSpace

### test_onnx.py

验证ONNX模型并对比PyTorch输出。

**使用方法**:
```bash
python test_onnx.py <onnx_model_path> [image_path] [pytorch_output_path]

# 示例:
python test_onnx.py ../model/rcan_x4.onnx ../model/input_256x256.png ../model/test_pytorch_output.png
```

**输出**:
- `../model/test_onnx_output.png` - ONNX推理结果
- 控制台输出对比指标 (MAE, PSNR)

**成功标准**:
- MAE < 1.0: 优秀匹配
- MAE < 5.0: 可接受匹配
- PSNR > 80 dB: 良好质量

### convert.py

将ONNX模型转换为RKNN格式。

**使用方法**:
```bash
# FP16模式 (不推荐 - 有色彩失真):
python convert.py <onnx_model_path> <platform> fp [output_rknn_path]

# INT8模式 (推荐):
python convert.py <onnx_model_path> <platform> i8 [output_rknn_path]

# 示例:
python convert.py ../model/rcan_x4.onnx rk3576 i8 ../model/rcan_x4_i8.rknn
```

**参数**:
- `<onnx_model_path>`: ONNX模型文件路径
- `<platform>`: 目标平台 (rk3576, rk3588, rk3568等)
- `dtype`: 模型精度
  - `fp`: FP16精度 (不推荐)
  - `i8`: INT8量化 (推荐)
- `[output_rknn_path]`: 输出RKNN文件路径 (可选)

**量化数据集**:
INT8量化需要校准图像。将图像放在 `../dataset/calibration/` 目录:
- 推荐数量: 8-10张图像
- 支持格式: .png, .jpg, .jpeg, .bmp
- 图像类型: 多样化的图像 (渐变、纹理、自然场景等)

## 完整转换流程

```bash
# 1. 测试PyTorch模型
python test_pytorch.py ../model/rcan_bix4_pytorch.pt ../model/input_256x256.png

# 2. 导出ONNX (256x256输入)
python export_onnx.py ../model/rcan_bix4_pytorch.pt ../model/rcan_x4.onnx 256 256

# 3. 验证ONNX模型
python test_onnx.py ../model/rcan_x4.onnx ../model/input_256x256.png

# 4. 转换为INT8 RKNN (推荐)
python convert.py ../model/rcan_x4.onnx rk3576 i8 ../model/rcan_x4_i8.rknn
```

## 模型信息

### RCAN架构
- **Residual Groups**: 10
- **RCABs per group**: 20
- **Feature channels**: 64
- **Reduction ratio**: 16
- **Upsampling**: PixelShuffle → DepthToSpace

### 算子支持
所有使用的算子都在RKNN Toolkit2中受支持:
- `Conv`: 卷积层
- `Add`: 残差连接
- `Mul`: 注意力缩放
- `Relu`: 激活函数
- `Sigmoid`: 通道注意力门控
- `GlobalAveragePool`: 注意力池化 (batch_size=1限制)
- `DepthToSpace`: PixelShuffle上采样
- `Concat`: 特征拼接

### 输入/输出
- **输入**: RGB, NCHW, float32, [0, 255]
- **输入尺寸**: 固定 (默认256×256)
- **输出**: RGB超分辨率图像 (scale × 输入尺寸)
- **内置预处理**: MeanShift层用于RGB归一化

## 常见问题

### Q: export_onnx.py报错 "Model file not found"

**A**: 检查PyTorch模型文件路径是否正确，确保你在`python/`目录下运行脚本。

### Q: test_onnx.py显示 "Shape mismatch"

**A**: 脚本会自动调整输入图像大小。如果仍有问题，确保输入图像是有效的RGB PNG/JPG文件。

### Q: convert.py量化失败

**A**:
1. 确保校准图像在 `dataset/calibration/` 目录
2. 图像文件扩展名必须是小写的 `.png`, `.jpg` 等
3. 检查图像文件是否损坏
4. 尝试使用FP16模式代替INT8

### Q: INT8模型质量不佳

**A**:
1. 增加校准图像数量 (8-10张)
2. 使用更多样化的校准图像
3. 考虑使用FP16模式 (虽然速度较慢)

## 环境要求

```bash
pip install torch torchvision onnx onnxruntime onnxsim rknn-toolkit2 pillow numpy
```

- Python: 3.10+
- PyTorch: 2.0+
- RKNN Toolkit: 2.3.0+

## 输出文件

转换完成后，`model/`目录应包含:
- `rcan_bix4_pytorch.pt` (60 MB) - PyTorch原始模型
- `rcan_x4.onnx` (61 MB) - ONNX模型
- `rcan_x4_i8.rknn` (53 MB) - INT8 RKNN模型 ✅推荐
- `test_pytorch_output.png` - PyTorch推理结果
- `test_onnx_output.png` - ONNX推理结果
- `input_256x256.png` - 测试输入图像
