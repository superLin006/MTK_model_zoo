# Real-ESRGAN for RK3576 NPU

Real-ESRGAN超分辨率模型在Rockchip RK3576 NPU上的完整实现，支持FP16和INT8量化。

## 项目概述

本项目实现了Real-ESRGAN（4x超分辨率）模型从PyTorch到RKNN的完整移植，包括：
- ✅ PyTorch模型导出为ONNX
- ✅ ONNX转换为RKNN（FP16/INT8）
- ✅ C++推理实现（RK3576 Android）
- ✅ INT8量化优化（PSNR 33.34 dB）

## 性能指标

| 模型 | 推理时间 | 模型大小 | 内存使用 | PSNR | 质量评级 |
|------|---------|---------|---------|------|---------|
| **FP16** | 12.7秒 | 264 MB | 940 MB | 51.06 dB | 优秀 |
| **INT8** | 5.2秒 | 134 MB | 465 MB | 33.34 dB | 良好 |

**INT8优势**: 速度提升59%，内存减少50%，质量良好（PSNR > 30dB）

## 目录结构

```
realesrgan/
├── README.md                           # 本文件
├── MIGRATION_REPORT.md                 # 完整移植报告
├── python/                             # Python端工具
│   ├── realesrgan_arch.py             # 模型架构
│   ├── export_onnx.py                 # ONNX导出
│   ├── convert.py                     # RKNN转换（支持FP16/INT8）
│   ├── test_pytorch.py                # PyTorch推理测试
│   ├── test_onnx.py                   # ONNX推理测试
│   └── requirements.txt               # Python依赖
├── cpp/                                # C++推理代码
│   ├── main.cc                        # 主程序
│   ├── realesrgan.h                   # 头文件
│   ├── rknpu2/realesrgan.cc           # RKNN推理实现
│   └── CMakeLists.txt                 # 编译配置
├── model/                              # 模型文件
│   ├── RealESRGAN_x4plus.pth          # PyTorch权重
│   ├── RealESRGAN_x4plus_510x339.onnx # ONNX模型
│   ├── RealESRGAN_x4plus_510x339_fp16.rknn  # FP16 RKNN模型
│   ├── RealESRGAN_x4plus_510x339_i8.rknn    # INT8 RKNN模型
│   ├── input_510x339.png              # 测试输入
│   └── output_correct.png             # 参考输出
├── dataset/                            # Calibration数据
│   └── calibration_510x339/           # INT8量化校准数据集
├── outputs/                            # 示例输出
│   ├── fp16_output.png                # FP16推理结果
│   ├── int8_output.png                # INT8推理结果
│   └── fp16_vs_int8_comparison.png    # 对比图
├── build-android.sh                    # Android编译脚本
└── push_and_run.sh                     # 设备部署脚本

```

## 快速开始

### 1. 环境准备

#### Python环境
```bash
cd python
pip install -r requirements.txt
```

需要的包：
- PyTorch >= 1.8
- ONNX >= 1.12
- rknn-toolkit2 >= 2.3.0
- opencv-python
- numpy

#### Android NDK
```bash
export ANDROID_NDK_PATH=/path/to/your/ndk
```

### 2. 模型转换

#### 导出ONNX（可选，已提供）
```bash
cd python
python export_onnx.py
```

#### 转换为RKNN FP16
```bash
python convert.py \
  ../model/RealESRGAN_x4plus_510x339.onnx \
  rk3576 \
  fp \
  ../model/RealESRGAN_x4plus_510x339_fp16.rknn
```

#### 转换为RKNN INT8
```bash
python convert.py \
  ../model/RealESRGAN_x4plus_510x339.onnx \
  rk3576 \
  i8 \
  ../model/RealESRGAN_x4plus_510x339_i8.rknn \
  --dataset ../dataset/calibration_510x339
```

**重要**: INT8量化需要提供calibration数据集，数据必须是[0, 1]归一化的NCHW格式。

### 3. C++编译

```bash
# 编译Android版本
./build-android.sh
```

生成的可执行文件：`build/build_rk3576_android/rknn_realesrgan_demo`

### 4. 设备部署与测试

#### 自动部署（推荐）
```bash
./push_and_run.sh
```

这个脚本会自动：
- 推送可执行文件
- 推送RKNN模型
- 推送测试图片
- 推送运行时库
- 在设备上运行推理
- 拉取结果

#### 手动部署
```bash
# 连接设备
adb devices

# 创建目录
adb shell "mkdir -p /data/local/tmp/rknn_realesrgan_demo/model"

# 推送文件
adb push build/build_rk3576_android/rknn_realesrgan_demo /data/local/tmp/rknn_realesrgan_demo/
adb push model/RealESRGAN_x4plus_510x339_fp16.rknn /data/local/tmp/rknn_realesrgan_demo/model/
adb push model/input_510x339.png /data/local/tmp/rknn_realesrgan_demo/model/
adb push 3rdparty/runtime/rk3576/Android/arm64-v8a/librknnrt.so /data/local/tmp/rknn_realesrgan_demo/

# 运行FP16推理
adb shell "cd /data/local/tmp/rknn_realesrgan_demo && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           ./rknn_realesrgan_demo model/RealESRGAN_x4plus_510x339_fp16.rknn model/input_510x339.png"

# 运行INT8推理
adb shell "cd /data/local/tmp/rknn_realesrgan_demo && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           ./rknn_realesrgan_demo model/RealESRGAN_x4plus_510x339_i8.rknn model/input_510x339.png"

# 拉取结果
adb pull /data/local/tmp/rknn_realesrgan_demo/output_sr.png ./
```

## 输入输出规格

### 输入
- **尺寸**: 510×339（固定）
- **格式**: RGB888 或 RGBA8888
- **数据范围**: [0, 255] uint8
- **自动转换**: RGBA会自动转换为RGB

### 输出
- **尺寸**: 2040×1356（4x超分辨率）
- **格式**: RGB888
- **数据范围**: [0, 255] uint8
- **输出文件**: output_sr.png 或 output_sr.ppm

## 模型选择指南

### 使用FP16的场景
- ✅ 离线处理，对速度要求不高
- ✅ 需要最高质量（PSNR > 50dB）
- ✅ 有充足的内存和算力资源
- 推理时间: ~12.7秒
- 模型大小: 264 MB
- 内存使用: ~940 MB

### 使用INT8的场景
- ✅ 实时或近实时应用
- ✅ 移动设备或嵌入式平台
- ✅ 批量处理，需要高吞吐
- ✅ 内存受限环境
- 推理时间: ~5.2秒（59%加速）
- 模型大小: 134 MB（50%减少）
- 内存使用: ~465 MB（50%减少）
- 质量: PSNR 33.34 dB（良好）

## Python端测试

### 测试PyTorch模型
```bash
cd python
python test_pytorch.py
```

### 测试ONNX模型
```bash
python test_onnx.py
```

### 测试输出
- 输出图片: `python/output_pytorch_test.png`, `python/output_onnx_test.png`
- 对比报告: 控制台输出PSNR和差异统计

## INT8量化说明

### 关键要点
INT8量化已完全优化，关键技术点：

1. **Calibration数据预处理**:
   - 必须归一化到 [0, 1]
   - 格式转换为NCHW
   - 保存为.npy文件

2. **量化参数**:
   - 输入: zp=-128, scale=0.003922 (≈1/255)
   - 输出: zp=-96, scale=0.006289

3. **质量保证**:
   - PSNR 33.34 dB（良好质量）
   - 平均差异 4.13 像素
   - 视觉效果接近FP16

详细技术说明请参考 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)

## 已知限制

1. **输入尺寸固定**: 必须是510×339，其他尺寸需要预先resize
2. **输入格式**: 仅支持RGB888和RGBA8888
3. **内存需求**: FP16需要~940MB，INT8需要~465MB
4. **PNG保存**: 设备端可能缺失PNG库，会fallback到PPM格式

## 故障排除

### 推理输出全黑
- **原因**: 未设置输入tensor的type和fmt
- **解决**: 已在代码中修复（realesrgan.cc:321-322）

### INT8输出颜色失真
- **原因**: Calibration数据范围错误
- **解决**: 使用[0, 1]归一化的数据重新量化

### 编译错误：RGA库未找到
- **原因**: Android平台缺少RGA硬件库
- **解决**: 已通过DISABLE_RGA条件编译修复

### 输入尺寸不匹配
- **解决**: 使用OpenCV resize到510×339
```python
img = cv2.resize(img, (510, 339))
```

## 技术细节

### 模型架构
- **Backbone**: RRDB (Residual in Residual Dense Block)
- **Upsampling**: Nearest + Conv（RKNN友好）
- **Blocks**: 23个RRDB块
- **Scale**: 4x超分辨率

### 数据流
```
输入图像 (510×339 RGB)
    ↓ 归一化 [0, 1]
    ↓ RKNN推理
    ↓ 输出 [-0.15, 1.31]
    ↓ Clip [0, 1]
    ↓ ×255 → uint8
输出图像 (2040×1356 RGB)
```

### 量化精度
| 层类型 | FP16 | INT8 |
|--------|------|------|
| Conv | FP16 | INT8 (w8a8) |
| LeakyReLU | FP16 | INT8 |
| Add/Concat | FP16 | INT8 |
| Upsample | FP16 | INT8 |
| 总体PSNR | 51.06 dB | 33.34 dB |

## 参考资料

- [Real-ESRGAN官方仓库](https://github.com/xinntao/Real-ESRGAN)
- [RKNN-Toolkit2文档](https://github.com/rockchip-linux/rknn-toolkit2)
- [完整移植报告](MIGRATION_REPORT.md)

## 许可证

本项目遵循Apache 2.0许可证。

## 致谢

- Real-ESRGAN模型来自 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- RKNN工具链由Rockchip提供

## 更新日志

### 2026-01-29
- ✅ 完成PyTorch → ONNX → RKNN完整流程
- ✅ 实现FP16推理（PSNR 51.06 dB）
- ✅ 优化INT8量化（PSNR 33.34 dB）
- ✅ 完成RK3576设备测试
- ✅ 修复所有已知问题
