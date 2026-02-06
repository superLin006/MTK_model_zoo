# EDSR MTK Python - 模型转换

将PyTorch模型转换为MTK NPU可用的DLA格式。

## 转换步骤

```bash
# Step 1: PyTorch → TorchScript
python step1_pt_to_torchscript.py --input data/models/edsr/EDSR_x4.pt

# Step 2: TorchScript → TFLite
python step2_torchscript_to_tflite.py --torchscript ../../models/EDSR_x4_core_256x256.pt

# Step 3: TFLite → DLA
python step3_tflite_to_dla.py --tflite ../../models/EDSR_x4_256x256.tflite --platform MT8371
```

## 输出

- `models/EDSR_x4_256x256_MT8371.dla` - MTK NPU模型文件
