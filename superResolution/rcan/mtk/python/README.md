# RCAN MTK Python - 模型转换

将PyTorch模型转换为MTK NPU可用的DLA格式。

## 转换步骤 (510x339输入)

```bash
# Step 1: PyTorch → TorchScript
python step1_pt_to_torchscript.py --checkpoint data/models/rcan/RCAN_BIX4.pt --input_height 339 --input_width 510

# Step 2: TorchScript → TFLite
python step2_torchscript_to_tflite.py --torchscript ../../models/RCAN_BIX4_core_339x510.pt

# Step 3: TFLite → DLA
python step3_tflite_to_dla.py --tflite ../../models/RCAN_BIX4_339x510.tflite --platform MT8371
```

## 输出

- `models/RCAN_BIX4_339x510_MT8371.dla` - MTK NPU模型文件
