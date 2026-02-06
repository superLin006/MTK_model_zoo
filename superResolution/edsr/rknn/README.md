# EDSR RKNN实现

EDSR超分辨率模型在RKNN平台上的实现。

## 快速开始

### 模型转换
```bash
cd python
python convert.py
```

### C++推理
```bash
cd cpp
./build.sh
./edsr_demo ../model/edsr.rknn input.png output.png
```

## 支持平台
- RK3588
- RK3566  
- RK3568
