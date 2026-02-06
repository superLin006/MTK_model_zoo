# EDSR MTK C++ - NPU推理实现

使用MTK Neuron API实现EDSR超分辨率推理。

## 编译

```bash
./build.sh
```

## 部署

```bash
# 自动部署并测试
./deploy_with_sdk_lib.sh --test

# 或手动部署
adb push jni/libs/arm64-v8a/edsr_inference /data/local/tmp/
adb push ../../models/EDSR_x4_256x256_MT8371.dla /data/local/tmp/
adb shell "cd /data/local/tmp && ./edsr_inference EDSR_x4_256x256_MT8371.dla input.png output.png"
```

## 性能

- 输入: 256x256 RGB
- 输出: 1024x1024 RGB
- 推理时间: ~7000ms
- FPS: 0.14
