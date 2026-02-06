# RCAN MTK C++ - NPU推理实现

使用MTK Neuron API实现RCAN超分辨率推理。

## 编译

```bash
./build.sh
```

## 部署

```bash
# 自动部署并测试
./deploy_with_sdk_lib.sh --test

# 或手动部署
adb push jni/libs/arm64-v8a/rcan_inference /data/local/tmp/
adb push ../../models/RCAN_BIX4_339x510_MT8371.dla /data/local/tmp/
adb shell "cd /data/local/tmp && ./rcan_inference RCAN_BIX4_339x510_MT8371.dla input.png output.png"
```

## 性能

- 输入: 510x339 RGB
- 输出: 2040x1356 RGB
- 推理时间: ~4000ms
- FPS: 0.25
