#!/bin/bash

set -e

TARGET_SOC="rk3576"
BUILD_DIR=build/build_${TARGET_SOC}_android
DEVICE_DIR=/data/local/tmp/rknn_realesrgan_demo

echo "=========================================="
echo "Real-ESRGAN Push and Run Script"
echo "=========================================="

# Check if device is connected
if ! adb devices | grep -q device$; then
    echo "ERROR: No Android device connected!"
    echo "Please connect your RK3576 device via USB and enable ADB debugging."
    exit 1
fi

echo "✓ Android device detected"
echo ""

# Create directory on device
echo "Creating directory on device..."
adb shell "mkdir -p ${DEVICE_DIR}/model"

# Push executable
echo "Pushing executable..."
adb push ${BUILD_DIR}/rknn_realesrgan_demo ${DEVICE_DIR}/

# Push RKNN model
echo "Pushing RKNN model..."
if [ -f "model/RealESRGAN_x4plus_510x339_fp16.rknn" ]; then
    adb push model/RealESRGAN_x4plus_510x339_fp16.rknn ${DEVICE_DIR}/model/
else
    echo "ERROR: RKNN model not found at model/RealESRGAN_x4plus_510x339_fp16.rknn"
    exit 1
fi

# Push test image
echo "Pushing test image..."
if [ -f "model/input_510x339.png" ]; then
    adb push model/input_510x339.png ${DEVICE_DIR}/model/
else
    echo "WARNING: Test image not found, you'll need to provide your own input"
fi

# Push RKNN runtime libraries
echo "Pushing RKNN runtime library..."
RKNN_LIB="${RKNN_MODEL_ZOO:-/path/to/rknn_model_zoo}/3rdparty/runtime/${TARGET_SOC}/Android/arm64-v8a/librknnrt.so"
if [ -f "${RKNN_LIB}" ]; then
    adb push ${RKNN_LIB} ${DEVICE_DIR}/
else
    echo "WARNING: librknnrt.so not found at ${RKNN_LIB}"
    echo "         Make sure the library is already on the device"
fi

echo ""
echo "=========================================="
echo "Files pushed successfully!"
echo "=========================================="
echo ""
echo "Running inference on device..."
echo "=========================================="

# Set library path and run
adb shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && chmod +x rknn_realesrgan_demo && ./rknn_realesrgan_demo model/RealESRGAN_x4plus_510x339_fp16.rknn model/input_510x339.png"

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
echo ""
echo "Pulling output image from device..."
adb pull ${DEVICE_DIR}/output_sr.png ./output_device.png || adb pull ${DEVICE_DIR}/output_sr.ppm ./output_device.ppm

if [ -f "./output_device.png" ]; then
    echo "✓ Output image saved to: ./output_device.png"
elif [ -f "./output_device.ppm" ]; then
    echo "✓ Output image saved to: ./output_device.ppm"
    echo "  (Convert to PNG: convert output_device.ppm output_device.png)"
else
    echo "⚠ Failed to pull output image from device"
fi

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
