#!/bin/bash

set -e

TARGET_SOC="rk3576"
BUILD_TYPE=Release
ANDROID_ABI=arm64-v8a

if [[ -z ${ANDROID_NDK_PATH} ]]; then
    echo "ANDROID_NDK_PATH not set, trying to find it..."
    # Try common locations
    if [[ -d ~/Android/Sdk/ndk ]]; then
        export ANDROID_NDK_PATH=$(ls -d ~/Android/Sdk/ndk/* 2>/dev/null | head -1)
    elif [[ -d ~/opts/ndk ]]; then
        export ANDROID_NDK_PATH=$(ls -d ~/opts/ndk/* 2>/dev/null | head -1)
    fi

    if [[ -z ${ANDROID_NDK_PATH} ]]; then
        echo "Please set ANDROID_NDK_PATH, such as: export ANDROID_NDK_PATH=~/opts/ndk/android-ndk-r18b"
        exit 1
    fi

    echo "Found NDK at: ${ANDROID_NDK_PATH}"
fi

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# Build
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SOC}_android

if [[ ! -d "${BUILD_DIR}" ]]; then
    mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ${ROOT_PWD}/cpp \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib"

make -j4
make install
cd -

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Output directory: ${BUILD_DIR}/install/rknn_realesrgan_demo"
echo "Binary: ${BUILD_DIR}/install/rknn_realesrgan_demo/rknn_realesrgan_demo"
echo ""
echo "Next steps:"
echo "  1. Push to device: ./push_and_run.sh"
echo "=========================================="
