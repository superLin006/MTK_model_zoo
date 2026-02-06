#!/bin/bash

# Whisper MTK NPU - Host Build Script
# Builds the whisper_test executable for host (x86-64) testing

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

echo "========================================"
echo "  Whisper MTK NPU - Host Build"
echo "========================================"
echo ""

# Create build directory
if [ -d "${BUILD_DIR}" ]; then
    echo "[INFO] Cleaning existing build directory..."
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo "[INFO] Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

# Build
echo "[INFO] Building..."
make -j$(nproc)

# Install
echo "[INFO] Installing..."
make install

# Check if executable was created
if [ -f "${BUILD_DIR}/bin/whisper_test" ]; then
    echo ""
    echo "========================================"
    echo "  Build Successful!"
    echo "========================================"
    echo ""
    echo "Executable: ${BUILD_DIR}/bin/whisper_test"
    ls -lh "${BUILD_DIR}/bin/whisper_test"
    echo ""
    echo "To run:"
    echo "  cd ${BUILD_DIR}/bin"
    echo "  ./whisper_test <model_dir> <audio_file>"
    echo ""
else
    echo ""
    echo "[ERROR] Build failed - executable not found"
    exit 1
fi
