#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Real-ESRGAN ONNX model to RKNN format."""

import sys
import os
import argparse
import numpy as np
import cv2
from rknn.api import RKNN


def collect_calibration_data(dataset_path, img_size, max_images=20):
    """Collect calibration images for quantization.

    Args:
        dataset_path: Path to calibration dataset directory
        img_size: (height, width) for input images
        max_images: Maximum number of images to collect

    Returns:
        dataset: List of preprocessed image arrays
    """
    print(f"\n--> Collecting calibration data from {dataset_path}...")

    if not os.path.exists(dataset_path):
        print(f"WARNING: Calibration dataset not found: {dataset_path}")
        print(f"         Using synthetic data instead")
        # Generate synthetic calibration data
        h, w = img_size
        dataset = [np.random.uniform(0, 255, (1, 3, h, w)).astype(np.float32) for _ in range(max_images)]
        print(f"    Generated {len(dataset)} synthetic images")
        return dataset

    # Collect real images
    dataset = []
    img_files = [f for f in os.listdir(dataset_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not img_files:
        print(f"WARNING: No images found in {dataset_path}")
        print(f"         Using synthetic data instead")
        h, w = img_size
        dataset = [np.random.uniform(0, 255, (1, 3, h, w)).astype(np.float32) for _ in range(max_images)]
        print(f"    Generated {len(dataset)} synthetic images")
        return dataset

    h_target, w_target = img_size
    for img_file in img_files[:max_images]:
        img_path = os.path.join(dataset_path, img_file)
        try:
            # Read and preprocess image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Resize to target size
            img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)

            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1] (Real-ESRGAN expects normalized input)
            img = img.astype(np.float32) / 255.0

            # HWC to CHW
            img = np.transpose(img, (2, 0, 1))

            # Add batch dimension
            img = np.expand_dims(img, axis=0)

            dataset.append(img)
        except Exception as e:
            print(f"    Warning: Failed to load {img_file}: {e}")
            continue

    print(f"    Collected {len(dataset)} calibration images")
    return dataset


def convert_to_rknn(onnx_path, platform, quantize_mode, output_path, dataset_path=None):
    """Convert ONNX model to RKNN format.

    Args:
        onnx_path: Path to ONNX model file
        platform: Target platform (e.g., 'rk3576', 'rk3588')
        quantize_mode: Quantization mode ('fp' for FP16, 'i8' for INT8)
        output_path: Path to save RKNN model
        dataset_path: Path to calibration dataset directory (required for INT8)
                   or None to use generated synthetic data
    """
    print("=" * 70)
    print("Real-ESRGAN ONNX to RKNN Conversion")
    print("=" * 70)
    print(f"ONNX model:     {onnx_path}")
    print(f"Output RKNN:    {output_path}")
    print(f"Platform:       {platform}")
    print(f"Quantize mode:  {quantize_mode.upper()}")
    if dataset_path:
        print(f"Dataset:        {dataset_path}")
    print("=" * 70)

    # Check ONNX file
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}")
        sys.exit(1)

    # Create RKNN object
    print("\n--> Creating RKNN instance...")
    rknn = RKNN(verbose=True)

    # Configure RKNN
    print("\n--> Configuring RKNN...")

    # Determine quantization settings
    # RKNN Toolkit2 v2.3.0 uses: 'w8a8', 'w8a16', 'w16a16i', 'w16a16i_dfp', 'w4a16'
    if quantize_mode.lower() == 'fp':
        # For FP16, we don't do quantization, just use FP16 precision
        do_quantization = False
        target_dtype = None
        print("    Using FP16 mode (no quantization, native FP16)")
    elif quantize_mode.lower() == 'i8':
        do_quantization = True
        target_dtype = 'w8a8'  # INT8: weight and activation both 8-bit
        print("    Using INT8 mode (w8a8)")
    else:
        print(f"ERROR: Unknown quantize mode: {quantize_mode}")
        print("       Supported modes: 'fp' (FP16), 'i8' (INT8)")
        sys.exit(1)

    # Build config based on quantization mode
    config_kwargs = {
        'target_platform': platform,
        'optimization_level': 3,
    }

    # Only add quantization parameters if quantizing
    if do_quantization:
        config_kwargs['quantized_dtype'] = target_dtype
        config_kwargs['quantized_algorithm'] = 'normal'
        config_kwargs['quantized_method'] = 'channel'

        # CRITICAL: Tell RKNN the input range for proper quantization
        # Real-ESRGAN expects [0, 1] normalized input
        # We provide the model with [0, 1] input during calibration
        # mean_values/std_values are used to describe the data distribution
        # For [0, 1] input: mean=0.5, std=0.5 (so normalized range becomes [-1, 1] internally)
        # But actually, we want to preserve [0, 1] range, so don't normalize
        # Instead, use mean_values=0, std_values=1 to keep original [0, 1] range
        config_kwargs['mean_values'] = [[0, 0, 0]]  # No mean subtraction
        config_kwargs['std_values'] = [[1, 1, 1]]   # No std division

        print("    INT8 quantization settings:")
        print(f"      dtype: {target_dtype}")
        print(f"      algorithm: normal")
        print(f"      method: channel")
        print(f"      mean_values: {config_kwargs['mean_values']}")
        print(f"      std_values: {config_kwargs['std_values']}")
        print(f"      (Input range: [0, 1] preserved)")

    ret = rknn.config(**config_kwargs)

    if ret != 0:
        print(f"ERROR: RKNN config failed!")
        sys.exit(1)
    print("✓ RKNN configured successfully")

    # Load ONNX model
    print("\n--> Loading ONNX model...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"ERROR: Load ONNX model failed!")
        sys.exit(1)
    print("✓ ONNX model loaded successfully")

    # Build RKNN model
    print("\n--> Building RKNN model...")
    print("    This may take several minutes...")

    # Prepare dataset for quantization
    dataset = None
    dataset_txt = None
    if do_quantization and quantize_mode.lower() == 'i8':
        # Get input size from ONNX model (need to parse or use default)
        default_dataset_path = dataset_path or '../dataset/calibration'

        # Try to get model input shape
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
            img_size = (input_shape[2], input_shape[3])  # (H, W)
            print(f"    Detected input size from ONNX: {img_size[1]}x{img_size[0]}")
        except:
            img_size = (339, 510)  # Default size (H, W)
            print(f"    Using default input size: {img_size[1]}x{img_size[0]}")

        # CRITICAL FIX: Create preprocessed numpy arrays for calibration
        # Real-ESRGAN expects [0, 1] normalized input, NOT [0, 255]
        # We need to load images, normalize to [0, 1], and provide as numpy arrays

        print("\n    Preparing calibration dataset (normalized to [0, 1])...")
        calibration_images = []

        # Check if dataset_path exists and is a directory
        if dataset_path and os.path.isdir(dataset_path):
            # Use real images from directory
            img_files = [f for f in os.listdir(dataset_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if img_files:
                import cv2
                for img_file in img_files[:20]:  # Limit to 20 images for calibration
                    img_path = os.path.join(dataset_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize if needed
                        if img.shape[0] != img_size[0] or img.shape[1] != img_size[1]:
                            img = cv2.resize(img, (img_size[1], img_size[0]))

                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Normalize to [0, 1] - THIS IS THE KEY FIX
                        img = img.astype(np.float32) / 255.0

                        # Convert HWC to CHW (RKNN expects NCHW format for calibration)
                        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

                        # Add batch dimension
                        img = np.expand_dims(img, axis=0)  # CHW -> NCHW

                        calibration_images.append(img)

                print(f"    Loaded {len(calibration_images)} calibration images from {dataset_path}")
            else:
                print(f"    WARNING: No images found in {dataset_path}")

        # If no images loaded, create synthetic calibration data
        if not calibration_images:
            print(f"    Creating synthetic calibration data...")
            # Create diverse synthetic images in [0, 1] range
            for i in range(10):
                # Random noise with different patterns
                if i < 5:
                    # Random uniform noise (HWC format)
                    img = np.random.uniform(0.0, 1.0, (img_size[0], img_size[1], 3)).astype(np.float32)
                else:
                    # Gradient patterns (HWC format)
                    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
                    for c in range(3):
                        img[:, :, c] = np.linspace(0, 1, img_size[0])[:, None] * np.linspace(0, 1, img_size[1])[None, :]

                # Convert HWC to NCHW for RKNN
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                img = np.expand_dims(img, axis=0)  # CHW -> NCHW

                calibration_images.append(img)
            print(f"    Created {len(calibration_images)} synthetic calibration images")

        # RKNN expects dataset as a .npy file path or text file with .npy paths
        # Save preprocessed images as .npy files
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix='rknn_calib_')
        temp_txt = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')

        print(f"    Saving preprocessed calibration data to {temp_dir}...")
        for i, img in enumerate(calibration_images):
            npy_path = os.path.join(temp_dir, f'calib_{i:03d}.npy')
            np.save(npy_path, img)
            temp_txt.write(f'{npy_path}\n')

        temp_txt.close()
        dataset_txt = temp_txt.name
        print(f"    Calibration dataset ready: {len(calibration_images)} images, range [0, 1]")
        print(f"    Dataset list: {dataset_txt}")

    # Build with preprocessed dataset (numpy arrays in [0, 1] range)
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset_txt, rknn_batch_size=1)

    # Clean up temporary files
    if do_quantization and dataset_txt:
        try:
            os.unlink(dataset_txt)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            print(f"    Cleaned up temporary calibration files")
        except:
            pass
    if ret != 0:
        print(f"ERROR: Build RKNN model failed!")
        sys.exit(1)
    print("✓ RKNN model built successfully")

    # Export RKNN model
    print(f"\n--> Exporting RKNN model to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"ERROR: Export RKNN model failed!")
        sys.exit(1)
    print(f"✓ RKNN model exported successfully")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Optionally run inference test
    print("\n--> Testing RKNN model inference...")
    try:
        # Initialize runtime
        ret = rknn.init_runtime(target=None)  # Run on simulator/PC
        if ret != 0:
            print(f"    Warning: Failed to initialize runtime (this is expected on PC without NPU)")
        else:
            # Create dummy input
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
            except:
                input_shape = [1, 3, 510, 339]

            dummy_input = np.random.uniform(0, 255, input_shape).astype(np.float32)
            outputs = rknn.inference(inputs=[dummy_input])

            if outputs is not None and len(outputs) > 0:
                print(f"✓ RKNN inference test passed")
                print(f"    Input shape:  {dummy_input.shape}")
                print(f"    Output shape: {outputs[0].shape}")
            else:
                print(f"    Warning: Inference returned empty output")
    except Exception as e:
        print(f"    Warning: RKNN inference test failed: {e}")
        print(f"    (This is normal on PC without NPU, test on device)")

    # Release RKNN
    rknn.release()

    print("\n" + "=" * 70)
    print("Conversion Completed Successfully!")
    print("-" * 70)
    print(f"  RKNN model saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Platform: {platform}")
    print(f"  Quantization: {quantize_mode.upper()}")
    print("=" * 70)
    print("\n✓ RKNN conversion PASSED!")
    print("\nNext steps:")
    print("  1. Copy the RKNN model to your device")
    print("  2. Run C++ inference on device")
    print(f"  3. Or test with Python: python test_rknn.py --model_path {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Real-ESRGAN ONNX model to RKNN format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to FP16 mode (recommended, good quality and performance)
  python convert.py ../model/RealESRNet_x4_510x339.onnx rk3576 fp \\
                    ../model/RealESRNet_x4_510x339_fp16.rknn

  # Convert to INT8 mode (faster, may reduce quality)
  python convert.py ../model/RealESRNet_x4_510x339.onnx rk3576 i8 \\
                    ../model/RealESRNet_x4_510x339_i8.rknn \\
                    --dataset ../dataset/calibration

  # For RK3588
  python convert.py ../model/RealESRNet_x4_510x339.onnx rk3588 fp \\
                    ../model/RealESRNet_x4_510x339_rk3588_fp16.rknn

Quantization modes:
  fp  - FP16 mode (mixed FP16/INT8), best quality
  i8  - INT8 mode, best performance, requires calibration dataset

Supported platforms:
  rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rk1808, rv1126
        """
    )
    parser.add_argument(
        'onnx_path',
        type=str,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        'platform',
        type=str,
        help='Target platform (e.g., rk3576, rk3588)'
    )
    parser.add_argument(
        'quantize_mode',
        type=str,
        choices=['fp', 'i8'],
        help='Quantization mode: fp (FP16) or i8 (INT8)'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path to save RKNN model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to calibration dataset directory (required for INT8 mode)'
    )

    args = parser.parse_args()

    # Validate INT8 requires dataset
    if args.quantize_mode == 'i8' and args.dataset is None:
        print("WARNING: INT8 mode without calibration dataset specified")
        print("         Will use synthetic data or default dataset path")

    # Convert to RKNN
    convert_to_rknn(
        args.onnx_path,
        args.platform,
        args.quantize_mode,
        args.output_path,
        args.dataset
    )


if __name__ == '__main__':
    main()
