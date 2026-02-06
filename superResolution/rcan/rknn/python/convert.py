#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert ONNX RCAN model to RKNN format."""

import sys
import os
from rknn.api import RKNN

# RKNN quantization dataset
# For super-resolution, we use representative images for quantization
DATASET_PATH = './dataset.txt'
DEFAULT_QUANT = False  # FP16 recommended for super-resolution quality


def create_dataset_file():
    """Create a dataset file for RKNN quantization."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try to use calibration images from dataset directory
    dataset_dir = os.path.join(script_dir, '../dataset/calibration/')
    image_files = []

    # Check if dataset directory exists
    if os.path.exists(dataset_dir):
        for fname in os.listdir(dataset_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(dataset_dir, fname))

    # Also check for test image in model directory
    model_dir = os.path.join(script_dir, '../model/')
    for fname in ['test_input.png', 'test_input_256x256.png']:
        test_img = os.path.join(model_dir, fname)
        if os.path.exists(test_img):
            image_files.append(test_img)

    if image_files:
        # Filter only images with valid extensions and use absolute paths
        valid_images = []
        for img_path in image_files:
            # Use absolute path to avoid any path issues
            abs_path = os.path.abspath(img_path)
            ext = os.path.splitext(abs_path)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                valid_images.append(abs_path)

        if valid_images:
            # Use absolute path for dataset file - put it in project root
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            abs_dataset_path = os.path.join(project_root, 'dataset.txt')
            with open(abs_dataset_path, 'w') as f:
                for img_path in valid_images[:10]:  # Use up to 10 images
                    f.write(img_path + '\n')
            print(f"✓ Created dataset file with {len(valid_images)} images: {abs_dataset_path}")
            # Print first few paths for debugging
            for i, path in enumerate(valid_images[:3]):
                print(f"  [{i+1}] {path}")
            # Also set DATASET_PATH to absolute path for later use
            global DATASET_PATH
            DATASET_PATH = abs_dataset_path
            return True
        else:
            print(f"⚠ Warning: No valid images found in dataset directories")
            return False
    else:
        print(f"⚠ Warning: No calibration images found for quantization")
        print(f"  Searched in: {dataset_dir}")
        print(f"  Searched for: {test_img}")
        return False


def parse_arg():
    """Parse command line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python convert.py <onnx_model_path> <platform> [dtype] [output_rknn_path]")
        print("\nArguments:")
        print("  onnx_model_path: Path to ONNX model file")
        print("  platform: Target platform (rk3576, rk3588, rk3568, etc.)")
        print("  dtype: Model precision - 'fp' for FP16 (default) or 'i8' for INT8 quantized")
        print("  output_rknn_path: Output RKNN file path (optional, auto-generated if not specified)")
        print("\nExamples:")
        print("  python convert.py ../model/rcan_x4.onnx rk3576")
        print("  python convert.py ../model/rcan_x4.onnx rk3576 fp ../model/rcan_x4_fp.rknn")
        print("  python convert.py ../model/rcan_x4.onnx rk3576 i8 ../model/rcan_x4_i8.rknn")
        print("\nNote: FP16 ('fp') is recommended for super-resolution to maintain quality.")
        print("      INT8 ('i8') is faster but may lose visual quality.")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'fp']:
            print(f"ERROR: Invalid dtype: {model_type}")
            print("       dtype should be 'fp' (FP16) or 'i8' (INT8)")
            exit(1)
        do_quant = (model_type == 'i8')

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        # Auto-generate output path
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        dtype_suffix = '_i8' if do_quant else '_fp'
        output_path = os.path.join(os.path.dirname(model_path), f"{base_name}{dtype_suffix}.rknn")

    return model_path, platform, do_quant, output_path


def print_conversion_info(onnx_path, platform, do_quant, output_path):
    """Print conversion configuration."""
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print("=" * 60)
    print("RKNN Conversion Configuration:")
    print("-" * 60)
    print(f"  ONNX model:      {onnx_path}")
    print(f"  ONNX size:       {file_size_mb:.2f} MB")
    print(f"  Target platform: {platform}")
    print(f"  Precision:       {'INT8 (quantized)' if do_quant else 'FP16'}")
    print(f"  Output:          {output_path}")
    print("=" * 60)


def main():
    model_path, platform, do_quant, output_path = parse_arg()

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: ONNX model not found: {model_path}")
        exit(1)

    print_conversion_info(model_path, platform, do_quant, output_path)

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    # For RCAN super-resolution:
    # - Input is RGB image with range [0, 255]
    # - Model has built-in MeanShift for normalization
    # - So we don't need to normalize here
    print('\n--> Configuring RKNN model')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],  # No normalization (model has built-in)
        target_platform=platform,
        optimization_level=3,
        quantized_algorithm='normal',
        quantized_method='channel',
    )
    print('done')

    # Load ONNX model
    # RCAN typically uses 256x256 input (can be adjusted)
    # Check input size from ONNX model first
    import onnx
    onnx_model = onnx.load(model_path)
    input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_h = input_shape[2].dim_value
    input_w = input_shape[3].dim_value

    print(f'\n--> Loading ONNX model')
    print(f'    Input size from ONNX: {input_h}x{input_w}')

    input_size_list = [[1, 3, input_h, input_w]]  # NCHW format
    ret = rknn.load_onnx(model=model_path, input_size_list=input_size_list)
    if ret != 0:
        print('ERROR: Load ONNX model failed!')
        exit(ret)
    print('done')

    # Build model
    print('\n--> Building RKNN model')
    if do_quant:
        # Create dataset for quantization
        dataset_file = None
        if create_dataset_file():
            dataset_file = os.path.abspath(DATASET_PATH)
        else:
            print('WARNING: No dataset file available for quantization')
            print('         Quantization may not work properly')
            print('         Consider using FP16 mode instead')

        print(f'        Using INT8 quantization with dataset...')
        ret = rknn.build(do_quantization=True, dataset=dataset_file)
    else:
        # FP16 mode (recommended for super-resolution)
        print('        Using FP16 precision (recommended for quality)...')
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print('ERROR: Build RKNN model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('\n--> Exporting RKNN model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('ERROR: Export RKNN model failed!')
        exit(ret)
    print('done')

    # Print final info
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    input_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0

    print("\n" + "=" * 60)
    print("RKNN Conversion Completed Successfully!")
    print("-" * 60)
    print(f"  ONNX size:    {input_size_mb:.2f} MB")
    print(f"  RKNN size:    {output_size_mb:.2f} MB")
    print(f"  Compression:  {compression_ratio:.2f}x")
    print(f"  Output file:  {output_path}")
    print("=" * 60)

    if do_quant:
        print("\nNote: INT8 model may have quality degradation.")
        print("      Test the output quality and compare with FP16 version.")
    else:
        print("\nNote: FP16 model preserves visual quality.")
        print("      For faster inference, consider INT8 quantization.")

    # Release
    rknn.release()


if __name__ == '__main__':
    main()
