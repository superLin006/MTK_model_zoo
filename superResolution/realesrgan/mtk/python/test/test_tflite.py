#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试TFLite模型推理（使用MTK TFLite工具）
"""

import sys
import os
import numpy as np
import cv2
import argparse

# 尝试导入MTK的TFLite解释器
try:
    from mtk_converter import TFLiteExecutor
    MTK_TFLITE_AVAILABLE = True
except ImportError:
    MTK_TFLITE_AVAILABLE = False
    print("❌ 错误: mtk_converter未安装")
    print("   请确保在MTK-superResolution conda环境中运行")
    sys.exit(1)


def preprocess(img_path, target_size=None):
    """Load and preprocess image for Real-ESRGAN.

    Args:
        img_path: Path to input image
        target_size: (height, width) to resize input. If None, use original size.

    Returns:
        img_tensor: Preprocessed image tensor (1, 3, H, W) in range [0, 1]
        original_size: Original image size (W, H)
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Store original size
    h_orig, w_orig = img.shape[:2]
    original_size = (w_orig, h_orig)

    # Resize if specified
    if target_size is not None:
        h_target, w_target = target_size
        img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] range
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img, original_size


def postprocess(output_tensor):
    """Convert model output tensor to image.

    Args:
        output_tensor: Model output (1, 3, H, W) in range [0, 1]

    Returns:
        output_img: Output image in BGR format (uint8)
    """
    # Remove batch dimension
    output = output_tensor.squeeze(0)

    # Clamp to [0, 1] and convert to [0, 255]
    output = np.clip(output, 0, 1)
    output = (output * 255.0).round().astype(np.uint8)

    # CHW to HWC
    output = np.transpose(output, (1, 2, 0))

    # Convert RGB to BGR for OpenCV
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output


def test_tflite_model(tflite_path, img_path, output_path, input_size=None):
    """Test TFLite Real-ESRGAN model inference using MTK TFLiteExecutor.

    Args:
        tflite_path: Path to .tflite model file
        img_path: Path to input image
        output_path: Path to save output image
        input_size: (height, width) for model input
    """
    print("=" * 70)
    print("Real-ESRGAN MTK TFLite模型推理测试")
    print("=" * 70)
    print(f"模型:  {tflite_path}")
    print(f"输入:  {img_path}")
    print(f"输出: {output_path}")
    print("=" * 70)

    # Check files
    if not os.path.exists(tflite_path):
        print(f"错误: 模型文件不存在: {tflite_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"错误: 输入图像不存在: {img_path}")
        sys.exit(1)

    # Load TFLite model using MTK TFLiteExecutor
    print("\n--> 加载MTK TFLite模型...")
    executor = TFLiteExecutor(tflite_path)
    print(f"  ✓ 模型加载成功")

    # Note: MTK TFLiteExecutor doesn't expose input/output details easily
    # We'll infer from the actual data during inference

    # Preprocess input
    print("\n--> 预处理输入图像...")
    img_tensor, original_size = preprocess(img_path, input_size)
    img_tensor = img_tensor.astype(np.float32)
    print(f"    输入张量形状: {img_tensor.shape}")
    print(f"    输入值范围: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

    # Run inference
    print("\n--> 运行推理...")
    import time
    start = time.time()
    output_list = executor.run([img_tensor])
    elapsed = time.time() - start

    # Get output (MTK TFLiteExecutor returns a list)
    output_tensor = output_list[0] if isinstance(output_list, list) else output_list
    print(f"    输出张量形状: {output_tensor.shape}")
    print(f"    输出值范围: [{output_tensor.min():.2f}, {output_tensor.max():.2f}]")
    print(f"    推理耗时: {elapsed*1000:.1f}ms")

    # Postprocess output
    print("\n--> 后处理输出...")
    output_img = postprocess(output_tensor)
    print(f"    输出图像形状: {output_img.shape}")

    # Save output
    print(f"\n--> 保存输出到 {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, output_img)

    # Print summary
    h_in, w_in = img_tensor.shape[2:]
    h_out, w_out = output_img.shape[:2]
    scale = h_out // h_in

    print("\n" + "=" * 70)
    print("推理成功完成!")
    print("-" * 70)
    print(f"  输入尺寸:     {w_in}x{h_in}")
    print(f"  输出尺寸:    {w_out}x{h_out}")
    print(f"  超分倍数:   x{scale}")
    print(f"  推理耗时:    {elapsed*1000:.1f}ms")
    print(f"  输出文件:    {output_path}")
    print("=" * 70)
    print("\n✓ MTK TFLite推理测试通过!")


def main():
    parser = argparse.ArgumentParser(
        description='Test Real-ESRGAN TFLite model inference with MTK tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_tflite.py --tflite ../../models/RealESRGAN_x4plus_128x128.tflite --input ../../test_data/input_128x128.png

  python test_tflite.py --tflite ../../models/RealESRGAN_x4plus_128x128.tflite --input ../../test_data/input_128x128.png --output ../../test_data/output/my_output.png
        """
    )
    parser.add_argument(
        '--tflite',
        type=str,
        required=True,
        help='TFLite模型文件路径'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入图像路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output_tflite.png',
        help='输出图像保存路径'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='输入尺寸 (height width)。如不指定将从模型自动推断。'
    )

    args = parser.parse_args()

    # Convert input_size to tuple if provided
    input_size = tuple(args.input_size) if args.input_size else None

    # Run test
    test_tflite_model(
        tflite_path=args.tflite,
        img_path=args.input,
        output_path=args.output,
        input_size=input_size
    )


if __name__ == '__main__':
    main()
