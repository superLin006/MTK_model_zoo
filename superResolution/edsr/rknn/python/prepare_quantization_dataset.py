#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备EDSR量化校准数据集

量化需要一组有代表性的图像来统计激活值分布。
建议准备10-50张不同场景的图像。
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse


def download_sample_images():
    """
    下载一些样本图像用于量化校准

    你也可以使用自己的图像数据集
    """
    print("=" * 60)
    print("准备量化校准数据集")
    print("=" * 60)

    # 这里我们使用PIL生成一些测试图像
    # 实际使用时，应该用真实的、多样化的图像

    dataset_dir = "../dataset/calibration"
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"\n数据集目录: {dataset_dir}")

    # 如果目录已有图像，询问是否重新生成
    existing_images = [f for f in os.listdir(dataset_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if existing_images:
        print(f"\n已存在 {len(existing_images)} 张图像")
        response = input("是否重新生成样本图像? (y/N): ")
        if response.lower() != 'y':
            print("使用现有图像")
            return dataset_dir

    # 生成多样化的测试图像
    patterns = [
        ('uniform_bright', lambda: np.ones((256, 256, 3), dtype=np.uint8) * 200),
        ('uniform_dark', lambda: np.ones((256, 256, 3), dtype=np.uint8) * 50),
        ('gradient_h', lambda: np.repeat(np.linspace(0, 255, 256).reshape(1, 256, 1), 256, axis=0).repeat(3, axis=2).astype(np.uint8)),
        ('gradient_v', lambda: np.repeat(np.linspace(0, 255, 256).reshape(256, 1, 1), 256, axis=1).repeat(3, axis=2).astype(np.uint8)),
        ('checkerboard', lambda: create_checkerboard()),
        ('color_bars', lambda: create_color_bars()),
        ('natural_scene_sim', lambda: create_natural_sim()),
    ]

    print(f"\n生成 {len(patterns)} 张测试图像...")
    for name, generator in patterns:
        img_arr = generator()
        img = Image.fromarray(img_arr, mode='RGB')
        img_path = os.path.join(dataset_dir, f'{name}.png')
        img.save(img_path)
        print(f"  ✓ {img_path}")

    print(f"\n✓ 生成完成，共 {len(patterns)} 张图像")
    print(f"\n建议:")
    print("  1. 将你自己的真实图像（256x256 RGB）放到 {dataset_dir} 目录")
    print("  2. 图像应该覆盖不同的场景：人脸、风景、文字、建筑等")
    print("  3. 建议准备 10-50 张图像")

    return dataset_dir


def create_checkerboard():
    """创建棋盘格图案"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    square_size = 32
    for i in range(256 // square_size):
        for j in range(256 // square_size):
            if (i + j) % 2 == 0:
                img[i*square_size:(i+1)*square_size,
                    j*square_size:(j+1)*square_size] = 255
    return img


def create_color_bars():
    """创建彩色条纹"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 255, 255],# White
    ]
    bar_width = 256 // len(colors)
    for i, color in enumerate(colors):
        img[:, i*bar_width:(i+1)*bar_width] = color
    return img


def create_natural_sim():
    """模拟自然场景（随机噪声+渐变）"""
    # 基础渐变
    base = np.zeros((256, 256, 3), dtype=np.float32)
    y, x = np.ogrid[:256, :256]
    base[:, :, 0] = x / 256.0 * 255  # R gradient
    base[:, :, 1] = y / 256.0 * 255  # G gradient
    base[:, :, 2] = ((x + y) / 2.0) / 256.0 * 255  # B gradient

    # 添加噪声
    noise = np.random.normal(0, 20, (256, 256, 3))
    img = base + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def validate_dataset(dataset_dir):
    """验证数据集"""
    print("\n" + "=" * 60)
    print("验证数据集")
    print("=" * 60)

    images = [f for f in os.listdir(dataset_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print(f"❌ 错误: {dataset_dir} 中没有图像文件")
        return False

    print(f"\n找到 {len(images)} 张图像")

    # 验证每张图像
    valid_count = 0
    for img_name in images:
        img_path = os.path.join(dataset_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            if w == 256 and h == 256:
                valid_count += 1
                print(f"  ✓ {img_name}: {w}x{h}")
            else:
                print(f"  ⚠ {img_name}: {w}x{h} (需要256x256，将被resize)")
                # 自动resize
                img_resized = img.resize((256, 256), Image.LANCZOS)
                img_resized.save(img_path)
                print(f"    已自动resize为256x256")
                valid_count += 1

        except Exception as e:
            print(f"  ❌ {img_name}: 读取失败 - {e}")

    print(f"\n有效图像: {valid_count}/{len(images)}")

    if valid_count < 5:
        print("\n⚠️  建议: 至少准备10张图像以获得更好的量化效果")

    return valid_count > 0


def main():
    parser = argparse.ArgumentParser(description='准备EDSR量化校准数据集')
    parser.add_argument('--dataset-dir', type=str, default='../dataset/calibration',
                        help='数据集目录路径')
    parser.add_argument('--skip-generate', action='store_true',
                        help='跳过生成样本图像，仅验证现有数据集')

    args = parser.parse_args()

    if not args.skip_generate:
        dataset_dir = download_sample_images()
    else:
        dataset_dir = args.dataset_dir
        if not os.path.exists(dataset_dir):
            print(f"❌ 错误: 目录不存在 {dataset_dir}")
            sys.exit(1)

    # 验证数据集
    if validate_dataset(dataset_dir):
        print("\n" + "=" * 60)
        print("✅ 数据集准备完成！")
        print("=" * 60)
        print(f"\n下一步:")
        print(f"  运行量化脚本: python convert_quantized.py")
    else:
        print("\n❌ 数据集验证失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
