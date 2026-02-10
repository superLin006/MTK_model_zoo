#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤3: 将Whisper TFLite转换为DLA格式
使用MTK ncc-tflite编译器
"""

import argparse
import os
import json
import subprocess
import time
from pathlib import Path


def compile_dla(
    tflite_path: str,
    output_dir: str = None,
    platform: str = 'MT8371',
    model_name: str = None
):
    """
    将TFLite编译为DLA格式

    参数:
        tflite_path: TFLite模型路径
        output_dir: 输出目录（默认与输入相同）
        platform: 目标平台 (MT8371, MT6899, MT6991)
        model_name: 模型名称（用于日志）
    """
    if output_dir is None:
        output_dir = os.path.dirname(tflite_path)

    model_desc = model_name if model_name else os.path.basename(tflite_path)
    
    print("="*70)
    print(f"步骤3: {model_desc} TFLite -> DLA")
    print("="*70)
    print(f"  输入: {tflite_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  目标平台: {platform}")
    print("="*70)

    # MTK SDK路径
    sdk_path = "/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk"
    ncc_tool = f"{sdk_path}/host/bin/ncc-tflite"

    # 检查工具是否存在
    if not os.path.exists(ncc_tool):
        print(f"❌ 错误: 找不到ncc-tflite工具")
        print(f"   期望路径: {ncc_tool}")
        return None

    # 平台配置
    platform_configs = {
        'MT8371': {'arch': 'mdla5.3,edma3.6', 'l1': '256', 'mdla': '1'},
        'MT6899': {'arch': 'mdla5.5,edma3.6', 'l1': '2048', 'mdla': '2'},
        'MT6991': {'arch': 'mdla5.5,edma3.6', 'l1': '7168', 'mdla': '4'},
    }

    if platform not in platform_configs:
        print(f"❌ 错误: 不支持的平台 {platform}")
        print(f"   支持的平台: {list(platform_configs.keys())}")
        return None

    cfg = platform_configs[platform]

    # 构建输出路径
    basename = os.path.basename(tflite_path).replace('.tflite', f'_{platform}.dla')
    dla_path = os.path.join(output_dir, basename)

    print(f"\n编译DLA模型...")
    print(f"  ncc-tflite: {ncc_tool}")
    print(f"  架构: {cfg['arch']}")
    print(f"  L1缓存: {cfg['l1']} KB")
    print(f"  MDLA数量: {cfg['mdla']}")

    # 设置环境变量
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f"{sdk_path}/host/lib:" + env.get('LD_LIBRARY_PATH', '')

    # 构建命令
    cmd = [
        ncc_tool,
        tflite_path,
        f'--arch={cfg["arch"]}',
        f'--l1-size-kb={cfg["l1"]}',
        f'--num-mdla={cfg["mdla"]}',
        '--relax-fp32',      # 放宽FP32精度要求
        '--opt-accuracy',    # 优化精度
        '--opt-footprint',   # 优化内存占用
        '-o', dla_path
    ]

    print(f"\n执行命令:")
    print(f"  {' '.join(cmd)}")
    print(f"\n开始编译...")

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        # 显示输出
        if result.stdout:
            print(f"\n编译日志:")
            lines = result.stdout.split('\n')
            for line in lines:
                # 过滤关键信息
                if any(keyword in line.lower() for keyword in 
                       ['compiling', 'optimi', 'tensor', 'error', 'warning', 
                        'size', 'performance', 'layer', 'mdla']):
                    print(f"  {line}")

        if result.returncode == 0 and os.path.exists(dla_path):
            dla_size_mb = os.path.getsize(dla_path) / 1024 / 1024
            elapsed = time.time() - start

            print(f"\n  ✓ 编译成功!")
            print(f"  输出: {basename}")
            print(f"  大小: {dla_size_mb:.1f} MB")
            print(f"  耗时: {elapsed:.1f}s")
            
            return dla_path

        else:
            print(f"\n  ❌ 编译失败!")
            if result.stderr:
                print(f"\n错误信息:")
                print(result.stderr)
            return None

    except subprocess.TimeoutExpired:
        print(f"\n  ❌ 编译超时 (>10分钟)")
        return None
    except Exception as e:
        print(f"\n  ❌ 编译异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='步骤3: 将Whisper TFLite转换为DLA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step3_tflite_to_dla.py \\
      --encoder_tflite ./models/encoder_base_80x3000.tflite \\
      --decoder_tflite ./models/decoder_base_448.tflite \\
      --platform MT8371
        """
    )

    parser.add_argument('--encoder_tflite', type=str, required=True,
                       help='Encoder TFLite模型路径')
    parser.add_argument('--decoder_tflite', type=str, required=True,
                       help='Decoder TFLite模型路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认与输入相同）')
    parser.add_argument('--platform', type=str, default='MT8371',
                       choices=['MT8371', 'MT6899', 'MT6991'],
                       help='目标平台 (默认: MT8371)')

    args = parser.parse_args()

    # 获取输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.encoder_tflite)

    os.makedirs(args.output_dir, exist_ok=True)

    # 编译Encoder
    print("\n" + "="*70)
    print("编译Whisper Encoder")
    print("="*70)
    
    encoder_dla = compile_dla(
        args.encoder_tflite,
        args.output_dir,
        args.platform,
        model_name="Encoder"
    )

    if encoder_dla is None:
        print("\n❌ Encoder编译失败，停止")
        return

    # 编译Decoder
    print("\n" + "="*70)
    print("编译Whisper Decoder")
    print("="*70)
    
    decoder_dla = compile_dla(
        args.decoder_tflite,
        args.output_dir,
        args.platform,
        model_name="Decoder"
    )

    if decoder_dla is None:
        print("\n❌ Decoder编译失败")
        return

    # 总结
    print("\n" + "="*70)
    print("✓ 所有DLA编译完成!")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  Encoder DLA: {os.path.basename(encoder_dla)}")
    print(f"  Decoder DLA: {os.path.basename(decoder_dla)}")

    print(f"\n相关文件:")
    print(f"  Token Embedding: token_embedding.npy (需要在C++端加载)")

    print(f"\n下一步:")
    print(f"  1. 开发C++推理代码")
    print(f"  2. 使用DLA模型在MT8371设备上运行")
    print(f"  3. 实现Embedding查表（C++端）")
    print(f"  4. 实现自回归解码循环（C++端）")


if __name__ == '__main__':
    main()
