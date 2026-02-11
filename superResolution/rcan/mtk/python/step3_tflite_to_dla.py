#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤3: 将TFLite转换为DLA格式
使用MTK ncc-tflite编译器
"""

import argparse
import os
import json
import subprocess
import time
from pathlib import Path

# 项目根目录（step3 在 .../rcan/mtk/python/ 下，向上4级到 MTK_models_zoo）
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]  # MTK_models_zoo/


def compile_dla(
    tflite_path: str,
    output_dir: str = None,
    platform: str = 'MT8371'
):
    """
    将TFLite编译为DLA格式

    参数:
        tflite_path: TFLite模型路径
        output_dir: 输出目录（默认与输入相同）
        platform: 目标平台 (MT8371, MT6899, MT6991)
    """
    if output_dir is None:
        output_dir = os.path.dirname(tflite_path)

    print("="*70)
    print("步骤3: TFLite -> DLA")
    print("="*70)
    print(f"  输入: {tflite_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  目标平台: {platform}")
    print("="*70)

    # MTK SDK路径（优先使用环境变量，否则使用项目内相对路径）
    sdk_path = os.environ.get(
        'MTK_NEURON_SDK',
        str(_PROJECT_ROOT / '0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk')
    )
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
    # EDSR_x4_256x256.tflite -> EDSR_x4_256x256_MT8371.dla
    basename = os.path.basename(tflite_path).replace('.tflite', f'_{platform}.dla')
    dla_path = os.path.join(output_dir, basename)

    print(f"\n[1/2] 编译DLA模型...")
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
        '--fc-to-conv',      # 全连接层转卷积
        '-o', dla_path
    ]

    print(f"\n执行命令:")
    print(f"  {' '.join(cmd)}")

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
            print(f"\n编译输出:")
            # 只显示关键信息
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['Compiling', 'optimi', 'tensor', 'Error', 'Warning', 'Size']):
                    print(f"  {line}")

        if result.returncode == 0 and os.path.exists(dla_path):
            dla_size_mb = os.path.getsize(dla_path) / 1024 / 1024
            elapsed = time.time() - start

            print(f"\n  ✓ 编译成功!")
            print(f"  输出: {basename}")
            print(f"  大小: {dla_size_mb:.1f} MB")
            print(f"  耗时: {elapsed:.1f}s")

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

    # 更新info.json
    print(f"\n[2/2] 更新模型信息...")

    # 查找对应的info文件
    base_name = os.path.basename(tflite_path).replace('.tflite', '')
    info_path = os.path.join(output_dir, f"{base_name}_info.json")

    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)

        info['files']['dla'] = basename
        info['target_platform'] = platform
        info['compilation'] = {
            'mdla_arch': cfg['arch'],
            'l1_size_kb': cfg['l1'],
            'num_mdla': cfg['mdla']
        }

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"  ✓ 更新: {os.path.basename(info_path)}")
    else:
        print(f"  ⚠ 未找到info文件: {info_path}")

    print("\n" + "="*70)
    print("✓ DLA编译完成!")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  DLA: {basename}")

    print(f"\n下一步:")
    print(f"  开发C++推理代码，使用DLA模型在MT8371设备上运行")

    return dla_path


def main():
    parser = argparse.ArgumentParser(
        description='步骤3: 将TFLite转换为DLA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step3_tflite_to_dla.py --tflite ../models/EDSR_x4_256x256.tflite
  python step3_tflite_to_dla.py --tflite ../models/EDSR_x4_256x256.tflite --platform MT6899
        """
    )

    parser.add_argument('--tflite', type=str, required=True,
                       help='TFLite模型路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--platform', type=str, default='MT8371',
                       choices=['MT8371', 'MT6899', 'MT6991'],
                       help='目标平台 (默认: MT8371)')

    args = parser.parse_args()

    compile_dla(
        tflite_path=args.tflite,
        output_dir=args.output_dir,
        platform=args.platform
    )


if __name__ == '__main__':
    main()
