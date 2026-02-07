"""
RCAN测试工具函数

提供统一的输出保存接口，确保符合 python_output_management.md 规范
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from test_config import OUTPUT_DIR, DEBUG_DIR


def save_output(stage, test_name, data, format="json"):
    """
    保存测试输出

    Args:
        stage: "baseline" | "torchscript" | "tflite" | "dla"
        test_name: 测试用例名称（如 "butterfly", "baby"）
        data: 输出数据（dict、str或PIL.Image）
        format: "json" | "txt" | "png"

    Returns:
        Path: 保存的文件路径

    Example:
        >>> save_output("baseline", "butterfly", {"psnr": 28.5}, format="json")
        >>> save_output("baseline", "butterfly", output_image, format="png")
    """
    stage_dir = OUTPUT_DIR / stage

    if format == "json":
        file = stage_dir / f"{test_name}.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif format == "txt":
        file = stage_dir / f"{test_name}.txt"
        with open(file, "w", encoding="utf-8") as f:
            f.write(data)
    elif format == "png":
        file = stage_dir / f"{test_name}.png"
        if isinstance(data, np.ndarray):
            data = Image.fromarray(data.astype(np.uint8))
        data.save(file)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Saved: {file}")
    return file


def save_debug(name, data):
    """
    保存中间输出（给C++对比用）

    Args:
        name: 描述性名称（如 "preprocessed_input", "model_output"）
        data: numpy数组或可转换为numpy的数据

    Returns:
        Path: 保存的文件路径

    Example:
        >>> preprocessed = preprocess(image)
        >>> save_debug("preprocessed_input", preprocessed)
        [DEBUG] Saved preprocessed_input: shape=(1, 3, 256, 256), dtype=float32
        → /path/to/debug/preprocessed_input.npy
    """
    # 确保数据是numpy数组
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    file = DEBUG_DIR / f"{name}.npy"
    np.save(file, data)

    print(f"[DEBUG] Saved {name}: shape={data.shape}, dtype={data.dtype}")
    print(f"        → {file}")
    return file


def load_debug(name):
    """
    加载debug数据

    Args:
        name: 描述性名称（不带.npy后缀）

    Returns:
        np.ndarray: 加载的数据
    """
    file = DEBUG_DIR / f"{name}.npy"
    if not file.exists():
        raise FileNotFoundError(f"Debug file not found: {file}")

    data = np.load(file)
    print(f"[DEBUG] Loaded {name}: shape={data.shape}, dtype={data.dtype}")
    return data


def calculate_psnr(img1, img2):
    """
    计算PSNR

    Args:
        img1: 第一张图像（numpy array）
        img2: 第二张图像（numpy array）

    Returns:
        float: PSNR值
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compare_outputs(baseline_name, test_name, test_stage):
    """
    对比测试输出与baseline

    Args:
        baseline_name: baseline文件名（如 "butterfly"）
        test_name: 测试文件名
        test_stage: "torchscript" | "tflite" | "dla"

    Returns:
        dict: 对比结果
    """
    baseline_img = OUTPUT_DIR / "baseline" / f"{baseline_name}.png"
    test_img = OUTPUT_DIR / test_stage / f"{test_name}.png"

    baseline = np.array(Image.open(baseline_img))
    test = np.array(Image.open(test_img))

    psnr = calculate_psnr(baseline, test)
    mae = np.mean(np.abs(baseline.astype(float) - test.astype(float)))

    result = {
        "baseline_shape": baseline.shape,
        "test_shape": test.shape,
        "psnr": float(psnr),
        "mae": float(mae),
        "match": np.allclose(baseline, test, atol=1.0),
    }

    return result


def save_comparison_report(baseline_name, test_name, test_stage):
    """
    生成并保存对比报告

    Args:
        baseline_name: baseline文件名
        test_name: 测试文件名
        test_stage: "torchscript" | "tflite" | "dla"
    """
    result = compare_outputs(baseline_name, test_name, test_stage)

    report = []
    report.append(f"# Comparison Report: {test_stage.upper()} vs Baseline")
    report.append(f"Baseline: {baseline_name}")
    report.append(f"Test: {test_name}")
    report.append("")
    report.append("## Image Comparison")
    report.append(f"Baseline Shape: {result['baseline_shape']}")
    report.append(f"Test Shape:     {result['test_shape']}")
    report.append(f"PSNR:           {result['psnr']:.2f} dB")
    report.append(f"MAE:            {result['mae']:.2f}")
    report.append(f"Close Match:    {'✓' if result['match'] else '✗'}")

    report_file = OUTPUT_DIR / test_stage / "diff_vs_baseline.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"✓ Comparison report saved: {report_file}")
    return report_file
