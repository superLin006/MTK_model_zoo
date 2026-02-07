"""
测试工具函数

提供统一的输出保存接口，确保符合 python_output_management.md 规范
"""

import json
import numpy as np
from pathlib import Path
from test_config import OUTPUT_DIR, DEBUG_DIR


def save_output(stage, test_name, data, format="json"):
    """
    保存测试输出

    Args:
        stage: "baseline" | "torchscript" | "tflite" | "dla"
        test_name: 测试用例名称（如 "test_en", "test_zh", "jfk"）
        data: 输出数据（dict或str）
        format: "json" | "txt"

    Returns:
        Path: 保存的文件路径

    Example:
        >>> save_output("baseline", "test_en", {"text": "Hello"}, format="json")
        >>> save_output("baseline", "test_en", "Hello world", format="txt")
    """
    stage_dir = OUTPUT_DIR / stage

    if format == "json":
        file = stage_dir / f"{test_name}.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:  # txt
        file = stage_dir / f"{test_name}.txt"
        with open(file, "w", encoding="utf-8") as f:
            f.write(data)

    print(f"✓ Saved: {file}")
    return file


def save_debug(name, data):
    """
    保存中间输出（给C++对比用）

    Args:
        name: 描述性名称（如 "encoder_output", "preprocessed_mel"）
        data: numpy数组或可转换为numpy的数据

    Returns:
        Path: 保存的文件路径

    Example:
        >>> mel = preprocess(audio)
        >>> save_debug("preprocessed_mel", mel)
        [DEBUG] Saved preprocessed_mel: shape=(80, 3000), dtype=float32
        → /path/to/debug/preprocessed_mel.npy
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


def compare_outputs(baseline_name, test_name, test_stage):
    """
    对比测试输出与baseline

    Args:
        baseline_name: baseline文件名（如 "test_en"）
        test_name: 测试文件名
        test_stage: "torchscript" | "tflite" | "dla"

    Returns:
        dict: 对比结果
    """
    baseline_file = OUTPUT_DIR / "baseline" / f"{baseline_name}.json"
    test_file = OUTPUT_DIR / test_stage / f"{test_name}.json"

    with open(baseline_file, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test = json.load(f)

    # 简单对比
    result = {
        "baseline_text": baseline.get("text", ""),
        "test_text": test.get("text", ""),
        "match": baseline.get("text") == test.get("text"),
        "baseline_tokens": baseline.get("tokens", []),
        "test_tokens": test.get("tokens", []),
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
    report.append("## Text Comparison")
    report.append(f"Baseline: {result['baseline_text']}")
    report.append(f"Test:     {result['test_text']}")
    report.append(f"Match:    {'✓' if result['match'] else '✗'}")
    report.append("")
    report.append("## Tokens")
    report.append(f"Baseline: {result['baseline_tokens'][:10]}... ({len(result['baseline_tokens'])} total)")
    report.append(f"Test:     {result['test_tokens'][:10]}... ({len(result['test_tokens'])} total)")

    report_file = OUTPUT_DIR / test_stage / "diff_vs_baseline.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"✓ Comparison report saved: {report_file}")
    return report_file
