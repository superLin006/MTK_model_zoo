"""
EDSR测试输出路径配置

统一管理所有输出目录路径，确保符合 python_output_management.md 规范
"""

from pathlib import Path

# 目录路径
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR = TEST_DIR / "outputs"

# 各阶段输出目录
BASELINE_DIR = OUTPUT_DIR / "baseline"
TORCHSCRIPT_DIR = OUTPUT_DIR / "torchscript"
TFLITE_DIR = OUTPUT_DIR / "tflite"
DLA_DIR = OUTPUT_DIR / "dla"
DEBUG_DIR = OUTPUT_DIR / "debug"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"

# 测试图像目录
TEST_IMAGE_DIR = PROJECT_ROOT.parent.parent / "test_images"

# 确保所有目录存在
for d in [BASELINE_DIR, TORCHSCRIPT_DIR, TFLITE_DIR, DLA_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 打印配置（仅在直接运行时）
if __name__ == "__main__":
    print("📁 EDSR Test Configuration")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Output Dir:   {OUTPUT_DIR}")
    print(f"  Baseline:     {BASELINE_DIR}")
    print(f"  TorchScript:  {TORCHSCRIPT_DIR}")
    print(f"  TFLite:       {TFLITE_DIR}")
    print(f"  DLA:          {DLA_DIR}")
    print(f"  Debug:        {DEBUG_DIR}")
    print(f"  Models:       {MODELS_DIR}")
    print(f"  Test Images:  {TEST_IMAGE_DIR}")
