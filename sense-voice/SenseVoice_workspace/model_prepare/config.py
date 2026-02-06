# SenseVoice 模型配置

# PYTORCH 模式开关
# 0 = 导出模式 (用于 TFLite 转换和验证)
# 1 = 原生模式 (用于 PyTorch 推理和基准测试)
PYTORCH = 0

# 说明:
# - PYTORCH=0 时:
#   * SAVE_PT 模式: 导出 TorchScript 模型（移除不兼容的输出）
#   * CHECK_TFLITE 模式: 验证 TFLite 模型
#
# - PYTORCH=1 时:
#   * PYTORCH 模式: 运行原生 PyTorch 模型，保存完整输出用于对比
#   * 保存所有中间层输出
#
# 验证流程:
# 1. PYTORCH=0: python3 main.py --mode=SAVE_PT
# 2. PYTORCH=0: python3 pt2tflite.py ...
# 3. PYTORCH=1: python3 main.py --mode=PYTORCH      # 保存基准
# 4. PYTORCH=0: python3 main.py --mode=CHECK_TFLITE # 验证 TFLite
# 5. python3 compare_outputs.py                     # 对比输出
