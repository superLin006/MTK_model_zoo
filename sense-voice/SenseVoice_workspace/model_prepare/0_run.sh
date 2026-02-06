#!/bin/bash
set -ex

# SenseVoice-Small ASR test
# 使用 FunASR 提取特征进行测试
python3 test_converted_models.py \
    --audio="../audios/test_en.wav" \
    --language="auto"
