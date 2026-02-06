#!/bin/bash
set -ex

# Save SenseVoice encoder to TorchScript
python3 main.py --mode="SAVE_PT" \
    --model_path="../models/sensevoice-small" \
    --audio_path="../audios/test_en.wav"
