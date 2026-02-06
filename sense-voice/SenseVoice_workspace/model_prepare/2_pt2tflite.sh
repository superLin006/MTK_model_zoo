#!/bin/bash
set -ex

# Convert TorchScript to TFLite
python3 pt2tflite.py \
    -i model/sensevoice_complete.pt \
    -o model/sensevoice_complete.tflite \
    --float 1
