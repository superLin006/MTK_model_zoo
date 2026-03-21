# Moonshine Streaming Small - Android 测试包

Moonshine Streaming Small 英语语音识别，MTK MT8371 NPU推理。
固定输入窗口：10s（不足10s自动补零）。

## 环境要求

- Android ARM64 设备（API ≥ 29），芯片 MT8371
- 已开启 USB 调试
- 设备上已存在 `/data/local/tmp/zipformer_mtk_test/`（提供 MTK 运行时库）

## 快速开始

```bash
# 推送测试包
adb shell "rm -rf /data/local/tmp/moonshine_test"
adb push . /data/local/tmp/moonshine_test
adb shell "chmod +x /data/local/tmp/moonshine_test/bin/moonshine_test"
adb shell "chmod +x /data/local/tmp/moonshine_test/test_data/run_test.sh"

# 运行（自动测试 test_data/ 下所有 .wav）
adb shell "cd /data/local/tmp/moonshine_test && sh test_data/run_test.sh"

# 指定音频文件
adb shell "cd /data/local/tmp/moonshine_test && sh test_data/run_test.sh /path/to/audio.wav"
```

## 目录结构

```
deliver/
├── bin/
│   └── moonshine_test          # ARM64 可执行文件
├── lib/
│   └── libc++_shared.so        # C++ 运行时
├── models/
│   ├── moonshine_encoder.dla   # Encoder NPU模型（输入[1,2000,80] → 输出[1,500,620]）
│   ├── moonshine_decoder.dla   # Decoder NPU模型（自回归，每步一次）
│   ├── embed_tokens.npy        # Token embedding权重 [32768, 512]
│   ├── pos_emb_weight.npy      # 位置编码权重 [4096, 620]
│   ├── proj_weight.npy         # Encoder→Decoder adapter投影 [512, 620]
│   ├── log_k.npy               # AsinhCompression参数
│   └── vocab.txt               # 词表（32000词条）
└── test_data/
    ├── run_test.sh             # 测试脚本
    └── test_en.wav             # 示例音频（5.86s英语）
```

## 性能指标（MT8371，5.86s音频）

| 指标 | 数值 |
|------|------|
| Init Time | ~190 ms |
| Encoder | ~115 ms |
| Decoder（25 tokens） | ~395 ms |
| Inference Time | ~660 ms |
| RTF | 0.11x |
| Peak RSS | ~253 MB |

## 命令行用法

```
moonshine_test <encoder.dla> <decoder.dla> <embed_tokens.npy>
               <pos_emb.npy> <proj_weight.npy> <log_k.npy>
               <vocab.txt> <audio.wav>
```
