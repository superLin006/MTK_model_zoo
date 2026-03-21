# Baseline 测试结果

## 模型信息
- 模型: Moonshine Streaming Small
- 模型路径: /home/xh/projects/MTK_models_zoo/moonshine/mtk/models/moonshine-streaming-small/
- 模型类型: Encoder-Decoder ASR (streaming)
- 固定形状版本: **10s** (T_AUDIO_FIXED=160000, NUM_FRAMES=2000, T_ENC=500, MAX_DEC_LEN=128)
- Encoder输入shape: [1, 2000, 80] (MTK NPU 固定输入，已经过CPU预处理)
- Encoder输出shape: [1, 500, 620] (10s window → 500 frames, hidden_size=620)
- Decoder: autoregressive, vocab_size=32768, max_dec_len=128
- 权重文件: model.safetensors

## 测试数据
- 测试文件: test_en.wav
- 音频时长: 5.61s
- 采样率: 16000Hz
- 采样点数: 89784 (padded to 89840)

## 测试结果
- 转录文本: "I think it's a very important thing to do. I think it's a very important thing to do."
- 生成tokens数: 26 (含BOS/EOS)
- 推理时间: ~745-1054ms (CPU, 两次运行)
- Encoder输出shape: [1, 281, 620]

## 环境信息
- Conda环境: MTK-moonShine
- Python版本: 3.10
- PyTorch版本: 2.6.0+cu124
- Transformers版本: 5.3.0.dev0
- soundfile: 0.13.1
- safetensors: 0.7.0

## 架构要点（供后续转换参考）

### 模型层级结构
```
MoonshineStreamingForConditionalGeneration
├── model
│   ├── encoder   ← model.model.encoder
│   └── decoder   ← model.model.decoder
└── proj_out
```

### Audio Frontend
- 输入: raw waveform float32, 16kHz
- 预处理: AutoProcessor (processor) 将音频转为 input_values tensor
- 输入长度会被 pad 到 16 的倍数
- 50Hz features: encoder内部做 2x causal stride-2 conv，输出帧率约 50fps

### Encoder
- 输入: [B, T_audio] raw waveform
- 输出: [B, T_enc, 620], T_enc ≈ T_audio / (16000/50) = T_audio * 50 / 16000
- sliding-window attention (无绝对位置编码)
- num_hidden_layers: 10, hidden_size: 620

### Decoder
- 输入: encoder输出 + 已生成token序列
- autoregressive, causal Transformer with RoPE
- num_hidden_layers: 10, hidden_size: 512
- vocab_size: 32768

### 分离策略（后续NPU转换）
- Encoder: 单独导出NPU (输入 raw waveform, 固定长度)
- Decoder: 单独导出NPU (每步一个token, KV-cache)
- 前后处理: CPU

## 遇到的问题
1. `model.encoder` 不存在 → 正确路径为 `model.model.encoder`
   MoonshineStreamingForConditionalGeneration 将 encoder/decoder 包裹在 `model.model` 内

## 保存的文件
- `python/test/outputs/baseline/test_en.json` - 完整推理结果
- `python/test/outputs/baseline/test_en.txt` - 转录文本
- `python/test/outputs/debug/preprocessed_audio.npy` - 预处理后音频 shape=(1, 89840)
- `python/test/outputs/debug/encoder_output.npy` - encoder输出 shape=(1, 281, 620)

## MTK NPU 转换后文件 (10s 版本)

| 文件 | 大小 | 说明 |
|------|------|------|
| models/moonshine_encoder.pt | 205.2 MB | TorchScript, 输入 [1,2000,80] |
| models/moonshine_decoder.pt | 264.6 MB | TorchScript, max_dec_len=128 |
| models/moonshine_encoder.tflite | 197.4 MB | TFLite, 输入 [1,2000,80] |
| models/moonshine_decoder.tflite | 264.5 MB | TFLite |
| models/moonshine_encoder.dla | 106.5 MB | DLA (mdla5.3,edma3.6) |
| models/moonshine_decoder.dla | 132.8 MB | DLA (mdla5.3,edma3.6) |

### Debug npy shapes (10s 版本)
- `preprocessed_frames.npy`: shape=(1, 2000, 80)
- `encoder_output.npy`: shape=(1, 500, 620)
- `decoder_first_logits.npy`: shape=(1, 1, 32768)
