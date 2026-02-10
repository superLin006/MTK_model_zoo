# Python 模型转换

将 Whisper Base 转换为 MTK NPU (MT8371) 可执行的 DLA 格式，共三步。

## 转换流程

```
PyTorch (.pt) → TorchScript (.pt) → TFLite (.tflite) → DLA (.dla)
```

```bash
python step1_pt_to_torchscript.py   # 导出 TorchScript
python step2_torchscript_to_tflite.py  # 转换 TFLite
python step3_tflite_to_dla.py       # 编译 DLA
```

步骤 1 完成后建议先验证：

```bash
python test/test_pt.py
```

## 生成文件

```
models/
├── encoder_base_80x3000_MT8371.pt       (79 MB)
├── encoder_base_80x3000_MT8371.tflite   (79 MB)
├── encoder_base_80x3000_MT8371.dla      (40 MB)
├── decoder_base_448_MT8371.pt           (199 MB)
├── decoder_base_448_MT8371.tflite       (198 MB)
├── decoder_base_448_MT8371.dla          (100 MB)
├── token_embedding.npy                  (102 MB)
└── embedding_info.json
```

生成后将 `.dla` 和 `.npy` 复制到 `mtk/models/` 供 C++ 使用：

```bash
cp models/encoder_base_80x3000_MT8371.dla ../models/
cp models/decoder_base_448_MT8371.dla ../models/
cp models/token_embedding.npy ../models/
cp models/position_embedding.npy ../models/
```

## 模型说明

**Encoder**
- 输入：梅尔频谱 `[1, 80, 3000]`（80 通道 × 3000 帧 = 30s）
- 输出：特征 `[1, 1500, 512]`

**Decoder（KV Cache 模式，每步单 token）**
- 主要输入：`token_embeddings [1,1,512]`、`encoder_output [1,1500,512]`、`past_self_keys/values [6,1,448,512]`
- 输出：`logits [1,1,51865]`、`new_self_keys/values [6,1,1,512]`

### MTK 适配要点

| 问题 | 解决方案 |
|------|---------|
| GATHER 算子不支持 | 移除 `nn.Embedding`，token lookup 在 C++ 中完成 |
| 5D tensor 限制 | KV Cache 改为 4D `[num_layers, batch, seq_len, d_model]` |
| `tril` 不支持 | causal mask 预计算为 buffer |

## 环境

- conda env：`MTK-whisper-kv`
- SDK：NeuroPilot SDK 8.0.10
- 目标平台：MT8371 (MDLA 5.3)
