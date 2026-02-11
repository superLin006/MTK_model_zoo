# Python 模型转换

将 Whisper 模型转换为 MTK NPU (MT8371) 可执行的 DLA 格式，共三步。

支持模型：**base**、**large-v3-turbo**（已验证）

## 转换流程

```
PyTorch (.pt) → TorchScript (.pt) → TFLite (.tflite) → DLA (.dla)
```

## 快速开始

### large-v3-turbo（当前使用）

```bash
cd python/

# Step 1: 导出 TorchScript + 嵌入权重
python step1_pt_to_torchscript.py --model large-v3-turbo --models-dir models_large_turbo

# Step 1 验证（可选，建议）
python test/test_pt.py --model large-v3-turbo --models-dir models_large_turbo

# Step 2: TorchScript → TFLite
python step2_torchscript_to_tflite.py --model large-v3-turbo \
    --d-model 1280 --n-layers 4 --n-mels 128 --models-dir models_large_turbo

# Step 3: TFLite → DLA
python step3_tflite_to_dla.py --model large-v3-turbo \
    --n-mels 128 --models-dir models_large_turbo
```

### base

```bash
# Step 1
python step1_pt_to_torchscript.py --model base --models-dir models

# Step 1 验证
python test/test_pt.py --model base --models-dir models

# Step 2
python step2_torchscript_to_tflite.py --model base \
    --d-model 512 --n-layers 6 --n-mels 80 --models-dir models

# Step 3
python step3_tflite_to_dla.py --model base \
    --n-mels 80 --models-dir models
```

## 模型参数对比

| 参数 | base | large-v3-turbo |
|------|------|----------------|
| `--d-model` | 512 | 1280 |
| `--n-layers` | 6 | 4 |
| `--n-mels` | 80 | 128 |
| vocab_size | 51865 | 51866 |
| initial tokens | `[SOT, lang, 50359, 50363]` | `[SOT, lang, 50360, 50364]` |

> **注意**：large-v3-turbo 使用 `<\|startoflm\|>`(50360) + `<\|0.00\|>`(50364) 作为解码起始序列，
> 与 base 的 `<\|transcribe\|>`(50359) + `<\|notimestamps\|>`(50363) 不同。

## 生成文件

### large-v3-turbo → `models_large_turbo/`

```
models_large_turbo/
├── encoder_large-v3-turbo_128x3000_MT8371.pt       (TorchScript, ~200 MB)
├── encoder_large-v3-turbo_128x3000_MT8371.tflite   (~2430 MB)
├── encoder_large-v3-turbo_128x3000_MT8371.dla       (1217 MB)
├── decoder_large-v3-turbo_448_MT8371.pt             (~654 MB)
├── decoder_large-v3-turbo_448_MT8371.tflite         (~654 MB)
├── decoder_large-v3-turbo_448_MT8371.dla            (327 MB)
├── token_embedding.npy                              (251 MB, 51866 × 1280)
├── position_embedding.npy                           (2.2 MB, 448 × 1280)
├── mel_128_filters.txt                              (25728 行，每行一个 float)
├── vocab.txt
├── model_config.json
└── embedding_info.json
```

### base → `models/`

```
models/
├── encoder_base_80x3000_MT8371.dla     (40 MB)
├── decoder_base_448_MT8371.dla         (100 MB)
├── token_embedding.npy                 (102 MB, 51865 × 512)
├── position_embedding.npy
├── mel_80_filters.txt                  (16080 行，每行一个 float)
└── vocab.txt
```

## Mel 滤波器文件说明

mel filter 文件格式：**每行一个 float**，行数 = n_mels × 201。

- `mel_80_filters.txt`：16080 行（80 × 201）
- `mel_128_filters.txt`：25728 行（128 × 201）

如需重新生成（例如文件损坏）：

```python
import whisper, torch
mel = whisper.audio.mel_filters(torch.device('cpu'), n_mels=128)  # 或 80
with open('models_large_turbo/mel_128_filters.txt', 'w') as f:
    for v in mel.numpy().flatten():
        f.write(f'{v:.18e}\n')
```

## 部署到 C++ 使用

生成 `.dla`、`.npy`、`.txt` 后，供 `cpp/deploy_and_test.sh` 自动推送到设备。
部署脚本默认读取 `models_large_turbo/`，切换 base 需修改脚本中的 `MODELS_DIR`。

## MTK 适配要点

| 问题 | 解决方案 |
|------|---------|
| GATHER 算子不支持 | 移除 `nn.Embedding`，token lookup 在 C++ 中完成 |
| 5D tensor 限制 | KV Cache 改为 4D `[num_layers, batch, seq_len, d_model]` |
| `tril` 不支持 | causal mask 预计算为 buffer |
| encoder 大模型 OOM | `relax_fp32=False`（encoder），`relax_fp32=True`（decoder） |

## 环境

- conda env：`MTK-whisper-kv`
- SDK：NeuroPilot SDK 8.0.10
- 目标平台：MT8371 (MDLA 5.3)

## test/test_pt.py 修改说明

相较于原始版本，`test_pt.py` 做了以下修正，以支持 large-v3-turbo 正确验证：

**1. initial_tokens 按模型区分**

```python
# vocab_size > 51000 → large-v3-turbo
if self.dims.n_vocab > 51000:
    initial_tokens = [50258, 50259, 50360, 50364]  # SOT, en, SOT_LM, TIMESTAMP_BEGIN
else:
    initial_tokens = [50258, 50259, 50359, 50363]  # SOT, en, TRANSCRIBE, NO_TIMESTAMPS
```

原因：large-v3-turbo 使用 `<|startoflm|>`(50360) + `<|0.00|>`(50364) 作为解码起始序列，
与 base/small 的 `<|transcribe|>`(50359) + `<|notimestamps|>`(50363) 不同。
使用错误的 initial_tokens 会导致解码器陷入循环，无法生成正常文本。

**2. EOT 检测硬编码**

```python
EOT_TOKEN = 50257
if next_token == EOT_TOKEN or next_token >= 50257:
    break
```

原因：`baseline_model.tokenizer` 属性不存在，无法动态获取 EOT。
50257 是所有多语言 Whisper 模型的 EOT token，`>= 50257` 同时过滤其他特殊 token。

**3. 文本相似度宽松比较**

```python
import re
def normalize(s):
    return re.sub(r'[^\w\s]', '', s.lower()).strip()
normalized_match = normalize(baseline_text) == normalize(torchscript_text)
passed = text_match or normalized_match or similarity > 0.90
```

原因：large-v3-turbo 输出带标点和大写，base 输出为小写无标点，直接字符串比较会误报失败。
归一化后比较或相似度 > 90% 均视为通过。

**4. n_mels 从 model_config.json 自动读取**

```python
config_path = models_dir / "model_config.json"
if config_path.exists():
    n_mels = json.load(open(config_path)).get("n_mels", 80)
```

原因：避免硬编码 80，large-v3-turbo 需要 128。
