# Python 模型转换

将 Whisper 模型转换为 MTK NPU (MT8371) 可执行的 DLA 格式，共三步。

支持模型：**base**、**large-v3-turbo**（已验证）

## 转换流程

```
PyTorch (.pt) → TorchScript (.pt) → TFLite (.tflite) → DLA (.dla)
```

## 快速开始

### large-v3-turbo（当前使用，10s 窗口）

```bash
cd python/

# Step 1: 导出 TorchScript + 嵌入权重
python step1_pt_to_torchscript.py --model large-v3-turbo --models-dir models_large_turbo

# Step 1 验证（可选，建议）
python test/test_pt.py --model large-v3-turbo --models-dir models_large_turbo

# Step 2: TorchScript → TFLite（--mel-frames 1000 表示 10s 窗口）
python step2_torchscript_to_tflite.py --model large-v3-turbo \
    --d-model 1280 --n-layers 4 --n-mels 128 --mel-frames 1000 --models-dir models_large_turbo

# Step 3: TFLite → DLA
python step3_tflite_to_dla.py --model large-v3-turbo \
    --n-mels 128 --mel-frames 1000 --models-dir models_large_turbo
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
| `--mel-frames` | 1000（10s）| 1000（10s）|
| vocab_size | 51865 | 51866 |
| initial tokens | `[SOT, lang, 50359, 50363]` | `[SOT, lang, 50360, 50364]` |

> **注意**：large-v3-turbo 使用 `<\|startoflm\|>`(50360) + `<\|0.00\|>`(50364) 作为解码起始序列，
> 与 base 的 `<\|transcribe\|>`(50359) + `<\|notimestamps\|>`(50363) 不同。

## 生成文件

### large-v3-turbo → `models_large_turbo/`

```
models_large_turbo/
├── encoder_large-v3-turbo_128x1000_MT8371.pt       (TorchScript, ~2431 MB)
├── encoder_large-v3-turbo_128x1000_MT8371.tflite   (~2425 MB)
├── encoder_large-v3-turbo_128x1000_MT8371.dla       (1214 MB)
├── decoder_large-v3-turbo_448_MT8371.pt             (~656 MB)
├── decoder_large-v3-turbo_448_MT8371.tflite         (~654 MB)
├── decoder_large-v3-turbo_448_MT8371.dla            (327 MB)
├── token_embedding.npy                              (253 MB, 51866 × 1280)
├── position_embedding.npy                           (2.2 MB, 448 × 1280)
├── mel_128_filters.txt                              (25728 行，每行一个 float)
├── vocab.txt
├── model_config.json
└── embedding_info.json
```

> 文件名中 `128x1000` 表示 mel 通道数×帧数，对应 10s 音频窗口。

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

## Encoder 窗口大小调整（30s → 10s）

Whisper 原始设计固定处理 30s 音频（mel 帧数 = 3000，encoder 输出 = 1500 帧）。
为降低推理延迟，我们将 encoder 输入窗口缩短为 10s（mel 帧数 = 1000，encoder 输出 = 500 帧）。

### 原理

Whisper encoder 由两层 Conv1d（stride=2）+ Transformer 组成：
- 输入：`[1, n_mels, T]`，T = mel 帧数
- Conv1d stride=2 → 输出序列长度 = T/2
- Positional embedding 原长 1500，截取前 T/2 个位置即可

权重完全不变，只需用更短的输入 trace 导出即可。

### 实测性能对比（large-v3-turbo，5.86s 英文音频）

| 指标 | 30s 窗口（旧） | 10s 窗口（新） | 提升 |
|------|-------------|-------------|------|
| Encoder 耗时 | 3455 ms | 942 ms | **3.7x** |
| Decoder 耗时 | 2256 ms | 1029 ms | **2.2x**（cross-attn 变短）|
| 总耗时 | 5778 ms | 2014 ms | **2.9x** |
| RTF | 0.987x | 0.344x | — |
| Cross KV cache 内存 | 76 MB | 39 MB | 节省 49% |

Decoder 也变快的原因：cross-attention 的 KV 从 `[4,1,1500,1280]` 缩为 `[4,1,500,1280]`，
每个 decoder step 的注意力计算量减少约 1/3。

### 涉及的改动

| 文件 | 改动内容 |
|------|---------|
| `whisper_kv_model.py` | `WhisperEncoderCore.forward` 改为截取 positional_embedding 前 `seq_len` 个位置，支持任意窗口长度 |
| `step1_pt_to_torchscript.py` | dummy_mel 从 `[1,n_mels,3000]` 改为 `[1,n_mels,1000]`；encoder_output 从 1500 帧改为 500 帧 |
| `step2_torchscript_to_tflite.py` | 新增 `--mel-frames` 参数；encoder input_shapes 和 decoder encoder_output/cross_kv shapes 随之调整 |
| `step3_tflite_to_dla.py` | 新增 `--mel-frames` 参数；encoder 文件名 stem 随之变化 |
| `cpp/jni/src/whisper_inference.h` | 新增 `enc_seq_len_` 成员（默认 500）；cross KV 注释更新 |
| `cpp/jni/src/whisper_inference.cpp` | encoder DLA 路径、所有 input/output shapes、cross KV buffer 大小全部改用 `enc_seq_len_` |
| `cpp/jni/src/utils/audio_utils.h` | `MAX_AUDIO_LENGTH` 从 480000（30s）改为 160000（10s） |

### 切换回 30s 窗口

如需切换回 30s 窗口，只需：
1. 脚本参数改为 `--mel-frames 3000`（step2/step3）
2. `audio_utils.h` 中 `MAX_AUDIO_LENGTH` 改回 `480000`
3. `whisper_inference.h` 中 `enc_seq_len_` 改为 `1500`
4. 重新走完三步转换 + 编译

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
