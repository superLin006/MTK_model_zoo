# 算子兼容性分析 - Moonshine Streaming Small

**平台**: MT8371 (MDLA 5.3)
**模型**: Moonshine Streaming Small (Encoder-Decoder ASR)
**分析日期**: 2026-03-21
**参考SDK**: NeuroPilot SDK 8.0.10

---

## 确认的模型参数（moonshine-streaming-small）

| 组件 | 参数 |
|------|------|
| Encoder hidden_size | 620 |
| Encoder num_layers | 10 |
| Encoder num_attention_heads | 8 |
| Encoder head_dim | 64 (620/8 向下取整, 注意: 620/8=77.5, 实际head_dim=64) |
| Encoder intermediate_size | 2480 |
| Encoder sliding_windows | [[16,4],[16,4],[16,0],[16,0],[16,0],[16,0],[16,0],[16,0],[16,4],[16,4]] |
| Encoder hidden_act | gelu |
| Decoder hidden_size | 512 |
| Decoder num_layers | 10 |
| Decoder num_attention_heads | 8 |
| Decoder head_dim | 64 |
| Decoder intermediate_size | 2048 |
| Decoder vocab_size | 32768 |
| Decoder hidden_act | silu |
| Decoder RoPE | partial_rotary_factor=0.5, theta=10000 |
| Encoder输出 → Decoder投影 | Linear(620 → 512) |
| 位置编码（Encoder侧跨模态） | pos_emb: nn.Embedding(4096, 620) 加到 encoder_out |

---

## 架构拆分策略

```
[CPU端]
  原始音频 → AutoProcessor → input_values [1, T_audio]

[NPU: Encoder]
  输入: input_values [1, T_fixed]   (固定长度, 例如 T_fixed = 89840 for ~5.6s)
  输出: encoder_out [1, T_enc, 620]  (T_enc = T_fixed // 80 // 4 ≈ T_fixed * 50 / 16000)

[CPU端]
  embed_tokens(input_ids) → decoder_embed [1, 1, 512]
  RoPE cos/sin 预计算 → 查表传入
  causal_mask 预计算 → 传入
  encoder_attn_mask 预计算 → 传入
  KV cache 管理（数组写入/读出）

[NPU: Decoder（单步）]
  输入:
    decoder_embed: [1, 1, 512]
    encoder_out: [1, T_enc, 512]  (经 proj 投影后)
    past_keys: [10, 1, max_dec_len, 512]
    past_values: [10, 1, max_dec_len, 512]
    pos_embed (cos/sin): [1, 1, 32] (partial RoPE, 32 = 64 * 0.5)
    attn_mask: [1, 1, 1, max_dec_len+1]
    encoder_attn_mask: [1, 1, 1, T_enc]
  输出:
    logits: [1, 1, 32768]
    new_keys: [10, 1, 1, 512]
    new_values: [10, 1, 1, 512]

[CPU端]
  argmax(logits) → next token
  更新 KV cache
  重复直到 EOS
```

---

## 不支持 / 有限制的算子

| 算子 | 使用位置 | 问题 | 替换方案 |
|------|---------|------|---------|
| **GATHER** | Decoder: `embed_tokens(input_ids)` | GATHER 不在支持列表中 | CPU 端 embedding 查表，仅传 embed 结果给 NPU |
| **GATHER** | Decoder: `pos_emb(arange(...))` — 给 encoder_out 加位置编码的 nn.Embedding | GATHER 不支持 | 预计算位置编码 tensor，作为 buffer 或 CPU 端常数直接加，或从 NPU 侧直接通过 ADD 接收预计算的 pos_embed 张量 |
| **动态 arange/position_ids** | Decoder: `torch.arange(...)`, `position_ids = ...` | 动态计算不可 trace | CPU 端预计算 position_ids，仅传入 cos/sin 结果 |
| **create_causal_mask（tril/逻辑运算）** | Decoder: `create_causal_mask(...)` 动态构建 | 含 GREATER/LOGICAL_AND 等逻辑算子 | 预计算 causal mask，注册为 buffer，切片传入 |
| **create_bidirectional_mask（逻辑运算）** | Encoder: 每层 sliding_window mask | 含逻辑比较 dist >= 0, dist < N | 预计算所有层的 sliding window mask（固定长度），注册为 buffer |
| **masked_fill** | `shift_tokens_right` 中的 `masked_fill_` | 含 EQUAL 逻辑 | 此处只在训练用，推理不触发 |
| **dynamic RoPE** (`inv_freq @ position_ids`) | Decoder: `MoonshineStreamingRotaryEmbedding.forward()` | 含 matmul + cat + cos + sin，动态 position_ids | 预计算所有位置的 cos/sin 表，CPU 按步骤查表传入 |
| **AsinhCompression** (`torch.asinh`) | Encoder Embedder | asinh 不在支持列表中 | 需展开为 LOG + SQRT 组合：`asinh(x) = log(x + sqrt(x^2 + 1))` — 但 LOG 也不支持，因此**必须移到 CPU 或用近似** |
| **LOG** | asinh 展开需要 | MDLA 5.3 不支持 LOG | 将 AsinhCompression 整体移到 CPU，或用 F.silu 近似，或 fuse 进 Linear 前归一化 |
| **conv1d mask 逻辑** | `MoonshineStreamingCausalConv1d.forward()` mask 分支 | 含 `conv1d(mask)` + `mask > 0`（GREATER） | 推理时不传 mask（mask=None），causal padding 已由 `left_pad` 保证；仅当 batch padding 时需要，推理单句无需 |
| **5D KV Cache** | Decoder 标准 HF DynamicCache | HF Cache 内部使用5D | 重写为 4D KV Cache（参考 Helsinki mtk_model.py）|
| **TOPK** | 推理解码 argmax/greedy | 如用 beam search 则需 TOPK | Greedy 解码用 argmax，CPU 端实现；beam search 全部 CPU 端 |
| **chunk（split）** | Decoder MLP: `fc1(x).chunk(2, dim=-1)` | SPLIT_V 支持，但 chunk 等价于 SPLIT，需验证 | 替换为显式 `torch.split(x, x.shape[-1]//2, dim=-1)` 或手动切片 `x[..., :N], x[..., N:]` 以确保静态 shape |

### 关键说明：AsinhCompression 问题

`MoonshineStreamingAsinhCompression` 使用了：
```python
torch.asinh(torch.exp(self.log_k) * x)
```
- `torch.exp` → EXP，MDLA 支持
- `torch.asinh` → 等价于 `log(x + sqrt(x^2+1))`，需要 LOG，MDLA **不支持 LOG**

**因此 AsinhCompression 必须移到 CPU 端**，作为 Encoder 前处理的一部分。

### Encoder CMVN + AsinhComp + Linear 整体前处理建议

以下三步可全部在 CPU 端完成，得到 `hidden_states` 后再送入 NPU：
1. `MoonshineStreamingFrameCMVN`（均值/标准差归一化，用 MEAN+SUB+POW+MEAN+ADD+SQRT+DIV 实现，MDLA 支持）
   - **但 CMVN 包含多个 reduce 操作，且 frame 维度是动态的，建议整体 CPU 处理**
2. `MoonshineStreamingAsinhCompression`（含 LOG，CPU）
3. `nn.Linear(frame_len, hidden_size, bias=False)` + `silu`（FULLY_CONNECTED + SiLU，MDLA 支持）
   - 可移入 NPU

**实际建议**: 将 `MoonshineStreamingEncoderEmbedder` 中 CMVN 和 Asinh 步骤在 CPU 完成，Linear(frame_len→620) + SiLU 开始进 NPU，之后 conv1 + conv2 + Transformer layers 全部在 NPU。

---

## 模型修改方案

### Encoder 修改

#### 拆分边界
```
CPU：
  input_values [1, T] → reshape [B, T//frame_len, frame_len]
  → CMVN (均值中心化 + RMS 归一化)
  → AsinhCompression (k * x → asinh)
  → 结果: [B, num_frames, frame_len]  → 传入 NPU

NPU (MTKEncoder):
  输入: x_frames [1, num_frames, frame_len]
  → Linear(frame_len=80, 620) + SiLU  → [1, num_frames, 620]
  → transpose → [1, 620, num_frames]
  → CausalConv1d(stride=2): [1, 1240, num_frames//2]  (causal pad=left_pad)
  → SiLU
  → CausalConv1d(stride=2): [1, 620, num_frames//4]
  → transpose → [1, num_frames//4, 620]
  → 10x EncoderLayer (self-attn + MLP，sliding window mask 预计算传入)
  → LayerNorm
  输出: [1, T_enc, 620]
```

#### Encoder 注意事项
1. **CausalConv1d 的 causal padding**：使用 `F.pad(x, (left_pad, 0))` 实现，只在左侧补零，MTK PAD 算子支持，可进 NPU。
2. **Sliding window mask**：每层有不同的 `(left=16, right=4)` 或 `(left=16, right=0)` 窗口。固定 T_enc 后，预计算 10 个 mask `[1, 1, T_enc, T_enc]`，传入或注册为 buffer。mask 用加法而非 masked_fill。
3. **LayerNorm 变体**：`MoonshineStreamingLayerNorm` = `nn.LayerNorm(elementwise_affine=False)` + `gamma * (normed + unit_offset=1.0)`。等价于标准 LayerNorm，可用 MEAN + SUB + SQUARED_DIFFERENCE + MEAN + RSQRT + MUL + MUL。MDLA 支持所有这些算子，但标准 LayerNorm（layer_norm op 本身）需验证是否作为 FULLY_CONNECTED 融合。安全起见，分解为 MEAN/SUB/RSQRT/MUL 链。
4. **GRU hidden_size 不整除 num_heads**：encoder hidden_size=620, num_heads=8, 620/8=77.5 不整除。实际 head_dim=64（从 config 读到），因此 `q_proj` 输出不是 620 而是 `num_heads * head_dim = 8 * 64 = 512`，剩余的 `620 - 512 = 108` 维不参与注意力（`v_proj` 输出同样是 512）。这是上游模型的设计，移植时按实际权重 shape 复制即可。

### Decoder 修改

#### 关键改动列表

1. **去除 embed_tokens（GATHER）**
   ```python
   # 原：inputs_embeds = self.embed_tokens(input_ids)
   # 改：embed_tokens 权重导出到 CPU，CPU 查表后传入 decoder_embed [1, 1, 512]
   ```

2. **去除动态 RoPE 计算，改为输入预计算 cos/sin**
   ```python
   # 原：position_embeddings = self.rotary_emb(hidden_states, position_ids)
   # 改：CPU 预计算全部位置的 cos/sin 表 [max_dec_len, 32]
   #     每步 decode 时，CPU 查表取出当前位置的 cos/sin [1, 1, 32]，作为输入传给 NPU
   ```
   partial_rotary_factor=0.5，head_dim=64，实际旋转维度 = 64 * 0.5 = 32。
   cos/sin shape: [1, 1, 32]（broadcast 到 [B, heads, 1, 64] 时 repeat_interleave）

3. **去除 create_causal_mask（含逻辑运算）**
   ```python
   # 改：预计算 max_dec_len x max_dec_len 的 causal mask，存为 buffer
   # 每步取 attn_mask [1, 1, 1, cache_len+1]，0=有效，-1e9=无效
   ```

4. **去除 encoder pos_emb + GATHER**
   ```python
   # 原：pos_emb = nn.Embedding(4096, 620); encoder_out += pos_emb(arange(T_enc))
   # 改：CPU 端预计算 pos_emb 权重表，按 T_enc 切片后作为 ADD 张量传入
   #     或：在 encoder 输出后、decoder 输入前，CPU 加上位置编码
   ```

5. **encoder proj（620→512 Linear）**
   ```python
   # decoder 内部有 self.proj = nn.Linear(620, 512)
   # 可以移入 NPU decoder 模型内部（FULLY_CONNECTED 支持）
   # 也可以 CPU 端做，只需一次矩阵乘法
   # 建议：移入 NPU，作为 decoder 的第一层
   ```

6. **4D KV Cache 设计（参考 Helsinki V2）**
   ```python
   # 输入:
   past_keys:   [10, 1, max_dec_len, 512]   # 4D
   past_values: [10, 1, max_dec_len, 512]   # 4D
   # 输出:
   new_keys:    [10, 1, 1, 512]   # 4D，每步 1 个新 K
   new_values:  [10, 1, 1, 512]   # 4D
   # C++ 端负责写入 KV cache 数组
   ```

7. **Decoder MLP (GLU 结构)**
   ```python
   # MoonshineStreamingDecoderMLP:
   # fc1: Linear(512, 2048*2=4096) → x, gate = chunk(2, -1)
   # output = act_fn(gate) * x → fc2: Linear(2048, 512)
   # 等价于 Gated Linear Unit (GLU with silu)
   # SPLIT_V + MUL + SiLU 均支持，可进 NPU
   # 注意: chunk 改为显式 split 保证静态 shape trace
   ```

8. **标准 LayerNorm（Decoder）**
   `nn.LayerNorm(config.hidden_size, bias=False)` — 标准 LayerNorm 无 bias，MDLA 内有 LayerNorm 相关融合，或拆分为 MEAN/SUB/RSQRT/MUL 链。

---

## 修改示例代码（关键片段）

### 1. Encoder 前处理（CPU 端）

```python
def preprocess_audio_cpu(input_values: np.ndarray, model) -> np.ndarray:
    """
    CPU 端完成 CMVN + AsinhCompression，输出 x_frames [1, num_frames, frame_len]
    frame_len = int(round(16000 * 5.0 / 1000)) = 80
    """
    frame_len = 80
    T = input_values.shape[-1]
    num_frames = T // frame_len
    x = input_values[:, :num_frames * frame_len].reshape(1, num_frames, frame_len)

    # CMVN
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    rms = np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    x_normed = centered / rms

    # AsinhCompression: k = exp(log_k), y = asinh(k * x)
    k = np.exp(model.embedder.comp.log_k.item())
    x_comp = np.arcsinh(k * x_normed)

    return x_comp.astype(np.float32)   # [1, num_frames, 80]
```

### 2. Decoder 中的 RoPE 预计算（CPU 端）

```python
def precompute_rope_table(config, max_len: int = 512) -> tuple:
    """
    返回 cos_table, sin_table，shape 均为 [max_len, 32]
    partial_rotary_factor=0.5, head_dim=64, rot_dim=32
    """
    rot_dim = int(config.head_dim * config.rope_parameters['partial_rotary_factor'])
    theta = config.rope_parameters['rope_theta']
    inv_freq = 1.0 / (theta ** (np.arange(0, rot_dim, 2) / rot_dim))

    positions = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)           # [max_len, rot_dim//2]
    emb = np.concatenate([freqs, freqs], axis=-1)   # [max_len, rot_dim]
    cos_table = np.cos(emb).astype(np.float32)      # [max_len, 32]
    sin_table = np.sin(emb).astype(np.float32)      # [max_len, 32]
    return cos_table, sin_table

# 使用：step i 时传入 cos_table[i:i+1] → [1, 32] → reshape 供 NPU 使用
```

### 3. Causal Mask 预计算（CPU 端，作为 buffer）

```python
def create_decoder_self_attn_mask(cache_len: int, max_dec_len: int) -> np.ndarray:
    """
    返回 [1, 1, 1, max_dec_len+1]，0=有效，-1e9=无效
    cache_len: 当前已缓存 token 数
    """
    mask = np.full((1, 1, 1, max_dec_len + 1), -1e9, dtype=np.float32)
    mask[:, :, :, :cache_len] = 0.0   # 历史 cache 有效
    mask[:, :, :, -1] = 0.0           # 当前 token 有效
    return mask
```

### 4. Sliding Window Mask（Encoder，作为模型 buffer）

```python
def create_sliding_window_mask(seq_len: int, left: int, right: int) -> torch.Tensor:
    """
    返回 [1, 1, seq_len, seq_len]，0=参与，-1e9=不参与
    left: 向左看 left 帧，right: 向右看 right 帧
    """
    mask = torch.full((seq_len, seq_len), -1e9)
    for q in range(seq_len):
        k_min = max(0, q - left + 1)
        k_max = min(seq_len, q + right + 1)
        mask[q, k_min:k_max] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
```

---

## 固定形状策略

### Encoder

| 参数 | 值 | 说明 |
|------|-----|------|
| 固定音频长度 | 89840 samples (~5.61s) | 与 baseline.md 一致；可设多档 |
| frame_len | 80 | 5ms @ 16kHz |
| num_frames（CPU前处理输出） | 1123 | 89840 // 80 |
| T_enc（NPU encoder输出） | 281 | 1123 → conv1(stride2) → 562 → conv2(stride2) → 281 (含 causal padding，精确值需测量) |

**建议**: 先固定一档 89840 samples（5.61s），后续可按需增加 160000 samples（10s）档。

### Decoder

| 参数 | 值 | 说明 |
|------|-----|------|
| 固定 encoder_out 长度 | 281 | 与 encoder 输出对应 |
| max_dec_len (KV cache) | 64 | 最大生成 token 数，可根据场景调整 |
| decoder 输入 seq_len | 1 | 每步只输入 1 个 token |
| past_keys shape | [10, 1, 64, 512] | 4D，10层，max_dec_len=64 |

---

## 分模型导出方案

```
moonshine_encoder_npu.tflite
  输入: x_frames [1, 1123, 80]   (CPU 完成 CMVN + Asinh 后的结果)
  输出: encoder_out [1, 281, 620]

moonshine_decoder_npu.tflite
  输入(共 5 + 10*2 = 25 inputs):
    decoder_embed:       [1, 1, 512]
    encoder_out:         [1, 281, 512]     (proj 620→512 在 decoder 内)
    pos_embed_enc:       [1, 281, 620]     (encoder pos_emb，可 CPU 加，不入 NPU)
    cos_input:           [1, 1, 32]
    sin_input:           [1, 1, 32]
    attn_mask:           [1, 1, 1, 65]    (max_dec_len+1)
    encoder_attn_mask:   [1, 1, 1, 281]
    past_keys:           [10, 1, 64, 512]
    past_values:         [10, 1, 64, 512]
  输出:
    logits:              [1, 1, 32768]
    new_keys:            [10, 1, 1, 512]
    new_values:          [10, 1, 1, 512]
```

**替代方案**：将每层 KV 分别作为独立 input/output（避免 4D [num_layers,...] 被视为大 tensor 导致内存问题），参考 sherpa-onnx 方案：per-layer 独立 KV 输入/输出。

---

## 风险评估

### 整体风险等级：**中等偏高**

| 风险点 | 等级 | 原因 |
|--------|------|------|
| AsinhCompression (torch.asinh / LOG) | **高** | LOG 不支持，必须移 CPU；改变了 Encoder 进 NPU 的边界 |
| Encoder head_dim 不整除 | **中** | 620 / 8 = 77.5，实际 head_dim=64，q_proj 输出 512 ≠ 620，需验证 attention 计算是否 match |
| Sliding Window Mask 动态生成 | **中** | 含逻辑运算，必须预计算并固定 T_enc |
| RoPE 动态计算 | **中** | 含 matmul+cos+sin+position_ids，需全部预计算为 lookup table |
| Decoder MLP (GLU/chunk) | **低** | chunk 等价于 SPLIT，支持，但需替换为显式 split 避免 trace 问题 |
| 5D KV Cache | **低** | 有成熟参考（Helsinki V2），改为 4D 方案成本可控 |
| vocab_size=32768 的 FULLY_CONNECTED | **低** | 最后 lm_head Linear(512, 32768)，数据量大但算子支持 |
| GATHER (embed_tokens, pos_emb) | **低** | 有成熟解决方案（CPU 查表），在 Helsinki 中已验证 |

### 关键验证点

1. **Encoder head_dim 验证**：实际运行 `model.model.encoder.layers[0].self_attn.q_proj.weight.shape` 确认是 `[512, 620]` 还是 `[620, 620]`
2. **conv1/conv2 输出帧数**：固定 T_audio 后，实测 T_enc 精确值（causal padding 影响计算）
3. **AsinhCompression 参数**：`log_k` 是可训练参数，需从权重文件中导出
4. **encoder_hidden_states 的 pos_emb**：`pos_emb = nn.Embedding(4096, 620)` 是 Decoder 内的参数（非 Encoder 内），加法发生在 decoder.forward() 内，处理方案：在 CPU 端或将 pos_emb 加法移入 NPU encoder 输出端或 decoder NPU 模型入口

---

## 参考项目对应关系

| 组件 | 参考项目 |
|------|---------|
| Encoder (Conv + Transformer) | SenseVoice (encoder-only ASR) |
| Decoder (KV Cache, 4D tensor) | Helsinki V2 (mtk_model.py) |
| Embedding CPU 查表 | Helsinki V2 |
| Attention Mask 设计 | Helsinki V2 (PORTING_NOTES.md) |
| RoPE 预计算 | Moonshine (原始) 类似 Llama RoPE |
| Audio Frontend CPU | SenseVoice (kaldi-native-fbank 前处理) |

---

## 下一步行动

1. 编写 `mtk_model.py`，实现 `MTKMoonshineEncoder` 和 `MTKMoonshineDecoder`
2. 编写 `convert.py`，导出 TorchScript → TFLite
3. 编写 `test_converted.py`，数值验证（逐层对比）
4. 编写 C++ 推理代码，参考 Helsinki `helsinki.cc`

---

**分析人**: operator-analyst v2.1 (Claude Sonnet 4.6)
**最后更新**: 2026-03-21
