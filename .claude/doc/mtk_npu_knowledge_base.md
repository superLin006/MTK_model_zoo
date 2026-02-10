# MTK NPU 知识库 - 已知问题与最佳实践

本文档记录 MTK NPU 模型移植过程中的所有已知问题、解决方案和最佳实践。

**重要：Claude Code 在生成代码时会自动参考此文档！**

---

## 📋 目录

1. [平台限制](#平台限制)
2. [不支持的算子](#不支持的算子)
3. [Tensor 形状限制](#tensor-形状限制)
4. [常见陷阱](#常见陷阱)
5. [最佳实践](#最佳实践)
6. [参考实现](#参考实现)

---

## 🚫 平台限制

### MT8371 特定限制

#### 1. 不支持 5D Tensor

**问题：**
```python
# ❌ 错误：MT8371 不支持 5D tensor
past_key: [num_layers, batch, num_heads, seq_len, head_dim]  # 5D
```

**解决方案：**
```python
# ✅ 正确：重新设计为 4D tensor
past_key: [num_layers, batch, seq_len, d_model]  # 4D
# 其中 d_model = num_heads * head_dim
```

**来源：** Helsinki 项目 (PORTING_NOTES.md - 问题1)

**适用场景：** 所有使用 KV Cache 的 Transformer 模型

---

## 🔗 算子支持参考

**完整算子列表**: `/home/xh/projects/MTK_models_zoo/.claude/doc/mtk_mdla_operators.md`

查看完整支持列表：
```bash
cat /home/xh/projects/MTK_models_zoo/.claude/doc/mtk_mdla_operators.md
```

---

## ❌ 不支持的算子

### 1. GATHER 算子

**问题：**
```python
# ❌ Embedding 层使用 GATHER 算子
embedding = nn.Embedding(vocab_size, d_model)
output = embedding(token_ids)  # GATHER 操作
```

**解决方案：**
```python
# ✅ 方案A：导出 embedding weights，CPU 端查找
# Python 端：
torch.save(model.embedding.weight, 'embedding_weights.bin')

# C++ 端：
void embed_tokens(const int64_t* token_ids, int seq_len, float* output) {
    for (int i = 0; i < seq_len; i++) {
        int64_t token_id = token_ids[i];
        const float* src = embedding_weights_.data() + token_id * d_model_;
        memcpy(output + i * d_model_, src, d_model_ * sizeof(float));
    }
}
```

**来源：** Helsinki 项目 (PORTING_NOTES.md - 问题2)

**适用场景：** 所有 NLP 模型（Transformer、BERT、GPT 等）

---

### 2. masked_fill 算子

**问题：**
```python
# ❌ 不支持 masked_fill
attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
```

**解决方案：**
```python
# ✅ 使用加法代替
# 将 mask 从 0/1 改为 0/-1e9
mask = torch.zeros(seq_len, seq_len)
mask[mask == 0] = -1e9  # 无效位置

attn_weights = attn_weights + mask  # 直接相加
```

**来源：** Helsinki 项目 (PORTING_NOTES.md - 问题3)

**适用场景：** 所有使用 attention mask 的模型

---

### 3. tril 算子

**问题：**
```python
# ❌ 不支持 torch.tril
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
```

**解决方案：**
```python
# ✅ 预计算 causal mask，注册为 buffer
def __init__(self):
    # 预计算 causal mask
    causal_mask = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        for j in range(i + 1, max_seq_len):
            causal_mask[i, j] = -1e9

    # 注册为 buffer（会被序列化）
    self.register_buffer('causal_mask', causal_mask)

def forward(self, x):
    seq_len = x.size(1)
    mask = self.causal_mask[:seq_len, :seq_len]
    attn_weights = attn_weights + mask
```

**来源：** Helsinki 项目 (PORTING_NOTES.md - 问题3)

---

## 📐 Tensor 形状限制

### 固定形状要求

**问题：**
MTK NPU 编译需要固定的输入形状，不支持动态形状。

**解决方案：**

1. **音频模型（ASR）**
```python
# 固定音频长度
fixed_audio_duration = 10  # 秒
fixed_frames = 166  # 对应 10 秒的帧数

# Padding 策略
if actual_frames < fixed_frames:
    # Pad 到固定长度
    padded = F.pad(features, (0, 0, 0, fixed_frames - actual_frames))
else:
    # 截断到固定长度
    padded = features[:, :fixed_frames, :]
```

2. **文本模型（NLP）**
```python
# 固定序列长度
max_seq_len = 64

# Encoder self-attention mask (处理 padding)
def create_encoder_mask(actual_len, max_len):
    mask = torch.zeros(1, 1, max_len, max_len)
    mask[:, :, :, actual_len:] = -1e9  # Mask padding positions
    return mask
```

**来源：** SenseVoice 和 Helsinki 项目

---

## ⚠️ 常见陷阱

### 1. Position Embedding 重复添加

**问题：**
```python
# Python 模型内部已添加 position embedding
class MTKEncoder(nn.Module):
    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds + self.embed_positions(...)  # 内部添加

# C++ 端又添加了一次
void embed_tokens(...) {
    memcpy(...);
    // ❌ 错误：重复添加 position embedding
    for (int j = 0; j < d_model_; j++) {
        dst[j] += position_embeddings_[i * d_model_ + j];
    }
}
```

**解决方案：**
```cpp
// ✅ C++ 端只做 token embedding，不添加 position
void embed_tokens(const int64_t* token_ids, int seq_len, float* output) {
    for (int i = 0; i < seq_len; i++) {
        const float* src = embedding_weights_.data() + token_id * d_model_;
        memcpy(output + i * d_model_, src, d_model_ * sizeof(float));
        // 不添加 position，模型内部会处理
    }
}
```

**来源：** Helsinki 项目 (PORTING_NOTES.md - 问题6)

---

### 2. final_logits_bias 缺失

**问题：**
HuggingFace 的 MarianMT 等模型在输出层有 `final_logits_bias`，容易遗漏。

```python
# ❌ 漏掉 bias
logits = self.lm_head(hidden_states)

# ✅ 正确实现
logits = self.lm_head(hidden_states) + self.final_logits_bias
```

**解决方案：**
```python
class MTKDecoder(nn.Module):
    def __init__(self):
        # 添加 final_logits_bias buffer
        self.register_buffer('final_logits_bias', torch.zeros(1, vocab_size))

    def forward(self, ...):
        logits = self.lm_head(hidden_states) + self.final_logits_bias
        return logits

# 加载权重时复制
mtk_decoder.final_logits_bias.copy_(hf_model.final_logits_bias)
```

**来源：** Helsinki 项目 (PORTING_NOTES.md - 问题4)

---

### 3. Encoder Padding 处理

**问题：**
Encoder 处理 padding tokens 时，如果不加 mask，padding 位置会参与 attention 计算，导致输出错误。

**解决方案：**
```python
# Python 端
def create_encoder_self_attn_mask(actual_src_len, src_seq_len):
    """
    Shape: [1, 1, src_seq_len, src_seq_len]
    有效位置 = 0, padding 位置 = -1e9
    """
    mask = torch.zeros(1, 1, src_seq_len, src_seq_len)
    mask[:, :, :, actual_src_len:] = -1e9  # padding columns
    return mask

# C++ 端
void create_encoder_self_attn_mask(int actual_src_len, float* output) {
    const float NEG_INF = -1e9f;
    for (int r = 0; r < src_seq_len_; r++) {
        for (int c = 0; c < src_seq_len_; c++) {
            output[r * src_seq_len_ + c] = (c < actual_src_len) ? 0.0f : NEG_INF;
        }
    }
}
```

**来源：** Helsinki 项目 (PORTING_NOTES.md - 问题5)

---

## ✅ 最佳实践

### 1. 模型结构设计原则

**职责分离：**
```
CPU 端：
  ✓ 音频/图像特征提取（Fbank, Mel-spectrogram）
  ✓ Tokenization / Embedding lookup
  ✓ 复杂的后处理逻辑（Beam Search, CTC 解码）
  ✓ 动态逻辑（条件判断、循环）

NPU 端：
  ✓ 矩阵运算（Linear, MatMul）
  ✓ 卷积操作
  ✓ Attention 机制
  ✓ 激活函数（ReLU, GELU, Softmax）
```

---

### 2. Attention Mask 设计

#### Encoder Self-Attention Mask
```python
# Shape: [batch, 1, src_len, src_len]
# 用于屏蔽 padding positions
mask[:, :, :, actual_len:] = -1e9
```

#### Decoder Self-Attention Mask (with KV Cache)
```python
# Shape: [batch, 1, 1, cache_len + 1]
# 当前 query 只有 1 个 token
mask[:, :, :, :cache_len] = 0      # past cache 有效
mask[:, :, :, cache_len:-1] = -1e9  # 未使用的 cache 位置
mask[:, :, :, -1] = 0               # 当前 token 有效
```

#### Decoder Cross-Attention Mask
```python
# Shape: [batch, 1, 1, src_len]
# 屏蔽 encoder output 的 padding
mask[:, :, :, actual_src_len:] = -1e9
```

---

### 3. 数值验证流程

**逐层对比：**
```python
def validate_model(pytorch_model, mtk_model, test_input):
    """逐层对比输出，确保数值一致"""

    # 1. 对比 encoder output
    pt_encoder_out = pytorch_model.encoder(test_input)
    mtk_encoder_out = mtk_model.encoder(test_input)
    diff = (pt_encoder_out - mtk_encoder_out).abs().max()
    assert diff < 1e-4, f"Encoder diff: {diff}"

    # 2. 对比 decoder layers
    for i, (pt_layer, mtk_layer) in enumerate(
        zip(pytorch_model.decoder.layers, mtk_model.decoder.layers)
    ):
        pt_out = pt_layer(...)
        mtk_out = mtk_layer(...)
        diff = (pt_out - mtk_out).abs().max()
        assert diff < 1e-4, f"Layer {i} diff: {diff}"

    # 3. 对比 final output
    pt_logits = pytorch_model(test_input)
    mtk_logits = mtk_model(test_input)
    diff = (pt_logits - mtk_logits).abs().max()
    assert diff < 1e-3, f"Final diff: {diff}"
```

---

### 4. C++ 端实现检查清单

**必须验证的点：**
- [ ] Embedding 是否正确（不要重复添加 position）
- [ ] Attention mask 的 shape 和值是否正确
- [ ] KV Cache 的拼接逻辑是否正确
- [ ] 内存是否正确释放（无泄漏）
- [ ] 数值是否与 Python 端一致（固定输入测试）

---

## 📚 参考实现

### SenseVoice (ASR - Encoder Only)

**优点：**
- 音频预处理管道（kaldi-native-fbank）
- CTC 解码实现
- 固定长度音频处理

**关键文件：**
- `torch_model.py` - 自定义模型结构
- `test_converted_models.py` - 验证脚本
- `sensevoice/sensevoice.cc` - C++ 推理实现

**适用场景：**
- ASR 模型（Whisper Encoder, Wav2Vec2 等）
- Encoder-only 架构
- 需要音频特征提取

---

### Helsinki (Translation - Encoder-Decoder)

**优点：**
- 4D KV Cache 实现（避免 5D 限制）
- Encoder-Decoder 协同
- Embedding CPU 端处理
- 完整的 Attention Mask 设计

**关键文件：**
- `mtk_model.py` - 自定义模型结构（4D KV Cache）
- `PORTING_NOTES.md` - 详细的问题记录
- `helsinki/helsinki.cc` - C++ 推理实现

**适用场景：**
- 翻译模型（M2M100, NLLB 等）
- Encoder-Decoder 架构
- 需要 KV Cache 的生成模型

---

## 🔄 知识库更新流程

每次遇到新问题时，请按以下格式添加：

```markdown
### X. 新问题标题

**问题：**
描述问题现象和错误信息

**解决方案：**
给出具体的代码示例

**来源：** 项目名称 (日期)

**适用场景：** 哪些模型会遇到这个问题
```

---

## 📝 版本历史

- **v1.0** (2026-01-19): 初始版本，整合 Helsinki 和 SenseVoice 经验
- 后续更新：每次新项目完成后，添加新的经验

---

**最后更新**: 2026-01-19
**维护者**: 算法工程师 + Claude Code
