# Helsinki 模型 MTK NPU 移植问题记录

本文档记录了将 Helsinki (MarianMT) 翻译模型移植到 MediaTek MT8371 NPU 过程中遇到的关键问题及解决方案。

---

## 问题 1: MT8371 不支持 5D Tensor

### 问题描述
原始 KV Cache 设计使用 5D tensor：
```python
past_key: [num_layers, batch, num_heads, seq_len, head_dim]  # 5D
```

MT8371 的 MDLA 5.3 不支持 5D tensor，导致模型转换失败。

### 解决方案
将 KV Cache 重新设计为 4D tensor，合并 `num_heads` 和 `head_dim` 到 `d_model`：
```python
past_key: [num_layers, batch, seq_len, d_model]  # 4D, d_model = num_heads * head_dim
```

### 关键代码修改
```python
# mtk_model.py
class MTKDecoderKVCacheV2Fixed(nn.Module):
    # KV Cache shape: [num_layers, batch, max_cache_len, d_model]
    # 而不是: [num_layers, batch, num_heads, max_cache_len, head_dim]
```

---

## 问题 2: GATHER 算子不支持

### 问题描述
Embedding 层使用 GATHER 算子从词表中查找 token embedding，MT8371 NPU 不支持此算子。

### 解决方案
将 Embedding 操作放在 CPU 端执行：
1. 导出 embedding weights 为单独的 `.bin` 文件
2. C++ 端手动实现 embedding 查找

### 关键实现
```cpp
// helsinki.cc
void HelsinkiTranslator::embed_tokens(const int64_t* token_ids, int seq_len, float* output) {
    for (int i = 0; i < seq_len; i++) {
        int64_t token_id = token_ids[i];
        const float* src_embed = embedding_weights_.data() + token_id * d_model_;
        memcpy(output + i * d_model_, src_embed, d_model_ * sizeof(float));
    }
}
```

---

## 问题 3: masked_fill 和 tril 算子不支持

### 问题描述
Transformer 的 attention mask 通常使用 `masked_fill` 和 `tril` 算子：
```python
# 原始代码
attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
```

MTK 转换器不支持这些算子。

### 解决方案
1. **预计算 mask**：在模型初始化时预计算 causal mask，存储为 buffer
2. **使用加法代替 masked_fill**：将 mask 从 0/1 改为 0/-1e9，直接相加

### 关键代码修改
```python
# 预计算 causal mask
def __init__(self):
    # 创建 causal mask buffer: 0 表示有效，-1e9 表示无效
    causal_mask = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        for j in range(i + 1, max_seq_len):
            causal_mask[i, j] = -1e9
    self.register_buffer('causal_mask', causal_mask)

def forward(self, x, attn_mask=None):
    # 使用加法代替 masked_fill
    attn_weights = attn_weights + self.causal_mask[:seq_len, :seq_len]
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
```

---

## 问题 4: final_logits_bias 缺失导致输出错误

### 问题描述
移植后的模型输出 logits 与 HuggingFace baseline 存在固定偏差（约 8.7246），导致翻译结果错误。

### 排查过程
1. 逐层对比 encoder output → 完全匹配
2. 逐层对比 decoder layer output → 完全匹配
3. 对比最终 logits → 存在固定偏差

### 根因分析
HuggingFace 的 MarianMTModel 在 `lm_head` 输出后会加上 `final_logits_bias`：
```python
# HuggingFace 原始代码
lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
```

我们的移植版本漏掉了这个 bias。

### 解决方案
```python
# mtk_model.py
class MTKDecoderKVCacheV2Fixed(nn.Module):
    def __init__(self):
        # 添加 final_logits_bias buffer
        self.register_buffer('final_logits_bias', torch.zeros(1, config.vocab_size))

    def forward(self, ...):
        logits = self.lm_head(hidden_states) + self.final_logits_bias
        return logits, new_keys, new_values

# 加载权重时复制 bias
mtk_decoder.final_logits_bias.copy_(hf_model.final_logits_bias)
```

---

## 问题 5: Encoder Padding 导致输出错误

### 问题描述
输入句子长度不足 64 时需要 padding，但 encoder 没有正确处理 padding tokens，导致：
- 短句子翻译结果错误
- 不同长度的句子产生相同的错误输出

### 排查过程
```python
# 测试发现
inputs_no_pad = embed[:, :actual_len, :]  # 只用实际 tokens
inputs_with_pad = embed[:, :64, :]        # 包含 padding

encoder_out_no_pad = encoder(inputs_no_pad)  # 正确
encoder_out_with_pad = encoder(inputs_with_pad)  # 错误！

# 差异来自 self-attention 计算时 padding 位置参与了计算
```

### 解决方案
为 Encoder 添加 self-attention mask，屏蔽 padding 位置：

```python
# mtk_model.py - Encoder attention mask
def create_encoder_self_attn_mask(actual_src_len, src_seq_len):
    """
    Shape: [1, 1, src_seq_len, src_seq_len]
    有效位置 = 0, padding 位置 = -1e9
    """
    mask = torch.zeros(1, 1, src_seq_len, src_seq_len)
    mask[:, :, :, actual_src_len:] = -1e9  # padding columns
    return mask

# Encoder attention 中使用
attn_weights = attn_weights + attn_mask  # 加上 mask
attn_weights = F.softmax(attn_weights, dim=-1)
```

C++ 端对应实现：
```cpp
void HelsinkiTranslator::create_encoder_self_attn_mask(int actual_src_len, float* output) {
    const float NEG_INF = -1e9f;
    for (int r = 0; r < src_seq_len_; r++) {
        for (int c = 0; c < src_seq_len_; c++) {
            output[r * src_seq_len_ + c] = (c < actual_src_len) ? 0.0f : NEG_INF;
        }
    }
}
```

---

## 问题 6: Position Embedding 重复添加

### 问题描述
C++ 端翻译结果与 Python baseline 不一致，所有句子输出相同的错误 tokens。

### 排查过程
对比 Python 和 C++ 的数据流，发现 encoder input embedding 值不同。

### 根因分析
Encoder 模型内部已经添加了 position embedding：
```python
# mtk_model.py - MTKEncoderNoEmbed
def forward(self, inputs_embeds, attn_mask):
    hidden_states = inputs_embeds + self.embed_positions(...)  # 内部添加
```

但 C++ 端的 `embed_tokens()` 也添加了一次：
```cpp
// 错误代码
void embed_tokens(...) {
    memcpy(dst, src_embed, ...);
    // 错误：又加了一次 position embedding
    for (int j = 0; j < d_model_; j++) {
        dst[j] += position_embeddings_[i * d_model_ + j];
    }
}
```

### 解决方案
C++ 端的 `embed_tokens()` 只做纯 token embedding，不添加 position：
```cpp
void HelsinkiTranslator::embed_tokens(const int64_t* token_ids, int seq_len, float* output) {
    for (int i = 0; i < seq_len; i++) {
        const float* src_embed = embedding_weights_.data() + token_id * d_model_;
        memcpy(output + i * d_model_, src_embed, d_model_ * sizeof(float));
        // 不添加 position embedding，encoder 内部会添加
    }
}
```

---

## 问题 7: Decoder Self-Attention Mask 设计

### 问题描述
Decoder 使用 KV Cache 时，attention mask 的设计与传统 causal mask 不同。

### 分析
传统 causal mask（无 KV Cache）：
```
Query 位置 0: 只能看 Key 位置 0
Query 位置 1: 只能看 Key 位置 0, 1
Query 位置 2: 只能看 Key 位置 0, 1, 2
...
```

KV Cache 模式下：
- Query 始终是当前单个 token：`[batch, 1, d_model]`
- Key/Value 是 past cache + 当前 token：`[batch, cache_len + 1, d_model]`

### 解决方案
设计 KV Cache 专用的 attention mask：
```python
def create_attn_mask(cache_len, max_cache_len):
    """
    Shape: [1, 1, 1, max_cache_len + 1]
    - 位置 0 到 cache_len-1: 有效 (past cache)
    - 位置 cache_len 到 max_cache_len-1: 无效 (未使用的 cache 位置)
    - 位置 max_cache_len (最后): 有效 (当前 token)
    """
    mask = torch.full((1, 1, 1, max_cache_len + 1), -1e9)
    mask[:, :, :, :cache_len] = 0        # past cache 有效
    mask[:, :, :, -1] = 0                # 当前 token 有效
    return mask
```

C++ 实现：
```cpp
void HelsinkiTranslator::create_attn_mask(int cache_len, float* output) {
    const float NEG_INF = -1e9f;
    int total_len = max_cache_len_ + 1;

    std::fill(output, output + total_len, NEG_INF);
    for (int i = 0; i < cache_len; i++) {
        output[i] = 0.0f;  // past cache
    }
    output[total_len - 1] = 0.0f;  // current token
}
```

---

## 问题 8: Cross-Attention Encoder Mask

### 问题描述
Decoder 的 cross-attention 需要 mask 掉 encoder output 中的 padding 位置。

### 解决方案
```python
def create_encoder_attn_mask(actual_src_len, src_seq_len):
    """
    Shape: [1, 1, 1, src_seq_len]
    用于 decoder cross-attention
    """
    mask = torch.zeros(1, 1, 1, src_seq_len)
    mask[:, :, :, actual_src_len:] = -1e9
    return mask
```

---

## 总结：关键经验教训

### 1. 平台限制要提前调研
- 5D tensor 限制
- 不支持的算子列表（GATHER, masked_fill, tril 等）
- 提前规划替代方案

### 2. 数值对齐是调试关键
- 逐层对比 Python 和 C++ 的中间结果
- 使用固定输入进行 deterministic 测试
- 注意 bias、scale 等容易遗漏的参数

### 3. Attention Mask 设计复杂
- Encoder self-attention mask（处理 padding）
- Decoder self-attention mask（KV Cache 模式）
- Decoder cross-attention mask（encoder padding）
- 每种 mask 的 shape 和语义都不同

### 4. Position Embedding 容易重复
- 明确模型内部是否已添加 position embedding
- C++ 端和 Python 端的职责划分要清晰

### 5. HuggingFace 模型细节
- 仔细阅读源码，注意 `final_logits_bias` 等隐藏参数
- 使用 `model.generate()` 时的默认参数（beam search vs greedy）
- tokenizer 的特殊 token 处理

---

## 调试工具和方法

### 1. 逐层对比脚本
```python
# 对比两个模型的中间输出
def compare_outputs(model1, model2, inputs):
    with torch.no_grad():
        out1 = model1(inputs)
        out2 = model2(inputs)
        diff = (out1 - out2).abs().max().item()
        print(f"Max diff: {diff}")
```

### 2. 固定输入测试
```python
# 使用固定 seed 确保可复现
torch.manual_seed(42)
test_input = torch.randn(1, 64, 512)
```

### 3. C++ 调试输出
```cpp
// 打印中间结果
std::cout << "[DEBUG] Step " << step << ": token " << current_token << std::endl;
std::cout << "[DEBUG] Logits[0:5]: ";
for (int i = 0; i < 5; i++) std::cout << logits_[i] << " ";
std::cout << std::endl;
```

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `mtk_model.py` | MTK 优化的 PyTorch 模型，包含 4D KV Cache |
| `convert.py` | 模型转换脚本（PT → TFLite → DLA）|
| `test_model.py` | 一致性测试脚本 |
| `helsinki.cc` | C++ NPU 推理实现 |
| `compile_helsinki_fp.sh` | DLA 编译脚本 |

---

*最后更新: 2025-01-15*
