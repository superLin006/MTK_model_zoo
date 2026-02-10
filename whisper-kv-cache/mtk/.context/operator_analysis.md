# Whisper-KV-Cache 算子兼容性分析

**项目**: Whisper Base with KV Cache
**目标平台**: MediaTek MT8371 (MDLA 5.3)
**分析日期**: 2026-02-09
**模型架构**: Encoder-Decoder Transformer with KV Cache

---

## 执行摘要

经过对 OpenAI Whisper Base 模型的详细分析，该模型可以成功移植到 MTK MT8371 NPU，但需要针对以下关键算子进行修改：

### 风险评估
- **风险等级**: 中等
- **主要挑战**:
  1. Embedding 层分离（GATHER 不支持）
  2. KV Cache 4D 设计（5D Tensor 不支持）
  3. Causal Mask 预计算（tril 不支持）
  4. Cross-Attention K,V 缓存管理

### 移植策略
- **Encoder**: 完整移植到 NPU（所有算子支持）
- **Decoder**: Embedding 分离到 CPU，Transformer 核心移植到 NPU
- **KV Cache**: 在 C++ host 端管理，作为模型输入/输出

---

## 1. 模型架构分析

### 1.1 Encoder 架构

```
输入: Mel-spectrogram [batch, 80, 3000] (30s 音频)
  ↓
Conv1d(80 → 512, kernel=3, padding=1) + GELU
  ↓
Conv1d(512 → 512, kernel=3, stride=2, padding=1) + GELU
  ↓ (3000 → 1500 frames)
Positional Embedding (Sinusoidal) [1500, 512]
  ↓
6 × Transformer Block:
  - LayerNorm
  - Multi-Head Self-Attention (8 heads, 64 head_dim)
  - LayerNorm
  - Feed-Forward (512 → 2048 → 512)
  - GELU 激活
  ↓
LayerNorm
  ↓
输出: Encoder features [batch, 1500, 512]
```

**算子清单**:
- ✅ Conv1d (CONV_2D after reshape)
- ✅ GELU (支持)
- ✅ Linear (FULLY_CONNECTED)
- ✅ LayerNorm (通过 MEAN + SUB + MUL + ADD 实现)
- ✅ MatMul (BATCH_MATMUL)
- ✅ Softmax (SOFTMAX)
- ✅ Transpose (TRANSPOSE)
- ✅ Add (ADD)

### 1.2 Decoder 架构

```
输入: Token IDs [batch, seq_len]
  ↓
❌ Token Embedding (GATHER - 不支持，需分离)
  ↓
Positional Embedding (Learnable) [448, 512]
  ↓
6 × Transformer Block:
  - LayerNorm
  - Multi-Head Self-Attention + KV Cache
  - LayerNorm
  - Multi-Head Cross-Attention + KV Cache
  - LayerNorm
  - Feed-Forward (512 → 2048 → 512)
  - GELU 激活
  ↓
LayerNorm
  ↓
LM Head (shared weight with embedding)
  ↓
输出: Logits [batch, seq_len, 51865]
```

**算子清单**:
- ❌ nn.Embedding (GATHER - 不支持)
- ✅ Linear, LayerNorm, MatMul, Softmax, GELU (全部支持)
- ⚠️ Causal Mask (tril 不支持，需预计算)

### 1.3 KV Cache 结构

```python
# 原始设计（可能是 5D）
past_key: [num_layers, batch, num_heads, seq_len, head_dim]  # 5D - 不支持!

# MTK 兼容设计（4D）
past_key: [num_layers, batch, seq_len, d_model]  # 4D - 支持
# 其中 d_model = num_heads * head_dim = 8 * 64 = 512
```

---

## 2. 不支持的算子及解决方案

### 2.1 GATHER 算子（Embedding 层）

#### 问题描述
```python
# ❌ Decoder 使用 nn.Embedding 进行 token 查表
self.token_embedding = nn.Embedding(51865, 512)
x = self.token_embedding(token_ids)  # 内部调用 GATHER
```

MTK MDLA 5.3 不支持 GATHER 算子，导致 Embedding 层无法在 NPU 上运行。

#### 解决方案：分离到 CPU 端

**Python 端修改**：
```python
class WhisperDecoderCore(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()

        # ❌ 删除 token_embedding 层
        # self.token_embedding = nn.Embedding(n_vocab, n_state)

        # ✅ 改为接受 embeddings 作为输入
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        # ... 其他层保持不变

    def forward(self, token_embeddings: Tensor, xa: Tensor):
        """
        token_embeddings: [batch, seq_len, n_state] - 已经查表的 embeddings
        xa: [batch, n_audio_ctx, n_state] - encoder 输出
        """
        seq_len = token_embeddings.shape[1]
        x = token_embeddings + self.positional_embedding[:seq_len]

        # ... Transformer blocks
        return logits
```

**导出 Embedding 权重**：
```python
def export_embedding_weights(whisper_model, output_dir):
    """导出 token embedding 权重为 .npy 文件"""
    token_embedding = whisper_model.decoder.token_embedding.weight.data
    # Shape: [51865, 512], Size: ~100 MB
    np.save(os.path.join(output_dir, 'token_embedding.npy'),
            token_embedding.cpu().numpy())

    # 保存元数据
    metadata = {
        'vocab_size': 51865,
        'embedding_dim': 512,
        'dtype': 'float32'
    }
    json.dump(metadata, open(os.path.join(output_dir, 'embedding_info.json'), 'w'))
```

**C++ 端实现**：
```cpp
class WhisperKVCache {
private:
    std::vector<float> embedding_weights_;  // [51865 * 512]
    int vocab_size_ = 51865;
    int d_model_ = 512;

public:
    void load_embedding_weights(const std::string& npy_path) {
        // 加载 .npy 文件到 embedding_weights_
    }

    void embed_tokens(const int64_t* token_ids, int seq_len, float* output) {
        for (int i = 0; i < seq_len; i++) {
            int64_t token_id = token_ids[i];
            const float* src = embedding_weights_.data() + token_id * d_model_;

            // 复制对应的 embedding vector
            memcpy(output + i * d_model_, src, d_model_ * sizeof(float));

            // 注意：不添加 position embedding，模型内部会处理
        }
    }
};
```

#### 影响
- **精度**: 无影响（仅改变计算位置，不改变逻辑）
- **性能**: Embedding 查表在 CPU 上很快（~0.1ms），可忽略
- **内存**: 增加 ~100 MB 权重文件

#### 参考实现
- `/home/xh/projects/MTK_models_zoo/whisper/mtk/python/whisper_model.py` - WhisperDecoderCore
- `/home/xh/projects/MTK_models_zoo/helsinki/helsinki_workspace/model_prepare/mtk_model.py` - MTKDecoderNoEmbed

---

### 2.2 tril 算子（Causal Mask）

#### 问题描述
```python
# ❌ Decoder 使用 torch.tril 生成 causal mask
mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
```

MTK 转换器不支持 `tril`/`triu` 算子。

#### 解决方案：预计算为 buffer

**Python 端修改**：
```python
class WhisperDecoderCore(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()

        # ✅ 预计算 causal mask 并注册为 buffer
        # 格式: 有效位置 = 0, 无效位置 = -inf
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        # ... 其他层

    def forward(self, token_embeddings, xa):
        # ...
        for block in self.blocks:
            x = block(x, xa, mask=self.mask)  # 直接使用预计算的 mask
        # ...
```

**Attention 中使用加法而非 masked_fill**：
```python
class MultiHeadAttention(nn.Module):
    def qkv_attention(self, q, k, v, mask=None):
        # 计算 attention scores
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        # ✅ 使用加法代替 masked_fill
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]  # 直接相加

        w = F.softmax(qk, dim=-1)
        return (w @ v)
```

#### KV Cache 模式下的 Mask 设计

在使用 KV Cache 时，mask 的设计与传统 causal mask 不同：

```python
def create_decoder_self_attn_mask_kv_cache(cache_len, max_cache_len):
    """
    KV Cache 模式下的 self-attention mask

    Query: [batch, 1, d_model] - 当前单个 token
    Key/Value: [batch, cache_len + 1, d_model] - past cache + 当前 token

    Mask shape: [1, 1, 1, max_cache_len + 1]
    """
    mask = torch.full((1, 1, 1, max_cache_len + 1), -1e9)

    # 前 cache_len 个位置有效（past cache）
    mask[:, :, :, :cache_len] = 0

    # 中间未使用的 cache 位置无效
    # mask[:, :, :, cache_len:max_cache_len] = -1e9  (已默认)

    # 最后一个位置有效（当前 token）
    mask[:, :, :, -1] = 0

    return mask
```

**C++ 端实现**：
```cpp
void WhisperKVCache::create_decoder_self_attn_mask(int cache_len, float* output) {
    const float NEG_INF = -1e9f;
    int total_len = max_cache_len_ + 1;

    // 全部初始化为 -1e9
    std::fill(output, output + total_len, NEG_INF);

    // 前 cache_len 个位置有效
    for (int i = 0; i < cache_len; i++) {
        output[i] = 0.0f;
    }

    // 最后一个位置有效（当前 token）
    output[total_len - 1] = 0.0f;
}
```

#### 影响
- **精度**: 无影响（mask 语义完全一致）
- **性能**: 预计算在模型加载时完成，推理时无开销
- **内存**: 增加 ~1 MB (448 × 448 × 4 bytes)

#### 参考实现
- `/home/xh/projects/MTK_models_zoo/whisper/mtk/python/whisper_model.py` - WhisperDecoderCore
- `/home/xh/projects/MTK_models_zoo/helsinki/PORTING_NOTES.md` - 问题 3 和问题 7

---

### 2.3 5D Tensor 限制（KV Cache）

#### 问题描述

MT8371 MDLA 5.3 不支持 5D Tensor，但传统 KV Cache 设计使用 5D：

```python
# ❌ 可能的 5D 设计（不支持）
past_key: [num_layers, batch, num_heads, seq_len, head_dim]
# Shape: [6, 1, 8, 448, 64] - 5D
```

#### 解决方案：4D 重设计

**方案：合并 num_heads 和 head_dim 到 d_model**：

```python
# ✅ 4D 设计（支持）
past_key: [num_layers, batch, seq_len, d_model]
# Shape: [6, 1, 448, 512] - 4D
# 其中 d_model = num_heads * head_dim = 8 * 64 = 512
```

**实现细节**：

```python
class DecoderSelfAttentionWithCache(nn.Module):
    def __init__(self, d_model, num_heads, max_cache_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, 1, d_model]
        past_key: torch.Tensor,       # [batch, max_cache_len, d_model] - 4D
        past_value: torch.Tensor,     # [batch, max_cache_len, d_model] - 4D
        attn_mask: torch.Tensor,      # [1, 1, 1, max_cache_len + 1]
    ):
        batch_size = hidden_states.shape[0]

        # 计算当前 token 的 Q, K, V
        query = self.q_proj(hidden_states)      # [batch, 1, d_model]
        new_key = self.k_proj(hidden_states)    # [batch, 1, d_model]
        new_value = self.v_proj(hidden_states)  # [batch, 1, d_model]

        # 拼接 past cache + current
        # 注意：past_key 中只有前面的有效部分会被使用（通过 mask 控制）
        key = torch.cat([past_key, new_key], dim=1)    # [batch, max_cache_len + 1, d_model]
        value = torch.cat([past_value, new_value], dim=1)

        # Reshape 为 multi-head 格式进行 attention
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, 1, head_dim]

        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, max_cache_len + 1, head_dim]

        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attn_mask  # 应用 mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)

        output = self.out_proj(attn_output)

        return output, new_key, new_value  # 返回新的 K, V 供下次缓存
```

**KV Cache 管理（C++ 端）**：

```cpp
class WhisperKVCache {
private:
    // KV Cache buffers (4D)
    struct LayerCache {
        std::vector<float> self_key;    // [batch * max_cache_len * d_model]
        std::vector<float> self_value;  // [batch * max_cache_len * d_model]
        std::vector<float> cross_key;   // [batch * 1500 * d_model] (encoder output)
        std::vector<float> cross_value;

        int current_len = 0;  // 当前缓存的 token 数量
    };

    LayerCache layer_caches_[6];  // 6 个 decoder layers
    int max_cache_len_ = 448;
    int d_model_ = 512;

public:
    void update_cache(int layer_idx, const float* new_key, const float* new_value) {
        auto& cache = layer_caches_[layer_idx];

        if (cache.current_len < max_cache_len_) {
            // 将新的 K, V 添加到缓存末尾
            int offset = cache.current_len * d_model_;
            memcpy(cache.self_key.data() + offset, new_key, d_model_ * sizeof(float));
            memcpy(cache.self_value.data() + offset, new_value, d_model_ * sizeof(float));
            cache.current_len++;
        } else {
            // Cache 已满，可以选择：
            // 1. 停止生成（推荐）
            // 2. 循环覆盖（会破坏语义）
            // 3. 重新编码（昂贵）
            std::cerr << "Warning: KV Cache is full!" << std::endl;
        }
    }

    void reset_cache() {
        for (auto& cache : layer_caches_) {
            cache.current_len = 0;
            std::fill(cache.self_key.begin(), cache.self_key.end(), 0.0f);
            std::fill(cache.self_value.begin(), cache.self_value.end(), 0.0f);
        }
    }
};
```

#### 影响
- **精度**: 无影响（逻辑等价，只是维度组织不同）
- **性能**: 无影响（内存布局优化可能略有提升）
- **内存**:
  - 每层 self-attention cache: 2 × (448 × 512 × 4) ≈ 1.8 MB
  - 每层 cross-attention cache: 2 × (1500 × 512 × 4) ≈ 6.1 MB
  - 总计 6 层: ~47 MB

#### 参考实现
- `/home/xh/projects/MTK_models_zoo/helsinki/helsinki_workspace/model_prepare/mtk_model.py` - MTKDecoderSelfAttentionKVCacheV2
- `/home/xh/projects/MTK_models_zoo/helsinki/PORTING_NOTES.md` - 问题 1

---

### 2.4 Cross-Attention K,V 缓存优化

#### 问题描述

Cross-Attention 的 Key 和 Value 是从 Encoder 输出计算得到的，在整个解码过程中保持不变，但每次解码都会重新计算，造成浪费。

#### 解决方案：首次计算后缓存

**Python 端设计**：

```python
class DecoderCrossAttentionWithCache(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,     # [batch, 1, d_model]
        encoder_output: torch.Tensor,    # [batch, 1500, d_model]
        cached_cross_key: Optional[torch.Tensor] = None,    # [batch, 1500, d_model]
        cached_cross_value: Optional[torch.Tensor] = None,  # [batch, 1500, d_model]
        cross_attn_mask: Optional[torch.Tensor] = None,     # [1, 1, 1, 1500]
    ):
        batch_size = hidden_states.shape[0]

        # 计算 Query（每次都需要重新计算）
        query = self.q_proj(hidden_states)  # [batch, 1, d_model]

        # Key 和 Value：首次计算，之后使用缓存
        if cached_cross_key is None:
            # 首次计算
            cross_key = self.k_proj(encoder_output)    # [batch, 1500, d_model]
            cross_value = self.v_proj(encoder_output)  # [batch, 1500, d_model]
        else:
            # 使用缓存
            cross_key = cached_cross_key
            cross_value = cached_cross_value

        # Multi-head attention computation
        # ... (与 self-attention 类似)

        return output, cross_key, cross_value  # 返回 K, V 供缓存
```

**C++ 端缓存管理**：

```cpp
class WhisperKVCache {
private:
    struct LayerCache {
        // ... self-attention cache

        // Cross-attention cache (computed once from encoder output)
        std::vector<float> cross_key;    // [batch * 1500 * d_model]
        std::vector<float> cross_value;  // [batch * 1500 * d_model]
        bool cross_cached = false;       // 标记是否已缓存
    };

    LayerCache layer_caches_[6];

public:
    void cache_cross_attention(int layer_idx, const float* key, const float* value) {
        auto& cache = layer_caches_[layer_idx];

        int size = 1500 * d_model_;  // encoder_seq_len * d_model
        memcpy(cache.cross_key.data(), key, size * sizeof(float));
        memcpy(cache.cross_value.data(), value, size * sizeof(float));

        cache.cross_cached = true;
    }

    bool is_cross_cached(int layer_idx) const {
        return layer_caches_[layer_idx].cross_cached;
    }
};
```

#### 推理流程

```cpp
// 第一次解码（SOT sequence）
for (int layer = 0; layer < 6; layer++) {
    // Decoder layer 推理
    // 输出包含 cross_key, cross_value

    // 缓存 cross-attention K,V
    kv_cache_.cache_cross_attention(layer, cross_key, cross_value);
}

// 后续解码步骤
for (int layer = 0; layer < 6; layer++) {
    if (kv_cache_.is_cross_cached(layer)) {
        // 直接使用缓存的 cross K,V，不传入 encoder_output
    }
}
```

#### 影响
- **精度**: 无影响（完全等价）
- **性能**: 显著提升（避免重复计算 K,V projection）
- **内存**: 每层增加 ~6.1 MB，6 层共 ~37 MB

---

## 3. 支持的算子

以下算子在 MTK MDLA 5.3 上完全支持，无需修改：

| 算子 | PyTorch API | MTK MDLA 算子 | 使用位置 |
|------|-------------|---------------|----------|
| 卷积 | nn.Conv1d | CONV_2D | Encoder (Conv1d → Conv2d reshape) |
| 线性层 | nn.Linear | FULLY_CONNECTED | 所有层 |
| 矩阵乘法 | torch.matmul, @ | BATCH_MATMUL | Attention |
| 激活函数 | nn.GELU | GELU | Encoder, Decoder MLP |
| 归一化 | nn.LayerNorm | MEAN + SUB + MUL + ADD | 所有层 |
| Softmax | F.softmax | SOFTMAX | Attention |
| 加法 | +, torch.add | ADD | Residual, Mask |
| 转置 | .transpose(), .permute() | TRANSPOSE | Attention reshape |
| Reshape | .view(), .reshape() | RESHAPE | Multi-head 变换 |
| 拼接 | torch.cat | CONCATENATION | KV Cache 拼接 |

---

## 4. 模型修改方案总结

### 4.1 Encoder 修改

**修改点**：
1. ✅ 保持原始结构，无需修改（所有算子支持）
2. ✅ Position embedding 注册为 buffer（而非 Parameter）

**修改后的输入/输出**：
```
输入: mel_spectrogram [batch, 80, 3000]
输出: encoder_features [batch, 1500, 512]
```

### 4.2 Decoder 修改

**修改点**：
1. ❌ 删除 `nn.Embedding` 层
2. ✅ 改为接受 `token_embeddings` 作为输入
3. ✅ Position embedding 保持为 Parameter（但会转为 buffer）
4. ✅ Causal mask 预计算为 buffer
5. ✅ KV Cache 设计为 4D
6. ✅ 添加 Cross-Attention K,V 缓存机制

**修改后的输入/输出（单步推理）**：

```python
# 首次推理（SOT sequence）
inputs:
  - token_embeddings: [1, 3, 512]  # SOT tokens: [50258, 50259, 50359]
  - encoder_output: [1, 1500, 512]
  - past_self_keys: [6, 1, 448, 512] (全零，未使用)
  - past_self_values: [6, 1, 448, 512] (全零，未使用)
  - self_attn_mask: [1, 1, 1, 448+1] (前3个有效，其余无效)

outputs:
  - logits: [1, 3, 51865]
  - new_self_keys: [6, 1, 3, 512]  # 3个 token 的 K
  - new_self_values: [6, 1, 3, 512]
  - cross_keys: [6, 1, 1500, 512]  # 缓存供后续使用
  - cross_values: [6, 1, 1500, 512]

# 后续推理（每次1个 token）
inputs:
  - token_embeddings: [1, 1, 512]  # 当前 token
  - encoder_output: [1, 1500, 512]
  - past_self_keys: [6, 1, 448, 512] (前 cache_len 个有效)
  - past_self_values: [6, 1, 448, 512]
  - cached_cross_keys: [6, 1, 1500, 512]  # 重用
  - cached_cross_values: [6, 1, 1500, 512]
  - self_attn_mask: [1, 1, 1, 448+1]

outputs:
  - logits: [1, 1, 51865]
  - new_self_keys: [6, 1, 1, 512]  # 当前 token 的 K
  - new_self_values: [6, 1, 1, 512]
```

---

## 5. C++ 实现架构

### 5.1 类设计

```cpp
class WhisperKVCache {
public:
    // 初始化
    void init(const std::string& encoder_dla_path,
              const std::string& decoder_dla_path,
              const std::string& embedding_weights_path);

    // Encoder 推理（每句话执行一次）
    void encode(const float* mel_spectrogram,
                float* encoder_output);

    // Decoder 推理（自回归生成）
    std::vector<int> decode(const std::vector<int>& initial_tokens,
                           int max_length = 448);

    // 重置 KV Cache（新句子开始时）
    void reset_cache();

private:
    // RKNN contexts
    rknn_context encoder_ctx_;
    rknn_context decoder_ctx_;

    // Embedding weights (CPU 端)
    std::vector<float> embedding_weights_;  // [51865 * 512]
    int vocab_size_ = 51865;
    int d_model_ = 512;
    int num_layers_ = 6;
    int encoder_seq_len_ = 1500;
    int max_cache_len_ = 448;

    // KV Cache buffers (4D)
    struct LayerCache {
        std::vector<float> self_key;    // [max_cache_len * d_model]
        std::vector<float> self_value;  // [max_cache_len * d_model]
        std::vector<float> cross_key;   // [1500 * d_model]
        std::vector<float> cross_value;
        int current_len = 0;
        bool cross_cached = false;
    };
    LayerCache layer_caches_[6];

    // 辅助方法
    void embed_tokens(const int64_t* token_ids, int seq_len, float* output);
    void create_decoder_self_attn_mask(int cache_len, float* output);
    void create_decoder_cross_attn_mask(int actual_src_len, float* output);
    void update_cache(int layer_idx, const float* new_key, const float* new_value);
    int sample_token(const float* logits);  // Greedy sampling
};
```

### 5.2 推理流程

```cpp
std::vector<int> WhisperKVCache::decode(
    const std::vector<int>& initial_tokens,
    int max_length
) {
    std::vector<int> generated_tokens = initial_tokens;  // 通常是 [50258, 50259, 50359]
    reset_cache();

    // 第一步：处理 SOT sequence
    int sot_len = initial_tokens.size();  // 通常是 3

    // 1. Embedding lookup (CPU)
    std::vector<float> token_embeds(sot_len * d_model_);
    embed_tokens(initial_tokens.data(), sot_len, token_embeds.data());

    // 2. 创建 self-attention mask
    std::vector<float> self_attn_mask(max_cache_len_ + 1);
    create_decoder_self_attn_mask(0, self_attn_mask.data());  // cache_len = 0

    // 3. 准备 decoder 输入
    rknn_input inputs[num_decoder_inputs];
    // inputs[0]: token_embeds [1, 3, 512]
    // inputs[1]: encoder_output [1, 1500, 512]
    // inputs[2]: past_self_keys [6, 1, 448, 512] (全零)
    // inputs[3]: past_self_values [6, 1, 448, 512] (全零)
    // inputs[4]: self_attn_mask [1, 1, 1, 449]
    // ... (还有 cross_attn 相关输入)

    // 4. Decoder 推理
    rknn_run(decoder_ctx_, nullptr);

    // 5. 获取输出
    rknn_output outputs[num_decoder_outputs];
    // outputs[0]: logits [1, 3, 51865]
    // outputs[1]: new_self_keys [6, 1, 3, 512]
    // outputs[2]: new_self_values [6, 1, 3, 512]
    // outputs[3]: cross_keys [6, 1, 1500, 512]
    // outputs[4]: cross_values [6, 1, 1500, 512]

    // 6. 更新 cache
    for (int layer = 0; layer < num_layers_; layer++) {
        // 将新的 K, V 添加到 cache
        update_cache(layer,
                    new_self_keys + layer * sot_len * d_model_,
                    new_self_values + layer * sot_len * d_model_);

        // 缓存 cross-attention K, V
        cache_cross_attention(layer,
                            cross_keys + layer * encoder_seq_len_ * d_model_,
                            cross_values + layer * encoder_seq_len_ * d_model_);
    }

    // 7. 采样下一个 token
    float* last_logits = logits + (sot_len - 1) * vocab_size_;
    int next_token = sample_token(last_logits);
    generated_tokens.push_back(next_token);

    // 自回归生成循环
    while (generated_tokens.size() < max_length) {
        // 1. Embedding lookup (只需当前 token)
        std::vector<float> current_embed(d_model_);
        embed_tokens(&next_token, 1, current_embed.data());

        // 2. 更新 mask (cache_len++)
        int cache_len = layer_caches_[0].current_len;
        create_decoder_self_attn_mask(cache_len, self_attn_mask.data());

        // 3. Decoder 推理（现在输入只有 1 个 token）
        // inputs[0]: token_embeds [1, 1, 512]
        // inputs[2]: past_self_keys [6, 1, 448, 512] (前 cache_len 个有效)
        // ... 使用缓存的 cross K,V

        rknn_run(decoder_ctx_, nullptr);

        // 4. 获取输出
        // outputs[0]: logits [1, 1, 51865]
        // outputs[1]: new_self_keys [6, 1, 1, 512]
        // outputs[2]: new_self_values [6, 1, 1, 512]

        // 5. 更新 cache
        for (int layer = 0; layer < num_layers_; layer++) {
            update_cache(layer,
                        new_self_keys + layer * d_model_,
                        new_self_values + layer * d_model_);
        }

        // 6. 采样
        next_token = sample_token(logits);

        // 7. 检查结束符
        if (next_token == EOT_TOKEN) {
            break;
        }

        generated_tokens.push_back(next_token);
    }

    return generated_tokens;
}
```

---

## 6. 风险评估

### 6.1 技术风险

| 风险项 | 风险等级 | 描述 | 缓解措施 |
|--------|---------|------|----------|
| KV Cache 维度设计 | 中 | 4D 设计需要仔细验证数值一致性 | 逐层对比 Python/C++ 输出 |
| Mask 设计错误 | 中 | Self/Cross-Attention mask 设计复杂 | 参考 Helsinki 实现，单元测试 |
| 内存溢出 | 低 | KV Cache 占用 ~47 MB | MT8371 有足够内存 |
| 性能不达标 | 低 | Encoder/Decoder 算子都支持，NPU 利用率高 | 预期 Encoder ~100ms, Decoder ~10ms/token |
| 量化精度损失 | 中 | FP16/INT8 量化可能影响准确率 | 从 FP32 开始，逐步量化 |

### 6.2 实现复杂度

| 模块 | 复杂度 | 工作量估计 | 依赖 |
|------|--------|-----------|------|
| Encoder 导出 | 低 | 1 天 | TFLite 转换 |
| Decoder 导出（无 KV Cache） | 中 | 2 天 | Embedding 分离 |
| Decoder 导出（带 KV Cache） | 高 | 3-4 天 | 4D 设计，多输入/输出 |
| C++ Encoder 推理 | 低 | 1 天 | RKNN API |
| C++ Decoder 推理（单步） | 中 | 2 天 | Embedding lookup |
| C++ KV Cache 管理 | 高 | 3 天 | 内存管理，cache 更新逻辑 |
| 端到端测试 | 中 | 2 天 | 数值验证，性能测试 |
| **总计** | - | **14-15 天** | - |

---

## 7. 参考项目

### 7.1 Helsinki (MarianMT Translation)

**路径**: `/home/xh/projects/MTK_models_zoo/helsinki/`

**可复用经验**:
- ✅ 4D KV Cache 设计 (`mtk_model.py` - MTKDecoderSelfAttentionKVCacheV2)
- ✅ Encoder-Decoder 协同工作
- ✅ Embedding 分离到 CPU
- ✅ Attention Mask 设计（Self/Cross）
- ✅ C++ 端 KV Cache 管理 (`helsinki.cc`)
- ✅ 详细的 PORTING_NOTES.md 记录了所有踩坑经验

**关键文件**:
```
helsinki/
├── helsinki_workspace/model_prepare/
│   ├── mtk_model.py              # 4D KV Cache 实现
│   ├── convert.py                # 模型转换脚本
│   └── test_model.py             # 数值验证
├── helsinki_mtk_cpp/
│   └── helsinki/helsinki.cc      # C++ 推理实现
└── PORTING_NOTES.md              # 问题记录（必读！）
```

### 7.2 现有 Whisper 项目（无 KV Cache）

**路径**: `/home/xh/projects/MTK_models_zoo/whisper/mtk/python/whisper_model.py`

**可复用代码**:
- ✅ WhisperEncoderCore（完全可用）
- ✅ WhisperDecoderCore（需添加 KV Cache 逻辑）
- ✅ Embedding 导出函数
- ✅ 权重加载函数

### 7.3 SenseVoice (ASR Encoder-only)

**路径**: `/home/xh/projects/MTK_models_zoo/sense-voice/`

**可参考**:
- ✅ 音频预处理（Mel-spectrogram）
- ✅ C++ 端音频处理流程

---

## 8. 下一步行动

### Phase 1: 模型导出（Python）

1. **修改 Decoder 架构**（2 天）
   - [ ] 删除 nn.Embedding 层
   - [ ] 添加 4D KV Cache 输入/输出
   - [ ] 实现 DecoderSelfAttentionWithCache
   - [ ] 实现 DecoderCrossAttentionWithCache
   - [ ] 数值验证（与原始 Whisper 对比）

2. **导出模型**（2 天）
   - [ ] 导出 Encoder: PyTorch → TorchScript → TFLite → DLA
   - [ ] 导出 Decoder: PyTorch → TorchScript → TFLite → DLA
   - [ ] 导出 token_embedding.npy
   - [ ] 生成元数据文件

### Phase 2: C++ 实现（5 天）

3. **Encoder 推理**（1 天）
   - [ ] 加载 Encoder DLA 模型
   - [ ] 实现前向推理
   - [ ] 验证输出（与 Python 对比）

4. **Decoder 推理（无 Cache）**（2 天）
   - [ ] 加载 Decoder DLA 模型
   - [ ] 实现 embedding lookup
   - [ ] 实现单步推理
   - [ ] 验证输出

5. **KV Cache 管理**（2 天）
   - [ ] 实现 LayerCache 结构
   - [ ] 实现 cache 更新逻辑
   - [ ] 实现 mask 生成
   - [ ] 端到端自回归生成

### Phase 3: 测试与优化（3 天）

6. **功能测试**（1 天）
   - [ ] 使用 test_en.wav 测试
   - [ ] 使用 test_zh.wav 测试
   - [ ] 对比 baseline 结果

7. **性能测试**（1 天）
   - [ ] Encoder 性能（目标 <150ms）
   - [ ] Decoder 性能（目标 <15ms/token）
   - [ ] 端到端延迟分析

8. **量化优化**（1 天）
   - [ ] FP16 量化测试
   - [ ] INT8 量化测试（可选）
   - [ ] 准确率验证

---

## 9. 预期性能

### 9.1 性能目标

| 阶段 | CPU Baseline | NPU 目标 | 说明 |
|------|-------------|----------|------|
| Encoder | ~330ms | <150ms | Conv + Transformer |
| Decoder (首次) | ~50ms | <30ms | 处理 SOT 序列 (3 tokens) |
| Decoder (单步) | ~40ms | <15ms | 每个 token（带 KV Cache） |
| 总计 (30s 音频) | ~1.5s | <1.0s | 假设生成 50 tokens |

### 9.2 内存占用

| 项目 | 大小 | 说明 |
|------|------|------|
| Encoder DLA | ~80 MB | FP32 权重 |
| Decoder DLA | ~100 MB | FP32 权重 |
| Embedding weights | ~100 MB | 51865 × 512 × 4 bytes |
| KV Cache (runtime) | ~47 MB | 6 layers × self + cross |
| 音频 buffer | ~1 MB | 30s 音频 |
| **总计** | ~328 MB | 可接受 |

---

## 10. 结论

Whisper-KV-Cache 模型可以成功移植到 MTK MT8371 NPU，主要修改点包括：

1. ✅ **Embedding 层分离**：将 nn.Embedding 移到 CPU 端，导出权重为 .npy 文件
2. ✅ **4D KV Cache 设计**：避免 5D Tensor 限制，将 cache 设计为 `[num_layers, batch, seq_len, d_model]`
3. ✅ **Causal Mask 预计算**：使用 register_buffer 预计算 mask，避免 tril 算子
4. ✅ **Cross-Attention K,V 缓存**：首次计算后缓存，避免重复计算

**参考 Helsinki 项目的成功经验**，预计在 15 个工作日内完成移植，最终性能预期：
- Encoder: <150ms
- Decoder: <15ms/token
- 端到端 30s 音频: <1.0s

**关键风险点**已有成熟的解决方案，技术可行性高。

---

**文档版本**: v1.0
**作者**: Claude Code + operator-analyst subagent
**最后更新**: 2026-02-09
