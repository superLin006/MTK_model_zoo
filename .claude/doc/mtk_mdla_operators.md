# MTK MDLA 5.3 算子支持列表

**针对平台**: MDLA 5.3 (MT8371 Genio 700)
**数据来源**: NeuroPilot SDK 8.0.10 官方文档
**更新日期**: 2026-01-19

---

## ✅ MDLA 5.3 支持的算子 (65个)

### A-D
- **ABS** - 绝对值
- **ADD** - 加法
- **ARG_MAX** - 最大值索引
- **ARG_MIN** - 最小值索引
- **AVERAGE_POOL_2D** - 平均池化
- **BATCH_MATMUL** - 批量矩阵乘法
- **BATCH_TO_SPACE_ND** - Batch 转 Space
- **CAST** - 类型转换
- **CONCATENATION** - 张量拼接
- **CONV_2D** - 2D 卷积
- **DEQUANTIZE** - 反量化
- **DEPTH_TO_SPACE** - Depth 转 Space
- **DEPTHWISE_CONV_2D** - 深度可分离卷积
- **DIV** - 除法

### E-H
- **ELU** - ELU 激活
- **EXP** - 指数运算
- **EXPAND_DIMS** - 维度扩展
- **FULLY_CONNECTED** - 全连接层
- **GELU** - GELU 激活
- **HARD_SWISH** - Hard Swish 激活

### L
- **L2_NORMALIZATION** - L2 归一化
- **L2_POOL_2D** - L2 池化
- **LOGISTIC** - Sigmoid 激活
- **LSTM (QLSTM)** - LSTM 层

### M-P
- **MAXIMUM** - 最大值
- **MAX_POOL_2D** - 最大池化
- **MEAN** - 均值
- **MINIMUM** - 最小值
- **MIRRORPAD** - 镜像填充
- **MUL** - 乘法
- **NEG** - 取负
- **PACK** - 打包
- **PAD** - 填充
- **POW** - 幂运算
- **PRELU** - PReLU 激活

### Q-R
- **QUANTIZE** - 量化
- **REDUCE_MAX** - Reduce 最大值
- **REDUCE_MIN** - Reduce 最小值
- **RELU** - ReLU 激活
- **RELU6** - ReLU6 激活
- **RELU_N1_TO_1** - ReLU [-1, 1] 范围
- **RESHAPE** - 形状变换
- **RESIZE_BILINEAR** - 双线性插值调整大小
- **RESIZE_NEAREST** - 最近邻插值调整大小
- **RSQRT** - 平方根倒数

### S
- **SLICE** - 切片
- **SOFTMAX** - Softmax 激活
- **SPACE_TO_BATCH_ND** - Space 转 Batch
- **SPACE_TO_DEPTH** - Space 转 Depth
- **SPLIT** - 分割
- **SPLIT_V** - 可变分割
- **SQRT** - 平方根
- **SQUARE** - 平方
- **SQUARED_DIFFERENCE** - 平方差
- **SQUEEZE** - 维度压缩
- **STRIDED_SLICE** - 步长切片
- **SUB** - 减法
- **SUM** - 求和

### T-U
- **TANH** - Tanh 激活
- **TILE** - 张量重复
- **TRANSPOSE** - 转置
- **TRANSPOSE_CONV** - 转置卷积
- **UNPACK** - 解包

---

## ❌ MDLA 5.3 不支持的算子 (22个)

### 数学运算
- **CEIL** - 向上取整
- **FLOOR** - 向下取整
- **LOG** - 对数运算
- **ROUND** - 四舍五入

### 逻辑运算 (全部不支持)
- **EQUAL** - 等于比较
- **GREATER** - 大于比较
- **GREATER_EQUAL** - 大于等于比较
- **LESS** - 小于比较
- **LESS_EQUAL** - 小于等于比较
- **LOGICAL_AND** - 逻辑与
- **LOGICAL_NOT** - 逻辑非
- **LOGICAL_OR** - 逻辑或
- **NOT_EQUAL** - 不等于比较

### Tensor 操作
- **GATHER** - 索引查找（⚠️ 影响 Embedding 层）
- **SELECT** - 选择操作
- **TOPK_V2** - Top-K 选择（⚠️ 影响 Beam Search）

### 激活函数
- **LEAKY_RELU** - Leaky ReLU
- **LOG_SOFTMAX** - Log Softmax

### 其他
- **CHANNEL_SHUFFLE** - 通道混洗
- **FILL** - 填充固定值
- **LOCAL_RESPONSE_NORMALIZATION** - 局部响应归一化
- **REDUCE_ANY** - Reduce Any 操作

---

## ⚠️ 关键不支持算子及解决方案

### 1. GATHER (最重要)

**问题描述**:
```python
# ❌ Embedding 层使用 GATHER，不支持
embedding = nn.Embedding(vocab_size, d_model)
output = embedding(token_ids)  # 内部调用 GATHER
```

**解决方案**:
```python
# ✅ 方案：导出 embedding weights，CPU 端查找

# Python 端导出
torch.save(model.embedding.weight, 'embedding_weights.bin')
np.savetxt('embedding_weights_meta.txt', [vocab_size, d_model])

# C++ 端实现
void embed_tokens(const int64_t* token_ids, int seq_len, float* output) {
    for (int i = 0; i < seq_len; i++) {
        int64_t token_id = token_ids[i];
        const float* src = embedding_weights_.data() + token_id * d_model_;
        memcpy(output + i * d_model_, src, d_model_ * sizeof(float));
    }
}
```

**影响范围**: 所有 NLP 模型（BERT, GPT, Transformer 等）

---

### 2. 逻辑运算符 (8个全部不支持)

**问题描述**:
```python
# ❌ 不支持任何逻辑比较
mask = (x > 0)  # GREATER
mask = (x == 0)  # EQUAL
result = mask1 & mask2  # LOGICAL_AND
```

**解决方案**:
```python
# ✅ 方案A：使用算术替代（仅适用于简单情况）
# 例如：ReLU 可以用 max(0, x) 而不是 if x > 0

# ✅ 方案B：预计算 mask 并注册为 buffer
class Model(nn.Module):
    def __init__(self):
        # 预计算 causal mask
        causal_mask = torch.zeros(max_len, max_len)
        for i in range(max_len):
            for j in range(i + 1, max_len):
                causal_mask[i, j] = -1e9
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x):
        # 使用加法代替 masked_fill
        attn = attn + self.causal_mask[:seq_len, :seq_len]

# ✅ 方案C：将条件逻辑移到 CPU 端
```

**影响范围**: Attention Mask, 条件判断, 动态逻辑

---

### 3. TOPK_V2

**问题描述**:
```python
# ❌ Beam Search 需要 Top-K 操作
top_k_values, top_k_indices = torch.topk(logits, k=beam_size)
```

**解决方案**:
```python
# ✅ 方案：CPU 端实现 Beam Search

# Python 端测试可以用 torch.topk
# C++ 端手动实现
std::vector<int> topk(const float* data, int size, int k) {
    std::vector<std::pair<float, int>> pairs;
    for (int i = 0; i < size; i++) {
        pairs.push_back({data[i], i});
    }
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                     [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<int> indices;
    for (int i = 0; i < k; i++) {
        indices.push_back(pairs[i].second);
    }
    return indices;
}
```

**影响范围**: Beam Search 解码, Top-K 采样

---

### 4. LOG_SOFTMAX

**问题描述**:
```python
# ❌ 直接使用 log_softmax 不支持
log_probs = F.log_softmax(logits, dim=-1)
```

**解决方案**:
```python
# ✅ 方案：分解为 SOFTMAX + LOG (但 LOG 也不支持!)
# 实际上：移到 CPU 或用 SOFTMAX + 手动计算 log

# 更好的方案：避免使用 log_softmax
# 如果只需要最终 token，用 argmax
token = torch.argmax(logits, dim=-1)
```

**影响范围**: 某些损失函数, 概率计算

---

## 📐 MDLA 5.3 特殊限制

### 1. Tensor 维度限制
- **最大维度**: 4D
- **不支持**: 5D 及以上张量

**影响**: KV Cache 设计
```python
# ❌ 5D 不支持
past_key: [num_layers, batch, num_heads, seq_len, head_dim]

# ✅ 改为 4D
past_key: [num_layers, batch, seq_len, d_model]
# 其中 d_model = num_heads * head_dim
```

### 2. 动态形状限制
- **要求**: 编译时必须固定输入形状
- **不支持**: 动态 batch size、动态序列长度

**解决方案**:
```python
# 固定形状并做 padding
fixed_seq_len = 64

if actual_len < fixed_seq_len:
    padded = F.pad(input, (0, 0, 0, fixed_seq_len - actual_len))
else:
    padded = input[:, :fixed_seq_len, :]
```

---

## 💡 模型设计最佳实践

### NLP 模型 (Transformer)

**职责分离**:
```
CPU 端:
  ✓ Tokenization
  ✓ Embedding lookup (GATHER 替代)
  ✓ Position Encoding (如果用 TILE)
  ✓ 最终 Argmax/Beam Search (TOPK_V2 替代)
  ✓ 所有条件判断 (逻辑运算替代)

NPU 端:
  ✓ Encoder/Decoder layers
  ✓ Self-Attention
  ✓ Cross-Attention
  ✓ Feed-Forward
  ✓ LayerNorm
  ✓ 激活函数
```

### ASR 模型

**职责分离**:
```
CPU 端:
  ✓ 音频特征提取 (Fbank/Mel-spectrogram)
  ✓ Token embedding
  ✓ CTC/Greedy 解码

NPU 端:
  ✓ CNN 编码器
  ✓ Transformer 编码器
  ✓ CTC 输出层 (logits)
```

### Vision 模型

**职责分离**:
```
CPU 端:
  ✓ 图像预处理 (resize, normalize)
  ✓ NMS 后处理
  ✓ 分类/检测结果解析

NPU 端:
  ✓ CNN 主干网络
  ✓ 特征提取
  ✓ 分类/回归头
```

---

## 🎯 常见问题速查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Embedding 不工作 | GATHER 不支持 | CPU 端手动查找 |
| masked_fill 报错 | 逻辑运算不支持 | 用加法 mask |
| 5D KV Cache 失败 | 最大 4D | 重新设计为 4D |
| Beam Search 报错 | TOPK_V2 不支持 | CPU 端实现 |
| 动态长度不工作 | 固定形状要求 | Padding 到固定长度 |
| tril/triu 不支持 | 逻辑运算 | 预计算 mask |

---

## 🔗 相关文档

- **知识库**: `/home/xh/projects/MTK_models_zoo/.claude/doc/mtk_npu_knowledge_base.md`
- **SDK 路径**: `/home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk`
- **参考项目**: SenseVoice, Helsinki

---

**维护者**: 算法工程师 + Claude Code
**最后更新**: 2026-01-19
