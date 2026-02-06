# Whisper MTK NPU 项目经验总结

## 关键Bug修复记录 (2025-02-05)

### 问题描述
C++ mel频谱图计算产生完全错误的值，导致推理结果错误。

**症状**：
- 预期输出：`"Mr. Quilter is the apostle of the middle classes..."`
- 实际输出：`"(whistling)"`
- Mel频谱图与Python差异：0.77 (均值绝对差)

### 根本原因

音频预处理流程与RK参考实现有**三个关键差异**：

#### 1. 错误的处理顺序
```cpp
// ❌ 错误实现
compute_stft(audio, stft_result);           // STFT: [num_freqs x num_frames]
compute_magnitudes(stft_result, mags);      // 直接计算magnitude
transpose_float(mags, mags_transposed);     // 转置float数组

// ✅ 正确实现（RK风格）
compute_stft(audio, stft_result);           // STFT: [num_frames x num_freqs]
transpose_complex(stft_result, stft_t);     // 先转置复数数组 -> [num_freqs x num_frames]
compute_magnitudes(stft_t, mags);           // 再计算magnitude
```

**教训**：转置复数数组和转置magnitude数组的内存布局不同！

#### 2. 丢弃最后一帧
```cpp
// ❌ 错误：使用所有帧
for (int j = 0; j < num_frames; j++) {
    magnitudes[k++] = compute_magnitude(stft[i][j]);
}

// ✅ 正确：丢弃最后一帧
for (int j = 0; j < num_frames - 1; j++) {  // 注意：num_frames - 1
    magnitudes[k++] = compute_magnitude(stft[i][j]);
}
```

**原因**：STFT最后一帧可能不完整或为padding，Python/RK都会丢弃。

**教训**：`< n` vs `< n-1` 这种看似微小的差异会导致完全错误的结果！

#### 3. 矩阵维度不匹配
```cpp
// ❌ 错误：使用错误的列数
matmul(mel_filters, magnitudes, result,
       N_MELS, num_freqs, num_frames);  // 维度错误

// ✅ 正确：使用 num_frames - 1
matmul(mel_filters, magnitudes, result,
       N_MELS, num_freqs, num_frames - 1);  // 正确维度
```

**结果**：维度不匹配导致mel滤波器权重应用到错误的频率bin上。

### 正确的实现流程

```cpp
void audio_preprocess(audio_buffer_t* audio, float* mel_filters,
                     std::vector<float>& x_mel) {
    // Step 1: 计算STFT [num_frames x num_freqs]
    int num_freqs = N_FFT / 2 + 1;  // 201
    fftwf_complex* stfts_result = compute_stft(...);

    // Step 2: 转置STFT复数数组 [num_freqs x num_frames]
    fftwf_complex* stfts_result_t = malloc(...);
    transpose_complex(stfts_result, num_frames, num_freqs, stfts_result_t);

    // Step 3: 计算magnitude，丢弃最后一帧 [num_freqs x (num_frames-1)]
    std::vector<float> magnitudes(num_freqs * (num_frames - 1));
    for (int i = 0; i < num_freqs; i++) {
        for (int j = 0; j < num_frames - 1; j++) {  // ← 关键！
            magnitudes[k++] = compute_magnitude(stfts_result_t[i * num_frames + j]);
        }
    }

    // Step 4: 矩阵乘法 [80 x 201] @ [201 x (n-1)] = [80 x (n-1)]
    matmul(mel_filters, magnitudes.data(), cur_x_mel,
           N_MELS, num_freqs, num_frames - 1);  // ← 使用 num_frames - 1

    // Step 5: Log变换
    clamp_and_log_max(cur_x_mel, N_MELS, num_frames - 1);

    // Step 6: Padding到3000帧
    pad_to_3000(cur_x_mel, x_mel);
}
```

### 修复结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 输出文本 | "(whistling)" | "Mr. Quilter is the apostle..." ✅ |
| 首个token | 522 (错误) | 2221 (正确) ✅ |
| Mel频谱图差异 | 0.77 | 0.003 ✅ |
| 推理时间 | ~800ms | ~700-800ms ✅ |

## 最佳实践与经验教训

### 1. ⚠️ 永远不要假设代码是正确的

**错误做法**：
- "这段代码看起来和参考实现类似，应该没问题"
- "我理解原理，可以用自己的方式实现"

**正确做法**：
- 逐行对比参考实现
- 精确复制，包括看似"冗余"的代码
- 验证正确后再优化

### 2. 📊 逐层验证中间结果

**必须做**：
```cpp
// 在每个关键步骤后保存中间结果
save_debug_file("1_stft.bin", stft_result);
save_debug_file("2_transposed.bin", stft_transposed);
save_debug_file("3_magnitudes.bin", magnitudes);
save_debug_file("4_mel.bin", mel_output);

// 与Python参考对比
python compare.py 1_stft.bin python_stft.bin
```

**教训**：不要等到最终输出错误才开始调试！

### 3. 🔍 注意循环边界的细微差异

```cpp
// 这种差异看起来很小，但致命！
for (int i = 0; i < n; i++)      // ❌
for (int i = 0; i < n - 1; i++)  // ✅
```

**检查清单**：
- ✅ 循环是 `< n` 还是 `< n-1`?
- ✅ 数组索引是 `[i][j]` 还是 `[j][i]`?
- ✅ 是 `i * cols + j` 还是 `j * rows + i`?

### 4. 📐 始终打印和验证矩阵维度

```cpp
// 在矩阵乘法前打印维度
std::cout << "matmul: [" << rows_A << " x " << cols_A
          << "] @ [" << cols_A << " x " << cols_B << "] -> ["
          << rows_A << " x " << cols_B << "]" << std::endl;

// 验证：A的列数必须等于B的行数
assert(cols_A == rows_B);
```

### 5. 📖 仔细阅读参考实现的每个细节

**案例**：RK代码中的 `num_frames - 1`

```cpp
// RK whisper/cpp/process.cc:291
std::vector<float> magnitudes(MELS_FILTERS_SIZE * (cur_num_frames_of_stfts - 1));
```

**错误思维**：
- "这个 -1 可能是个bug或可以优化的地方"
- "我可以用更简洁的方式实现"

**正确思维**：
- "为什么参考实现要 -1？"
- "这是STFT的特性还是Whisper的要求？"
- "先完全复制，理解后再决定是否修改"

### 6. 🧪 对比数值结果，不只是代码逻辑

```python
# 对比关键数值
cpp_mel_first_10 = [-0.854, -0.854, -0.854, ...]
python_mel_first_10 = [0.119, -0.095, -0.110, ...]

# 立即发现问题！差异太大，肯定哪里错了
```

### 7. 🎯 从源头开始验证，不要跳步

调试顺序：
1. ✅ STFT输出是否正确？
2. ✅ 转置后的STFT是否正确？
3. ✅ Magnitude计算是否正确？
4. ✅ 矩阵乘法结果是否正确？
5. ✅ Log变换是否正确？
6. ✅ 最终mel频谱图是否正确？

**不要直接跳到第6步！**

## 调试策略总结

### 发现问题的过程

1. **最终输出错误** → Decoder输出 "(whistling)"
2. **对比token序列** → 首个token 522 vs 2221
3. **对比encoder输出** → 发现encoder输出不对
4. **对比mel频谱图** → 发现0.77的巨大差异
5. **对比首10个值** → 完全不同的模式
6. **阅读RK源码** → 发现 `num_frames - 1`
7. **重构代码** → 完全按照RK实现
8. **验证** → 成功！

### 有效的调试技巧

```cpp
// 1. 添加详细的调试输出
std::cout << "[DEBUG] stft shape: [" << num_frames << " x " << num_freqs << "]" << std::endl;
std::cout << "[DEBUG] first 10 magnitudes: ";
for (int i = 0; i < 10; i++) std::cout << mags[i] << " ";
std::cout << std::endl;

// 2. 保存中间结果
std::ofstream file("debug_output.bin", std::ios::binary);
file.write((char*)data, size * sizeof(float));

// 3. 对比Python输出
import numpy as np
cpp_data = np.fromfile("cpp_output.bin", dtype=np.float32)
py_data = np.fromfile("python_output.bin", dtype=np.float32)
print(f"Difference: {np.mean(np.abs(cpp_data - py_data))}")
```

## 参考资料

- **RK Whisper实现**: `/home/xh/projects/rknn_model_zoo/examples/whisper/cpp/process.cc`
  - 关键函数：`log_mel_spectrogram()`, `compute_magnitudes()`, `transpose()`

- **Python Whisper**: `whisper.log_mel_spectrogram()`
  - 位置：`whisper/audio.py`

- **修改的文件**:
  - `utils/audio_utils.cpp`: 主要修改
  - `src/whisper_inference.cpp`: 移除调试代码

## 总结

这次bug修复最重要的教训是：

> **在移植复杂算法时，必须精确复制参考实现的每一个细节，包括看似可以"优化"的部分。先确保正确性，再考虑优化。**

关键要点：
1. ✅ 转置复数数组，而不是转置magnitude数组
2. ✅ 使用 `num_frames - 1`，丢弃最后一帧
3. ✅ 矩阵乘法使用正确的维度
4. ✅ 逐层验证，从源头开始调试
5. ✅ 对比数值结果，不只看代码逻辑

---

📝 *记录于 2025-02-05，用于避免后续项目踩同样的坑*
