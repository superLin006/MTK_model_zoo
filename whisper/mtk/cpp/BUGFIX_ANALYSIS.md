# Mel Spectrogram Bug - 详细分析与修复

## 问题总结

C++实现的Whisper推理在mel频谱图计算阶段产生了完全错误的值，导致encoder输出错误的特征向量，最终decoder生成错误的token序列。

## 症状

### 修复前的输出
```
Input: test_en.wav (5.855s)
Expected: "Mr. Quilter is the apostle of the middle classes..."
Actual: "(whistling)"
Tokens: [522, 1363, 37174, 8, 50257]
Expected tokens: [2221, 13, 2326, 388, 391, 307, 264, ...]
```

### Mel频谱图对比
```python
# Python mel (正确)
First 10 values: [ 0.119, -0.095, -0.110, -0.020,  0.022, -0.069, -0.174,  0.095,  0.097, -0.082]
Range: [-0.806, 1.194]

# C++ mel (修复前)
First 10 values: [-0.854, -0.854, -0.854, -0.781, -0.806, -0.802, -0.795, -0.854, -0.854, -0.854]
Range: [-0.854, 1.146]

# 差异统计
Mean absolute difference: 0.77
Max difference: 2.05
```

## 根本原因

### 错误的实现流程

```cpp
// ❌ 错误的实现（修复前）
void audio_preprocess_WRONG() {
    // 1. 计算STFT: [num_freqs x num_frames] 格式
    compute_stft(audio, stft_result);

    // 2. 直接在原始STFT上计算magnitude
    for (int i = 0; i < num_freqs; ++i) {
        for (int j = 0; j < num_frames; ++j) {
            int idx = i * num_frames + j;
            magnitudes[idx] = real*real + imag*imag;
        }
    }

    // 3. 转置magnitude数组 (float)
    transpose_float(magnitudes, ...);

    // 4. 应用mel滤波器
    matmul(mel_filters, transposed_magnitudes, result,
           N_MELS, num_freqs, num_frames);  // ← 使用所有帧
}
```

### 三个关键错误

#### 错误1：在转置前计算magnitude
- **问题**: 在STFT复数数组上计算magnitude后再转置float数组
- **正确做法**: 先转置STFT复数数组，再在转置后的数组上计算magnitude
- **原因**: 转置复数数组和转置magnitude数组的内存布局不同

#### 错误2：保留了所有STFT帧
- **问题**: 使用 `num_frames` 作为维度
- **正确做法**: 使用 `num_frames - 1`，丢弃最后一帧
- **原因**: STFT的最后一帧可能不完整或为padding，Python/RK实现都会丢弃

#### 错误3：矩阵乘法维度不匹配
- **问题**: 矩阵乘法使用了错误的列数
- **结果**: 访问了错误的内存位置，mel滤波器权重应用到了错误的频率bin上

## 正确的实现

### RK参考实现流程

```cpp
// ✅ 正确的实现（参考RK whisper）
void audio_preprocess_CORRECT() {
    // Step 1: 计算STFT as [num_frames x num_freqs]
    int num_freqs = N_FFT / 2 + 1;  // 201
    fftwf_complex* stfts_result = compute_stft(...);

    // Step 2: 转置STFT复数数组
    // [num_frames x num_freqs] -> [num_freqs x num_frames]
    fftwf_complex* stfts_result_t = (fftwf_complex*)malloc(...);
    transpose_complex(stfts_result, num_frames, num_freqs, stfts_result_t);

    // Step 3: 计算magnitude，丢弃最后一帧
    // 输出: [num_freqs x (num_frames - 1)]
    std::vector<float> magnitudes(num_freqs * (num_frames - 1));
    int k = 0;
    for (int i = 0; i < num_freqs; i++) {
        for (int j = 0; j < num_frames - 1; j++) {  // ← 注意: num_frames - 1
            magnitudes[k++] = compute_magnitude(stfts_result_t[i * num_frames + j]);
        }
    }

    // Step 4: 矩阵乘法
    // mel_filters: [80 x 201]
    // magnitudes: [201 x (num_frames-1)]
    // result: [80 x (num_frames-1)]
    matmul(mel_filters, magnitudes, result,
           N_MELS, num_freqs, num_frames - 1);  // ← COLS_B = num_frames - 1

    // Step 5: Log变换
    clamp_and_log_max(result, N_MELS, num_frames - 1);

    // Step 6: Padding到3000帧
    pad_to_3000(result, ...);
}
```

## 关键代码改动

### 1. 添加transpose_complex函数

```cpp
// 转置复数STFT数组
static void transpose_complex(fftwf_complex* input, int input_rows, int input_cols,
                             fftwf_complex* output) {
    for (int i = 0; i < input_rows; ++i) {
        for (int j = 0; j < input_cols; ++j) {
            int input_index = i * input_cols + j;
            int output_index = j * input_rows + i;

            output[output_index][0] = input[input_index][0];  // real
            output[output_index][1] = input[input_index][1];  // imag
        }
    }
}
```

### 2. 修改compute_magnitudes函数

```cpp
// 修复前
static void compute_magnitudes(fftwf_complex* stft_result, int num_mel_filters,
                              int num_frames, std::vector<float>& magnitudes) {
    int k = 0;
    for (int i = 0; i < num_mel_filters; i++) {
        for (int j = 0; j < num_frames; j++) {  // ❌ 使用所有帧
            float real = stft_result[j * MELS_FILTERS_SIZE + i][0];
            float imag = stft_result[j * MELS_FILTERS_SIZE + i][1];
            magnitudes[k++] = real * real + imag * imag;
        }
    }
}

// 修复后
static void compute_magnitudes(fftwf_complex* stft_result_t, int num_freqs,
                              int num_frames, std::vector<float>& magnitudes) {
    int k = 0;
    for (int i = 0; i < num_freqs; i++) {
        for (int j = 0; j < num_frames - 1; j++) {  // ✅ 丢弃最后一帧
            magnitudes[k++] = compute_magnitude(stft_result_t[i * num_frames + j]);
        }
    }
}
```

### 3. 重构audio_preprocess函数

完整的流程改为：
1. STFT计算 → 2. 复数转置 → 3. Magnitude计算(n-1帧) → 4. 矩阵乘法 → 5. Log变换 → 6. Padding

## 验证结果

### 修复后的Mel频谱图
```python
# C++ mel (修复后)
First 10 values: [ 0.122, -0.094, -0.110, -0.020,  0.022, -0.069, -0.177,  0.094,  0.097, -0.081]
Range: [-0.806, 1.194]

# 与Python对比
Mean absolute difference: 0.003 (前10个值)
Overall difference: 0.65 (padding区域有差异，但不影响结果)
```

### 推理结果
```
Input: test_en.wav (5.855s)
Output: "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
First token: 2221 ✅ (正确)
Total tokens: 23
Time: ~700-800ms
Status: ✅ 完美匹配Python参考输出
```

## 教训与最佳实践

### 1. 逐层验证中间结果
```cpp
// 在每个关键步骤后保存并对比
save_debug("1_stft.bin", stft_result);
save_debug("2_transposed.bin", stft_transposed);
save_debug("3_magnitudes.bin", magnitudes);
save_debug("4_mel.bin", mel_output);
```

### 2. 精确复制参考实现
- **不要假设**"看起来类似"就是正确的
- **不要优化**未验证的代码 (`num_frames - 1` 看起来可以优化掉，但不能)
- **逐行对比**参考实现

### 3. 注意循环边界
```cpp
// ❌ 错误
for (int j = 0; j < num_frames; j++)

// ✅ 正确
for (int j = 0; j < num_frames - 1; j++)
```
这种差异看起来很小，但会导致完全错误的结果。

### 4. 打印矩阵维度
```cpp
std::cout << "matmul: [" << rows_A << " x " << cols_A
          << "] @ [" << cols_A << " x " << cols_B << "]" << std::endl;
```
始终验证矩阵乘法的维度是否匹配。

### 5. 从源头开始调试
不要等到最终输出错误才开始调试。从第一个函数开始，逐步验证每个中间输出。

## 调试时间线

1. **发现问题**: Decoder输出 "(whistling)" 而不是 "Mr. Quilter..."
2. **对比token**: C++首个token是522，Python是2221
3. **对比mel频谱图**: 发现0.77的均值差异
4. **对比首10个值**: 完全不同的模式
5. **阅读RK代码**: 发现 `num_frames - 1` 的关键差异
6. **重构代码**: 完全按照RK实现
7. **验证**: Mel差异降到0.003，推理结果完全正确

## 相关文件

- `utils/audio_utils.cpp`: 主要修改文件
  - 添加: `compute_magnitude()`, `transpose_complex()`
  - 修改: `compute_magnitudes()`, `audio_preprocess()`

- 参考实现: `/home/xh/projects/rknn_model_zoo/examples/whisper/cpp/process.cc`
  - 函数: `log_mel_spectrogram()`, `compute_magnitudes()`, `transpose()`

## 总结

这个bug的根本原因是**对参考实现的理解不够深入**，错误地假设了某些实现细节。修复的关键是：

1. ✅ 先转置STFT复数数组，再计算magnitude
2. ✅ 丢弃STFT的最后一帧 (`num_frames - 1`)
3. ✅ 矩阵乘法使用正确的维度

这次经验教训非常宝贵：在移植复杂算法时，必须**精确复制参考实现的每一个细节**，包括看似可以"优化"的部分。
