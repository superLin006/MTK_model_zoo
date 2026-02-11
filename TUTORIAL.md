# MTK Model Zoo 使用教程

> **MTK NPU 算法移植 — 多 Agent 协作流水线使用指南**

本教程将指导您如何使用 MTK Model Zoo 中的 Claude Code Subagent 系统，快速完成深度学习模型从 PyTorch 到 MTK NPU 的完整移植流程。

---

## 目录

1. [环境准备](#1-环境准备)
2. [克隆项目](#2-克隆项目)
3. [使用提示词开始移植](#3-使用提示词开始移植)
4. [提示词模板](#4-提示词模板)
5. [常见问题](#5-常见问题)

---

## 1. 环境准备

### 必需工具

#### MTK NeuroPilot SDK
- **版本**: 8.0.10 或更高
- **下载**: [MTK NeuroPilot 官网](https://neuropilot.mediatek.com/)
- **放置位置**: 解压到项目根目录下的 `0_Toolkits/` 目录

```
MTK_models_zoo/
└── 0_Toolkits/
    └── neuropilot-sdk-basic-8.0.10-build20251029/
        └── neuron_sdk/
```

> 所有 Python 脚本和 Shell 脚本会自动从该相对路径查找 SDK，**无需手动配置路径**。
> 如需使用自定义路径，可设置环境变量：
> ```bash
> export MTK_NEURON_SDK=/your/custom/neuron_sdk
> export MTK_CONVERTER_PATH=/your/custom/neuron_sdk/host/lib/python
> ```

#### Android NDK
- **推荐版本**: r25c
- **下载**: [Android NDK 官网](https://developer.android.com/ndk/downloads)
- **配置**: 运行构建脚本前设置环境变量

```bash
export ANDROID_NDK=/path/to/android-ndk-r25c
```

### Python 环境

- **版本**: Python 3.10
- **推荐**: 使用 Conda 创建独立环境

```bash
# 按需创建对应的 Conda 环境
conda create -n MTK-whisper python=3.10
conda create -n MTK-superResolution python=3.10

# 激活环境后安装依赖
conda activate MTK-superResolution
pip install torch torchvision pillow opencv-python
```

---

## 2. 克隆项目

```bash
git clone https://github.com/superLin006/MTK_model_zoo.git
cd MTK_model_zoo

# 将 MTK SDK 放入项目目录
# 0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/

# 将模型权重放入对应子项目的 models/ 目录
# 例如: superResolution/realesrgan/mtk/models/RealESRGAN_x4plus.pth
```

---

## 3. 使用提示词开始移植

克隆并准备好 SDK 后，即可使用 Claude Code 配合提示词模板开始移植。

**工作流程**：

```
提示词输入
    ↓
Agent 自动工作
    ↓
├─ 项目初始化 (project-initializer)
├─ 算子分析 (operator-analyst)
├─ Python 端转换 (python-converter)
│   ├─ .pt → TorchScript
│   ├─ TorchScript → TFLite
│   └─ TFLite → DLA
├─ C++ 实现 (cpp-implementer)
└─ Android 部署 (android-deployer)
    ↓
完成移植
```

**使用步骤**：

1. 在项目根目录打开 Claude Code
2. 复制对应的提示词模板（见下节）
3. 将您的实际路径替换提示词中的占位符后发送
4. Agent 自动完成整个移植流程

---

## 4. 提示词模板

### 4.1 Whisper 移植提示词模板

适用于：**语音识别模型**（Whisper、SenseVoice 等）

![提示词示例](3_assert/prompts_2.png)

<details>
<summary>点击展开完整提示词</summary>

```
### Whisper 移植到 MTK NPU 推理测试

**1 完整且丰富的上下文**

**目标：** whisper 移植到 MTK8371 这块芯片上用 NPU 跑起来，推送到检测到的 Android 设备在 Android 环境下完成测试。

**历史以及计划：** 整个移植的流程可以分为三个大的步骤

#### 第一步（Python端）：从 pytorch.bin/pt → tflite → dla

**(1) 转换脚本**
- 每个环节需要单独的转换脚本：
  - `step1_pt_to_torchscript.py`
  - `step2_torchscript_to_tflite.py`
  - `step3_tflite_to_dla.py`
- 可以把 openai whisper 官方仓库 clone 下来，再下载 base 模型
- 它里面有原始模型的架构定义、模型数据权重、测试数据等以及对应的推理代码

**(2) 测试脚本**
- 每个转换节点完成后需要单独测试：
  - `test_pytorch.py`
  - `test_pt.py`
  - `test_tflite.py`
- 这个部分的 `test_pytorch.py` 可以参考 whisper 官方项目的 python 推理脚本

**(3) 对比测试**
- 最后完整的对比不同格式模型差异：`test_compare.py`
- 即拿到 .pt 和 .tflite 格式都推理测试一遍，检查输出是否正确
- 正确后再进行下一个节点的转换（dla 文件除外，后面 C++ 端再做验证）
- 参考 edsr/mtk 的目录格式重组
- 工作主要在 python 端

#### 第二步（C++端）：将 Python 推理代码转换为 C++ 推理代码

**(1) 实现要点**
- 参考 rk 的 whisper 项目的 C++ 部分相关代码以及复用他用到的第三方库
- 严格复刻 python 端的行为（`test_torchscript.pt`）
- 参考 edsr 项目了解 MTK API 接口是如何调用的
- 使用正确的 MTK API 接口替换 RKNN API 接口进行推理

#### 第三步（Android端）：编译好相关的 C++ 代码（针对 MT8371 平台），推送到检测到的 Android 设备完成推理测试

---

**2 目前的环境**

**Conda 环境：** MTK-whisper，Python = 3.10

**项目根目录：** MTK_models_zoo/（所有路径均相对于此）

**其他资源目录如下：**
- (1) MTK SDK: `0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk`
- (2) 目标平台 MT8371 运行时库: `0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/mt8371`
- (3) Android NDK: 通过 `$ANDROID_NDK` 环境变量指定
- (4) 第三方库文件: `1_third_party/`
- (5) MTK DLA 支持的算子信息列表: `.claude/doc/mtk_mdla_operators.md`
- (6) 中央知识库（已知问题和最佳实践）: `.claude/doc/mtk_npu_knowledge_base.md`
- (7) whisper 官方仓库: `whisper/whisper-official/`（脚本中已通过 `__file__` 自动定位）

---

**3 当前的意图**

1. 参考 mtk/edsr 项目，新建目录结构，检查环境，安装移植算法必要的依赖
2. 根据计划先完成 python 端的工作，有问题再分析解决
3. 完成 Python 端的工作后，我逐一检查输出结果，没有问题后，我再给出具体的下一步任务

---

**4 注意事项以及限制**

1. **模型分离处理**
   - 由于算法模型要跑在 MTK NPU 上，没办法一步实现端到端的输出
   - 导出模型的中间格式 tflite 以及转换后的 dla 格式，都只负责推理
   - 前处理和后处理都需要分离出来单独处理（其他的案例都是这么做的）

2. **算子支持问题**
   - 如果导出的模型没办法正常工作，或者报错了，那么可能是算子不支持的问题
   - 就要我们自己定义模型机构或者修改算子，再重新导出来重新测试
   - 我们先像其他案例一样，按照一般的方法正常导出
   - Helsinki 这个案例就是这么做的：`helsinki/`

3. **固定形状限制**
   - 由于 MTK NPU 不支持动态形状
   - 所以在转换为 tflite 格式的时候就得输入固定形状 (30s)

4. **代码复用原则**
   - 对于有参考代码的部分，能复用就复用
   - 这样不用重复造轮子，效率和准确性也能有效提升

5. **使用 MTK 工具**
   - 我们在转 tflite 格式以及用 tflite 格式模型推理测试的时候都得用 MTK 提供的它自己的工具
   - 不能用标准的 tflite 工具，不然会报错
   - 他有他自定义的算子，MTK 他的工具已经集成在环境中了
```

</details>

**适用场景**：Whisper、SenseVoice、其他 Transformer-based 音频模型

---

### 4.2 Real-ESRGAN 移植提示词模板

适用于：**超分辨率模型**（Real-ESRGAN、EDSR、RCAN 等）

![提示词示例](3_assert/prompts_1.png)

<details>
<summary>点击展开完整提示词</summary>

```
### Real-ESRGAN 移植到 MTK NPU 推理测试

**1 完整且丰富的上下文**

**目标：** Real-ESRGAN 移植到 MTK8371 这块芯片上用 NPU 跑起来，推送到检测到的 Android 设备在 Android 环境下完成测试。

**历史以及计划：** 整个移植的流程可以分为三个大的步骤

#### 第一步（Python端）：从 pytorch.bin/pt → tflite → dla

**(1) 转换脚本**
- 每个环节需要单独的转换脚本：
  - `step1_pt_to_torchscript.py`
  - `step2_torchscript_to_tflite.py`
  - `step3_tflite_to_dla.py`
- 可以参考 rk 的 realesrgan 项目，它里面有原始模型的架构定义、模型数据权重、测试图片等以及对应的推理代码

**(2) 测试脚本**
- 每个转换节点完成后需要单独测试：
  - `test_pytorch.py`
  - `test_pt.py`
  - `test_tflite.py`
- 这个部分的 `test_pytorch.py` 可以直接复制 rk 项目的 realesrgan 的 `test_pytorch.py`

**(3) 对比测试**
- 最后完整的对比不同格式模型差异：`test_compare.py`
- 即拿到 .pt 和 .tflite 格式都推理测试一遍，检查输出是否正确
- 正确后再进行下一个节点的转换（dla 文件除外，后面 C++ 端再做验证）
- 参考 edsr 的目录格式重组，以及 edsr 项目的 python 代码
- 工作主要在 python 端

#### 第二步（C++端）：将 Python 推理代码转换为 C++ 推理代码

**(1) 实现要点**
- 参考 rk 的 realesrgan 项目的 C++ 部分相关代码
- 参考 edsr 项目了解 MTK API 接口是如何调用的
- 使用正确的 MTK API 接口替换 RKNN API 接口进行推理

#### 第三步（Android端）：编译好相关的 C++ 代码（针对 MT8371 平台），推送到检测到的 Android 设备完成推理测试

---

**2 目前的环境**

**Conda 环境：** MTK-superResolution，Python = 3.10

**项目根目录：** MTK_models_zoo/（所有路径均相对于此）

**其他资源目录如下：**
- (1) MTK SDK: `0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk`
- (2) 目标平台 MT8371 运行时库: `0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/mt8371`
- (3) Android NDK: 通过 `$ANDROID_NDK` 环境变量指定
- (4) 第三方库文件及工具: `1_third_party/`
- (5) MTK DLA 支持的算子信息列表: `.claude/doc/mtk_mdla_operators.md`
- (6) 中央知识库（已知问题和最佳实践）: `.claude/doc/mtk_npu_knowledge_base.md`
- (7) rk realesrgan 项目（参考）: `superResolution/realesrgan/rknn/`

---

**3 当前的意图**

1. 参考 mtk/edsr 项目，新建目录结构，检查环境，安装移植算法必要的依赖
2. 根据计划先完成 python 端的工作，有问题再分析解决
3. 完成 Python 端的工作后，我逐一检查输出结果，没有问题后，我再给出具体的下一步任务

---

**4 注意事项以及限制**

1. **模型分离处理**
   - 由于算法模型要跑在 MTK NPU 上，没办法一步实现端到端的输出
   - 导出模型的中间格式 tflite 以及转换后的 dla 格式，都只负责推理
   - 前处理和后处理都需要分离出来单独处理（其他的案例都是这么做的）

2. **算子支持问题**
   - 如果导出的模型没办法正常工作，或者报错了，那么可能是算子不支持的问题
   - 就要我们自己定义模型机构或者修改算子，再重新导出来重新测试
   - 我们先像其他案例一样，按照一般的方法正常导出
   - Helsinki 这个案例就是这么做的：`helsinki/`

3. **固定形状限制**
   - 由于 MTK NPU 不支持动态形状
   - 所以在转换为 tflite 格式的时候就得输入固定形状 (510x339)

4. **代码复用原则**
   - 对于有参考代码的部分，能复用就复用
   - 这样不用重复造轮子，效率和准确性也能有效提升

5. **使用 MTK 工具**
   - 我们在转 tflite 格式以及用 tflite 格式模型推理测试的时候都得用 MTK 提供的它自己的工具
   - 不能用标准的 tflite 工具，不然会报错
   - 他有他自定义的算子，MTK 他的工具已经集成在环境中了

6. **Real-ESRGAN 算子特性**
   - Real-ESRGAN 是基于 RRDB（Residual in Residual Dense Block）的纯 CNN 架构
   - 主要使用的算子都是常规卷积操作（Conv2d、LeakyReLU、Add 等）
   - 这些算子 MTK 支持都很好
   - 需要注意的算子：PixelShuffle（用于上采样）、LeakyReLU（激活函数）

7. **前后处理要求**
   - 前处理：归一化到 [0,1] 范围
   - 后处理：输出 clip 到 [0,1] 范围，×255 转回 uint8
   - 这些都在模型外部单独实现
```

</details>

**适用场景**：Real-ESRGAN、EDSR、RCAN、其他 CNN-based 图像处理模型

---

### 4.3 自定义提示词

如果您要移植其他类型的模型，可以参考上述模板的结构进行自定义：

```
1. 完整且丰富的上下文
   - 目标描述
   - 移植流程（3大步骤）
   - 参考项目（使用项目内相对路径）

2. 目前的环境
   - Conda 环境名称
   - 资源目录（使用相对于项目根目录的路径）

3. 当前的意图
   - 具体要做的事情
   - 检查点

4. 注意事项以及限制
   - 模型特性
   - MTK NPU 限制
   - 最佳实践
```

---

## 5. 常见问题

### Q1: SDK 路径找不到怎么办？

**A**: 确认 SDK 已解压到项目根目录下的 `0_Toolkits/` 目录，路径结构为：
```
0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk/
```
或者手动设置环境变量：
```bash
export MTK_NEURON_SDK=/your/path/to/neuron_sdk
```

### Q2: 转换过程中报错"算子不支持"

**A**:
1. 查看 `.claude/doc/mtk_mdla_operators.md` 确认算子支持情况
2. 参考 `.claude/doc/mtk_npu_knowledge_base.md` 查找解决方案
3. 如果算子确实不支持，需要修改模型结构（参考 `helsinki/` 案例）

### Q3: TFLite 转换失败

**A**:
- 确保使用 MTK 提供的转换工具，不要使用标准 TFLite 工具
- 检查输入形状是否固定（MTK NPU 不支持动态形状）
- 超分辨率模型固定输入为 510x339（宽x高）

### Q4: C++ 编译失败

**A**:
- 确认已设置 `export ANDROID_NDK=/path/to/ndk`
- 确认 MTK SDK 存在于 `0_Toolkits/` 目录
- 参考各子项目 `cpp/` 目录下的 `README.md`

### Q5: 如何查看推理测试结果？

**A**:
- Python 端测试输出保存在各子项目的 `test_data/output/` 目录
- 例如：`superResolution/realesrgan/mtk/test_data/output/`
- C++ 端测试日志通过 `deploy_and_test.sh` 输出到终端

---

## 快速检查清单

在开始移植前，请确认：

- [ ] 已克隆 MTK Model Zoo 项目
- [ ] MTK SDK 已解压到 `0_Toolkits/` 目录
- [ ] `export ANDROID_NDK=/path/to/ndk` 已设置
- [ ] Python 环境已创建并激活
- [ ] 原始模型权重文件已放入 `models/` 目录
- [ ] 测试数据已准备好（图像/音频）
- [ ] Claude Code 可以正常运行

---

## 相关文档

- **Subagent 系统说明**: `.claude/subagents/README.md`
- **MTK 算子支持列表**: `.claude/doc/mtk_mdla_operators.md`
- **最佳实践知识库**: `.claude/doc/mtk_npu_knowledge_base.md`
- **项目概览**: `README.md`
