# MTK NPU Subagent 系统

## Subagent 列表

| Subagent | 版本 | 职责 | 输出文件 |
|---------|------|------|---------|
| **project-initializer** | v2.1 | 项目初始化、环境配置、baseline测试 | `{project}/mtk/.context/baseline.md` |
| **operator-analyst** | v2.1 | 算子兼容性分析、具体修改方案 | `{project}/mtk/.context/operator_analysis.md` |
| **python-converter** | v2.1 | Python端转换（.pt→TorchScript→TFLite→DLA） | DLA模型、debug输出 |
| **cpp-implementer** | v5.2 | C++推理实现、编译、部署、测试 | 最终可执行程序 |

## 设计原则

1. **每个subagent自闭环**：做完自己的事 → 自己验证 → 有问题自己修 → 确认无误后返回
2. **做一步验一步**：每个Step都有明确的验证标准和失败修复策略
3. **精简指令**：模板只包含执行指令，不包含说明性文字
4. **Context传递机制**：通过 `.context/` 目录的md文件在subagent间传递信息

## 信息注入机制

### 用户提供的上下文信息

用户提示词中通常包含以下信息，主Agent需要将其注入给相关subagent：

| 信息类型 | 示例 | 接收的subagent |
|---------|------|---------------|
| 项目路径 | `whisper-kv-cache` | 全部 |
| 模型信息 | "带KV cache的whisper base模型" | 全部 |
| 目标平台 | `MT8371` | python-converter, cpp-implementer |
| Conda环境 | `MTK-whisper, python=3.10` | project-initializer, python-converter |
| MTK SDK路径 | `/home/xh/projects/MTK_models_zoo/0_Toolkits/...` | python-converter, cpp-implementer |
| Android NDK | `/home/xh/Android/Ndk/android-ndk-r25c` | cpp-implementer |
| 参考项目 | `/home/xh/projects/rknn_model_zoo/examples/whisper` | operator-analyst, cpp-implementer |
| 特殊要求 | "带KV cache"、"固定形状30s" | operator-analyst, python-converter |

### 主Agent注入方式

启动subagent时，主Agent需要在prompt中包含：

```
启动 {subagent_name}，注入以下信息：

## 项目信息
- 项目路径: /home/xh/projects/MTK_models_zoo/whisper-kv-cache
- 模型: OpenAI Whisper Base (带KV cache)
- 目标平台: MT8371

## 环境配置
- Conda环境: MTK-whisper (python 3.10)
- 需要clone并切换: MTK-clip-encoder

## 资源路径
- MTK SDK: /home/xh/projects/MTK_models_zoo/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk
- Android NDK: /home/xh/Android/Ndk/android-ndk-r25c
- RK Whisper参考: /home/xh/projects/rknn_model_zoo/examples/whisper

## 前序Context（如果有）
{读取 .context/*.md 文件内容}

## 你的任务
{从 subagent 模板中复制任务描述}
```

## Context 传递机制

### 目录结构
```
{project}/mtk/.context/
├── baseline.md              # project-initializer 生成，包含baseline结果
├── operator_analysis.md     # operator-analyst 生成，包含不支持的算子和修改方案
└── ...                      # 其他中间上下文
```

### 传递流程
```
用户上下文 → 主Agent注入 → subagent执行 → 生成.context/md → 主Agent读取并传递给下个subagent
```

### 主Agent职责
1. **解析用户提示词**：提取项目路径、模型信息、环境配置、资源路径等
2. **注入Context**：将解析出的信息 + `.context/*.md` 文件传递给subagent
3. **调度顺序**：按标准流程依次启动subagent
4. **报告结果**：每步完成后向用户报告，确认后再继续

### Subagent职责
1. **读取注入信息**：从主Agent的prompt中获取项目信息、环境配置、资源路径
2. **读取Context**：从 `.context/` 目录读取前序subagent生成的信息
3. **执行任务**：按照各自模板的流程执行
4. **生成Context**：完成任务后，生成对应的md文件到 `.context/`
5. **自修复**：遇到问题尽量自己修复，不返回给主Agent

## 使用方式

```
用户 → 主Agent解析 → Task(模板 + 用户信息 + Context文件) → subagent执行 → 生成.context/md → 返回结果
```

## 标准流程

```
1. project-initializer
   输入: 项目路径、模型路径、Conda环境
   输出: baseline.md

2. operator-analyst
   输入: baseline.md、模型代码、参考项目路径
   输出: operator_analysis.md

3. python-converter
   输入: operator_analysis.md、SDK路径、目标平台配置
   输出: DLA模型、debug输出

4. cpp-implementer
   输入: Python端debug输出、NDK路径、SDK运行时库路径
   输出: 可执行程序、测试结果
```

每步的输出是下一步的输入。每步完成后主Agent向用户报告结果，用户确认后再进行下一步。

## Context 文件格式

### baseline.md 示例
```markdown
# Baseline 测试结果

## 模型信息
- 模型路径: xxx
- 输入shape: xxx
- 输出shape: xxx

## 测试结果
- 测试文件: xxx.wav
- 输出文本: xxx
- 推理时间: xxx
```

### operator_analysis.md 示例
```markdown
# 算子兼容性分析

## 不支持的算子
| 算子 | 使用位置 | 替换方案 |
|-----|---------|---------|
| torch.nn.Embedding | encoder.token_embed | 分离到CPU，改为输入 |

## 模型修改方案
1. 移除embedding层，改为输入token对应的embedding
2. masked_fill改为加法mask
...
```

## 配套资源

- 算子列表: `/home/xh/projects/MTK_models_zoo/.claude/doc/mtk_mdla_operators.md`
- 知识库: `/home/xh/projects/MTK_models_zoo/.claude/doc/mtk_npu_knowledge_base.md`
- 输出规范: `/home/xh/projects/MTK_models_zoo/.claude/standards/python_output_management.md`

## 版本历史

- v2.2 (2026-02-09): 增加"信息注入机制"章节，明确主Agent如何解析用户提示词并注入给subagent
- v2.1 (2026-02-09): 明确Context传递机制，增加.context目录说明
- v2.0 (2026-02-09): 全面重构，强化执行→验证→修复闭环，精简模板
- v1.0 (2026-02-04): 初始版本，基于Whisper项目验证
