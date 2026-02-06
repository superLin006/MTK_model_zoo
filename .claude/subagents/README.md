# MTK NPU Subagent 系统

这个目录包含MTK NPU算法移植工作流程的所有subagent模板。

## 📚 Subagent列表

| Subagent | 状态 | 版本 | 职责 |
|---------|------|------|------|
| **project-initializer** | ✅ 已验证 | v1.0 | 项目初始化、环境配置、baseline测试 |
| **operator-analyst** | ✅ 已验证 | v1.0 | 算子分析、兼容性检查、修改建议 |
| **python-converter** | ✅ 已验证 | v1.0 | Python端完整转换（.pt→.tflite→.dla） |
| **cpp-implementer** | ✅ 已验证 | v3.0 | C++推理实现、预处理精确复制、MTK API调用 |
| **android-deployer** | ⏳ 待验证 | v2.0 | Android部署、NPU测试 |

## 🚀 使用方式

### 方式1：主Agent自动调用（推荐）

用户提供详细的移植任务prompt，主Agent会：
1. 解析prompt提取信息
2. 选择对应的subagent模板
3. 组合模板+用户信息生成完整指令
4. 调用Task工具启动subagent
5. 汇总结果返回给用户

**示例流程**：
```
用户: "移植YOLO-v8到MTK NPU..."
   ↓
主Agent读取: project-initializer.md
   ↓
主Agent组合: 模板 + 用户信息
   ↓
主Agent调用: Task(subagent_type="general-purpose", prompt=完整指令)
   ↓
Subagent执行并返回结果
```

### 方式2：直接读取模板（高级）

高级用户可以直接读取模板，手动调用：
```bash
# 读取模板
cat /home/xh/projects/MTK/.claude/subagents/python-converter.md

# 组合信息并调用Task工具
```

## 📖 模板文件说明

每个模板文件包含：

### 1. Subagent身份
- 名称和简短描述
- 核心职责

### 2. 输入信息规范
- 必需信息
- 可选信息
- 来源（从哪个subagent或用户）

### 3. 工作流程
- 详细的步骤说明
- 每步的输入输出
- 检查点机制

### 4. 输出规范
- 文件结构
- 报告格式
- 命名规范

### 5. 特殊约束
- 环境要求
- 工具限制
- 最佳实践

### 6. 参考资源
- 参考项目路径
- 工具路径
- 知识库路径

## 🔧 模板维护

### 版本管理
- **v1.x**：已验证，稳定可用
- **v0.x**：草案，待验证

### 更新流程
1. 通过实战验证功能
2. 记录经验和改进点
3. 更新模板文档
4. 更新版本号

### 验证状态
- ✅ **已验证**：通过实际项目验证，稳定可用
- ⏳ **待验证**：框架已创建，需要通过实战验证
- 🔄 **迭代中**：正在根据反馈优化

## 📊 验证记录

### Whisper ASR 项目（2025-02-05）
- ✅ **project-initializer** (v1.0)：成功初始化，创建标准结构
- ✅ **operator-analyst** (v1.0)：准确识别Embedding问题，提供解决方案
- ✅ **python-converter** (v1.0)：完整转换，生成DLA模型
- ✅ **cpp-implementer** (v3.0)：实现C++推理，输出正确
- ⏳ **android-deployer** (v2.0)：待完整验证

**Python端经验**：
1. ⚠️ 初始尝试ONNX路径（错误） → 优化：明确使用mtk_converter
2. ✅ 检查点机制工作良好
3. ✅ Embedding分离方案成功

**C++端经验（重点）**：
1. 🔴 **预处理是最大痛点**：mel频谱图计算错误导致3小时调试
   - 症状：输出 "(whistling)" 而非正确文本
   - 原因：3个"微小"差异（转置顺序、丢帧、维度不匹配）

2. ✅ **逐层验证必不可少**：每个处理步骤都要保存中间结果对比
   - 对比STFT、转置、magnitude、mel输出
   - 与Python输出逐个对比，快速定位问题

3. ⚠️ **微小差异致命**：`< n` vs `< n-1` 这种差异会让结果完全错误
   - 必须精确复制Python实现，不要"自己实现"
   - 循环边界、数组索引、处理顺序都要精确匹配

4. ✅ **目录结构标准化**：统一采用EDSR模式（所有源码在 `jni/src/`）
   - 避免源码分散（src/, include/, utils/）
   - Android.mk路径简单，无需 `../` 跳转

**最终结果**：
- ✅ 英文音频测试通过：正确输出 "Mr. Quilter is the apostle..."
- ✅ 推理时间：~700-800ms (MT8371 NPU)
- ✅ 代码整洁：目录结构符合标准

详细分析见：`/home/xh/projects/MTK/whisper/mtk/cpp/LESSONS_LEARNED.md`

## 🎯 设计原则

### 1. 专业分工
每个subagent专注一个明确的领域，避免职责重叠。

### 2. 上下文隔离
每个subagent有独立、干净的上下文，只注入必需信息。

### 3. 质量优先
不追求全自动化，关键步骤设置检查点，用户可验证。

### 4. 可复用性
模板设计通用化，适用于不同类型的算法。

### 5. 持续改进
每次实战后更新模板，融入新的经验和最佳实践。

## 📁 目录结构

```
/home/xh/projects/MTK/.claude/
├── subagents/                       # Subagent模板
│   ├── README.md                    # 本文件
│   ├── USAGE_EXAMPLE.md             # 使用示例
│   ├── project-initializer.md       # 项目初始化 ✅ v1.0
│   ├── operator-analyst.md          # 算子分析 ✅ v1.0
│   ├── python-converter.md          # Python转换 ✅ v1.0
│   ├── cpp-implementer.md           # C++实现 ✅ v3.0 (重点强化)
│   └── android-deployer.md          # Android部署 ⏳ v2.0
│
├── standards/                       # 规范文档
│   └── python_output_management.md  # Python输出管理规范 ⭐
│
└── doc/                             # 知识库文档
    ├── mtk_mdla_operators.md        # MTK支持的算子列表
    └── mtk_npu_knowledge_base.md    # 已知问题和最佳实践
```

## 🔗 相关资源

### 知识库
- `/home/xh/projects/MTK/.claude/doc/mtk_mdla_operators.md` - MTK支持的算子列表
- `/home/xh/projects/MTK/.claude/doc/mtk_npu_knowledge_base.md` - 已知问题和最佳实践

### MTK工具和SDK
- SDK路径：`/home/xh/projects/MTK/0_Toolkits/neuropilot-sdk-basic-8.0.10-build20251029/neuron_sdk`
- 文档和示例在SDK目录中

### 参考项目
- SuperResolution：`/home/xh/projects/MTK/superResolution/`
- Helsinki (Transformer)：`/home/xh/projects/MTK/helsinki/`
- SenseVoice (ASR)：`/home/xh/projects/MTK/sense-voice/`

## 📝 反馈和改进

如果在使用过程中发现：
- 模板不清晰的地方
- 缺失的重要信息
- 更好的实现方式
- 新的最佳实践

请及时更新对应的模板文件，并在本README中记录。

---

**创建日期**：2026-02-04
**最后更新**：2025-02-05
**维护者**：MTK NPU移植团队
