# 贡献指南

感谢您对 MTK Model Zoo 项目的关注！

## 📋 贡献方式

### 1. 报告 Bug
- 使用 GitHub Issues 报告问题
- 提供详细的复现步骤
- 包含平台信息（芯片型号、SDK版本等）

### 2. 提交新模型
添加新的模型实现时，请遵循以下结构：

```
{model_name}/
└── mtk/
    ├── python/              # Python 端转换
    │   ├── models/          # 转换后的模型（.gitkeep）
    │   ├── test/            # 测试脚本
    │   ├── step1_*.py       # 转换步骤1
    │   ├── step2_*.py       # 转换步骤2
    │   ├── step3_*.py       # 转换步骤3
    │   └── README.md
    │
    ├── cpp/                 # C++ 推理实现
    │   ├── jni/
    │   │   ├── Android.mk
    │   │   ├── Application.mk
    │   │   └── src/         # 所有源码
    │   ├── build_android.sh
    │   ├── deploy_android.sh
    │   └── README.md
    │
    ├── models/              # 原始模型权重（.gitkeep）
    ├── test_data/           # 测试数据（.gitkeep）
    └── README.md            # 模型说明
```

### 3. 提交代码

#### 代码规范
- **Python**: 遵循 PEP 8
- **C++**: 使用一致的命名和格式
- **注释**: 关键逻辑需要中文注释

#### Commit 规范
```
<type>: <subject>

<body>
```

**Type 类型:**
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具相关

**示例:**
```
feat: 添加 YOLOv8 模型支持

- 实现 Python 端转换脚本
- 实现 C++ 推理代码
- 添加测试用例
- 更新文档
```

### 4. Pull Request 流程

1. **Fork 仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/MTK_model_zoo.git
   cd MTK_model_zoo
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/add-yolov8
   ```

3. **开发和测试**
   - 实现功能
   - 本地测试通过
   - 添加文档

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加 YOLOv8 模型支持"
   ```

5. **推送到 Fork**
   ```bash
   git push origin feature/add-yolov8
   ```

6. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 填写 PR 模板
   - 等待 Review

## 📝 文档要求

### 模型 README 必须包含：
- 模型简介
- 输入输出规格
- 性能基准（推理时间、精度）
- 已知问题和限制
- 参考链接

### Python 转换脚本必须包含：
- 详细的注释
- 错误处理
- 中间输出保存（用于 C++ 对比）
- 使用说明

### C++ 实现必须包含：
- 预处理精确复制 Python 实现
- 逐层验证和调试代码
- 内存管理（无泄漏）
- 错误处理

## 🔍 代码审查要点

### Python 端
- [ ] 使用 `mtk_converter` 转换（不经过 ONNX）
- [ ] 遵循输出管理规范（`.claude/standards/python_output_management.md`）
- [ ] 保存中间输出到 `test/outputs/debug/`
- [ ] 测试脚本可运行

### C++ 端
- [ ] 遵循标准目录结构（所有源码在 `jni/src/`）
- [ ] 预处理精确匹配 Python 实现
- [ ] 使用 MTK Neuron API
- [ ] 无内存泄漏
- [ ] 编译脚本可用

### 文档
- [ ] README 完整
- [ ] 代码有注释
- [ ] 更新项目根目录的 README（新增模型）

## 🚫 不要提交的内容

❌ **绝对不要提交:**
- 模型权重文件 (*.pt, *.pth, *.tflite, *.dla)
- 测试数据 (音频、图像文件)
- 编译产物 (libs/, obj/, __pycache__)
- 中间输出 (test/outputs/)
- MTK SDK

✅ **应该提交:**
- 源代码
- 配置文件
- 脚本文件
- 文档
- `.gitkeep` 占位文件

## 🤝 社区规范

- 尊重他人
- 建设性的讨论
- 保持耐心
- 乐于助人

## 📧 联系方式

- GitHub Issues: 项目相关问题
- Email: [您的邮箱]

---

再次感谢您的贡献！🎉
