# Subagent系统使用示例

本文档演示主Agent如何使用subagent模板来处理用户的移植任务。

---

## 📝 场景：用户请求移植新算法

### 用户输入（详细prompt）

```
YOLO-v8移植到MTK NPU推理测试

目标：YOLO-v8移植到MTK MT8371芯片上用NPU跑起来

环境：conda环境 MTK-detection python=3.10

官方仓库：https://github.com/ultralytics/ultralytics
模型：yolov8n.pt (6.2M参数)
测试图片：COCO验证集样本

参考项目：无（首个目标检测模型）

当前意图：
1. 初始化项目，检查环境
2. 完成Python端工作
3. 逐步验证输出
```

---

## 🤖 主Agent处理流程

### 步骤1：解析用户prompt

主Agent提取关键信息：
```json
{
  "algorithm": "YOLO-v8",
  "model_variant": "yolov8n",
  "type": "目标检测",
  "platform": "MT8371",
  "conda_env": "MTK-detection",
  "python_version": "3.10",
  "official_repo": "https://github.com/ultralytics/ultralytics",
  "model_file": "yolov8n.pt",
  "test_data": "COCO验证集样本",
  "reference_project": null,
  "current_intent": ["初始化", "Python端转换"]
}
```

### 步骤2：选择并读取模板

主Agent决定调用`project-initializer`：

```python
# 读取模板
template = read_file("/home/xh/projects/MTK/.claude/subagents/project-initializer.md")
```

### 步骤3：组合模板和用户信息

主Agent生成完整的subagent指令：

```python
full_prompt = f"""
{template}

---

## 当前任务信息（从用户提供）

算法名称: YOLO-v8
模型文件: yolov8n.pt
算法类型: 目标检测
官方仓库: https://github.com/ultralytics/ultralytics
目标平台: MT8371

Conda环境配置:
- 环境名: MTK-detection
- Python版本: 3.10
- 需要安装: ultralytics, torch, opencv-python

测试数据:
- 来源: COCO验证集样本
- 需要: 2-3张典型图片

参考项目:
- 目录结构参考: /home/xh/projects/MTK/superResolution/edsr/mtk

用户意图:
1. 创建项目目录结构
2. 配置conda环境
3. 下载模型和测试数据
4. 执行baseline测试

请按照模板流程执行项目初始化。
"""
```

### 步骤4：调用Task工具启动subagent

```python
Task(
    subagent_type="general-purpose",
    description="初始化YOLO-v8 MTK项目",
    prompt=full_prompt
)
```

### 步骤5：接收并汇总结果

Subagent返回：
```
✅ 项目初始化完成
- 目录: /home/xh/projects/MTK/detection/yolo-v8/mtk/
- 环境: MTK-detection (Python 3.10) ✓
- 模型: yolov8n.pt (6.2MB) ✓
- 数据: 3个测试图片 ✓
- Baseline: 测试通过 ✓

Baseline结果:
- image1.jpg: 检测到3个对象 (person, car, dog)
- image2.jpg: 检测到5个对象
- image3.jpg: 检测到2个对象

下一步: 算子分析
```

### 步骤6：主Agent反馈用户

```
YOLO-v8项目初始化完成！✓

完成的工作:
✓ 项目目录已创建
✓ Conda环境已配置
✓ 模型已下载 (yolov8n.pt, 6.2MB)
✓ 测试数据已准备
✓ Baseline测试通过

Baseline识别结果:
- image1.jpg: person, car, dog
- image2.jpg: 5个对象
- image3.jpg: 2个对象

下一步建议:
1. 算子分析 - 检查YOLO-v8的算子支持情况
2. Python端转换 - 开始模型转换

请告诉我："继续算子分析" 或 "继续Python转换"
```

---

## 🔄 完整工作流示例

### 阶段1：项目初始化 ✓

```
用户 → 主Agent → project-initializer → 结果
                  (读取模板 + 组合信息)
```

### 阶段2：算子分析

```
用户: "继续算子分析"
  ↓
主Agent读取: operator-analyst.md
  ↓
主Agent组合: 模板 + YOLO-v8信息 + 上一步结果
  ↓
主Agent调用: Task(..., operator-analyst的完整指令)
  ↓
返回: 算子支持报告
```

### 阶段3：Python转换

```
用户: "继续Python转换"
  ↓
主Agent读取: python-converter.md
  ↓
主Agent组合: 模板 + 用户信息 + 算子分析结果
  ↓
主Agent调用: Task(..., python-converter的完整指令)
  ↓
返回: DLA模型 + 测试报告
```

### 阶段4：C++实现（后续）

```
用户: "继续C++实现"
  ↓
主Agent读取: cpp-implementer.md
  ↓
... (类似流程)
```

---

## 💡 关键设计点

### 1. 模板是"专业知识库"
- 包含详细的流程、约束、最佳实践
- 基于实战经验不断优化
- 可复用于不同算法

### 2. 主Agent是"信息分发中心"
- 解析用户需求
- 选择合适的模板
- 组合信息
- 调用subagent
- 汇总结果

### 3. Subagent是"专业执行者"
- 接收完整指令（模板+具体信息）
- 独立工作上下文
- 专注执行
- 返回结构化结果

### 4. 用户参与关键决策
- 在检查点验证输出
- 决定是否继续下一步
- 提供反馈和调整

---

## 🎯 优势总结

### vs 单一大Agent
- ✅ 上下文更干净（每个subagent独立）
- ✅ 专业性更强（每个agent专注一个领域）
- ✅ 可维护性更好（模板独立更新）
- ✅ 复用性更高（模板可用于不同项目）

### vs 完全自动化
- ✅ 质量更可控（检查点机制）
- ✅ 用户参与决策（关键步骤确认）
- ✅ 问题可及时修正（不会一路错到底）
- ✅ 学习效果更好（用户理解每一步）

### vs 纯手动
- ✅ 效率更高（自动化执行）
- ✅ 一致性更好（遵循标准流程）
- ✅ 不易遗漏（模板包含完整步骤）
- ✅ 经验积累（模板持续优化）

---

**示例版本**: v1.0  
**日期**: 2026-02-04
