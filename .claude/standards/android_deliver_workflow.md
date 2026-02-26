# Android 推理测试交付规范

> 本文档用于指导 AI 自动化构建 Android ARM64 推理测试包。

---

## 流程概览

```
确认编译产物 → 整理 deliver/ 目录 → 本地验证 → 打包交付
```

---

## deliver/ 标准目录结构

```
deliver/
├── bin/                    # 可执行文件（ARM64）
├── lib/                    # 动态库（如有）
├── models/                 # 模型文件（按类型/语言分子目录）
├── test_data/              # 测试数据 + run_test.sh
└── README.md               # 使用说明
```

**规则：**
- 脚本必须用 `#!/bin/sh`（Android 无 bash）
- 每次从源头重新生成，不直接修改 deliver/

---

## 执行步骤

### Step 1：确认编译产物

检查编译输出目录，确认以下文件存在：
- `bin/` 下的可执行文件
- `lib/` 下的动态库（如有）
- 模型文件路径

### Step 2：创建 deliver/ 目录

```bash
rm -rf deliver
mkdir -p deliver/{bin,lib,models,test_data}
```

### Step 3：复制文件

```bash
# 可执行文件
cp <编译产物>/bin/* deliver/bin/
chmod +x deliver/bin/*

# 动态库（如有）
cp <编译产物>/lib/*.so deliver/lib/

# 模型文件（按项目实际情况整理）
cp -r <模型源目录>/* deliver/models/

# 测试数据
cp <测试数据>/* deliver/test_data/
```

### Step 4：创建 run_test.sh

根据项目类型选择对应模板：

**音频输入类（语音识别）：**
```sh
#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXEC="$ROOT_DIR/bin/<可执行文件>"
export LD_LIBRARY_PATH="$ROOT_DIR/lib:$LD_LIBRARY_PATH"

run_one() {
    audio="$1"
    echo "=== 测试: $(basename "$audio") ==="
    "$EXEC" <模型参数> "$audio"
}

ARG1="${1:-}"
if [ -z "$ARG1" ]; then
    for f in "$SCRIPT_DIR"/*.wav; do [ -f "$f" ] && run_one "$f"; done
else
    case "$ARG1" in /*) run_one "$ARG1" ;; *) run_one "$SCRIPT_DIR/$ARG1" ;; esac
fi
```

**文本输入类（翻译）：**
```sh
#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXEC="$ROOT_DIR/bin/<可执行文件>"
export LD_LIBRARY_PATH="$ROOT_DIR/lib:$LD_LIBRARY_PATH"

LANG_PAIR="${1:-<默认语言对>}"
INPUT="${2:-$SCRIPT_DIR/<默认输入文件>}"
"$EXEC" <模型参数> --input "$INPUT"
```

### Step 5：创建 README.md

```markdown
# <项目名> Android 测试包

## 环境要求
- Android ARM64 设备（API ≥ 29）
- 已开启 USB 调试

## 快速开始
```bash
# 推送
adb shell "rm -rf /data/local/tmp/<项目名>"
adb push . /data/local/tmp/<项目名>
adb shell "chmod +x /data/local/tmp/<项目名>/bin/*"
adb shell "chmod +x /data/local/tmp/<项目名>/test_data/*.sh"

# 运行
adb shell "cd /data/local/tmp/<项目名> && sh test_data/run_test.sh"
```

## 性能输出
- Init Time: 初始化耗时
- Inference Time: 推理耗时
- Peak RSS: 峰值内存
```

### Step 6：本地验证

```bash
adb devices
adb shell "rm -rf /data/local/tmp/<项目名>"
adb push deliver/ /data/local/tmp/<项目名>
adb shell "chmod +x /data/local/tmp/<项目名>/bin/*"
adb shell "chmod +x /data/local/tmp/<项目名>/test_data/*.sh"
adb shell "cd /data/local/tmp/<项目名> && sh test_data/run_test.sh"
```

### Step 7：打包

```bash
zip -r <项目名>_$(date +%Y%m%d).zip deliver/
```

---

## 性能输出要求

程序应输出以下指标（stdout 或 stderr）：
```
Init Time:       X.XXX s
Inference Time:  X.XXX s
Peak RSS:        XXX.X MB
```

---

## 常见问题

| 问题 | 解决 |
|-----|------|
| `bash: not found` | 用 `sh` 运行脚本 |
| `libXXX.so not found` | 设置 `LD_LIBRARY_PATH=./lib` |
| 立即 crash | 编译时用 `c++_static` |

---

## 项目配置（AI 填充）

执行时根据实际项目填充以下信息：

| 配置项 | 值 |
|-------|-----|
| 项目名 | |
| 可执行文件 | |
| 编译产物路径 | |
| 模型路径 | |
| 测试数据路径 | |
| 模型参数 | |
| 设备目录 | /data/local/tmp/<项目名> |
