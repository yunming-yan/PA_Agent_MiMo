# PA Agent — AI K线分析辅助工具（MiMo 魔改版）

**原作者 UP 主：[阿尔法量化价格行为](https://space.bilibili.com/437555998)**

**原项目地址：https://github.com/rosemarycox5334-debug/PA_Agent**

---

## 关于本版本

本版本是基于原版 PA Agent 的**魔改版本**，主要针对**小米 MiMo 模型**进行了深度适配和优化。

### 主要改动

#### 1. MiMo 模型深度适配
- 支持 `mimo-v2.5-pro[1m]` 模型名（自动剥离 `[1m]` 后缀，与 Claude Code 行为一致）
- 适配 Anthropic 协议，支持 1M 上下文窗口
- 优化 thinking budget 参数，符合 Anthropic API 规范

#### 2. 重试机制优化
- 新增**思考耗尽检测**：当模型思考过长导致内容截断时自动降级
- 自动调整 `reasoning_effort`：`max → high → medium → low`
- 避免重试时思考深度暴降（从 30 万 token 降到 1 万 token）

#### 3. UI 体验改进
- **决策面板**：添加 QScrollArea，窗口缩小时可滚动查看全部内容
- **未来走势预期**：添加 QScrollArea，解决内容截断问题
- **默认全屏启动**：避免非全屏状态下的显示问题

#### 4. 性能优化
- TradingView 连通性检测移到**后台线程**，不再阻塞 UI（原版会卡顿 69 秒）
- 线程等待超时从 5 秒降到 200 毫秒，快速切换不再冻结
- 决策树可视化 FX 定时器只在 tab 可见时运行，节省 CPU
- 增量分析记录扫描添加缓存，避免每秒重复文件 I/O

---

## 环境要求

- **操作系统**：Windows 10/11
- **Python**：3.10+
- **数据源**：MT5 / TradingView / A股（AkShare）

---

## 安装步骤（适用于 Claude Code 等工具）

### 方式一：使用 Claude Code 安装

1. 克隆仓库：
```bash
git clone https://github.com/YOUR_USERNAME/PA_Agent.git
cd PA_Agent
```

2. 使用 Claude Code 安装依赖：
```bash
# 在 Claude Code 中运行
pip install -e .
```

3. 配置模型：
```bash
# 复制示例配置
cp config/settings.example.json config/settings.json

# 编辑配置文件，填入你的 API 密钥
# model: "mimo-v2.5-pro[1m]"
# base_url: "https://token-plan-sgp.xiaomimimo.com/anthropic"
# api_key: "你的 API 密钥"
```

### 方式二：手动安装

1. 克隆仓库：
```bash
git clone https://github.com/YOUR_USERNAME/PA_Agent.git
cd PA_Agent
```

2. 创建虚拟环境：
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. 安装依赖：
```bash
pip install -e .
```

4. 配置模型：
```bash
# 复制示例配置
copy config\settings.example.json config\settings.json

# 编辑 config\settings.json，填入你的 API 密钥
```

---

## 配置说明

### MiMo 模型配置示例

```json
{
  "provider": {
    "model": "mimo-v2.5-pro[1m]",
    "base_url": "https://token-plan-sgp.xiaomimimo.com/anthropic",
    "api_key": "你的 API 密钥",
    "thinking": true,
    "reasoning_effort": "max",
    "context_window": 1048576,
    "max_output_tokens": 128000
  }
}
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `model` | 模型名称 | `mimo-v2.5-pro[1m]` |
| `base_url` | API 地址 | MiMo 官方地址 |
| `thinking` | 启用思考模式 | `true` |
| `reasoning_effort` | 思考深度 | `max` |
| `context_window` | 上下文窗口 | `1048576`（1M） |
| `max_output_tokens` | 最大输出 token | `128000` |

---

## 启动程序

```bash
# 方式一：使用批处理脚本
运行智能体.bat

# 方式二：直接运行
python -m pa_agent.main
```

---

## 测试配置环境

### 验证安装

```bash
# 测试导入
python -c "from pa_agent.main import main; print('OK')"

# 测试模型连接
python -c "
from pa_agent.config.settings import load_settings
from pa_agent.ai.deepseek_client import DeepSeekClient
s = load_settings()
client = DeepSeekClient(s.provider)
reply = client.chat([{'role': 'user', 'content': 'hi'}], thinking=False)
print(f'Model: {reply.content[:50]}')
"
```

### 运行测试

```bash
# 运行单元测试
python -m pytest tests/unit/ -q

# 运行集成测试
python -m pytest tests/integration/ -q
```

---

## 目录结构

```
PA_Agent/
├── pa_agent/
│   ├── ai/                    # AI 客户端、提示词、验证器
│   ├── config/                # 配置文件
│   ├── data/                  # 数据源（MT5、TradingView、AkShare）
│   ├── gui/                   # PyQt6 界面
│   ├── orchestrator/          # 两阶段分析流程
│   ├── records/               # 分析记录
│   └── util/                  # 工具函数
├── config/
│   ├── settings.example.json  # 配置示例
│   └── settings.json          # 用户配置（不提交）
├── tests/                     # 测试文件
└── README.md
```

---

## 常见问题

### Q: 为什么选择 MiMo 模型？
A: MiMo 是小米推出的国产大模型，支持 1M 上下文窗口，价格实惠，中文理解能力强，适合量化分析场景。

### Q: 如何获取 MiMo API 密钥？
A: 访问 [小米 MiMo 开放平台](https://mimo.mi.com/) 注册并获取 API 密钥。

### Q: 支持其他模型吗？
A: 支持任何 OpenAI 兼容接口的模型（DeepSeek、PackyAPI、云雾等），只需修改 `config/settings.json` 中的 `base_url` 和 `model`。

### Q: 程序卡顿怎么办？
A: 本版本已优化性能，如果仍卡顿，请检查：
1. 网络连接是否稳定
2. TradingView 数据源是否可达
3. 降低 `analysis_bar_count` 参数

---

## 致谢

- 原作者：[阿尔法量化价格行为](https://space.bilibili.com/437555998)
- 原项目：https://github.com/rosemarycox5334-debug/PA_Agent
- MiMo 模型：[小米 MiMo](https://mimo.mi.com/)

---

## 免责声明

本工具仅供学习和研究使用，不构成任何投资建议。使用本工具进行交易决策的风险由用户自行承担。
