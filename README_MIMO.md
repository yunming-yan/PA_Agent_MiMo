# PA Agent - MiMo 兼容版

> 基于 [PA Agent](https://github.com/rosemarycox5334-debug/PA_Agent) 的二次修改版本，解决了小米 MiMo 系列模型无法使用的问题。

## 修改内容

### 核心改动：支持 MiMo Anthropic 协议

MiMo API 使用 Anthropic Messages 格式进行通信，而原版 PA Agent 仅支持 OpenAI/DeepSeek 协议。本次修改在底层 `deepseek_client.py` 中增加了 Anthropic 协议支持：

1. **自动检测 MiMo 模型**：当 `base_url` 包含 `xiaomimimo` 时，自动切换到 Anthropic 协议
2. **新增 Anthropic SDK 调用**：使用 `anthropic` Python SDK 替代 `openai` SDK 进行 API 通信
3. **支持 Thinking 模式**：正确处理 MiMo 的推理（thinking）响应块
4. **代理兼容**：TradingView 数据源走代理，MiMo API 直连（通过 `NO_PROXY` 环境变量）

### 修改的文件

- `pa_agent/ai/deepseek_client.py` — 新增 Anthropic 协议支持（`_is_anthropic_provider`, `_make_anthropic_client`, `_anthropic_chat`, `_anthropic_stream_chat`）
- `运行智能体.bat` — 新增代理和 `NO_PROXY` 环境变量配置

## MiMo 模型信息

| 模型 | 上下文 | 最大输出 | 推理能力 |
|------|--------|---------|---------|
| mimo-v2.5-pro | 1M | 128K | ✅ |
| mimo-v2.5 | 1M | 128K | ✅ |
| mimo-v2-pro | 1M | 128K | ✅ |
| mimo-v2-omni | 256K | 128K | ✅ |
| mimo-v2-flash | 256K | 64K | ✅ |

## 部署方法

### 前置条件

- Windows 系统
- Python 3.11+ 已安装
- TradingView 数据源（免费，无需注册）或 MT5（需券商账户）

### 安装步骤

1. **克隆或下载本仓库**

2. **安装依赖**
   ```powershell
   cd PA_Agent
   pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
   pip install anthropic -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
   ```

3. **配置 API Key**

   启动程序后，在 GUI 设置中填入：
   - **Base URL**: `https://token-plan-sgp.xiaomimimo.com/anthropic`
   - **Model**: `mimo-v2.5-pro`
   - **API Key**: 在 [MiMo 平台](https://platform.xiaomimimo.com) 申请

4. **启动程序**
   ```powershell
   python run.py
   ```
   或双击 `运行智能体.bat`

### 代理配置

如果 TradingView 数据源需要代理，编辑 `运行智能体.bat`：
```bat
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897
set NO_PROXY=xiaomimimo.com
```

## 致谢

- 原项目：[PA Agent](https://github.com/rosemarycox5334-debug/PA_Agent)
- MiMo API 兼容性参考：[MiMo-API-Compat-Fix](https://github.com/Miku-cy/MiMo-API-Compat-Fix)

## License

遵循原项目许可证。
