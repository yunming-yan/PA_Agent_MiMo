# PA Agent - MiMo 兼容版

> 基于 [PA Agent](https://github.com/rosemarycox5334-debug/PA_Agent) 的二次修改版本，解决了小米 MiMo 系列模型无法使用的问题。
>
> **原作者 UP 主：[阿尔法量化价格行为](https://space.bilibili.com/437555998)** — 本修改仅为 MiMo 兼容性适配，核心功能均来自原作者。

## 修改内容

### 核心改动：支持 MiMo API

MiMo API 的认证方式与原版 PA Agent 支持的协议不同，本次修改在底层 `deepseek_client.py` 中增加了兼容支持：

1. **自动检测 MiMo 模型**：当 `base_url` 包含 `xiaomimimo` 时，自动切换到正确的调用方式
2. **新增 Anthropic SDK 支持**：通过 `anthropic` Python SDK 进行 API 通信
3. **支持 Thinking 模式**：正确处理 MiMo 的推理（thinking）响应块
4. **代理兼容**：TradingView 数据源可走代理，MiMo API 直连

### 修改的文件

- `pa_agent/ai/deepseek_client.py` — 新增 MiMo 兼容支持
- `运行智能体.bat` — 启动脚本

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
- 数据源：TradingView（免费，无需注册）或 MT5（需券商账户，见下方说明）

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

### 关于 MT5

MT5（MetaTrader 5）需要在券商开户才能使用，不是免费数据源。如果没有券商账户，**请使用 TradingView 数据源**，它是免费的，无需注册即可获取行情数据。

在程序 GUI 中切换数据源为 `TradingView` 即可。

### 代理配置

如果 TradingView 数据源需要代理（国内网络通常需要），编辑 `运行智能体.bat`，取消注释以下行：
```bat
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897
set NO_PROXY=xiaomimimo.com
```
将 `7897` 改为你的代理端口。

## 致谢

- **原作者**：[阿尔法量化价格行为](https://space.bilibili.com/437555998)（B 站 UP 主）
- **原项目**：[PA Agent](https://github.com/rosemarycox5334-debug/PA_Agent)
- MiMo API 兼容性参考：[MiMo-API-Compat-Fix](https://github.com/Miku-cy/MiMo-API-Compat-Fix)

## 打赏

打赏请给予原作者，本修改仅为小范围 MiMo 兼容性适配，核心工作均来自原作者。

## License

遵循原项目许可证。
