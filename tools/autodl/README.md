# AutoDL 远端实验连接

2026-09-14：本地 MCP 和 CLI 已安装，尚待真实实例 SSH 信息与认证完成远端联调。本轮未租卡、未运行训练。

## 来源与配置

- 选用社区 [AutoDL-Remote](https://github.com/haibarazz/AutoDL-Remote)，版本 0.9.0，固定提交 `578a1759580b37360defcb4a41e373d1649a523f`，MIT 许可；不是 AutoDL 官方 MCP。
- 官方支持 [SSH](https://api.autodl.com/docs/ssh/) 与 [文件传输](https://api.autodl.com/docs/scp/)。该 MCP 封装 SSH，无需 AutoDL API Token。
- 另有 [AUTODL-PLUGIN](https://github.com/wuzihuang/AUTODL-PLUGIN)，侧重账户余额和实例生命周期 API；本项目当前需求是代码传输和实验执行，因此未安装它。
- Codex 读取项目 [.codex/config.toml](../../.codex/config.toml)，配置方式依据 [Codex 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)。`codex mcp get autodl-remote --json` 已可识别；已运行的桌面会话若没有出现工具，需要重新打开项目任务以加载配置。
- CLI：`./tools/autodl/autodl-remote`。MCP 13 个工具包含账号、绑定、诊断、模型目录、文件、传输、执行、job、run、fleet、tmux、dashboard、shutdown。
- 上游源码存于 `vendor/AutoDL-Remote/`，不自动升级。唯一补丁：`APP_DIR` 增加 `AUTODL_REMOTE_APP_DIR` 环境覆盖；补丁保存在 `app-dir.patch`。

这是研发用 SSH 工具，能运行用户权限下的远端命令；不是医疗 Core 的隔离执行器，也不是安全沙箱。MCP 的文件参数检查不代表任意 shell 命令都被路径隔离。

## 租好卡后连接

先提供控制台 SSH 命令中的主机、端口和用户名。密码不要发到聊天；可使用 SSH key 或在本地交互终端保存到 macOS Keychain。

以账号名 `diabetes-gpu` 注册真实连接后，将本项目绑定至 `/root/autodl-tmp/diabetes-agent`（该目录目前仅是建议，尚未远端创建）。私有连接配置位于 `tools/autodl/private/`。若用密码，交互终端执行 `./tools/autodl/autodl-remote account password-save diabetes-gpu`，输入不回显。

接通验收顺序：

1. `account test diabetes-gpu` 和 `doctor` 核对 SSH、GPU、磁盘与工具。
2. 上传不含患者数据的小型测试文件，远端读取，再下载到新位置并核对 SHA-256。
3. 启动有唯一名称的短后台任务，验证状态、日志与退出码。
4. 确认训练代码、环境、数据范围和预算后再运行正式实验。

## 后续常用方式

以下从项目根目录运行，`train.py` 仅说明用法，尚未创建或执行：

```bash
./tools/autodl/autodl-remote --version
./tools/autodl/autodl-remote account list
./tools/autodl/autodl-remote doctor
./tools/autodl/autodl-remote put train.py train.py
./tools/autodl/autodl-remote exec --detach --name baseline-001 -- python -u train.py
./tools/autodl/autodl-remote job status baseline-001
./tools/autodl/autodl-remote job tail baseline-001 --lines 100
```

只传输具体文件/子目录，不整体同步项目根目录。数据、密钥、旧运行库和缓存分别处理；`.gitignore` 不等于传输排除规则。遵守 [.autodl-remote/CONVENTIONS.md](../../.autodl-remote/CONVENTIONS.md)。

## 本地验证与边界

```bash
implementation/.venv/bin/python tools/autodl/check_mcp.py
```

该检查使用 MCP Python SDK，通过 stdio 启动真实服务与 CLI，检查工具发现、账号列表和拒绝路径；没有远端网络请求。上游 MCP 测试使用模拟 CLI，仅证明协议、参数映射和错误处理。证据在 `runtime/`。真实上传、GPU 和远端实验仍待实例联调。
