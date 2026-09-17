# 本项目 AutoDL 约定

- 本地入口：`tools/autodl/autodl-remote`；Codex MCP 配置：`.codex/config.toml`。
- 用途是研发阶段远端训练、实验与文件传输，不注册到医疗主 Agent 或绕过 Core。
- 2026-09-14：本地已安装；尚无真实账号、SSH 主机/端口、项目绑定和远端验收。未租卡、未训练。
- 待租卡后建议远端根目录 `/root/autodl-tmp/diabetes-agent`；绑定前核对实际实例与目录。
- 账号配置保存在 `tools/autodl/private/`；SSH 连接复用 socket 在 `.adlc/`；临时文件在 `tools/autodl/runtime/`。不把密码/Token 写入源码、聊天、工具参数或日志。
- 只显式传输当前实验需要的文件；禁止将项目根目录整体上传，以免带上密钥、患者数据、旧运行库和大量缓存。数据上传另行核对许可、冻结划分及实验范围。
- 代码与可复用实验配置先在本地修改，再上传。训练用唯一 run 名称在后台启动；先读远端日志，再按需下载小型指标/结果。
- 当前只配置连接层；GPU 型号、镜像、Python/CUDA/PyTorch 版本及费用预算均待实际训练方案确定。
