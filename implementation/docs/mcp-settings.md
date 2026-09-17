# P03a：设置中的 MCP 状态与本地自检

批次 25，2026-09-11。用户在 P03 后补充的可视化需求，独立于尚未完成的 P04/P05。

App 左侧「设置」→「工具与模型连接」展示：预测工具定义、调用检查、真实预测模型，以及 SDK 版本、预测任务入口和 RL MCP 状态。工具定义可读不等于实际连接成功；只有手动自检通过才显示本地成功。真实预测模型仍未接入；批次 26 的 P04 已增加 RL MCP 工具定义，设置显示其定义和业务入口开关。默认桌面预测/RL Agent 任务入口仍关闭，本页自检只检查预测。不是远端 MCP server 管理器，也不是线上 DeepSeek 可用性检查。

## 固定接口

- 认证 `GET /mcp/status` 读取配置与该调用者本次服务内的最近检查，不创建病例、SDK 会话或付费请求。
- 认证 `POST /mcp/check` 不接受请求正文；不接受病例、路径、URL、模型或密钥。并发检查返回 `MCP_CHECK_BUSY`。与业务预测 Agent 开关独立。
- 原生桥接 `mcp_status` / `mcp_check` → Python 固定操作 → localhost API；bearer 保留在宿主。检查请求等待上限 35 秒，SDK 执行上限 20 秒；写请求不自动重放。

## 检查链及范围

每次检查创建本服务数据目录下独立临时账本，加载项目内置 synthetic-case.json，授予该合成检查自己的 prediction 许可。使用真实 Claude Agent SDK/CLI 和实际 MCP transport；上游始终是 `httpx.MockTransport` 的三段固定本地协议回复，使用明显标记的假凭据，不读取真实 key。SDK 请求预测后再查询同一 job，Core 启动独立数值 worker 并登记 fixture。两次证据引用必须一致且 SDK 完整结束才算通过。

用户库不产生病例、快照、许可、任务、数值结果或报告，仅写入检查开始/结束的受控审计。所有临时工作正常结束后回收；检查状态按调用者隔离，仅缓存于当前服务实例，重启后回到未检查。异常退出可能遗留本项目内的合成检查临时目录，不声称物理擦除。SDK 目录继续沿用已有登记/回收机制。

检查失败覆盖旧成功记录，审计失败不能显示成功；版本不符合当前固定 SDK 则拒绝运行。打开/刷新只读取状态，检查中按钮禁用；离开后重新进入可读取同一检查状态。服务读取失败时撤下旧成功状态，显示未知。绿色只表示上次本地合成链通过，不意味着真实数值模型、线上 LLM 或临床验证通过。

## 验证

- 11 项专项通过；全量 548 项通过，0 failures/errors/skipped，109.99 秒，1 条既有 anyio 提示。
- 实际 SDK 与数值 worker 两个 PID、同一 fixture 引用、原病例各表不变、临时数据回收；缺失/错误 SDK、并发、授权、正文禁止、审计失败、跨用户状态和重启重置。
- 独立 Python 桌面 client → 真实 localhost HTTP → SDK/MCP/Core/fixture 自检通过，用户表全部为零；`runtime/p03-ui-http/result.json`。假 token 清理，真实 key 未读取，收费 0 次。
- 前端及离线 Tauri 原生 App 构建通过，`runtime/p03-ui-build.txt`。实际点击验证使用隔离浏览器预览，替代的只是 JS 原生 invoke 适配层；后端 API 和 SDK/worker 使用真实实现。检查未运行、运行中禁用、成功、服务断连清除旧成功，以及 1120×820 / 760×650 布局。**本批没有原生窗口实点，不把浏览器预览作为原生运行验收。**
- 证据：`runtime/p03-ui-focused.xml`、`p03-ui-full.xml`、`p03-ui-full.txt`、`p03-ui-preview/requests.jsonl`、`p03-ui-preview/verification.json`。预览及 HTTP 账本都是新增隔离测试资料，未访问现用桌面库。
