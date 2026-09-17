# 桌面报告 Agent 任务

2026-09-11 批次 22 更新：报告入口新增明确外发授权和撤回，许可绑定当前病例/快照/用途，本次 Core 重启后须重新授权。原有默认启动路径现在也必须先授权；刷新、打开页面和单独勾选不调用模型。当前前端、桥接、TypeScript 与原生构建通过，并有本地 HTTP/SDK 离线验证；本轮未重复原生窗口实点或收费联调。详见 [数据与授权合同](data-policy.md)，下方原生运行数字为历史批次证据。


2026-09-10：完成首个原生 Agent 任务入口、异步状态和取消。**320 项回归通过**（新增 17 项，1 条既有依赖弃用提示）；前端及离线原生构建、实际窗口启动/取消/退出重开通过。

## 当前可用流程

选择完整的合成病例，打开“Agent 任务”，点击“启动报告 Agent”。Core 创建独立 eval run 并准备预测/策略占位证据；通过工程检查后，真正启动 Claude Agent SDK 0.2.152 / CLI 2.1.259，由 DeepSeek Flash 使用已验收的报告工具读取合同、检查状态和提交章节提案。SDK 正常完成后 Core 再校验、渲染并登记草稿，任务显示“工程草稿已准备，等待双审核”。

这是一个固定目标、受限工具的报告 Agent 入口，尚不是自由聊天或完整医疗任务规划器。证据准备由 Core 执行，章节提案由 SDK/LLM 提交，主 Agent 不能直接写报告正文、剂量或临床结论。预测/RL 和报告仍是 fixture；桌面流程本轮不自动执行审核或发布。

现有独立真实双 reviewer 实现继续保留。当前普通桌面 Core 的 run 仍冻结 fixture reviewer 版本，不能把已生成桌面草稿直接升级为真实双审证据。桌面双审调度、正确冻结真实 reviewer 配置，以及报告读取/导出将归入下一项全通道发布验收，不能仅新增两个按钮或更换旧 run 配置绕过审核绑定。

## 入口与权限

开发 App 为本地服务显式传入 `--enable-agent --enable-fixtures`；CLI 默认不启用 Agent。服务启动、打开页面和刷新不调用模型。凭据仍位于固定宿主私有路径，页面只读取配置是否就绪，不接触 key、bearer、端口或任意 URL。

| 原生命令 | 认证 HTTP 路由 | 范围 |
| --- | --- | --- |
| agent_configuration | GET /agent/configuration | 固定任务类型、模型 ID、凭据就绪与工具名 |
| agent_start | POST /agent/tasks | 仅病例、快照和幂等标识；202 立即返回任务元数据 |
| agent_latest | GET /cases/{case_id}/agent-task | 当前所有者该病例的最近任务 |
| agent_status | GET /agent/tasks/{task_id} | 当前所有者指定任务 |
| agent_cancel | POST /agent/tasks/{task_id}/cancel | 空正文，不能附带提案、角色或结果 |

请求合同 `AgentTaskRequest` 拒绝额外字段、路径、自由 prompt 和客户端模型配置。Core 再检查归属、快照、synthetic 来源、缺失点与 fixture 启用。浏览器 Origin 拒绝、无缓存响应和桌面 CSP 保持启用。

状态返回固定任务/阶段、Core 状态、当前有效性和最近 30 条白名单事件元数据。前端约每 1.5 秒只读刷新，隐藏时暂停；无法连接时明确撤销“当前可核验”状态，历史进度仅作历史显示。无草稿正文、原始模型自由文字、工具参数、候选值或 capability 流出。

## 任务生命周期

`agent_tasks` 是增量新增表，不改原有数据。幂等键绑定 owner/case/snapshot，同键重试返回原任务，完成或失败任务不会自动重新收费；全服务最多一个活动桌面任务，没有无界任务队列。

- 成功：QUEUED → RUNNING → SUCCEEDED，Core 停在 DRAFT_READY。
- 取消：Core 先取消 run 并 fence 作业，再确认 CANCELLING；SDK/worker 返回并回收后才确认 CANCELLED。终态写入与取消串行，避免完成/取消竞态留下永久 CANCELLING。
- 失败：模型异常、不完整 SDK 结果、缺少提案、预算或证据失败等成为固定原因码，不能提交迟到草稿。终态审计失败则状态查询返回 STORAGE_UNAVAILABLE，不继续显示永远运行或伪造成功；持久恢复交给独占服务重启处理。
- 重启：独占服务先 fence 原 Core 未决作业，再失效未完成 Agent 任务，记录 SERVICE_RESTARTED，不自动恢复模型调用。完整终态保持可查询。
- 正常退出：Rust 对 Core 发 SIGTERM，Core 取消活动任务、关闭 SDK runner 并等待任务线程结束；原生端给 5 秒清理窗口，未响应才强制结束。强杀 App、机器崩溃及该兜底分支的孤儿进程加固仍未完成，不能称作完整 OS 沙箱。

预算固定在 `runtime/desktop-core/agent/budget.json`，所有桌面报告任务共用并持久保留：最多 12 个请求、0.05 美元保守预留、每请求最多 384 输出 token。不会因新任务或重开而重置。真实 key 只由宿主读取，SDK 只有本次本地网关 capability；禁止 Pro 和模型回退。

## 验证证据

- 完整回归 320 passed，70.34 秒；专项 17 passed，7.22 秒。覆盖认证/越权、固定操作、幂等、来源/快照/缺失/配置、创建及完成审计失败、异步失败脱敏、重启和取消。
- 真实 SDK/CLI 配合离线假 provider 验证提案暂存后取消及关闭：进程组 SIGKILL、0 草稿。该故障验证没有调用官方 API。
- 原生实际任务：两个成功任务分别经 SDK 生成一份草稿，共 **6 次真实 Flash 请求**；第二次在操作取消前已完成，未伪称取消成功。第三次同次 UI 操作启动并取消，SDK worker 启动后约 0.312 秒被回收，0 草稿、没有增加模型请求。
- 成功 SDK worker 用时约 21.266/3.417 秒，exit 0；取消 SDK worker exit -9。所有对应进程组均实查已回收。Core 事件包含报告工具接受、SDK 会话完成和草稿登记；真实 provider 错误或模型响应不匹配会被 ReportAgent 拒绝，不能登记该成功草稿。
- 桌面只新增 3 个 Agent run、2 个草稿；**0 审核、0 发布**。原有 2 病例、2 快照、11 runs、21 jobs、14 artifacts、74 events/outbox 逐项保留；原数据库已先通过 SQLite backup 保存基线。
- 正常退出 App/Core 两进程均消失；重开后取消状态仍在，预算未变，不新增任务或模型请求。预算预留 0.0234828 美元，不是实际账单。

证据文件：`runtime/desktop/agent-before.sqlite3`、`agent-before.json`、`agent-live-http.json`、`agent-verification.json`、`agent-build.log`；测试为 `runtime/agent-task-tests.txt`、`test-output.txt`、`test-results.xml`。构建仍依赖本地项目及 Python runtime，不能当作可分发安装包。

```bash
.venv/bin/pytest -q tests/test_agent_tasks.py
CARGO_NET_OFFLINE=true scripts/desktop.sh build
scripts/desktop.sh open
```

本项完成后停下等待用户审核；下一项是全通道发布限制。未接真实预测/RL、医学规则或临床有效性验证。
