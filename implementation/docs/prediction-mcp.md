# P03：受控预测 MCP 作业工具

2026-09-11，批次 24。目标是证明真实 Claude Agent SDK 能通过 MCP 请求与查询 Core 管理的预测作业。当前使用真实 SDK/CLI 和离线 mock 上游验工程链，数值模型仍是明确标记的 fixture；未调用付费 Flash、未接真实预测模型或验证医疗有效性。

## 实际执行链

认证调用者指定已有 run → PredictionAgent 检查本病例当前快照的 prediction 外发许可 → 创建独立 SDK 父 job → 真实 SDK 通过内置 MCP server 调用固定工具 → 受限本地网关 → PredictionTools → Core 构造模型输入并启动独立数值 worker → Core 校验/登记产物 → MCP 返回状态及证据引用。

SDK 与数值 worker 是两个不同进程。SDK 不读取账本或原始病例，不接收服务 bearer 或真实 provider key，也不持有登记数值结果的工具。控制网关使用随机会话 capability，权限仍由宿主 Core 检查。旧报告 Agent 的固定流程保持不变；本轮没有将它冒称为 P05 的主 Agent 循环。

## 工具合同

| 工具 | 输入 | 输出与限制 |
| --- | --- | --- |
| `request_prediction` | 严格空对象 `{}` | 为宿主已绑定的 run 请求一次预测；调用同步等待 Core 的限时 worker。返回 `job_id/status/prediction_available` 和可用时的证据引用。不接受 case/run/snapshot、模型、单位、数值、producer 或 origin 参数 |
| `inspect_prediction` | 严格 `{ "job_id": "32 位小写十六进制 ID" }` | 只能查询该 SDK 父 job、同一 run 的 prediction 子 job。每次重新核验归属、当前快照、许可、配置、状态和权威产物绑定；其他病例/job、伪造 ID/额外字段被拒 |

成功时的 `evidence` 仅含 `artifact_id/artifact_hash/origin/producer/version`。Core 保存原始数值及完整 case/run/snapshot/job/输入摘要关联，LLM 不接收预测数值或原始输入。当前 origin 为 fixture；真实 SDK 不会把 fixture 自动变成真实模型结果。

`ok:true` 表示工具请求/查询被正确处理，不代表预测成功。只有 `prediction_available:true` 且重验通过才有证据引用；FAILED/FENCED 等状态不提供数值或替代结论。SDK 最终自由文本不会登记为预测，也不会从 API 返回。

同一 SDK 会话的重复请求复用同一个数值 job，成功和失败都不自动重新抽样。成功完成后重复 HTTP 请求返回同一受控回执，不再读取 key、启动 SDK 或追加调用费用（仍须满足当前授权/有效性）。未完成的 SDK job 不自动重开；新任务须使用新 run。普通 SDK 外预测 API 对没有活跃预测 SDK 管理的 run 继续可用。

取消使用现有归属用户 `POST /runs/{run_id}/cancel` 或 Core.cancel；并行数值执行周期性重验并终止。这一轮没有异步 SDK 排队调度器或单独的 MCP cancel 工具，不声称完成 P05 的通用任务循环。

## 权限、绑定与失败

新增独立用途 `prediction`，仅允许声明为 synthetic 的工程输入。report、medical_review、ethics_review 的授权均不能代替它。公开 data policy 加入该用途的最小外发字段：固定提示/工具 schema、job 状态/错误码/ID、受控产物引用、模型生成会话及 SDK 协议元数据。

许可仍绑定 case/snapshot/purpose/policy_hash/服务实例和修订。扩展政策内容会使旧内容摘要不匹配，不能隐式升级旧许可。重启需要重新授权；先检查许可再读取 key。SDK 的 prediction profile 无系统工具、子 Agent、RL、报告、审核、发布或登记结果工具，不扩展既有 report/review/smoke profile。

SDK 父 job 的 input_digest 绑定实际工具/提示词/模型/SDK/CLI 配置及 run/case/snapshot 和快照内容摘要。创建数值子 job 时，在同一事务复制冻结许可并写入 job_dependencies。执行、回收与 Core 登记时检查子/父 job、许可存在性/摘要、配置和快照；缺失绑定不能退化成不需授权的普通 fixture。SDK 外普通预测入口不能在活跃预测 SDK 的同一 run 中绕开父 job。

数值作业取消、超时、错误、撤权再授权、换快照、删除、父作业失效或配置变化均不能让迟到结果登记；审计无法提交则不算完成。数值产物已登记而 SDK 随后失败时保留审计事实，但整体 run 失效，不能继续 RL/报告/发布。不用 LLM 自报成功、零值或伪造证据补齐。

`research` 模式仍在真实模型未配置处拒绝，不回退到 fixture。失败原因保持具体错误码，例如 `EXECUTOR_TIMEOUT`、`MODEL_NOT_CONFIGURED`，不掩盖成模型成功。

## 服务入口与生命周期

服务默认不启用该 SDK 入口。操作者可显式使用 `serve --enable-prediction-agent`；工程 fixture 另需 `--enable-fixtures`，基础输注新输入另需 `--enable-basal-fixtures`。开关和服务启动不发起 LLM 调用。沿用本地私有 key 路径；没有 key 时返回受控错误，不索要或输出凭据。

1. 通过已有病例/运行 API 创建并冻结合成输入，得到 run。
2. 归属用户调用 `PUT /cases/{case_id}/data-permissions`，指定当前 snapshot_id、purpose=`prediction`、allowed=true、policy_version=`engineering-data-policy-v1`。
3. 认证调用 `POST /runs/{run_id}/prediction-agent`，请求无 body。接口默认关闭时返回 `AGENT_NOT_CONFIGURED`，无许可时不读取 key 或调用模型。
4. 可通过既有 run 状态/事件 API 检查经过 Core 的轨迹。此接口只返回受控预测状态及引用，不发布医疗结果。

新 SDK profile 继续仅允许 deepseek-flash、禁用 Pro/fallback，并沿用独立预算文件、输出上限和 SDK/worker 资源限制。SDK 工作目录被 P02b 的登记/回收机制覆盖；失败待清理、墓碑、备份新实例许可等规则不变。新增 job_dependencies 仅保留 job ID 关系，无原始内容；删除病例仍清除父/子许可和所有数值正文，必要运行/审计元数据保留。

当前桌面按钮仍使用已有报告流程，没有悄悄替换成 P03/P05 的新主循环。P04 才扩展 RL MCP，P06/P05 再接会话事实和主循环；P16 的真实资料保护要求继续保留。

## 验证证据

`tests/test_prediction_agent.py` 包含真实 SDK→MCP→独立数值 worker、同作业请求/查询与重复 HTTP 回执、实际 localhost HTTP 端到端、默认权限与用途隔离、跨病例/额外输入、产物 hash/origin/job 篡改、缺失父/子许可或依赖、取消/撤权/换快照/删除/配置变化、超时不重试、SDK 后置失败与审计回滚。

实际进程故障检查使用独立挂起探针制造取消/撤权/超时，单独标注为故障注入。测试中的 mock provider 不代表真实 DeepSeek 决策质量，fixture 不代表模型或临床有效性。最新结果见工作记录批次 24，证据目录 `runtime/p03-proof/`、全量 `runtime/p03-full.xml`。
