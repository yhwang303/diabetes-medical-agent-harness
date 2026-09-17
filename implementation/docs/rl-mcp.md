# P04：受控 RL MCP 作业工具

批次 26，2026-09-11。在 P03 预测 MCP 后增加独立 RL SDK profile。真实 Claude Agent SDK/CLI 和独立数值 worker 验证工程链；上游是离线 mock，预测/RL 继续 fixture，没有训练真实模型或验证医学效果。

## 实际链路

认证调用者指定已有 run → RLAgent 检查该病例/快照的独立 `rl` 外发许可 → Core 检查当前有效父预测（若由预测 SDK 发起，其父 SDK 必须完整成功）→ 创建 RL SDK 父 job → SDK 调用受控 MCP → Core 从账本装配快照、父预测正文与摘要 → 独立 policy worker → Core 校验/登记 → 工具返回候选引用。

RLAgent 的配置摘要绑定工具、提示词、SDK/CLI/模型版本、run/case/snapshot、快照内容与父预测产物摘要。RL SDK 不选择预测 payload/hash，也不能自行登记、覆盖或修改输出。数值执行仍可在 SDK 外进行；本步骤未实现 P05 主 Agent 循环。

## 固定工具

| 工具 | 输入 | 结果与限制 |
| --- | --- | --- |
| `request_rl` | 严格空对象 `{}` | 请求绑定 run 的唯一 RL 作业，限时等待；重复请求复用同一成功或失败作业，不自动重试抽样。不接受病例、快照、父预测、单位、剂量、模型或 producer/origin |
| `inspect_rl` | `{ "job_id": "32 位小写十六进制 ID" }` | 仅能查询同一 run 且依赖当前 RL SDK job 的 policy 作业；查询前重新检查父预测、权限、配置和当前产物 |

返回 `ok/job_id/status/policy_available/clinical_use/safety_checked`；可用时的 `evidence` 仅包含 `artifact_id/artifact_hash/origin/producer/version`。不可用时返回受控原因或错误码；不会返回原始快照、预测数值、基础率或剂量。SDK 自由文本不作为结果登记，也不由该 HTTP 入口返回。

`policy_available=true` 仅表示有经过结构及依赖校验的候选产物。该步骤明确 `safety_checked=false`、`clinical_use=false`：后续工程安全、报告、双审核、发布仍需各自通过；候选被接受不表示用户接受或实际用药。合法零候选仍须来源有效；弃权、unsupported、error、超时或异常输出不会改写成零剂量或“维持”。

## 权限与闸门

新增独立 `rl` 用途。report、medical_review、ethics_review、prediction 许可不能替代。只允许声明 synthetic 的工程输入；许可绑定病例/快照/政策内容/本次服务/修订，在读取 key 前检查，在 SDK 请求、工具、worker 执行及登记时重验。新增用途改变政策内容摘要，既有许可不能静默升级，新服务仍需重新授权。

数值子作业同事务复制父 SDK 的冻结许可并记录 job_dependencies；父 SDK 类型、同 run、权限存在性、配置及父预测都须一致。活跃 RL 会话不能从普通模型 API 绕开父 job。取消、删除、撤回再授权、换快照、配置变化、错绑/缺失记录、迟到结果均阻断。

P04 同时补强两种受控数值产物的后续读取：逐次检查父 SDK、冻结授权、依赖与配置；RL 消费预测、安全/草稿消费候选时，要求对应父 SDK 已完整成功。数值先写入而 SDK 后置失败时，历史审计可保留，但整个 run 失效，不能继续发布。普通历史 SDK 外 fixture 没有父 SDK/外发许可，继续按原合同核验，不因新工具而伪造授权。

父预测同病例/快照/输入/来源/执行器/版本/时效及 profile 检查由 Core 完成，RL 的 forecast_parent_hash 必须匹配 Core 装配的父预测。既有成人 T1D 基础输注、U/min↔U/h 转换、原始数值/单位、动作语义与工程安全规则保持原状；未扩大到 bolus、注射或其他人群。

## 服务与设置

- `serve --enable-rl-agent` 显式开启，默认关闭；开关本身不调用 LLM。需要的 fixture / basal fixture 开关仍须另行开启。
- 认证 `POST /runs/{run_id}/rl-agent` 无请求正文；未启用返回 AGENT_NOT_CONFIGURED，授权管理走已有 data-permissions API，purpose=`rl`。
- 成功 HTTP 重试重验当前依赖并返回相同回执，不再读取 key 或调用 SDK；失败不自动重开旧 job。取消走既有 run cancel API。
- 模型固定 deepseek-flash、禁止 Pro/fallback；沿用次数、输出、费用上限、独立 worker 与 SDK 目录登记/回收。真实 key 仅由宿主读取，不进入 SDK/日志。
- 设置页同步 RL 工具已定义、任务入口是否启用；真实模型仍未接入。已有本地自检仍仅检查预测，不把其绿色状态当成 RL 的实时连通性检查。该步骤未替换桌面报告固定流程或新增主循环。

## 验证证据

56 项 P04 专项覆盖旧/基础输注合同实际双 SDK 和四个独立 worker、重复请求/查询、实际 localhost HTTP、独立用途授权、父 SDK 未完成/后置失败、跨病例、伪造输入/来源/父摘要、权限/依赖缺失、配置变化、撤回/删除/取消、弃权及异常输出、审计故障。

实际 SDK 配合数值挂起探针验证取消/撤权/超时后的回收，SDK 目录清理，policy 产物为零；它们是明确的故障探针。正常链的数值执行使用真实独立 fixture worker，不用线程替身冒充此项证明。其他单项故障测试使用显式 FaultRunner 注入，分别标记。

专项结果 `runtime/p04-focused.xml` / `p04-focused.txt`，实际链证明 `runtime/p04-proof/`，旧库副本兼容 `runtime/p04-compatibility/result.json`。前端和离线 Tauri 构建 `runtime/p04-build.txt`，未宣称本轮原生窗口实点。全量 604 项通过（155.34 秒，0 failures/errors/skipped，1 条既有 anyio 提示），见 `runtime/p04-full.xml` / `p04-full.txt` 和根目录工作记录批次 26。
