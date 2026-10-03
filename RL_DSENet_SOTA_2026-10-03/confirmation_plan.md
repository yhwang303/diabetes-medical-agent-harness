# 独立 confirmation 闸门设计（待审核、未接入）

当前只有 `confirmation_gate.py` 和临时 fixture 机制检查。**没有建立真实最终方法清单，没有执行真实 freeze，没有生成实际 confirmation jobs/轨迹，没有改动 `evaluate_candidates.py`。** 当前评价器继续拒绝 confirmation。此文的接口不是最终模型选择，也不是确认结果或模型晋升证据。

## 最小接口

| 接口 | 作用 | 写入边界 |
|---|---|---|
| `validate_plan(plan)` / CLI `validate --plan PATH` | 核对显式最终选择、文件哈希、训练完成、完整开发证据及已保留的确认曝光记录 | 只读；不生成 confirmation 餐次或轨迹 |
| `freeze(plan_path)` / CLI `freeze --plan PATH` | 排他冻结整个比较集合 | 新建 `confirmation/<plan SHA>/freeze.json`、seal、空 events/runs；不启动模型或仿真 |
| `authorize(freeze_path,method_id)` | 核对全部冻结依赖，领取一个方法的唯一运行许可 | 首次生成其固定 60 jobs，排他新建 `runs/<id>/ticket.json` 并追加授权事件 |
| `validate_ticket(ticket_path)` | 执行前复核原始授权与全部依赖 | 只读；未来 runner 必须调用 |
| `record_result(ticket_path,summary_path=...)` 或 `failure_reason=...` | 登记全部结果或明确基础设施失败 | 追加一次结束事件；不覆盖首次输出，不改冻结选择 |

CLI 的最后两步分别为 `authorize --freeze PATH --method-id ID` 和 `finish --ticket PATH --summary PATH`，基础设施故障用 `--failure-reason TEXT`。本批不调用这些接口处理实际候选。临时测试目录内的 synthetic freeze/ticket 不能用作真实研究许可。

## 显式冻结清单

清单 `schema=1`，要求 `selection_finalized=true`、`confirmation_previously_generated=false`、`split="confirmation"`、`smoke=false`。清单必须另绑定一份**不再修改的开发选择说明**；不要引用持续更新的工作记录作为选择证明。

| 必需字段 | 内容 |
|---|---|
| `required_comparator_ids` | 审核人明确指定的主比较对照 ID，不由程序猜测 |
| `selected_internal_ids` | 开发结果后明确选定的内部候选 ID |
| `methods` | 恰好是以上两个不相交集合的并集；不允许遗漏、重复或额外临时方法 |
| `protocol`、`scorer`、`normalizer` | 各为 `{"path":项目相对路径,"sha256":精确文件哈希}` |
| `confirmation_runner` | 待审核的新确认 runner 的同样文件绑定；当前 development evaluator 和 gate 自身不能冒充 runner |
| `selection_evidence` | 冻结的开发选择说明文件绑定 |

每个 method 必须写明：`id`、`kind`、`method`（仅 retained 非 null）、固定 `batch_size`、`checkpoint`、`config`、`worker` 文件绑定，以及完整 `source_sha256` 和 `artifact_sha256` 映射。无权重的 hold/physiology 对应 null；legacy/retained 仍须显式列出实际固定 checkpoint，不能靠名字省略。配置为原 run/config；retained 的固定配置从实际依赖集合绑定。

源码集合精确等于当前评价器 `dependencies(...,allow_smoke=False)` 返回集合，加本 gate 与待审确认 runner；artifact 集合为该入口实际依赖，加下述训练完成文件。不能只写部分主要文件。归一化、scorer、protocol、原/new worker、world 与上游权重都必须有实文件和精确 SHA。路径仅允许项目内相对路径；运行时解析为绝对路径。

每个方法的 `development_evidence` 包含本方法的 `manifest` 和 `summary` 两个文件绑定。必须是同 checkpoint、同配置、同 kind/method 的完整新 60 例 development：原 development jobs 完全一致、无 smoke、非技术失败状态；60 条 raw 引用全部核 SHA。原生仿真提前终止可以保留为失败观测，但不能以技术失败或缺记录面板代替完整开发流程。选择标准及为何这些对照足够由审核人写入选择说明，gate 不替人选择。

对照清单应先明确临床任务基准、已保留部署和新任务训练对照分别承担什么比较问题。`method_registry.json` 是可用方法记录，不是自动全选名单；冻结几个具体适配实例，不能声称覆盖“全部 SOTA”。mean 仍是同权重、开发后提出的部署消融，不能改写为原预注册或新增外部算法。

## 训练与确认未曝光条件

- 新 PPO / world-PPO 及 mean 部署必须先完成整个 40 轮训练，之后仍只能选原预定第 8/16/32/40 轮之一。gate 同时核对选中迭代及第 40 轮的已完成训练 history/checkpoint 绑定；不会在第 16 轮仍训练时提前 freeze 第 8 轮。
- 新 world 必须 completed/configured 都为 4000，且 `budget_override=false`；新 IQL 必须是正式最终 20,000 更新。底层入口既有正式/smoke 门禁继续生效。
- 旧 legacy/retained 使用已经保留的历史最终选择清单，明确属于历史冻结权重，不伪造旧训练的新版 completion 文件。
- protocol 必须仍为 seed 260915、已曝光十名成人、三种相同外部 bolus factor、4320 min 总时长与 360 min warmup，确认 seeds 恰为 103901/103902，且不得与 train/world-validation/development/smoke 交叉。
- freeze 前检查本研究 `results/data/cache/confirmation` 中保留的 manifest、summary、config、ticket 及带确认 seed 的轨迹名；若已有确认曝光，拒绝初次 freeze。清单还要求显式声明从未提前生成。**文件扫描不能证明没有未记录或外部运行**，因此此声明和研究执行审计不可省略。

一次 freeze 后不能创建另一份替代选择。目录名是规范化 plan 的 SHA，seal 绑定 freeze 文件；每个授权事件再绑定 freeze 和 ticket SHA。确认结果出现后，清单、模型、配置、源码或开发证据变化都会阻断后续授权。

## 一次完整 60 例与失败保留

每个方法唯一 ticket 固定十成人×三 bolus factor×两确认 seed，共 60 jobs；顺序与原 development 构造方式相同，餐次仍由原 `scenario` 生成。job/患者元数据只交环境，策略仍只接收既定历史与 anchor。ticket 没有 smoke、子集、重抽 seed、换 checkpoint 或覆盖输出开关。

允许固定清单内不同方法继续完成，即使较早方法技术失败；不允许据已见确认结果换候选。已授权方法不能再次 authorize，已有输出目录不能覆盖，finish 不能改写。summary 必须列出全部 60 个固定 key，逐条 raw SHA、job 和原 scorer 重算必须一致。真实缺失记录保持空记录与 unknown tail，不能填零 BG、假定成功或把前缀当完整随访。

未来 summary provenance 至少携带 `freeze_sha256`、`ticket_sha256`、`protocol_sha256`、`scorer_sha256`、`normalization_sha256`、`worker_sha256`、`checkpoint_sha256`、`config_sha256` 和完整 `source_sha256`。失败 summary 也保留全部 60 行；如果基础设施连 summary 都未写出，使用显式 failure reason，gate 保留当前输出文件哈希并记录 60 例状态未知。

结果不符合合同、或执行期间依赖变化，结束事件记为 technical failure，保存已有文件而非静默放过。完成只是证据齐全，不会自动晋升模型、宣告收益或临床安全。

基础设施故障后，当前最小版本**拒绝自动重试**。这不意味着永久不可恢复：必要时应另制独立版本化恢复协议，经审核保留首次全部失败/已见结果、固定原模型和同一 jobs、单列新的执行尝试并解释原因。不得借恢复换模型、重选候选、丢弃差结果或覆盖首次记录。未实现自动清锁；进程崩溃留下锁也应按该恢复流程处理。

## 与评价器的未来整合点

待 root 审核后，新增唯一入口 `evaluate_confirmation.py --ticket PATH`，不接收 kind/config/checkpoint/jobs 覆盖参数。它读取 ticket 后，再调用明确的受控执行接口。建议后续为原评价器增加内部 `run_confirmation(ticket_path)`（或等价私有执行入口）：

1. 先 `validate_ticket`，只从 ticket 构造参数和完整 jobs；保留原 `make_jobs` 对普通 CLI 的 confirmation 拒绝，不使用 monkeypatch 绕过。
2. 输出目录已由 gate 排他保留且应仅有 ticket；执行器只能在其中新建 raw/source/manifest，绝不以 `exist_ok=True` 覆盖现有研究输出。初始化错误也走 finish failure。
3. 复用当前严格 dependencies、ready、动作和 scorer 检查，追加上述 confirmation provenance；主循环与原未知尾部/失败保留语义不变。
4. 写出全部 60 行 summary 后调用 `record_result`；异常路径也必须 finish，不能仅关闭进程留下“成功”标记。

**该整合尚未实施。** 未来入口源码必须先完成审查和独立机制检查，再进入计划的 source/runner SHA，之后才能实际 freeze。已生成确认结果后不能再改 runner 而沿用旧 seal。

报告侧仍需原评分器精确审核包、完整 jobs 对齐及逐面板 manifest 绑定的人工审核记录；gate 的完成事件不替代报告审查，也不构成自动发布许可。

## 本轮检查

`checks/check_confirmation_gate.py` 在临时目录使用人工计划和轨迹，39 项检查通过：只读校验、显式集合、源码/归一化/开发证据变更、首次曝光阻断、唯一 freeze/ticket、固定 60 例、失败保留、禁止重试/换选、允许继续已冻结方法、原 scorer 重算和完整训练门禁。检查前后原 evaluator/模型/worker/配置哈希未变，实际研究 confirmation 目录未创建或改变。证据为 `checks/confirmation_gate_mechanics.json`；不是正式确认实验。
