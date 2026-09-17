# 新会话交接：医疗 Agent / 强约束 Harness

> 2026-09-16 最新目标覆盖：以RL-DITR完整患者模型、策略/价值和beam规划为基底，先跑通后研究近期论文支持的创新；O07/ReBRAC不再作为我方主线。新训练仅seed260915、保留全量Loop及稀疏food/exercise；FQL/ReBRAC不再运行。多维评价与新场景先冻结，已有10个模拟成人均已曝光。任务未完成，见[本轮合同](RL_DITR创新_2026-09-16/任务与复现合同.md)。


> 2026-09-16 范围澄清：最终O07是固定参考有界残差ReBRAC，属于incremental任务适配；RL-DITR模型式路径的早期尝试未成为最终模型。仿真研究门槛通过不代表非incremental创新、SOTA或产品完成；M02d保留未完成。具体结构/损失/指标及ICML预测器接入边界见[模型说明](RL进阶对比_2026-09-15/模型结构与读表说明.html)。


> 2026-09-15 22:14 最新研究结论：O07固定历史参考±0.4 U/h残差ReBRAC（seed260915、50k、全部225训练患者）通过冻结后的6名独立模拟患者研究门槛，TIR94.97%、TBR70 2.05%、TBR54 0.52%、0/54终止；仅可作为Harness受控研究接入候选。adult009低血糖仍明显、只有单seed，未证明临床安全。独立JSON推理及Mac权重一致性已验；未注册Core，M02产品父项/M02a及双模型准入保持未完成。最终结果、失败尝试和论文表见[训练审核报告](RL进阶对比_2026-09-15/训练审核报告.html)。此条覆盖下方历史阶段的“尚未训练/策略未启动”状态，不删除其负结果。


> 2026-09-15 21:49 用户最新覆盖：所有模型均只训练一个seed，主比较使用260915；已停止O07额外seed进程，未启动的下一seed撤出。此前多seed记录仅作历史证据，不再列为当前必做要求。后续同一批独立评价情景用于所有方法，不涉及重新训练；单seed不证明随机初始化稳定性。


> 2026-09-15 21:41 用户最新研究范围：外部ReBRAC/FQL baseline各用一个训练seed的同数据可比结果即可，后续重点优化我方模型；未提交的额外baseline seeds已撤出。我方候选继续稳定性与独立评价。当前为个人历史参考+有界残差结构、行为/价值损失权重及去RL消融，全部225训练患者和1,653,421起点保留，性能阈值未改变。实际结果以[进阶训练审核报告](RL进阶对比_2026-09-15/训练审核报告.html)和研究协议最新条目为准；原方向比例不是验收门槛，真实双模型/Harness产品准入仍未完成。


> 2026-09-15 20:28 持续目标更新：用户已明确授权2–3个较新RL基线公平对比、有论文依据的损失/risk/模型结构实质改进、多seed训练和独立仿真，最终交付全部实验HTML与论文式对比表。当前ReBRAC/FQL真实离线策略及BC已实现并在AutoDL运行50k pilot；新实验入口为[研究协议](RL进阶对比_2026-09-15/研究协议.md)。旧80%动作方向门槛仅保留诊断，不作为新策略验收；独立闭环合同已在成绩产生前冻结。Loop全部225训练患者/1,653,421起点和food/exercise保持，旧轮次权重负结果保留；不能将新策略已启动写成旧轮次已有完整RL。当前总目标仍未完成，M02a/M02未勾选。此条覆盖下文旧仅复现不创新和停止策略训练的当前执行限制；产品可信规则与Harness闸门不修改。


> 2026-09-15 当前执行：用户已授权在AutoDL连续训练并按结果调整，使用[22维训练管线v2](Loop数据集/训练管线_v2/README.md)，food/exercise只加特征和mask，225名train/1,653,421起点不减少。实际训练及失败结果见[训练审核报告](RL训练_2026-09-15/训练审核报告.html)；本轮实验已结束：S00烟测、A01失败、A02/A03各完整1epoch及全量validation已验证。A03的30/60分钟RMSE为0.9973/1.6775，但+0.1U/h仅2.50%诊断样本预测下降；当前权重不能用于RL给药，策略阶段因患者模型动作响应未通过而未启动。M02a/M02仍未完成，Harness未接入真实模型。此条覆盖下文旧14维默认训练/未有SSH等状态。

> 用户最新方向：先按论文/作者代码复现RL-DITR训练与规划基线，完成研究闭环后再考虑模型结构或损失创新；当前DPO/门控均不实施。不得以自选简化Actor–Critic目标冒充原文；必要的Loop输入/连续基础率/时间适配写明。独立预测器F保留在产品双模型设计，首轮内部fR/fT/fP不直接替换；目前14维状态中没有F输出。本文后续旧“先创新/门控”内容由本条覆盖。

> 2026-09-14当前任务：D01b [Loop训练数据管线v1](Loop数据集/训练管线_v1/README.md)。已实现72×14核心输入、缺失掩码、train-only归一化、多时长动作索引及样本/轨迹读取。继续按RL-DITR的动作条件患者模型＋监督/模型式策略学习主线，下一阶段为真实患者模型训练与多步验证；不是直接在旧历史预测器上运行PPO。5分钟来自数据网格，动作可研究5/15/30/60分钟，最终产品周期未定。奖励以作者glu2risk固定commit核对，避免论文印刷括号差异。最终验收见该目录验收结果；CUDA/模型效果未测试，其他原TODO保留。

> 2026-09-14批次47：已完成 AutoDL 社区 MCP 本地配置与 SDK/CLI 自检（13工具、上游9项测试通过）；尚缺真实实例 SSH 地址/端口和认证，未远端传输或训练。后续连接从 [AutoDL说明](tools/autodl/README.md) 开始，使用项目包装器；不改变下述模型/产品未完成状态。

> 2026-09-14批次44最新：已完成 [Loop无插值预处理与最终审计](Loop数据集/预处理_v1/00_从这里开始.md)。实际train为225人/1,653,421条transition；validation55人、sealed_test32人、demo17人。329是四组合计。原始文件未改；真实模型与反事实/临床闭环未验收，D01父项仍未完成。下文此前“未处理/369候选”属于历史状态。

> 2026-09-14目录更新：Loop专属原始数据和历史材料已迁至 [Loop数据集](Loop数据集/00_从这里开始.md)。本轮仅整理目录与逐文件/逐字段说明，未执行事件重建、资格划分或训练；下文历史目录示意不代表当前Loop存放位置。

2026-09-12 批次42产品范围澄清：当前先以手动上传/补充资料、评估、受控建议、实际执行反馈和再次评估打通产品闭环；临床长期监测和设备自动上传属于后续扩展，不作为当前交付前置条件。本次不修改RL训练方式、数据集选型或启动其他TODO；Loop仍为首版RL/生理主池候选，其他来源用途分层。

2026-09-12 批次38最新结论：[真实数据集RL训练资格审计报告](07_真实数据集RL训练资格审计报告.md)。五来源已解压并完成关键表全量审计；Loop为唯一首版RL/生理主池候选（369为提名上限，非合格人数），DCLP3/Ohio/UOM用途分层，IOBP2整源外部部分评价。先资格冻结，再BC/TD3+BC和ReplayBG生理结构辨识，通过独立验证后开展模型式RL；当前Harness须新增真实来源/basal+bolus合同。未正式处理、未训练或改产品实现，M01/M02未完成。以下旧来源分工与下一步安排由本条及07覆盖；历史要求保留。

**2026-09-11 批次28当前入口：[RL真实联合数据集下载与构建方案](06_RL真实联合数据集下载与构建方案.md)。用户自行下载Loop/DCLP3/UOM，申请Ohio/DiaTrend；原包放`datasets/raw/`，随后先做资格/去重/划分审核。RL-DITR使用数据驱动患者模型（动态转移+预测），不依赖机理生理仿真器。真实联合数据为主，simglucose辅助；模型未训练、历史/真人执行权限未开放，其他产品TODO全部保留。以下批次27及26的下一项排期已被本条覆盖，历史证据保留。**

**2026-09-11 批次 27 最新覆盖说明：用户转为优先做 RL，本轮已完成公开数据选型来源审核，见 [审核报告](05_公开数据集与RL闭环可行性审核.md)。下一轮建议只审 T1D-UOM 泵子集可训练性，待本轮用户审核后开展；不沿用下方批次 26 的“下一项 P06”排期。真实模型仍未训练或接入；P06 等产品 TODO 保留。** 下方批次 26 的实现/测试证据保留用于追溯，本轮没有重跑工程测试。

更新：2026-09-11；批次 26 已完成 P04 受控 RL MCP，待用户审核。项目根目录：`/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案`。

最新检查点：新增 RLAgent 独立 SDK profile、request_rl / inspect_rl 两个工具；Core 装配同一 run/快照的父预测，SDK 只获取候选状态/引用。独立 rl 许可，RL SDK 父/policy 子 job、配置与父预测摘要绑定；候选不是安全通过或用药结论。工具返回 safety_checked=false，不返回剂量或报告。合同见 [RL MCP](implementation/docs/rl-mcp.md)。

Core 同时补强受控数值产物的后续读取：重验父 SDK、授权、依赖及配置；RL 消费预测、安全/草稿消费候选时必须等待相应 SDK 完整成功。产物已写入而 SDK 后置失败、撤权或绑定失效不能继续下游。SDK 外旧 fixture 保持兼容。

全量 **604 项通过**（新增 56 项，155.34 秒；1 条既有 anyio 提示）。实际双 SDK/四个独立 worker、两种输入合同、localhost HTTP 预测→RL 的独立授权/幂等，以及 SDK 配合数值挂起探针的取消/撤权/超时回收通过。上游均离线 mock，收费 0 次，真实预测/RL 未接入。证据 `implementation/runtime/p04-full.xml`、`p04-focused.xml`、`p04-proof/`；旧库全部行/历史报告与源 SHA 不变，见 p04-compatibility/result.json。未访问现用桌面库或真实 key。

POST /runs/{run_id}/rl-agent（无 body）由 serve --enable-rl-agent 显式开启，默认关闭；purpose=rl 授权走现有 data-permissions API。设置同步 RL 工具定义/入口开关，原自检仍仅检查预测。前端与离线原生构建通过（p04-build.txt），本轮未实点原生窗口；未实现 P05 主循环或 P16 完整资料防护。

下一批 **27 只做 P06：最小会话、病例事实与澄清/快照状态**。依赖顺序为 P04 → P06 → P05，等待本批用户审核后继续。

## 新会话先读这里

1. 读取本文件和 [AGENTS.md](AGENTS.md)，再查看 [总体方案第 7.2 节](01_项目总体方案.md#72-产品补充-todo此后执行顺序) 的 P06 与执行依赖。
2. **用户审核 P04 后，下一项只做 P06：最小会话/病例事实/澄清与快照版本衔接。** 先明确文本输入、事实确认、切病例/补资料/取消/重开的最小可验收范围；按既有规则若过大可拆分明确子项，保留父要求。不能直接跨入 P05 主 Agent 循环。
3. 不要重新跑一遍历史开发，不要一次做完全部新 TODO，不要启动 self-evolution。详细会话存档仅需追溯时读取，避免把全部历史重新灌入上下文。

## 用户已确定的目标与开发顺序

- 最终是面向用户的医疗 Agent：多模态生理资料输入 → 必要追问/核验 → 胰岛素建议及自然语言解释 → 实际用药、测量反馈与新评估。研究工作台只是中间验收载体。
- Agent 主导任务规划、工具选择、追问及有限恢复；Harness/Core 执行来源、依赖、适用性、安全和发布约束。个体化给药结论必须具备两模型有效结果；两份结果只是必要条件，不能跳过其余闸门。
- 架构目标：**SDK 主 Agent → 受控 MCP → Core → 独立预测/RL worker**。执行器留在 SDK 外与 Agent 通过 MCP 调用不冲突。Agent 不得直连裸模型、改数值、自造推理结果、自审或强制发布。
- 只使用公开数据集和公开仿真资源，遵守各自许可。RL-DITR 只借鉴方法，不复刻其原实验、私有医院队列，也不等待作者权重。
- 用户授权依据公开数据/模型选择胰岛素任务；v6 选择成人 T1D 基础输注研究场景，先在虚拟患者验收。不能把 U/min、U/h 和区间总量 U 偷换成单次注射剂量；现有预测器是否适配该仿真域仍未验证。
- 最新顺序：**完整必要链路 + 两个真实模型 + 各项 Harness 闸门通过 → 冻结基线/数据划分 → 用公开真实数据正式 self-evolution。** 现在先做 Harness、数值模型占位；两真实模型仍须在后续 M01/M02 完成，绝非永久省略。当前不训练/优化模型，不提前跑进化。
- 每轮只做一个具体 TODO；只有实际实现并验证才标 ☑️，否则保持 ☐。根目录工作记录即时更新，末尾固定保留 16 项“已完成 Harness 功能清单”，批次笔记插在其前面。

## 已完成的软件：以已有运行证据为准

| 部分 | 已实现 | 必须保留的限制 |
| --- | --- | --- |
| 合同与输入 | 严格类型/时序/单位/缺失检查、快照冻结、输入变更失效 | 已有研究输入、基础输注动作与合成 fixture 绑定；真实模型适用性及多模态事实合同仍缺 |
| Core / 账本 | SQLite 事务、run/job/attempt、幂等、有限重试、取消、fencing、版本及父依赖绑定、权威产物登记 | 不证明医学有效性；真实数值模型未配置 |
| 隔离执行 | 独立 worker、超时/取消/失效回收、资源/输出/并发限制 | 不是对恶意同 UID 任意代码的 OS 沙箱；崩溃/强杀孤儿加固未齐 |
| 数值模型占位 | 预测 fixture 与 RL fixture，来源隔离及故障演示 | 非真实预测、非训练策略、非医学剂量；research 不回退到 fixture |
| 报告 | 真实 SDK/Flash 报告 Agent，受限工具提交章节提案；受信 renderer 生成工程草稿 | 仅章节排序，不能称完整自然语言医疗建议；普通 UI/API 不泄露草稿 |
| 双角色审核 | medical/ethics 真实 SDK 会话；报告/证据/revision/job/角色/配置内容绑定、重审与失效 | 工程报告审核，非临床专家；桌面尚未配置真实双审 |
| 发布 | Core 统一发布、读取时重验、依赖/审核/审计失败阻断 | 桌面双审/发布/全通道展示和导出未打通 |
| 原生 App | Tauri + React，JSON 导入、时间线、证据卡、fixture 故障；真实报告任务启动/状态/取消/重开 | 桌面止于 DRAFT_READY；不是聊天规划器；不是可独立分发安装包 |
| SDK 与凭据 | Claude Agent SDK 0.2.152 / CLI 2.1.259、官方 Flash 实际联调、受控工具/网关、费用预算 | 禁止 Pro/回退；真实 key 留在宿主，不进入 SDK/源码/日志 |
| 审计 | Core 事件/outbox、SDK 会话/工具元数据及接受/拒绝记录 | 外部 Observation、自动候选改进/评价/晋级/回滚未实现 |

批次 16 历史软件基线：**320 tests，0 failures/errors/skipped，70.335 秒**；另有一条既有依赖弃用提示。XML 时间为 2026-09-10 16:02 +08:00。该历史记录保留；批次 20 已另跑 415 项，见顶部。

最后原生批次：两个成功报告任务、一个取消任务，共 6 次真实 Flash 请求，2 份待审工程草稿，取消任务无草稿，**0 审核、0 发布**；正常退出/重开及原有数据保留经过该批次核验。本轮未重新启动 App，不能据此保证它此刻正在运行。

证据（相对于项目根目录）：

- `implementation/runtime/test-results.xml`、`implementation/runtime/test-output.txt`
- `implementation/runtime/desktop/agent-verification.json`、`agent-before.sqlite3`、`agent-security-check.json`
- `implementation/docs/desktop-agent.md`、`review-binding.md`、`review-agent.md`、`report-agent.md`、`sdk-flash-smoke.md`、`worker-isolation.md`
- `工作记录.md` 批次 16 与末尾功能清单。批次 17 为产品方案复审，批次 18 为本交接，二者都不增加软件完成项。

## 目前真实调用链与关键源码

当前 `implementation/src/medical_harness/agent_tasks.py` 的 `_work` 先调用 `core.execute_models`，然后调用 `reporter.generate`。所以**真实 SDK 已接入，但两个数值模型尚不是由主 Agent 的业务 MCP 工具自主请求**。

- `implementation/src/medical_harness/contracts.py`：SnapshotInput、PolicyOutput、Proposal 和公开导出；`basal_inputs.py`/`basal_actions.py` 为 P01 新合同。
- `implementation/src/medical_harness/core.py`、`store.py`：受信状态、事务与依赖；不能为了新合同绕过现有门禁或破坏旧数据。
- `implementation/src/medical_harness/adapters.py`、`workers.py`、`worker_entry.py`：执行器、fixture 和独立 worker。
- `implementation/src/medical_harness/sdk_worker.py`：内嵌 MCP。烟测 `prediction_probe/release_probe/inspect_evidence`；报告 `inspect_report_contract/submit_report_proposal/inspect_report_status`；审核 `inspect_review_packet/submit_review`；预测 `request_prediction/inspect_prediction`；RL `request_rl/inspect_rl`。已有工具不等于 P05 主循环完成。
- `implementation/src/medical_harness/report_agent.py`、`review_agent.py`、`flash_gateway.py`：受限 SDK 运行、暂存后提交、绑定与宿主网关。
- `implementation/src/medical_harness/render.py`：保留 `fixture-template-v1`，新增 `basal-fixture-template-v1`，均输出 `EngineeringFixtureReport`；当前无产品级医学解释能力。
- `implementation/src/medical_harness/rl_agent.py`：独立 RL SDK 与工具；`implementation/docs/rl-mcp.md` 说明父预测/授权边界。
- `implementation/app/src/AgentPanel.tsx`、`InputWorkspace.tsx`、`EvidencePanel.tsx`；`implementation/app/src-tauri/src/main.rs`、`bridge.rs`：前端与原生受控桥接。

当前桌面服务显式启用 `--enable-fixtures --enable-agent`，新 run 仍冻结 fixture reviewer 配置；下一步接双审时必须正确新建 run，不能把旧草稿/旧 verdict 静默升级成真实审核。历史批次中的“下一步”只代表当时状态；以代码、工作记录最新批次和 v6 TODO 为准。

## 下一项 P06 的边界

按总体方案 7.2，先有最小会话、病例事实、澄清状态与快照版本，之后再做 P05 主 Agent 的任务恢复循环。P06 覆盖文本输入、关键事实确认、切病例/补资料/取消/重开；旧快照资格失效，重开不自动恢复收费任务。若需拆分，先保留父项并记录明确子 TODO，不一次跨多个事项。

先读 core.py 的快照与 run 失效逻辑、data_policy.py / data_lifecycle.py、InputWorkspace.tsx 和当前公开输入合同。新增对话或事实不能把建议写成实际用药，不能通过未确认文本直接扩展模型适用性；追问及无剂量状态说明不必等待双模型，但个体化给药结论仍受原双模型和发布闸门约束。

P03/P04 的 SDK profile 已可请求/查询受控模型，当前均为单任务入口，数值为 fixture；P05 才组合成主 Agent 任务循环。设置本地自检仅预测，默认桌面预测/RL 业务入口均未启用，不能把工具定义视为主循环已运行。

下一批从 **批次 27** 起，每轮一个 TODO 或其明确子项，验证后同步工作记录末尾 16 项清单并停止。

## 后续路线与 self-evolution

具体未完成项见 01 第 7.2 节，共 22 个新增/整合父项，P01/P02/P03/P04 工程合同已勾选，其余 18 个父项未勾选；另有用户补充的 P03a 设置组件已完成；原表的所有已完成与未完成要求均保留。执行依赖为 P01 → P02 → P03 → P04 → P06（会话事实状态）→ P05（主循环）→ P07…P16，随后按对应任务实现 M01/M02、V01/V02、O01；细分父项不授权跨多步执行。

- P05–P16：完整数据保护、主 Agent/会话、双审发布、RAG、语义解释、多模态、用户体验与反馈。
- M01/M02：真实预测器接口/适用域核验，自研 RL 及公开仿真评价。已有预测器的真实路径、窗口、人群仍待核对；RL 尚未训练。不要向私有医院队列寻求依赖。
- V01/V02：完整用户任务与真实双模型研究闭环、闸门正反验证、新环境原生交付/恢复。仿真有效不等于真人剂量产品准入。
- O01：完成上述必要链路、真实模型和门禁验收后接通并开展正式 self-evolution。观察准备可以随开发进行，但现在不跑进化。
- C01：实际患者服务的用途、专家规则、医学及适用准入证据。离线公开数据进化不等于向患者直接发布建议。

进化数据先固定为 `evolve_train`（失败归因/候选生成）、`evolve_validation`（候选选择）、`sealed_test`（冻结后最终评价）、`demo`（展示），患者划分先于切窗、跨源去重，并核对与数值模型训练集重叠。公开真实数据须满足许可及对应任务适用域；超范围只验资料处理和拒绝路径，不能放松门禁。仿真另验动作后果，故障/攻击用例另验硬约束，不能混成临床效果。

只允许改进主 Agent 提示、工具说明、检索查询和不改变医学语义的表达候选；隔离比较新旧版本，固定评价器/预算，满足安全与正常任务完成要求才晋级，可回滚。Core、权限、剂量来源/动作语义、安全规则、可信模板/renderer、评审规则和封存答案不允许自动修改；预测/RL 权重训练与该循环分开。

## 环境、运行与数据保护

- 项目根不是 Git 仓库，`implementation/` 才是；当前存在大量已暂存、已修改和未跟踪的既有实现，**不要 reset/clean，不要覆盖或批量暂存**。工作目录切换时必须明确。
- Python：`implementation/.venv/bin/python`；pytest 已在 pyproject 配置 `--basetemp=runtime/pytest-temp`。所有运行数据、缓存、日志、临时材料留在本目录，保留原有数据库。
- 私有凭据：`implementation/runtime/private/deepseek.key`，历史批次确认存在且 0600，批次 20 未访问，不复制内容；宿主读取，SDK/前端不得持有。官方白名单固定为项目已联调的 `deepseek-flash`，禁止 Pro。型号映射是先前核验记录，不需要本交接产生联网或模型费用；未来确需视觉时按用户指定 Flash 视觉路线另验，不自动换模型。
- 桌面报告预算：`implementation/runtime/desktop-core/agent/budget.json`，共享持久限制 12 次请求、0.05 美元保守预留；不能重置旧账伪造预算。各独立联调有各自账本；历史桌面批次预留为 0.0234828 美元，不是账单金额。
- 开发 App 位于 `implementation/app/src-tauri/target/debug/bundle/macos/Medical Harness.app`，历史批次确认文件存在，本轮未重验；它依赖本项目/Python，不是已完成的独立安装包。

在 `/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/implementation` 内执行（按实际变更选择必要命令，不要仅交接就全部重跑）：

```bash
.venv/bin/python -m pytest -q --junitxml=runtime/test-results.xml
CARGO_NET_OFFLINE=true scripts/desktop.sh build
scripts/desktop.sh open
```

真实 API smoke 会收费，某些脚本会覆盖最新导出文件；不要为了“恢复上下文”重复执行。先看已有证据及持久预算，仅在当前单项实现确需时做受限真实联调。

## 文档入口与本次完成状态

- [AGENTS.md](AGENTS.md)：持续开发约定。
- [01 总体方案](01_项目总体方案.md)：v6 产品合同、权责与全部 TODO；第 5 节已同步最新自进化顺序。
- [02 RL 方案](02_RL动态给药模型方案.md)：公开资源任务选择、自研算法、数据与评价；真实训练尚未开始。
- [04 产品审查](04_产品闭环审查与验收.md)：v5 缺口、源码证据与 v6 修订理由。
- [工作记录](工作记录.md)：批次 16 原生软件基线、17 方案复审、18 交接、19/20 的 P01 工程合同；已完成清单始终在末尾。
- `03_原长版方案_v3.5_保留.md` 只作历史参考，不作为当前执行顺序。
- 详细九段存档：[2026-09-10-codex-1655.md](summary/sessions/2026-09-10-codex-1655.md)（仅需历史原话时读取；凭据已脱敏）。

批次 26 已完成 P04；模型继续占位，完整用户任务与真实资料保护、医学验证仍未完成。下一项 P06，等待用户审核后继续。
