# P01a：版本化研究输入与用途合同

2026-09-11。本项接入 Core 的输入边界，不新增真实模型、临床规则、产品解释或桌面用药表单。后续批次 20 已完成 [P01b 工程动作和配置](basal-actions.md)，P01 工程合同验收通过；以下默认阻断仍成立，仅新增显式合成 fixture 执行路径。

## 当前可执行行为

`BasalResearchInput` 使用 `contract_version=basal-research-input-v1` 和 `use_profile=adult-t1d-basal-simulation-v1`。经现有 `POST /cases`、`POST /imports`、`PUT /cases/{id}/snapshot` 校验并冻结；归属用户可从 timeline 读取原始声明。新快照仍撤销旧任务及报告资格。

未开启 P01b 专用 fixture 配置时，完整新输入在启动 eval/research run 时返回 `INPUT_PROFILE_NOT_CONFIGURED`，无 run、job 或模型产物落库；桌面 Agent 在读取凭据和创建任务前也拒绝。既有 run 的消费入口另检查快照合同，不能仅凭旧 run 配置把新输入交给旧 fixture。证据卡不把这种输入显示为可执行或可发布。

输入的结构校验与来源字段一致性不等于鉴定了真实性。`source_ref`、主体标识、来源版本和定位均为导入者声明；Core 只对已冻结内容和摘要负责，没有签发模型证据。当前资料必须是合成/仿真研究数据，不开放真实个人健康资料输入；外发授权、真实药品、实际患者身份及数据生命周期另按 P02/P16 验收。

## 字段及当前消费者

| 字段 | 本轮执行的检查 | 后续用途及边界 |
| --- | --- | --- |
| cohort/source/use_profile | 仅 adult_t1d、synthetic/simulation、固定仿真基础输注用途 | 不支持 T2D、儿童、真实制剂或其他给药动作 |
| CGM history/unit/sampling/missing_mask | 保留 mg/dL、5 分钟网格、6–288 点、显式 null、时区和截止一致性 | 当前是存储与工程输入边界，非真实预测器已验证窗口；P01b 仅绑定工程 fixture 消费者 |
| provenance | 主体一致、来源种类与 source 一致、版本/定位非空、决策时已可用 | CGM 来源引用必须与 source_ref 一致；不自动读取定位指向的文件或 URL |
| treatment | 必填字段可为 null；有值时固定 generic_simulation_insulin/subcutaneous_pump/basal_only，并保留来源 | 不推断品牌、浓度或真实药品；供后续适用性规则与策略合同使用 |
| insulin_history | 必填字段可为 null；非空时声明覆盖起止和实际递送区间，截止等于 decision_time | 覆盖长度不视为已满足药效历史；必需长度保持未知，待模型配置验收 |
| deliveries | 仅 actual_delivery、正时长、不重复、不重叠、不留缺口；delivered_units 为非负有限数、单位 U | 这是过去区间的递送总量，不是新候选、命令率、接受状态或一次注射剂量；无自动换算/回填 |

CGM 来源的 available_at 不早于最后观测，也不能晚于 decision_time；当前 CGM 网格最后一点等于 decision_time，因此该来源可用时点与截止相同。治疗来源不得来自未来；输注来源不得早于其记录截止，截止不得晚于决策。时间比较按实际时刻处理，等价时区偏移可通过。

输注列表保持原始顺序，必须连续覆盖声明窗口。未知或不完整历史使用 `insulin_history=null`，不把空数组当作“没有用药”。若来源明确记录一个区间没有递送，可提交该区间 `delivered_units=0`；系统不从缺口推导零，不计算 IOB。当前每个历史窗口共用一份来源定位，混合来源逐条整合、纠错和反馈属于后续事实/反馈合同。

公开 `GET /contracts` 增量包含新 schema 和 input_profiles。消费者清单标为 `planned_only_no_model_or_medical_rule_binding`，prediction_window 和 required_insulin_history_minutes 为 null；客户端填写其他用途、版本、verified 或模型结果字段不能获得执行权限。

## 缺失与兼容

- CGM 缺失：允许冻结，输入/模型证据卡和启动请求报告 `MISSING_INPUT`。
- 治疗背景未知：明确 null，报告 `TREATMENT_REQUIRED`。
- 实际输注历史未知：明确 null，报告 `INSULIN_HISTORY_REQUIRED`。
- 新合同资料齐备但未开启 P01b 专用 fixture：报告 `INPUT_PROFILE_NOT_CONFIGURED`，不回退 fixture。
- 结构、来源、主体、时间、单位或用途非法：HTTP 固定 `INVALID_SCHEMA`，不回显原始事实；没有病例/快照写入。

原 `SnapshotInput` 不增加默认字段，不改变旧 JSON 或摘要。无 contract_version 的输入仍按原严格合同解析；它不能额外携带新治疗/历史字段。出现 contract_version 后显式分派，未知或 null 版本不会重试为旧合同。旧 synthetic eval 仍可完成已有工程报告和双审发布，不升级为新研究输入或医学报告。

无需本项数据库迁移；旧数据库使用现有增量初始化逻辑，既有行保留。历史报告重算仍使用原 renderer，P01a 未改变任何模型执行配置、可信模板或审核提示词。

## 示例及验证

`examples/basal-research-case.json` 是全合成样例，数值仅用于合同测试，没有真实递送、仿真执行或治疗含义。可导入和读取；默认不能启动模型任务，显式工程执行见 P01b。原 `examples/synthetic-case.json` 保留原样。

专项测试位于 `tests/test_basal_inputs.py`；完整结果为 `runtime/p01a-full.xml` 与 `runtime/p01a-full.txt`。另在 `runtime/p01a-compatibility/` 使用历史 `runtime/desktop/agent-before.sqlite3` 的独立副本核对旧行；原文件 hash 不变，未操作现用桌面库。旧副本初始化出现的 agent_tasks/review_configs 属于已有迁移逻辑，不是 P01a 新增表。

P01a 本批未进行真实 API 调用或原生窗口验收；输入子项本身不代表 P01 全部完成。P01b 后续补齐工程合同，但实际模型适用性、完整用药反馈与临床验证仍未完成。
