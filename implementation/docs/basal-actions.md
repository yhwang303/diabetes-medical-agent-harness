# P01b：基础输注动作、配置与证据绑定

2026-09-11，批次 20 完成 P01，批次 21 按用户要求增加等价单位转换。P01a/P01b 的工程合同已实现并验收，P01 父项完成。真实模型配置、仿真动作后果和医学有效性分别留在 M01/M02/C01，不能由此推导已支持真实患者给药。

## 开启方式与输入边界

新配置 `basal-fixture-profile-v2` 默认关闭，只允许完整的 `BasalResearchInput`、`source=synthetic`、`mode=eval`；需要同时开启 `--enable-fixtures --enable-basal-fixtures`。普通客户端不能上传 profile 或数值来授予执行资格。simulation、research、缺失的治疗/实际输注历史，以及旧 fixture 故障场景混用均被拒绝。桌面默认配置和 AgentTasks 入口仍不开放新合同；主 Agent 工具接入另按产品 TODO 推进。

在 implementation 目录运行独立本地验证：

```bash
.venv/bin/python -B scripts/basal_smoke.py
```

脚本创建唯一的 `runtime/basal-smoke/<id>/`，启动带认证的本地 HTTP 服务，运行四个独立 fixture worker，验证双审前阻断、双审后发布/读取、未认证请求与取消后失效，并关闭服务。没有真实 LLM 或数值模型调用。输入使用 `examples/basal-research-case.json`，历史数据不会覆盖。

## 动作语义与数值表达

新 `BasalRatePolicyOutput`（basal-policy-v2）只有 candidate/fixture_only 能承载 `BasalRateAction`；旧 `BasalPolicyOutput` / `BasalAction` 保留原 v1 合同。abstain/insufficient_input、unsupported/unsupported、error/model_error 必须 `action=null`；失败不能替换为零或“维持”。合法的显式零动作仍是候选值，必须通过相同闸门。

| 字段 | 强制合同 |
| --- | --- |
| action_kind / insulin / route | basal_rate / generic_simulation_insulin / subcutaneous_pump |
| action_unit / action_value | U/min 或 U/h；保留输出的原值/单位，有限非负数，不接受布尔或字符串；新合同不以固定小数位量化模型动作 |
| duration_minutes | 固定 5 分钟 |
| decision_time / end_time | 带时区；开始等于输入决策时点，结束严格为开始加 5 分钟 |
| 工程边界 | 原值 schema 上限 60，语义校验按单位等价限制至 1 U/min；工程规则进一步限制为 0.05 U/min（3 U/h）。比较采用精确分数；两者均不是医学阈值 |
| 受控转换 | U/h ÷ 60 得 U/min，反向 × 60；区间总量 = U/min × 持续分钟。保留源值/单位和精确分数，客户端不能传入转换结果；有限小数精确展示，循环小数以 28 位有效数字 half-even 近似展示并标记 |

输出明确标记合成回放、非单次注射、不外推全天、`action_executed=false`。发布与读取均不新增或改写输入中的 `actual_delivery`；实际执行反馈尚未接入。实际装置分辨率仍未验证，本轮不进行装置动作量化；通用仿真胰岛素不能解释为真实药品、品牌或浓度。

`basal-rate-conversion-v1` 将换算规则绑定到新 run 的版本和配置摘要。权威产物保留执行器输出的 `action_value` / `action_unit` 字段；报告引用该产物，并由代码生成 `conversion.source_value/source_unit/exact/approximate_fields`。转换精确性相对于已接收 JSON 数值的十进制表示，不声称恢复模型内部浮点计算之前的无限精度值。

`U` 本身无法说明它是区间总量还是一次注射，当前不能直接作为基础率单位转换；`mg/h` 等单位也不做未获定义的转换。来源合法仍需同时满足原有用途、父依赖、状态、期限、安全和双审要求。旧 v1 的 U/min 与 0.000001 工程精度限制只用于重验旧记录，新 v2 不沿用该任意量化条件。

## 当前配置和字段消费者

| 消费者 | 当前真实执行行为 | 未验证边界 |
| --- | --- | --- |
| Core 输入检查 | adult_t1d、synthetic、固定用途、CGM 网格/缺失、治疗和输注历史、来源及时间一致性；CGM 至少 6 点、输注历史覆盖至少 25 分钟 | 6 点/25 分钟只是 fixture 输入条件，不代表真实模型窗口或药效历史充分 |
| fixture.basal-predictor / basal-prediction-v1 | 读取最后 CGM 值，重复为 6 个、间隔 5 分钟的预测点；无预处理变换，不确定性 unavailable | 未训练，没有真实预测精度或仿真域适用性 |
| fixture.basal-rate-policy / basal-policy-v2 | 返回固定 U/h 工程值；读取决策时点、输入摘要、Core 父预测 hash 和 profile hash | 请求虽携带完整快照及预测，但没有使用它们学习或推理药效；训练依赖为空 |
| basal-fixture-rules-v2 | 检查上述用途、候选状态、动作语义、绑定及工程上限 | 不是医学安全规则，无 IOB 计算 |
| basal-fixture-template-v2 | 从 Core 验收的产物取值并换算，显示用途和时效；LLM 只能提交既有章节提案 | 未新增产品级解释或用药对话；RAG 仍待办 |

配置显式记录 `clinical_use=false`、`real_model_configuration=null`、`medical_validation=false`。真实模型仍需自己的入口、权重/训练依赖、预处理、域/窗口/不确定性和适用性证据，不能继承 fixture 配置获得资格。公开 catalog 分开保留 P01a 的 planned input_profiles 与新增 engineering_execution_profiles，后者仅描述该工程执行路径。

## 冻结、时效与发布

Core 在新 run 的同一事务中登记 `run_profiles` 内容和摘要，包含静态执行器/规则/模板配置及 run、病例、快照、输入 hash、决策时点、墙钟到期时间。预测请求/输出绑定 profile hash；策略再绑定 Core 已登记的父预测，不能自供模型证据。每次消费重验完整配置、版本、输入、作业与父依赖；配置改变、撤销、错绑、迟到、换快照和取消均失效。

有两个独立时间轴：输入的 `decision_time` 是合成回放时点，动作区间从该时点起计；现实处理期限为启动时间加 `min(run_ttl, 300)` 秒，报告的 `valid_until` 是这个 Unix 时间。后者仅限制工程作业/发布/读取，不将历史动作变成当前真人建议，也不证明临床有效期。job 自身截止仍独立检查。

受信 renderer、证据摘要、同一草稿修订、两角色审核、发布及读取复用既有闸门。修改展示总量或转换源值/精确分数后即使重算保存 hash，仍因模板重算不一致被拒绝。此轮审核为 fixture；现有真实 SDK/Flash reviewer 没有为新合同重复收费联调。

## 兼容与验证证据

旧 SnapshotInput、请求/产物 schema、旧模板和旧报告语义保留。批次 20 只增 `run_profiles` 表，批次 21 不迁移 schema；保留旧 v1 的输入配置、执行器、动作校验与模板，不回填旧 run 为新用途，不将旧证据升级；draft job 仅增返回其实际 template_version 的元数据，以便报告 Agent 使用对应模板。

- 批次 21 全量 440 项通过（新增 25 项单位专项），`runtime/unit-conversion-full.xml` / `unit-conversion-full.txt`，1 条既有依赖弃用提示。原 P01b 的 415 项记录保留。
- 本地 HTTP 及进程证据：`runtime/basal-smoke/1ac777739e624fb298ba06ccf0244eea/result.json`；其中取消后的报告文件只是留存测试证据，不能绕过 Core 当有效报告使用。
- 批次 21 已发布 v1 报告重验：`runtime/unit-conversion-compatibility/result.json`；使用历史到期前时钟只验兼容，返回完整发布包与原记录相等，全部旧行和原库文件保留，不恢复当前有效性。
- 旧库只读副本迁移：`runtime/p01b-compatibility/result.json`。全部旧行保留，原文件 hash 不变；该样本无旧草稿/发布记录，旧报告行为由回归覆盖。
- 未重跑原生 App、未执行仿真给药、未接真实预测/RL、未调用收费 LLM；不据工程测试宣称用户任务或临床闭环完成。
