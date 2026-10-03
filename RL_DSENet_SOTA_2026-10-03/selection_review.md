# 开发候选选择独立审查

日期：2026-10-04。本审查只读取已有 development 审计 JSON、方法定义、选择说明与必要源码；未读取或生成 confirmation 数据，未 freeze、未连接远端、未训练、未修改评分器或协议。本次只新增本文。

**结论：修订后的 [选择说明](confirmation_selection_20261004.md) 与现有证据一致，未发现阻断这次候选选择的错误或必须补做实验才能选型的遗漏。** 主候选 world-PPO40、第二 RL 候选 wide-PPO40 与其余 16 个固定对照构成可解释的 18 行集合。这个结论仅审查选择理由及其与已核 development 证据的一致性，不替代随后 gate 对完整训练预算、源码/权重/配置和确认未曝光条件的真实验证，也不表示研究目标或 SOTA 已实现。

## 实际核对范围

- 读取 25 行汇总：25 个面板、1,500 条病例记录，19 行可做完整随访比较，6 个 retained 部署合计 44 个失败病例。各面板 `scoring_verified_exact=true`，共享一个 scorer SHA 和一个 jobs SHA。
- 将 world-PPO 与 wide-PPO 各自的 8/16/32/40 加 D06 两份 5 行包，逐面板与 25 行包做对象精确比较，全部一致。两份各含 300 条病例记录。
- 核对 `confirmation_selected_panels_r1.json`：18 个 method ID 与 18 个 development panel 一一对应，ID 均存在于 registry，主候选/第二候选正确，绑定的 25 行包 SHA 正确。拟选集合有 12 个完整部署和 6 个 Observed 部署，全部为同一非 smoke development 60-job 面板。
- 从已核 JSON 的患者聚合、比较和事件字段重新计算/核对选择说明中的均值差、最优行、p3/p4 差异、低糖患者分布和失败总数；没有重新读取 raw 或再次执行评分器。`scoring_verified_exact` 是沿用已完成原评分审计的证据，不冒充本审查又做了一轮原始轨迹复算。
- `method_registry.json` 只用于方法定义与 ID，不使用其过时的运行状态快照推断训练或评价完成。

审查采用的 development jobs SHA 为 `92c50ab6c353023716a5e3fc7bfa76f0ac0999c9363f3cb82e0a5eaf0ed4a417`，原 scorer SHA 为 `0a6d44298d942bc09764f15e7ef45a71d3afbc7ba63cad3dae5425f0b22b8cfd`。

## 两条 PPO 路线的证据和门槛

`summarize_panels.py:247–277` 将原联合门槛与描述性指标家族分开，不能用后者覆盖前者。

| 比较 | 已核 development 结果 | 可支持的表述 |
|---|---|---|
| world-PPO40 相对 D06 | TIR +0.871212 pp、TBR70 −0.784933 pp、TBR54 −0.338805 pp；HBGI **+0.034466**。11 项描述指标为 9 改善、1 平、1 差；4 家族为 3 改善、1 取舍。原联合门槛因 HBGI 失败。 | 新 world 特征加真实 PPO 候选在该开发集相对项目旧部署呈现值得确认的低糖/TIR/波动取舍，不能称严格联合通过。 |
| wide-PPO40 相对 D06 | 原联合门槛通过；11 项描述指标为 10 改善、1 平、0 差，4 家族均改善。 | 保留一条通过原开发均值非退步标准的 RL 路线。该门槛不是患者逐例非退步、统计显著性或临床安全证明。 |
| world-PPO40 相对 wide-PPO40 | TIR +0.444024 pp、TBR70 −0.387205 pp、TBR54 −0.107323 pp、SD −0.906733、CV −1.020056 pp；HBGI **+0.049084**。 | 主次两个候选有不同取舍，同时冻结比事后只报告确认中较优者更可审查。这个比较不是只改变 world 一个组件的因果消融。 |

两个候选都选已经评完的固定第 40 轮，属于开发后选择，不能改称事前唯一候选。旧 P03/H02/D05 特征的 wide-PPO 改善不得归因于新 quantile world；两者表示、world 训练与预测依赖不同，共同真实 PPO 预算也不等于全部数据与计算预算相同。

选择说明第 11 行的患者配对区间与 JSON 一致：world40−D06 的 TIR 95% bootstrap CI 为 [−0.214646, 1.952862] pp，TBR70 为 [−1.742424, −0.079966] pp，TBR54 为 [−1.016414, 0] pp。TIR 区间跨零，严重低糖区间上界为零；不能把均值方向写成已经确定的总体收益，也不能因 HBGI 区间跨零而豁免原严格均值门槛。

world40 在 p4 的 TIR 下降 2.377946 pp，p3/p4 的 SD 分别上升 0.083961/0.378006。其非零 TBR54 集中在 patient9，60 条开发轨迹中仍有 **4 次持续至少 120 分钟的 BG<54 事件**。这支持保留逐患者与持续事件证据，不支持严重低糖已解决。十名已曝光虚拟成人的患者聚类区间也不替代训练多 seed 稳定性或新患者泛化。

## 全比较集与遗漏审查

18 行包含 hold、physiology、D06、8 个历史 retained frozen+projection、新任务 IQL、两条 PPO 和 4 个 MPC 部署。保留 physiology 与全部 MPC 很重要：它们提供比失败 retained 行更有区分力的控制参照。没有发现为支持这次有限任务确认必须临时增加训练/实验的遗漏；这不表示比较集合覆盖整个领域最强方法。

在拟选集合的 **12 个完整部署**、现有 family 的 **11 个描述指标**中，两条 PPO 均没有单独最优，仅 TAR250 与多行同为零：

| 指标 | 开发集合数值最优的完整部署 |
|---|---|
| TIR、TAR180、HBGI | point-world MPC |
| TBR70、CV、LBGI、低糖事件率 | physiology |
| TBR54、长严重低糖事件 | physiology 与三个 quantile MPC 部署共同为零 |
| SD | quantile-median MPC |
| TAR250 | 多个完整部署共同为零，包括两条 PPO |

这只是现有数值与 `1e-8` 比较容差下的描述，不是多重比较校正后的胜负检验。**相对 D06 的 9/11 或 10/11 改善，不能转换为全比较集多数最优。** 用户允许少数指标非最优并不证明当前结果已经实现该目标。修订后选择说明第 52 行已明确这一点。

补充原评分器已有的综合风险事实：同一已核包的 `complete.bg.risk.mean` 等于 LBGI+HBGI；逐行核对 12 个完整部署后，world40 的 **1.752581496825271 为其中最低**，其次 point MPC 为 1.7874653849185709、event-risk MPC 为 1.8006040259165619，wide40 为 1.8710909655383348。按训练风险权重构成的描述性组合 HBGI+2×LBGI，world40 为 2.2649482151623115，也在这 12 行中最低；这不是包含奖励缩放和终止惩罚的完整 PPO 累计回报。此处呈现已有分数及其确定组合，不增加新选择门槛、不修改 family 的 11 项统计，也不把相关综合指标计为额外独立胜利。一个综合风险数值领先可以说明总体风险取舍，但不能消除 HBGI 单项退步、4 次长严重事件或替代全指标/患者配对审查；“11 项 family 指标没有单独最优”仍成立，不能从 risk 最低推出多数指标最优或 SOTA。

6 个有失败的 retained 行继续保留，但仅有 Observed 前缀、覆盖率和未知尾部/TIR 上下界；不得参与完整随访指标最优排名，也不能拿它们的失败推断整个算法族上限。新 IQL 提供了明确 anchor/生理特征和宽动作的新任务训练参照，但其数据、表示和预算不同，不能声称对全部外部算法进行了同等充分调优。

没有把较早 checkpoint 和 mean16 放入确认并非自动构成隐藏负结果：选择说明保留了全部 8/16/32/40 与 mean 的 development 行，确认只锁定两个已选择终点及明确机制对照。早期 checkpoint 的低糖/高糖取舍不能被抹去，也不能把 18 行称作 18 个独立算法。未执行 BC 或 point-world PPO 不应补入成绩表。

## MPC 归因和 world 边界

保留 point、同一 quantile 权重的 median、expected、event-risk 四行是合理的机制比较。开发中 quantile-median、expected 和 event-risk 都没有 BG<54，因此严重低糖清零不能专归于概率积分或事件项。point/quantile 还存在输出头、训练损失和选中 best step 差异，不能把其差异单独归因于一个模块。MPC 行本身不是 RL。

world 预测绑定复算通过只说明 checkpoint 与保存预测的数值对应；不等于校准或控制安全。已有 quantile 诊断中，paired validation 的严重事件阈值 0.5 漏报 **18/55（32.7273%）**；natural validation 的严重阳性为 **0**，FNR 为 null、不可评估。开发控制成绩不消除这两个限制。

## 不激活 BC 条件如何解释

[原条件性设计](planner_actor_followup.md) 的“world-PPO 若已达到原研究目标就不启动 BC”，若具体解释为“world-PPO 已通过原严格联合门槛”，当前并不成立；wide40 的通过不能代替 world40 的 HBGI 失败。

不过 BC 是可选的事后追加方案，并没有“world-PPO 严格失败就必须新增训练”的反向义务。修订后选择说明第 15 行将不激活 BC 明确写成：固定开发预算结束，已有两个互补候选，停止事后扩展，先做一次冻结确认。该阶段决定与证据相容，不依赖宣称原目标或全表多数最优已经达成，因此**无阻断性冲突**。

确认不得反过来成为继续启动 BC、换 checkpoint、改阈值、删失败对照的开发数据。若结果未达到目标，应保留未达成结论；本审查没有批准后续追加实验。BC 工具继续保持未激活，工程边界见 [独立 BC 审查](planner_bc_independent_review.md)。

## 最终审阅证据快照

本表绑定的是本审查实际读到的修订版；不绑定尚未执行的 freeze 或确认结果。

| 文件 | SHA256 |
|---|---|
| `confirmation_selection_20261004.md` | `724eb0b7553469fc489e86d8edbda69fdc909faab8857574c92e0273cdc11c7d` |
| `checks/confirmation_selected_panels_r1.json` | `38844a5af85d1626bbdc7e21f4f35bd8d0a10b3d32a56bec59b45d442410e90a` |
| `checks/panels_development_25_rows_r1.json` | `13a849dc0785ceb54368c788e470b5b2024978fef70de35b883dd0a64e4f5bb4` |
| `checks/panels_development_all_world_ppo_checkpoints_r1.json` | `0fa697d8a47e3074ad302d64796be42af0d2aed5b13457942bfea369872732ac` |
| `checks/panels_development_all_wide_checkpoints_r1.json` | `8ec77c504f3e21bd10dd843d714464869d8c771f95c560c7f4d3eaad6e5a25de` |
| `checks/world_quantile_r1_best_diagnostics.json` | `2f0b5c077051800324ee665a79654b71d49544b060cb00f751064d366c5259be` |
| `summarize_panels.py` | `82c7a11db9d3316af1cfd9584e6175c594262127831ccad61ac77f16c32c6438` |
| `研究合同.md` | `b690904705cc2cc85ee23ea0eeafe9da8b657c9f4a5b508b918aa5e508ce0e69` |
| `planner_actor_followup.md` | `2131c502a6c697c39d01b01c7235580a78cae059ce30897f996d5ed0dcf573af` |

本审查建议是保留上述边界并按既有门禁完成后续统一确认，不重写评分或把选择审查当成自动模型晋升。
