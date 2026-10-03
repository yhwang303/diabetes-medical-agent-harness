# 2026-10-04 开发选择与一次确认范围

本文件记录在新确认场景生成前的具体选择。它不声称目标已经实现或模型晋升；独立选择审查以 selection_review.md 为证据，最终 source/weight/配置/开发面板 SHA 以随后 gate 验证的计划与 freeze 为准。确认运行后本选择不再修改，不根据确认结果改模型、预算、动作部署、比较集合或评分。

## 判断依据与主候选

固定的全部开发运行结束：25 行、1,500 条原始记录原评分精确核验；19 行完整，6 个旧适配方法共有44个提前终止病例。单训练 seed260915、十名已曝光虚拟成人、3天轨迹与外源 bolus factor 三档保持原合同。开发已经用于方法设计和检查点选择，后续确认是新场景验证，不是新患者验证或训练多种子复现。

**主候选为 quantile world + 真 BG PPO 第40轮，argmax 部署。** 它相对 D06 的 BG TIR 95.7260→96.5972%、TBR70 2.0013→1.2163%、TBR54 0.6965→0.3577%、TAR180 2.2727→2.1864%、SD 24.5850→23.1438、CV 20.4424→18.7050%，LBGI 0.8798→0.5124；HBGI 1.2057→1.2402。9 项描述改善、1平、1差并不等于9项独立成功；原严格联合门槛因HBGI未通过，继续原样报告。用户扩大目标允许少数指标不是最优，此主候选支持进入确认，不能倒写成严格门槛通过。

TIR 患者配对 bootstrap 95% CI 为 −0.2146～1.9529 pp，TBR70 为 −1.7424～−0.0800 pp，TBR54 为 −1.0164～0 pp；TIR/严重低糖的总体稳定改进尚无充分证据，未校正多重比较。p4 TIR 下降2.3779 pp，SD在p3/p4也上升；均值不是每患者保障。

**第二个内部RL候选为 wide-PPO第40轮。** 它在开发中通过原严格均值联合门槛（10改善1平0差），但多个主要控制指标不如主候选。它使用旧 P03/H02/D05 与生理特征，不把其改善归因于新 quantile world。选择两个固定终点的理由是分别保留严格均值非退步路线与新世界模型路线，不在确认后把更好的一行伪称事先唯一选定模型。

**当前不激活 BC 追加训练。** 固定开发预算已完整结束，已有两个互补的 PPO 候选；当前停止事后扩展，先通过一次冻结确认检验可重复性和比较范围。这是研究阶段选择，不是宣称 world-PPO40 已通过原严格门槛、实现全表多数最优或完成最终研究目标。BC工具只完成本地机制与源码检查，不生成真实缓存、不训练、不加入成绩表。此决定不宣称world40为SOTA，不以它为借口隐藏HBGI退步，也不意味着已完成端到端联合world/policy训练。

## 固定18行及作用

| 固定方法ID | 已完成开发面板 | 确认中的作用 |
|---|---|---|
| hold | dev_hold_r1 | 参考或任务适配对照 |
| physiology | dev_physiology_r1 | 参考或任务适配对照 |
| d06_frozen | dev_legacy_r1 | 参考或任务适配对照 |
| retained_bc | dev_retained_bc_r1 | 参考或任务适配对照 |
| retained_td3bc | dev_retained_td3bc_r1 | 参考或任务适配对照 |
| retained_iql | dev_retained_iql_r1 | 参考或任务适配对照 |
| retained_rebrac | dev_retained_rebrac_r1 | 参考或任务适配对照 |
| retained_fql | dev_retained_fql_r1 | 参考或任务适配对照 |
| retained_lom | dev_retained_lom_r1 | 参考或任务适配对照 |
| retained_gfp | dev_retained_gfp_r1 | 参考或任务适配对照 |
| retained_ditr | dev_retained_ditr_r1 | 参考或任务适配对照 |
| iql_wide | dev_iql_wide_r1 | 参考或任务适配对照 |
| ppo_wide | dev_ppo40_r1 | 第二RL候选：旧world特征 + 真实PPO |
| ppo_world_quantile | dev_world_ppo40_r1 | 主候选：新概率world + 真实PPO |
| mpc_point_world | dev_world_point_r1 | 机制对照：固定MPC部署，不能改名为RL |
| mpc_quantile_median | dev_world_quantile_median_r1 | 机制对照：固定MPC部署，不能改名为RL |
| mpc_expected | dev_world_expected_r1 | 机制对照：固定MPC部署，不能改名为RL |
| mpc_event_risk | dev_world_risk_r1 | 机制对照：固定MPC部署，不能改名为RL |

Hold衡量当前观测基础率本身；IOB/Zone为可执行生理规则参考，不称临床认证或充分调优最佳算法；D06为项目历史完整闭环部署参考。八个旧方法全部保留，显式 frozen+projection，不删除开发失败方法。新 IQL 使用新任务训练数据和固定20k更新，不能据固定预算结果定义IQL上限；其训练表示/数据覆盖/预算与PPO不相同。

四种MPC全部确认，避免只保留最好的规划器：point、同一quantile权重的median、expected、event-risk。Quantile median在开发中已无严重低糖，因此不得将严重低糖清零归功于事件项。point/quantile还改变输出头、损失和best step，现有对照无法单独归因于某一模块。新world-PPO只在固定world特征上进行真实模拟器PPO，不使用想象rollout，也不是端到端联合优化world与policy。

所有确认行都用原评分器、同一60个job、相同5min动作、观测输入、meal/bolus规则。源代码/权重/配置在首次生成确认场景前一次冻结，batch-size均为8。旧对照动作投影、固定训练预算差异与既有失败风险保持明确说明。相对D06的联合判断、患者配对区间、全部行的逐指标最优与取舍分别报告；不能以只与较弱RL行比较替代全表比较。

所有开发8/16/32/40、同权重mean消融及工程失败继续进入开发表/附录；未把这些部署变体都放进确认，是因为本次确认锁定最终检查点与明确机制对照，而非继续选型。wide16 mean仍是开发后提出的内部消融，不能计为新增算法。BC/world-point-PPO等未执行提案不得出现在实验成绩表。

## 可证与不可证

确认会说明固定18个任务适配实例在新随机餐食/传感器场景下的完整控制表现。任何技术失败、原生提前终止或缺失尾部都保留；失败前缀不得作完整排名，固定集合不得删行。确认分数不能反过来参与训练、阈值或检查点选择。

这项单种子已见虚拟患者研究，不能证明临床安全、患者泛化、充分调优的外部算法上限或整个领域SOTA。独立审查确认：在拟选集合的12个完整部署中，两个PPO候选在11个family描述指标上都没有单独最优，仅TAR250与多行共同为0；world40的TBR54均集中于patient9，仍有4次长严重低糖事件。当前“相对D06多数指标改善”不是“全比较集多数指标最好”。最终必须按实际全表报告是否达到用户期望；如果没有达到，保留未达成结论，不放宽评分制造成功。
