# 报告方法与证据口径（2026-10-03 快照）

完整定义、权重和文件哈希见 [method_registry.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/method_registry.json)。它登记方法和执行状态，不登记控制成绩。根任务最新回报：point/quantile world 正式训练均已过 2500/4000 步，wide PPO 已到 16/40 轮、预注册第 8 轮开发评价已启动，IQL wide 正式 20k 更新已启动；world PPO 仍只有 smoke。后续完成状态须用实际运行 manifest、completion、raw 和 summary 更新；本快照不随服务器自动变化。

| 报告名称 | 方法与 world 的关系 | 训练/动作差异 | 当前证据 |
|---|---|---|---|
| Hold | 固定已观测 warmup anchor | 无训练、单一动作 | 已接入；开发评价已启动，完成待核实 |
| IOB-aware zone PI(D) | 固定历史反馈及可解释 IOB/COB 代理；无学习 world | 项目固定经验参数；连续公共范围 | 已接入；不称作者官方复现 |
| D06 frozen | 原 P03/D05/H02 模型效用训练的有限计划 actor | 保留旧权重和 anchor±0.25；不扩宽 | 已接入，r2 smoke 完成；开发完成待核实 |
| BC、TD3+BC、IQL、ReBRAC、FQL、LOM、GFP frozen+projection | BC 为模仿；其余为旧离线 RL，无 learned-world 规划 | 旧 Loop+S1+S2 继续训练；原 0..20 输出再公共投影 | 已接入；58 项本地机制检查；根任务报告远端 8 方法依赖 SHA 预检通过 |
| RL-DITR* frozen+projection | 保留 learned patient model、policy 和 beam | 原 beam 后只投影首动作，计划内部未按新边界重算 | 同上；不能称 native 或边界内规划 |
| PPO wide | 冻结旧 world/reference 特征；新 actor 使用真实仿真 BG 奖励 | 9 个当前动作倍率 0、0.25…2；新训练 40×20 条预算 | 正式训练进行中；第 8 轮开发评价已启动 |
| Point-world median MPC | 新 point world 预测 + 固定风险 argmin | 9 个 6h 计划，执行首 5min；没有 RL actor | 已实现；等待 formal world 完成 |
| Quantile-world median / expected / event-risk MPC | 同一 quantile world 分别用中位数、期望风险、期望加事件风险 | 相同 9 计划；median 同权重行是拟议消融；expected/risk 已有配置 | 已实现机制；尚无正式控制成绩 |
| PPO with frozen quantile world | 冻结新 world 的上下文及动作响应特征；PPO 用真实 BG 奖励 | 新 actor/critic；不生成想象训练轨迹，不更新 world | 已实现，只有 smoke |
| PPO with frozen point world | 同上，q05 特征明确回退为 median | worker 支持；独立正式配置尚待冻结 | 拟评，未训练 |
| IQL wide | 无 DSENet/world；新离线 IQL | 1613 维可观测输入，连续公共范围；固定 20k 更新 | 已接入；280 例 replay、CUDA 4 步/冻结推理检查通过，正式 20k 训练中；闭环评价 smoke 待执行 |
| BC wide / DITR 候选内边界规划 | 追加公平性对照提案 | 前者需新训练；后者改变候选搜索 | 未实施，不能与已实现项混写 |

所有新评价都使用同一 60 条 development jobs、原环境和原 scorer。十名虚拟成人都已曝光，只能报告 known-patient 新场景结果。策略输入是过去 `72×22` 实际观测和可用的已观测 anchor；真实 BG 仅作训练标签/奖励或评价，未来餐食、隐藏 CR/生理参数、患者 ID 不进入策略。固定餐时 bolus 规则由外部环境执行，控制器只决定基础输注。

公共动作上限为 `min(20,2×anchor)` U/h，但各方法的有效支持不同：hold 是单点，D06 是窄离散支持，PPO 是 9 格，IQL/PI(D) 是连续动作，保留方法是原绝对率输出后的显式投影。共同信息许可和上限不等于相同表示、训练数据或预算。保留 FQL/GFP 继续使用逐病例 opaque RNG 流，病例标识只用于隔离随机流，不进入网络。

概率 world 的预测损失验证和闭环控制收益须分开报告。MPC 的 9 个计划为 hold、前 60/120min 的 0/0.5/1.5/2 倍后恢复；world PPO 的预测特征用 9 个 0/0.25…2 倍各持续 60min 的计划。因此 MPC 与 world PPO 对比并非只替换优化器。若比较风险目标，median/expected/event-risk 必须绑定同一 quantile checkpoint；另训 point-world 行同时改变学习目标/输出结构，应另列。

IQL wide 的 replay 预设为 120 条自然 train 加 wide PPO 前 8 轮 160 条 train，最终 `policy_020000.pt` 固定，不按开发结果择步数。其数据是离线收集而 PPO 使用新在线轨迹，不能据此声称训练预算完全相等。world 训练的自然/配对数据、world-validation 选择成本和旧 P03 训练历史也应列入方法说明。

最终表同时列低糖、高糖、TIR、提前终止、覆盖率和技术失败；有未知尾部的方法不能用已观察前缀充当完整随访胜利。保留方法另列每例/总体投影率及幅度，区分 raw、requested 和实际泵 delivered。旧 native/brake17 结果仅作历史附表；胜过某个失败适配实例不等于胜过其整个算法族。当前评价器继续拒绝 confirmation，须等最终模型与协议冻结后单独运行。
