# 规划器行为初始化后进行真实 PPO：条件性后续设计

2026-10-03；仅设计，未实现、未训练、未启动远端。此方案提出时，已有 MPC 的 development 结果及部分 PPO development 结果曝光，因此属于**后续探索方案，不是原预注册实验**。根任务报告概率 MPC 在完整 60 条开发轨迹上 TIR 约 96.8%、TBR54 为 0；本文未重新读取其原始轨迹，不将该进度信息当成新复核证据。现有 world-PPO 尚须完成固定训练及 8/16/32/40 面板。

**建议先完成现有 world-PPO。若它满足已经冻结的研究判断标准，就不新增这组实验。** 不能仅为追求更好表格继续训练。如果仍弱，唯一建议的新候选是：用冻结 risk MPC 在既有 train 历史上的首动作，给原 195 维 actor 做一次固定行为克隆初始化，然后用原真实模拟器 BG 奖励进行完整 PPO；同时保留该初始化的 BC-only 闭环对照。要回答的是“规划器提供的初始化能否帮助真实奖励学习”，不是给 MPC 的成绩改名为 RL。

## 源码事实：D06 已经做过什么

已只读核对 [旧训练器](../RL_DSENet_2026-09-17/train_policy_bounded.py)、[旧策略与计划](../RL_DSENet_2026-09-17/policy_bounded.py)、[旧部署 worker](../RL_DSENet_2026-09-17/bounded_policy_worker.py)：

- D06 冻结 D05/P03，在 Loop `train` 历史上计算七条计划的 world utility，优化 `J = Σ_a π(a|h) Q_world(h,a) − 0.05 KL(π || prior)`；没有 teacher 交叉熵，没有真实模拟器 BG PPO/GAE 更新。源码配置是一遍完整训练集、seed 260915、batch 256、AdamW 3e-4。
- `Q_world` 来自点 CGM 轨迹的 RL-DITR status 分数，按每小时 0.9 折扣；不是已经校准的真实 BG 事件概率。七条计划围绕固定观测 anchor，在三个 80 分钟块之一改变 ±0.25 U/h，执行时仅取第一步。
- 已有 actor、纯 planner 和 actor top-3 加 hold 后 world 重排三种部署。旧 actor 是 `306→128→7`，不是目前 `195→128→128→9`。
- 数学上，在单个状态不受网络函数类限制时，该 KL 正则目标的最优分布满足 `π*(a|h) ∝ prior(a) exp(Q_world(h,a)/0.05)`。这是对旧目标的解释，不表示旧源码实现了这个 soft teacher 或 BC。

因此，“world utility 引导 actor”“planner 到 policy”本身均不是本项目的新思想，也不作为新颖算法主张。下面的实证差异是**新概率 world 的规划行为初始化，接着由真实交互 BG 回报纠正**；属于已有策略提取/行为初始化与 PPO 的工程组合。

## 单一最小候选及计划映射

保留完成训练的 quantile world、P03、normalization、来源 SHA、risk MPC 配置；只使用一个固定 teacher，取现有 `mode=risk`，不在 expected/risk 的开发结果间做新一轮 teacher 搜索。选择 risk 是为了明确包含已实现的事件项，不是声称它优于 expected。teacher 代价原样为：边缘分位数数值积分的平均低权重 2/高权重 1 Kovatchev 风险，加 `2 p_BG70 + 4 p_BG54` 和 `0.02 mean((u/anchor−1)^2)`。这既不是联合轨迹 CVaR，也不是临床风险保证；事件概率的校准限制继续报告。

[paired_collect.plans](paired_collect.py) / [world_control_worker](world_control_worker.py) 的九条六小时计划如下。计划中干预结束后返回固定 warmup anchor；真实部署仍每五分钟重算，**不承诺持续执行 60 或 120 分钟**。

| MPC 计划编号 | 干预倍率 | 干预时长 | 首动作对应的原 actor 类别 |
|---:|---:|---:|---:|
| 0 | 1 | 全程 hold | 4 |
| 1 | 0 | 60 min | 0 |
| 2 | 0.5 | 60 min | 2 |
| 3 | 1.5 | 60 min | 6 |
| 4 | 2 | 60 min | 8 |
| 5 | 0 | 120 min | 0 |
| 6 | 0.5 | 120 min | 2 |
| 7 | 1.5 | 120 min | 6 |
| 8 | 2 | 120 min | 8 |

固定映射为 `[4,0,2,6,8,0,2,6,8]`；actor 原倍率仍为 `[0,.25,.5,.75,1,1.25,1.5,1.75,2]`。teacher 先按现有 `argmin` 与平局规则选完整计划，再映射该计划**已 clip 到 0–20 U/h 的首动作**。必须验证与 actor 对应类别的实际动作完全相同；大 anchor 的 clipping 别名也保留审计。不能直接克隆九个计划编号、平均 60/120 分钟动作或按重复计划数给某首动作额外权重。最小版本用一个 hard winner，不增加温度或 soft-score 调参。

继续复用 [ppo_world_worker.features](ppo_world_worker.py) 的全部 195 维：context 64、P03 48、physiology 28、anchor 1、九条 actor 自身 60 分钟计划的响应摘要 54。**不把九条 MPC 60/120 分钟计划偷偷换进这 54 维**，也不增加 teacher 选择结果作为部署输入。teacher 的 120 分钟后果并未逐项包含在现有摘要里，原历史/context 能否支持这种映射必须用克隆诊断检验；不能预设无损提取。若不能学到，则否决这个最小方案，不临时扩特征或改网络后继续用同一实验名称。

teacher 标签生成时可额外查询其九条计划；actor 部署时只计算现有 195 维，再由 actor argmax 决定真实动作，不查询 teacher、不做 MPC 重排或动作覆盖。195 维本身仍需 world 对九条 actor 计划预测，因而不是“无规划成本 actor”或已证实的推理加速。mean 部署另有独立消融，本方案不混用。

## 数据合同和固定预算

1. **示教状态只来自已有 natural `split=train` 数据。** 对 manifest、原轨迹及每个 NPZ 做 SHA 绑定，核对来源场景只为 103001–103004、已知十个成人和已冻结 bolus 设定。使用现成 `history[72,22]` 与固定观测 anchor；不读取 NPZ 的未来 CGM/BG、mask 所编码的结局或 `label_only_future_*` 来生成特征、teacher 动作、样本权重或筛选难例。BC 不需要未来完整窗口；合法历史的截尾样本也不能因未来低糖或终止被静默删除。
2. 103001–103003 的全部合法历史用于 BC；103004 按**完整场景/轨迹**留作一次 train 内部的克隆机制检查，不随机拆相邻窗口。此检查不是独立 world 验证，因为 world 训练已经接触过这些 train 场景。保留每条轨迹的 origin 顺序、覆盖率、类别计数；按真实来源密度训练，不重采样低糖或少数动作。相邻窗口不视为独立统计样本。
3. 不把已有 development 60 条状态、计划选择或真实未来作为示教数据；不从 development 日志挖掘“失败状态”加入 BC。`world_validation` 不参与 BC 或 PPO；confirmation 不读取。paired train 的真实分支未来也不作为 oracle teacher，本最小版本无需把配对 arms 扩成重复 BC 状态。已有 train 数据提供的 world 学习已通过冻结权重进入本方案。
4. **BC 固定三遍完整训练状态、batch 512、Adam 3e-4、gradient clip 0.5、seed 260915。** 从原 world-PPO 的网络结构和 prior 初始化开始，只训练 actor；world/P03 全冻结，可预计算 195 维和标签，缓存绑定全部来源。目标 `L_BC = −mean_s Σ_k y_k(s) log πθ(k|φ(s))`，`y = 0.95 one_hot(mapped_teacher_action) + 0.05/9`。5% label smoothing 只是固定的有限置信度初始化，给未示教的 .25/.75/1.25/1.75 类保留概率，不解释为安全探索。只保留第三遍结束权重作为候选；不按 development 或中间 epoch 选 BC checkpoint。
5. BC 结束存 `bc_init.pt` 及 actor SHA；该文件同时是 **BC-only** 部署权重和 **BC+RL** 的唯一 actor 起点。PPO critic 采用原零末层初始化，创建新的 actor/critic 优化器，不继承 BC Adam 状态、不继承弱 PPO checkpoint/value。将 Python/NumPy/Torch RNG 明确复位到同一个 260915 后开始真实 PPO，并保存边界 RNG；这保证配置可复查，不声称不同策略的采样轨迹或浮点运算逐位相同。
6. **PPO 固定 40×20 条三天真实模拟轨迹**，与当前 world-PPO 相同患者、场景 103201–103280、5 分钟动作、warmup、meal-bolus、reward、terminal penalty、gamma/GAE、lr、4 epochs/batch512、clip 和 entropy；只在新目录保存新结果。完整轨迹有 792 个部署动作，800 条最多 633,600 个真实 transition，native 提前终止导致实际数量更少，逐项报告，不补跑替换失败。额外 BC 更新和 teacher 查询成本单独列出，不能称与从 prior 开始的 PPO 完全相同总计算预算。

真实 PPO 使用原 `−(HBG risk + 2×LBG risk)/(12×10)` BG reward 和 native terminal −100。teacher 分数不当作真实 reward，不把 world 想象轨迹塞入 PPO buffer；保留原 GAE 与时限 bootstrap。PPO 阶段不加入新的 teacher KL/BC 正则或 planner 接管，才能把问题控制在“初始化是否有用”。FP32、两线程、TF32 off、cuDNN deterministic=False 等保留现行 world-PPO 运行设置。

## 必要对照与能够支持的结论

| 行 | 训练/部署 | 用来回答的问题 |
|---|---|---|
| 冻结 risk MPC | 现有 world；每五分钟直接选计划首动作 | teacher 自身的真实闭环表现；始终标为 MPC |
| 当前 world-PPO | 原 prior 起点；40×20 真 BG PPO；195 维 | 已有同结构、同真实交互预算参照 |
| BC-only | `bc_init.pt`；argmax actor；无 PPO 更新 | 行为提取本身保留了多少 teacher 能力 |
| BC+RL | 同一个 `bc_init.pt`；40×20 真 BG PPO；argmax actor | 真实奖励学习在同一个 BC 起点上的增量 |

wide PPO、hold、旧 D06 及其他既有行继续在统一大表保留，但它们不替代 BC-only 这个归因对照。本最小方案不再增加新的 point world、teacher 模式、持续时间、正则或多 seed 网格。

BC-only 仅评固定终点；BC+RL 仅评固定 8/16/32/40，完整 60 条相同 development 场景。训练和超参在看到这批新对照的闭环结果前冻结，不能因为 BC-only 的开发成绩调整 PPO 长度、reward 或 teacher。既有 checkpoint 选型与后续统一 confirmation 闸门仍适用；本文件不增加选型指标、改变原联合不退步标准或授权提前 confirmation。

所有行报告同一原 scorer 的真实 BG/CGM TIR、TBR70/54、持续低糖、失败、完整/未知尾部、TAR、波动、剂量与动作变动，保留 60 条逐轨迹配对结果和患者聚合。不要因 TBR54=0 就宣称安全，也不把十个已曝光虚拟成人写成新患者泛化。

只有 BC+RL 相比 BC-only 在原低糖/失败优先判断下有可复查的实际闭环增益，才能主张“真实 RL 改善了规划行为初始化”；只有 BC+RL 相比当前 world-PPO 有优势，才支持该初始化在本任务上有用。两项是不同问题。BC-only 接近 MPC、BC+RL 无增益，应写成“成功克隆，未证明 RL 增益”。即使 BC+RL 有增益，这组最小对照也不隔离概率 world 相对任意高质量 teacher 的专属贡献，不能把全部改善归因于新的 world 结构。

## 防止静默塌缩与明确否决条件

**先记录，不能靠强迫给药制造非 hold 动作。** 固定 label smoothing 与现有 entropy 0.01 是本方案仅有的探索设计，不保证不会塌缩。禁止非 hold 配额、遇低糖自动换动作、mean 替换 argmax 或失败后临时加 teacher KL。高 hold 占比本身不等于失败；低熵也可能来自状态适合 hold，需要与同状态 teacher 和真实闭环结果一起解释。

BC 完成后的固定 train 内部检查报告：teacher/actor 各类计数、overall agreement、teacher 非 hold 子集 agreement、各类召回、实际 U/h MAE、teacher 60/120 分钟来源比例、top-2 margin、hold 概率与 entropy。BC 和 PPO 均在同一个预先固定 train probe 上保存 logits/概率/实际 argmax、actor 参数 SHA；PPO 继续保存真实 BG reward、sampled action counts、KL、clip fraction、value loss 与 world 参数不变证据。可在该 probe 额外给 actor 首动作所属 teacher 计划计算最小 world cost，检查是否学成始终 hold；这是模型内诊断，不是真实反事实收益。

以下条件明确否决/停止，均不通过换名、补 seed、改预算或去掉失败轨迹修复结果：

1. **现有 world-PPO 已达原研究目标：不启动后续。** 若达标阈值在原协议中尚未数值化，先沿用主研究已声明的完整判断及限制，不用本方案事后创造阈值判现有方法失败。
2. 任一数据 split/源文件/权重 SHA 不闭合，或 teacher 标签用到了 dev、world_validation、confirmation、真实未来/隐藏状态：整组候选无效。技术失败保留产物并停止，不能转成 native terminal penalty。
3. 计划到首动作映射、clipping、duration、195 维顺序或冻结参数检验失败：不开始 BC/PPO。不能用“都叫九个动作”跳过该门槛。
4. 固定三遍 BC 后，train 内部机制检查的 overall action agreement <90%，或 teacher 非 hold 子集 agreement <80%：判定本最小表示/预算未提取到 teacher，不自动加 epoch 或扩特征后继续本实验。若检查集 teacher 非 hold 样本不足 100 个或来自不足 3 条完整来源轨迹，也停止为“证据不足”，不能用全 hold 的高 accuracy 过关。这些是本次设计拟冻结的工程门槛，未运行、非医学阈值，也不代表窗口统计独立。
5. PPO 没有新增真实 simulator transitions、reward/BG/动作数量不匹配、actor 不更新或 world/P03 被更新：不得标为 BC+RL。若固定 probe 显示 argmax 不变但概率变化，只报告概率变化；闭环收益必须由实际动作/轨迹证据支持。
6. BC+RL 相比 BC-only 没有符合原判断的增量，或仅靠更少完成轨迹得到低糖下降：否决“RL 带来改善”的主张。若 BC+RL 比当前 world-PPO 更差，否决“初始化优于现有路线”的主张；保留全部负结果。即便 BC-only 优秀，也只能报告 BC actor/MPC 提取，不把其行重命名成 RL 成功。

第 4 项未通过说明这一固定小方案失败，不证明 planner 无法被任何 actor 学习。若第 4 项通过而真实 BC-only 闭环偏离 MPC，也可能是状态分布变化、表示损失或递推误差，不能用离线 accuracy 替代闭环验证。本次不增加 DAgger、联合 world 更新、教师接管或预算自适应；若这些成为必要条件，应另作明确后续设计。

## 本次只读来源快照

| 文件 | SHA256 |
|---|---|
| `../RL_DSENet_2026-09-17/train_policy_bounded.py` | `971bab10b4b40b54ea77b4865c3ee524ec0432aa0687ddfc40e2062e25057da1` |
| `../RL_DSENet_2026-09-17/policy_bounded.py` | `73fb10be55505e02e13556a47b951f16091485a1ea672bf4b5413eb1fd77427e` |
| `../RL_DSENet_2026-09-17/bounded_policy_worker.py` | `3587eccef210e54ae1d471778d23cd0f1d86e2fe5acaeba8bbac1b4afc32ba0a` |
| `ppo_world_worker.py` | `c949d01d679baf58a95c231692975164890989861c9e17f891fd197028d0957e` |
| `ppo_wide_worker.py` | `282e7f58a36f94f6e13b6a84b258c3c3903662f49cedd3346708f012f7ee18eb` |
| `world_control_worker.py` | `88a40896b6e46b75b3a3612932831cc360a755cb36fbb3c5de800957c6ffffcb` |
| `paired_collect.py` | `c350557a654fc27f724770fb965ff4fa3a6abe12f72e6eacbe1b4c52163c1440` |
| `protocol.json` | `7f59e99eb5bd76fb430c0e6f488f75436315a2fffa3c415e7205e4f52a369b36` |

本文只新增方法设计；没有修改上述源码、训练记录、评分、协议或队列，没有读取 development/confirmation 原始轨迹用于训练，也没有执行任何新 BC/RL 或仿真。表内源码 hash 与计划映射在本地只读核对；设计中的预算、门槛和潜在收益尚无新实验结果。
