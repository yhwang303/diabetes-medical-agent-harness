# IQL_wide 基线适用范围与数值尺度审查

日期：2026-10-03。仅审查当前本地源码、作者 vendor、归档训练记录和已有开发评分核对包；未联网、训练、推理、修改源码或另选检查点。

**判断：当前 IQL_wide 可以作为一个真实训练、固定预算、同任务评价的 IQL 适配基线保留；证据不足以称为“充分调优的强 IQL”或本任务 SOTA。现在不必因开发成绩追加训练。** 应保留已冻结的最终 20k 结果和完整限制；如果需要更强的基线主张，先补充仅使用训练数据的只读诊断，再决定是否值得开展另行冻结的实验。不能把未知诊断写成已证实的缺陷，也不能用这些限制删掉或弱化不利比较。

## 1. 已有实证及来源范围

`checks/panels_development_new_iql_r1.json` 记录正式非 smoke 的 60/60 完整 episode，10 名已曝光虚拟成人，无失败、缺失尾部；与 D06 的 jobs/scorer 一致，`scoring_verified_exact=true`。患者均值 BG 指标如下：

| 方法 | TIR % | TBR70 % | TBR54 % | TAR180 % | LBGI | HBGI |
|---|---:|---:|---:|---:|---:|---:|
| IQL_wide，固定最终 20k | 94.1056 | 1.3742 | 0.5240 | 4.5202 | 0.6167 | 1.7069 |
| 原 D06 | 95.7260 | 2.0013 | 0.6965 | 2.2727 | 0.8798 | 1.2057 |

IQL 的低糖指标改善同时伴随 TIR 降低和 TAR180 升高，不能写成“低糖改善且未牺牲高糖/TIR”。这描述该固定实现的开发结果，不能推断 IQL 方法的性能上限，更不能以此推断 world/MPC 的改进机制。不同策略在低糖和高糖之间的取舍必须保留在主比较中。

正式训练元数据来自本地 `archives/completed_models_r1.tar.gz` 的 `results/IQL_wide/` 成员。审查通过内存读取，没有解包写出文件；completion/config/history/provenance 成员 SHA 与 `checks/completed_models_backup_r1.json` 一致。completion 记录 seed 260915、20,000 updates、5,120,000 sample visits、最终 `policy_020000.pt`，未使用 development/confirmation 选检查点。该权重 SHA 与开发评分包一致。源码配置、四个 IQL 实现文件、特征/协议及七个 vendor 文件也与归档 provenance 对应 SHA 一致。

本次没有重新运行评分器或加载 Torch 权重。旧 `iql_wide_design.md` 末尾仍保留实现阶段的“尚未训练/未接入”状态；在本次审查中，当前状态以已完成训练归档和真实开发评分证据为准，不修改旧文档。

## 2. 核心 IQL 机制确实存在

本地作者档案固定 commit `09d700248117881a75cb21f0adb95c6c8a694cb2`，MIT 许可。对照 `vendor/iql/actor.py`、`critic.py`、`learner.py`、`policy.py` 和 `configs/mujoco_config.py`，当前实现保留了：

- 以双 target Q 的最小值作目标的 expectile V 回归，expectile=0.7。
- 本步先更新 V，再以更新后的 V 和更新前 target Q 计算优势，随后更新 actor、Q，最后 target Q EMA。
- actor 目标是优势加权普通 Gaussian log likelihood，均值经过 tanh，标准差与状态无关；没有把它替换成确定性 MSE，也没有多加 tanh 分布 Jacobian。
- 权重 `w=min(exp(3A),100)`，其中 `A=min(targetQ1,targetQ2)−V`；actor cosine 学习率，target τ=0.005。

PyTorch 重写、默认层初始化、任务输入、动作映射、discount 和奖励与作者原任务不同。作者 MuJoCo 配置的 expectile/temperature 数字具有来源，但不等于它们已在本任务的奖励单位和数据上验证。`check_iql.py` 的已有 smoke 结果显示 actor、Q1、Q2、V 和两份 target Q 参数确有更新，评估冻结且重复动作一致；这排除了“只是未更新的占位实现”这一说法，不证明优化充分、权重选择合适或临床有效。

## 3. 奖励尺度与优势温度不能分开看

当前 IQL 与 PPO 使用相同的五分钟奖励、discount=.997 和真实原生终止定义：

```text
r = −(HBG_risk + 2×LBG_risk)/120
真实原生 terminal 的最后一步另减 100；时间限制结束仍 bootstrap。
A = min(targetQ1,targetQ2) − V
w = min(exp(βA),100)，β=3。
```

β 在这里是**乘数**，相当于逆温度；βA 必须无量纲。Q、V、A 的单位由奖励/折扣回报定义，而不是原始 mg/dL。若在理想精确求解条件下，把所有奖励项统一乘正数 c，策略回报排序及理论最优策略不变，Q、V、A 同时乘 c；expectile 回归本身也具有这种正尺度等变性。但原来的 actor 权重变成 `exp(βcA)`。要保持同一相对权重及 clipping 位置，需要同时令 `β′=β/c`，而不是保留 β=3 后宣称算法完全等价。

因此，“与 PPO 一样除以 120”保证双方使用同一奖励定义，**不保证 IQL 的有效优势选择强度与作者任务相同，也不保证与 PPO 的优化行为公平等价**。PPO 源码对整批 GAE advantage 做均值/标准差归一化；IQL 没有对 A 做同样处理，其指数权重直接感受 Q−V 的尺度。有限网络、随机初始化、学习率、估计误差和 optimizer 也使实际训练不具备理想的严格尺度不变性。

还要区分“全奖励统一缩放”和“仅改 reward_scale”：本实现额外的 terminal −100 不随 `/120` 自动缩放。只改连续奖励的除数而保留 −100，会改变终止惩罚相对于连续风险的权重，未必保留理论最优策略。这里不建议机械地把 β 乘以 120；作者任务奖励分布不同，当前也没有测出需要该倍数的证据。应先量测本任务的优势分布和重加权效果。

## 4. 正式日志能证明什么，尚不知道什么

`train_iql_wide.py` 每 100 次更新记录一次窗口均值；每次 batch=256、均匀有放回抽样。下表来自正式归档 `history.jsonl`，不是 smoke 或虚构分布：

| 记录末步数 | 窗口内 advantage_mean | 窗口内 advantage_weight_mean | action_bc_mse |
|---:|---:|---:|---:|
| 19,800 | −0.0136418 | 0.9729465 | 0.0606792 |
| 19,900 | −0.0318914 | 1.0079235 | 0.0603796 |
| 20,000 | −0.0095567 | 0.9795569 | 0.0596947 |

最后一行是最后 100 个更新、合计 25,600 次 sample visits 上的均值汇总，既不是唯一训练样本的完整分布，也不是在最终 checkpoint 上重新计算的固定集合诊断。每个样本对应当时变化中的模型，且可能重复采样。日志的 `action_bc_mse` 只是当前 IQL actor 对数据动作的 MSE，不是单独训练的 BC 对照成绩。

平均权重接近 1 **不能推出所有权重接近 1、有效样本量接近 batch size、没有高权重尾部或 IQL 等同 BC**；均值可能混合低权重与少数高权重。平均优势为负也不自动说明实现异常：上 expectile 的 V 会倾向高于条件 Q 分布的中心。三个窗口之间存在变化，同样不能凭局部均值认定发散或收敛。

当前未保存逐样本优势/权重分位数、方差、达到 100 上限的比例、有效样本量、不同数据来源/动作/低糖状态的权重质量分配，以及加权与不加权 actor 梯度差异。这些均为**未知**。小优势配合固定 β 可能使 actor 接近行为克隆，但目前只是需要诊断的假设；不能用开发结果反向把它写成原因。

## 5. 数据、表示、动作及预算的剩余差异

| 维度 | 当前 IQL_wide | 对比较的限制 |
|---|---|---|
| 数据 | 120 条自然探索训练轨迹 + PPO 前 8 轮的 160 条轨迹；280 episodes、221,760 transitions | 固定离线混合行为数据，没有之后的 PPO 数据，也没有自行在更新后的策略分布收集数据。数据覆盖和行为质量限制无法由“同一个奖励”消除。 |
| 抽样 | uniform transition sampling with replacement | 相邻 72 帧历史高度相关；221,760 transitions 不是同量独立场景。当前没有单独证明低糖/动作边界/外部 bolus 后状态的覆盖足够。 |
| 表示 | 72×22 展平 + 28 生理特征 + anchor/5，共 1,613 维；actor/V/双 Q 各 2×256 | 无 DSENet 或新 world 的预训练表示。PPO 使用冻结预测/历史表示，新 world 策略还使用模型预测特征；不属于仅替换 RL 算法、其余条件完全一致的消融。 |
| 可用信息 | 只有已观察 history 和 warmup anchor；真实 BG 只作奖励标签；无患者 ID、未来餐/bolus、CR/CF 隐藏参数 | 可观测合同一致值得保留，但 history 表示是否足以解决部分可观测性仍未验证；同一批已曝光虚拟患者不能称新患者泛化。 |
| 动作 | 连续 Gaussian 的确定性均值，`anchor×(mean+1)`，上限 min(20,2anchor)；每五分钟一次 | 与宽域 PPO/MPC 的外部支持范围相同，但 PPO 的九点 categorical、MPC 的九个未来计划、连续 IQL 均值是不同参数化。动作空间上限相同不意味着表达和探索机制相同。 |
| 预算与选择 | 单 seed、固定 20k updates、batch256；actor cosine 衰减至零；仅最终 checkpoint 正式评分 | 5,120,000 sample visits 约为 replay 大小的 23.09 倍，是重复访问量，不能当 23 次独立数据或自动证明收敛。PPO 的交互/更新/开发检查点预算不同；20k 完成只证明按约完成，不能反向认定预算充分或不足。 |

作者任务的 discount=.99，而这里为匹配 PPO 改为 .997，也会影响 Q/优势的尺度和估计难度。仅从固定 20k 或 actor 学习率到零不能判断收敛；需要训练内 Bellman/expectile 残差及策略稳定性等证据。训练过程没有开发集 checkpoint 搜索是一个清楚的选择规则，不应因最终开发表现不理想而事后换成 5k/10k。

数据量差异不必然只对 IQL 不利：它也获得了额外自然轨迹和前 8 轮 PPO 的多行为数据。表示、覆盖、算法和预算共同变化，现有实验不能把性能差分单独归因于任何一项，更不能用它估计“充分优化 IQL”的上限。

## 6. 最小可信度补充及是否需要现在重训

**第一步只读诊断，不需要任何额外训练或新 seed。** 在已保存的最终 20k checkpoint 上冻结全部模型，用预先固定的训练 replay 索引，或直接遍历训练 replay，保存索引/数据/权重/source SHA。不要读取 development/confirmation 的结果来选择样本、阈值或修改配置。最小输出包括：

1. A、βA、w 的 p0/p1/p5/p25/p50/p75/p95/p99/p100、均值/标准差、A>0 比例及上限触发率；同时给出归一化权重的 `ESS=(Σw)²/Σw²`、ESS/N 和前 1% 样本所占权重质量。ESS 是重加权集中度，不是把相关轨迹变成独立样本后的统计有效样本量。
2. 按已冻结的 natural/PPO-first8 来源、记录动作区间，以及训练标签 BG<70、BG<54 等分层重复这些汇总，并报告各层样本数及行为动作覆盖。BG/身份只在审计分层使用，不进入 actor。稀少样本须原样报告，不能以均值掩盖或补齐未观察区域。
3. 在同一冻结 actor、同一训练 batch 上，比较 w 与全 1 权重的 Gaussian NLL、目标梯度范数及方向；配合 Q−V、Bellman 残差和动作模仿误差判断重加权是否实质改变优化目标。此梯度诊断无需 optimizer.step，不产生新模型。即使梯度接近，也仅说明这些训练输入上的局部目标接近 BC，不能据此断言完整训练轨迹/闭环等价。

这些检查尚未实施。已有 5k/10k/20k checkpoint 可在相同训练集合上补充稳定性对照，但不能用新的开发排名替换已冻结的 20k 主结果。

**单独的 BC 训练是可选的第二阶段，不是当前必需动作。** 若以后要提出“IQL 的优势加权确实优于行为克隆”的机制主张，可在新的版本化方案里保持同 replay、表示、actor、优化器、20k 预算和同一既定 seed，仅将权重设为 1，得到匹配 BC 对照；不能拿旧任务 frozen BC 或上述 `action_bc_mse` 冒充该消融。它需要一次新训练，应先写清目的、预算和数据用途，再决定是否开展；本审查不实施，也不要求为当前报告补跑新 seed。

现阶段可以诚实报告“固定预算 IQL 任务适配基线”并保留完整开发结果，不必追加训练。若后续诊断发现数值异常，应独立登记、保留原结果，并说明新版本是否属 bug 修复或研究变更；不能因结果不利而选择性削弱基线，也不能把未量测的尺度假说包装成充分调优后的算法失败。候选模型本身的目标收益，仍由其独立的同任务比较、覆盖率、低糖/高糖共同指标和确认协议判断。

## 7. 可追溯证据

| 文件或归档成员 | SHA-256 |
|---|---|
| `checks/panels_development_new_iql_r1.json` | `ed627442f5e178fe34a5dfad934148207b2f9daed6b659546b137c1c12e0109e` |
| `checks/completed_models_backup_r1.json` | `1682ae145bd6f20b02fd7a044c5b2a710f522e698203afb2afae89deaca03fcc` |
| `checks/iql_mechanics_IQL_wide_smoke.json` | `26233d480a8bc3c34ca204b2285ee3c2e5c898f82784565d8e4388e3f2350bdc` |
| `iql_wide.py` | `2bd8ce3773e07ca0502fec17a63862d7ae16137ed37ec4b041d3a5768925b02c` |
| `prepare_iql_replay.py` | `069512184064c08817de933acf08f962d885796c7ffe40faaacbc533b327a5c7` |
| `train_iql_wide.py` | `58ec6e8bdff8bf353b26fa27d113e726fe67e8885e6b79a68fe8c2d17a4585f4` |
| 归档 `results/IQL_wide/history.jsonl` | `38dec7d74839317ad3e9124d4e7419fb56e8da6d32eaaa8bddbf1276e5c44f3c` |
| 归档 `results/IQL_wide/completion.json` | `9e91b044c50e948f9e8f15dd7512b349b782f431979284b6be8e26cb34cbe502` |
| 归档 `results/IQL_wide/provenance.json` | `f7f7b3296c2431e372bd6f3408ef75af345af3b9114c3780621dff1e0848d514` |

最终权重 SHA：`705b08a4da2c35a8759f3c70331e1fc0e37e27710000ac271c06032c4abaaab5`；训练 replay manifest SHA：`ca2c0456efaaf18391aa12e804e601093811da15220be25fe12717c379d109fa`。
