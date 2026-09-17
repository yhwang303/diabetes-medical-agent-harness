# 离线 RL 基线：原始来源与实现合同

核验日期：2026-09-15。范围：为 Loop 成人 T1D 基础输注任务选择真正的策略算法，不把更换血糖预测器称作 RL baseline。本文仅调研与实现建议，未运行训练。

## 1. 选择结论

优先跑 **ReBRAC（NeurIPS 2023）与 FQL（ICML 2025）**。两者覆盖确定性行为正则策略与流匹配多模态策略，均直接学习连续动作。第三个可选 Cal-QL（NeurIPS 2023）；它更偏离线预训练后的在线微调，且需要可靠轨迹 return-to-go，因此当前不应为凑数量草率接入。BC、固定基础率/既有控制器可另列 sanity/control 行，不计入“两个较新 RL 基线”。

| 方法 | 原始论文 | 作者代码与本地固定 commit | 本任务定位 |
|---|---|---|---|
| ReBRAC | [Revisiting the Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2305.09836)，NeurIPS 2023；[会议PDF](https://papers.nips.cc/paper/2023/file/26cce1e512793f2072fd27c391e04652-Paper-Conference.pdf) | [tinkoff-ai/ReBRAC](https://github.com/tinkoff-ai/ReBRAC)，`694385b72fcef2ae318e3f0ff41ca36122355343`，Apache-2.0 | 较低成本、强确定性策略基线 |
| FQL | [Flow Q-Learning](https://proceedings.mlr.press/v267/park25f.html)，ICML 2025，PMLR 267:48104–48127；[arXiv](https://arxiv.org/abs/2502.02538) | [seohongpark/fql](https://github.com/seohongpark/fql)，`e8cd16eb490332924dfa2492097219f181765933`，MIT | 较新、可表示复杂行为分布的策略基线 |
| Cal-QL，可选 | [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html)，NeurIPS 2023 | [nakamotoo/Cal-QL](https://github.com/nakamotoo/Cal-QL)，`ac6eafec22e8d60836573e1f488c7f626ce8a77e`；检出的仓库未发现 LICENSE 文件 | 若轨迹回报合同和评估预算允许，再做第三组；算法可依论文自实现，勿把无许可代码直接纳入产品 |

代码已仅作为本地研究来源保存于 `sources/` 的对应目录；没有修改作者仓库。FQL不是 ICLR 2025。ReBRAC准确标题也不是“Revisiting Design Choices in Offline Model-Based RL”，后者是另一篇论文。

## 2. 统一记号与不能省略的数据合同

`h` 为72×22历史（包含稀疏food/exercise及掩码），`a` 为5分钟区间基础输注速率（U/h），`r` 为同一冻结奖励定义，`h'` 为下一真实观测历史，`d` 表示真实终止/预先定义的研究回合终止。`sg` 为停止梯度，`Q1,Q2` 为独立评论家，横线表示目标网络。

1. 所有算法使用同一患者划分、仅train计算的归一化、同一时点可获得的信息、动作上限与单位转换。`a_norm=2*a_Uh/Amax−1` 是连续可逆缩放；不能静默裁剪数据集中大于上限的原始动作，须先冻结动作适用合同并报告覆盖。输出裁剪与安全投影对所有算法一致，同时单列投影前后表现。
2. 原 `RL训练_2026-09-15/data.py` 输出用于患者模型的未来12步动作/血糖；不能直接把随机采样的 `next_state` 当1步RL转移。RL批次须明确 `h_t,a_t,r_t,h_{t+1},done,bootstrap_mask,a_{t+1}`。
3. 不因food/exercise缺失筛患者/样本。观测缺口、真正终止和人为时间截断必须区分；不能给缺失 next_action 填0再用于ReBRAC正则，不能把每一个窗口末端假装成生理终止。允许合法转移资格不同于预测窗口资格，但须报告患者/转移数量、原因及所有方法共用集合。
4. 共享72×22原始历史后，如用编码器，基线也得到同样输入和可比容量/训练预算。固定编码器若来自我方患者模型，需单列该适配并提供至少一项不依赖它的检查；不能把自身表示优势暗藏在baseline数据入口中。
5. 当前真实历史是闭环观察数据。策略可进行离线训练，但历史action MSE/TD loss不能证明改变给药后变好；主比较需要冻结的独立仿真情景或可信独立评价。不要在同一个我方学习环境里训练并据其预测“证明优于所有基线”。

## 3. ReBRAC：准确更新

核对：[作者实现固定版本](https://github.com/tinkoff-ai/ReBRAC/blob/694385b72fcef2ae318e3f0ff41ca36122355343/src/algorithms/rebrac.py)，`update_actor` / `update_critic`。

下一动作与TD目标：

```text
ε = clip(N(0, σ²), −c, c)
ã' = clip(π̄(h') + ε, −1, 1)
y = r + γ(1−d) [min_i Q̄_i(h',ã') − β_Q ||ã'−a'_data||²]
L_Q = Σ_i E[(Q_i(h,a) − sg(y))²]
```

策略更新：

```text
q = min_i Q_i(h,π(h))
λ = sg(1 / E|q|)           # normalize_q=True
L_actor = E[β_π ||π(h)−a_data||² − λ q]
```

原实现每2次critic更新一次actor；随后对actor/critic目标参数做 `θbar←τθ+(1−τ)θbar`。动作误差在动作维度求和；本任务动作只有1维。额外epsilon若用于防除零必须记录为数值稳定适配。

| 作者默认项 | 数值 |
|---|---:|
| actor/critic lr | 1e-3 / 1e-3 |
| actor/critic隐藏层 | 各3×256 |
| critic LayerNorm / actor LayerNorm | True / False |
| batch | 1024 |
| γ / τ | .99 / .005 |
| βπ / βQ | 1 / 1，需按任务调参 |
| target noise σ / clip | .2 / .5（归一化动作空间） |
| 总更新 | 1000个日志epoch × 1000次更新 = 1M |

这里作者代码的epoch是固定1000次随机采样更新的日志单元，**不是遍历一遍整个数据集**。本项目报告应统一记录optimizer updates、seen transitions、equivalent dataset passes、墙钟时间。

## 4. FQL：准确更新

核对：[作者实现固定版本](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py)，`critic_loss` / `actor_loss` / `compute_flow_actions`。

包含一个行为流速度网络 `vθ(h,x,t)`、一个单步噪声条件actor `πφ(h,z)`、两个Q网络。它**并非直接对整个ODE积分过程反向传播Q目标**。

```text
z ~ N(0,I), t ~ U[0,1], x_t = (1−t)z + t a_data
L_flow = E ||vθ(h,x_t,t) − (a_data−z)||²

x_0=z; x_(k+1)=x_k+vθ(h,x_k,k/K)/K  for k=0..K−1
Fθ(h,z)=clip(x_K,−1,1)
L_distill = E ||πφ(h,z)−sg(Fθ(h,z))||²
L_actor = α L_distill − E mean_i Q_i(h,clip(πφ(h,z),−1,1))

 a' = clip(πφ(h',z'),−1,1)
 y = r + γ mask · Agg_i Q̄_i(h',a')
 L_Q = E_i,h,a [(Q_i(h,a)−sg(y))²]
```

流BC、单步actor、critic各只接收各自损失对应梯度；`Fθ`在蒸馏目标侧冻结，Q权重在actor损失中冻结但保留Q对动作的梯度。作者代码 actor Q聚合始终mean，target聚合 `q_agg` 默认mean，可设min；不能无说明把两处都改成min。

| 作者默认项 | 数值 |
|---|---:|
| lr / batch | 3e-4 / 256 |
| actor、flow、Q隐藏层 | 各4×512 |
| critic LayerNorm / actor LayerNorm | True / False |
| γ / τ / Euler K | .99 / .005 / 10 |
| α / target聚合 | 10 / mean |
| offline更新 | 1M |

[作者README](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/README.md)明确：新任务建议 `normalize_q_loss=True` 并从 α∈{.03,.1,.3,1,3,10}调起，**但该归一化没有用于原论文主实验**。使用时表名应标FQL（任务适配，Q归一化），同时保留未归一化起始配置。默认官方环境需Python≥3.9/JAX；若在现有Torch环境自实现，需模块级损失与梯度数值核对，不能称字节级复现。

## 5. Cal-QL：可选第三组，不能伪造MC回报

核对：[作者实现固定版本](https://github.com/nakamotoo/Cal-QL/blob/ac6eafec22e8d60836573e1f488c7f626ce8a77e/JaxCQL/conservative_sac.py)。主体为双Q SAC/CQL。CQL的候选动作来自均匀分布、π(h)、π(h')；每类默认10个，但都在当前h上计算Q。对后两类策略候选，在保守惩罚内部替换：

```text
Q_cal(h,ã) = max(Q(h,ã), G_MC(h))
L_conservative = α_CQL E[ T logsumexp(candidate_scores/T) − Q(h,a_data) ]
L_Qi = MSE(Q_i, TD_target) + L_conservative_i
L_actor = E[α_entropy logπ(a|h) − min_i Q_i(h,a)]
```

`candidate_scores`包含采样密度校正：随机项 `Q−log(.5^action_dim)`，策略项 `Q_cal−logπ`。作者实现不对随机均匀候选做MC下界截断。`max`在保守项内阻止低于参考回报的Q继续被压低，不是把整个critic输出硬设为MC回报。

默认critic lr3e-4、actor lr1e-4、γ=.99、τ=.005，熵自动调节，backup_entropy=False，max target backup=True（10个下一候选取双Q最小值后再选最大），CQL温度1、权重调用默认5。AntMaze脚本采用actor2×256/Q4×256、1M离线更新，后接1M在线步；这些task专用reward_scale=10/reward_bias=−5不能移植成胰岛素奖励依据。

Loop若只有有缺口的长观察流，必须先冻结连续episode和MC回报尾部定义；截断MC并不自动是参考行为真实value。可将有限时域任务及剩余时间纳入state，或使用明确估计且有误差报告的参考值；不能用零补未来、把终止误处理产生的偏差当校准。仅运行离线阶段须标明“Cal-QL offline-only”，不声称复现其online fine-tuning优势。

## 6. 公平且可执行的预算建议（本节是研究设计，不是论文既定超参）

- 第一阶段：每算法默认配置100k更新，随后200k/500k检查是否仍改善；能否停止按提前冻结的验证趋势，而非只给我方更多更新。GPU耗时先用1000次更新烟测计量，本文不臆报4090训练分钟数。
- 调参阶段：每方法等额3个配置、同样最大更新和验证次数；ReBRAC可围绕行为正则强度，FQL围绕α。奖励尺度/γ若改变，所有方法重跑同合同；不能只重算我方奖励得到更高分。
- 最终结果：至少3个训练seed，所有方法使用同一批配对评价情景；大表报告均值±标准差以及情景/患者聚类置信区间。1 seed只能标初步结果。论文原默认1M更新；若只跑100k，标题/局限须写“受限预算任务适配比较”。
- γ以5分钟为一步有实际含义：.99的折扣权重时间尺度约100步≈8.3小时；它不是“60分钟预测长度”的同义词。选择γ需与明确的控制评价时域共同冻结。
- 基线网络容量可保留原默认以评价“方法完整实现”，另报参数量与总GPU时间；如全部统一encoder/MLP容量，是“容量受控比较”，不要混称原默认复现。
- 表格不能预设我方获胜。建议列：方法、年份、输入/训练数据、参数量、更新数、GPU小时、TIR、TBR<70、TBR<54、TAR>180、平均血糖、血糖变异、总基础胰岛素、预设临床风险函数积分、动作支持越界率、控制器投影率、场景失败率。血糖阈值需统一单位，最终medical/clinical语义须由主线核验；此处只提出表结构。
- 增加策略的离线日志指标：行为动作MAE、TD误差、Q尺度、策略Q−行为Q、BC/蒸馏损失、动作分位数及饱和比例。这些是诊断列，不是疗效列。
- 主验收可合理宽松：预注册仿真条件下相对合理控制器改善主要指标且关键低血糖风险不明显恶化、边界失效可控、结果跨seed稳定。无需每个小扰动都满足人工单调比例；也不能用放松诊断阈值替代实际闭环策略评价。

## 7. 交付状态

已核对三种算法原始论文身份、作者仓库、实现关键公式和默认参数；已读取现有Loop患者模型loader以识别RL转移适配风险。没有运行训练、没有操作AutoDL、没有改变原模型/数据代码。下一步主线可直接固定ReBRAC/FQL任务合同与预算，再做实现核对和真实策略训练。

## 8. 对主线拟定合同的补充审核

拟定：完整72×22历史flatten进入各自MLP；全部合法起点参与行为学习；0..20 U/h连续线性映射到[-1,1]；50k更新pilot后按统一规则延长200k–1M。

**可行，建议下列处理：**

- 主表两方法critic使用同一合法转移交集：下一观测可用于bootstrap，且下一日志动作真实有效。全部合法起点仍进ReBRAC actor BC和FQL flow BC。分别记录原始起点总数、actor样本数、有效Q转移数、被mask原因。不能把跳过TD的样本target设为r，那隐式假设后续价值为0；应当整个TD项不参加该样本的loss均值，并按有效样本计分母。
- FQL本身不需要下一日志动作。可增加“FQL使用全部自身可用转移”补充行，以免强行丢掉其数据效率优势。若主表各用最大自然集合，必须透明注明Q转移数量不同，不称“完全同样的RL训练样本”。这不是因food/exercise稀疏而砍训练数据。
- 真实任务终止允许bootstrap=0且Q target=r；传感缺失、文件终点、episode人为截断不自动是这种终止。若定义固定有限时域MDP，截断方式和剩余时长必须成为明示合同。
- flatten不访问未来，公平共享完整原始信息；但不等于原论文状态维度。报告输入1584维、网络容量及优化预算，不把其称为作者基准字节复现。
- 0..20 U/h映射仅在原始合法动作均处范围内才无损。其导数为0.1 normalized per U/h：ReBRAC原σ=.2意味着2 U/h噪声，clip=.5意味着5 U/h。噪声、行为正则量级应有预注册任务适配扫描。不同归一化会同时改变BC/蒸馏损失大小，保留物理单位动作误差诊断很必要。
- 50k更新可作排错和趋势pilot；2个方法都按同样500/1000step的短段计时后确定GPU预算。主比较不得把我方多轮调参与baseline一次pilot并排宣称最终优势。
