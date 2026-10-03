# 低糖与高糖共同约束下的闭环控制文献及作者代码核验

日期：2026-10-03。用途：研究候选设计与强基线适配。状态：一手来源及下列源码已阅读；未安装、运行、训练这些外部项目，未验证本项目收益。

本文件遵守本目录《研究合同》：一个训练 seed **260915**；4090 指 GPU。只写本文件，不改旧权重、评分、轨迹及产品 Harness。关于当前 DSENet/H02、actor residual 与 brake 结果的描述来自本轮委派上下文，未在本子任务中重新计算，不作为新的实证结论。

## 1. 结论与优先顺序

最值得优先验证的是：**有实际胰岛素历史的 IOB/作用量估计 + 能及时恢复的预测低糖减量 + ROC/IOB 联合调节的 zone MPC；随后以同一执行合同训练动作条件世界模型和 RL。** 仅把低糖惩罚加大、提高暂停阈值或增大保守边界，会重复“少低糖但多高糖”的失败方向。

世界模型与 RL 的直接一手参照是 **G2P2C**，可复用的后继工程是 **RL4T1D**，GPU 环境是 **GluCoEnv**。本次对“GLUCOGYM”的精确名称及相关组合检索没有定位到可核验的一手血糖控制项目；不能将 GluCoEnv 静默改称 GLUCOGYM。

本轮建议先回答两个可验证的问题：

1. **控制权限是否足够？** 当前 residual ±0.25 U/h 在每 5 min 内只相当于相对 anchor ±0.02083 U，持续 1 h 的差额上限为 0.25 U。若低糖主要来自外部已给出的过量 meal bolus，事后改变小幅 basal 的能力有限；这只是控制权限推断，须用同起点的仿真干预验证，不能凭它宣布“不可能”。
2. **世界模型是否知道不同胰岛素动作的后果？** 同一个历史下输入不同未来动作序列，必须产生可验证的滞后响应与风险排序。一个只会预测参考轨迹、再叠加固定线性响应的模型，还没有证明这个能力。

下面五项是不同机制及其验证办法，**并非一次叠加所有组件**。M1、M2、M5优先用于建立可解释对照；M3、M4用于模型和学习任务的实质变化。

## 2. 五个具体机制

### M1：ROC + IOB 联合加权 zone MPC，处理持续高糖和餐后回落

**一手证据。** Deshpande、Doyle、Dassau，2023，*Glucose Rate-of-Change and Insulin-on-Board Jointly Weighted Zone Model Predictive Control*，DOI 10.1109/TCST.2023.3291573。该方法用预测 ROC 与归一化 IOB 联合调整超过上区间边界的代价；明显下降时收敛到 basal，ROC 接近零且 IOB 不足时加强纠正。原文也报告部分标称场景低糖风险略增，不能改述为“所有场景安全不退步”。[作者原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10958373/)

**针对当前问题的研究假设。** brake 后的高糖不能只看 CGM 高度：若已积累很多活性胰岛素，继续加量可能造成延迟低糖；若作用量不足而高糖平台持续，则继续刹车会延长 TAR。联合权重区分这两种状态。

**最小实现范围。** 用可观测 CGM 历史估计 ROC，用已实际执行的 basal 与 bolus 历史计算 IOB；控制模型预测多步 glucose，优化区间偏差和动作变化，施加共同动作上限。先构建固定权重 zone MPC，再加入一个 ROC/IOB 权重面，保留完整参数。不要把论文的 IOB 公式、归一化和区间定义省略后仍标注为原算法复现。

**可用性与代码。** 本次未找到与该论文对应、可核验许可的作者完整控制代码；应称“据公开方法适配实现”，不是“运行作者实现”。IOB 的开源工程参照可用 MIT 的 OpenAPS oref0，但其净 IOB 定义与该论文不必相同，必须单独说明。论文公开可读不等于代码已开源。

**最小消融。** 固定 zone 权重 → 仅 ROC 权重 → ROC+IOB 权重。三者共用模型、约束、初态和餐食；重点检查餐后 2–6 h 与长时间高糖区段，报告低糖、TAR、恢复时延、增量剂量，不能只比奖励。

### M2：带恢复条件与胰岛素记账的预测暂停，减少长时间低输注

**一手证据。** *A Modular Safety System for an Insulin Dose Recommender: A Feasibility Study* 的预测模块区分部分减量、暂停及恢复，并设置暂停总时长限制。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7189144/) 早期 Kalman 预测暂停研究也明确区分暂停/恢复阈值；该研究自行说明未严格评估反弹高糖，故这里只采用结构启发，不借其结果证明当前收益。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3570849/)

**作者工程证据。** oref0 同时考虑 IOB、预测轨迹、变化率及现有 temp basal；源码包括“已有明显负净 IOB 且 BG 上升快于预期时不继续低糖暂停”、预测风险触发有界 zero temp、趋势改善时恢复基础率、现有减量过多时缩短/提高 temp 的分支。可核验源码在 `determine-basal.js` 第 907–975 行。[冻结源码](https://github.com/openaps/oref0/blob/88cf032aa74ff25f69464a7d9cd601ee3940c0b3/lib/determine-basal/determine-basal.js#L907)

**研究假设。** 必须把“开始刹车”与“解除刹车”作为两个待测行为。恢复不应等 CGM 已进入明显高糖才开始，也不能仅因暂停计时已到就忽视持续低糖预测。比较继续低输注、恢复 basal 和小幅纠正的未来轨迹，记录每次切换理由、持续时长、实际少给量。

**观测合同。** IOB 只能来自已执行日志；它是模型估计，不是模拟器真实体内胰岛素。区分：总活性 bolus/基础输注的估计，以及相对计划 basal 的“净 IOB”，后者可为负；不能把负净 IOB 解释为体内不存在胰岛素。漏记 bolus 的场景必须独立测试。

**代码许可与边界。** oref0 为 MIT，核验了根 `LICENSE.txt`、`lib/iob/calculate.js` 与 `lib/determine-basal/determine-basal.js`。选择性移植机制不等于完整 OpenAPS。basal-only 合同比较时禁用 SMB/correction bolus，统一既有外部 meal bolus，不能给该基线或候选额外通道。完整 oref0 原生配置可做另一个任务的参考行。

**最小消融。** 同一低糖触发器：旧固定 brake → 加明确恢复规则 → 再加实际输注 IOB。报告每次暂停后 4 h TAR、低糖复发、暂停分钟、少给量；预注册时间窗以免挑有利片段。

### M3：G2P2C 式动作条件模型学习 + 短期规划

**一手证据。** Hettiarachchi 等，2024，*G2P2C—A modular reinforcement learning algorithm for glucose control by glucose prediction and planning in Type 1 Diabetes*，DOI 10.1016/j.bspc.2023.105839。以 PPO 为基础，交替进行模型辅助学习和短期模型规划。[作者机构论文页](https://www.utupub.fi/items/3cc24fa9-f965-4c25-ba03-f219e9851189) [作者代码](https://github.com/RL4H/G2P2C)

**源码已核验。** `GlucoseModel.forward(extract_state, action, mode)`显式拼接状态与动作；规划滚动更新 CGM 与 insulin 历史。默认 `feature_history=12`、`n_features=2`，meal/carb announcement 关闭，`planning_n_step=6`。这些是该 commit 的默认值，不代表可照搬到本项目的最佳设定。[模型](https://github.com/RL4H/G2P2C/blob/37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70/agents/g2p2c/models.py#L67) [参数](https://github.com/RL4H/G2P2C/blob/37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70/agents/g2p2c/parameters.py)

**适配方向。** 保留 DSENet 的历史表征作为待检验组件，另训练真正的动作条件滚动模型，输入过去 CGM、实际 basal/bolus、已发生且获准的餐食记录以及候选未来输注；训练数据应包含不同控制器及受限干预轨迹。短期学习模型处理局部动态，长期胰岛素作用可先保留独立生理尾部估计。不要从未校准的远期想象奖励反向训练出激进策略。

**本项目须补的证据。** 多步动作响应、低糖事件召回、最低值误差、候选排序和闭环收益都通过后，才让规划结果影响策略；不同动作只改变固定线性项的版本保留为消融。30–60 min 的规划不能单独证明覆盖了全部胰岛素尾部风险。

**许可与移植陷阱。** G2P2C、RL4T1D 均核验为 MIT。原 `models.py` 第 222–226 行明确硬编码 exponential pump mapping；移植到 residual 或双动作任务时，训练/规划/真实执行必须调用同一个受控映射，不能只改环境端。[硬编码位置](https://github.com/RL4H/G2P2C/blob/37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70/agents/g2p2c/models.py#L222)

**最小消融。** 同一 observation/action/reward/真实交互预算下：PPO；PPO+模型辅助损失但无规划；PPO+模型+规划。另以“同一规划器、旧线性响应 vs 新动作条件模型”隔离世界模型贡献。不能把增加真实仿真样本或更宽动作范围算作规划收益。

### M4：改进动作表示；必要时建立统一的 basal+bolus 新任务

**一手证据。** Hettiarachchi 等，2022，*Non-linear Continuous Action Spaces for Reinforcement Learning in Type 1 Diabetes*，DOI 10.1007/978-3-031-22695-3_39。论文研究胰岛素动态范围导致的探索困难及非线性动作映射；作者 RL4T1D `ControlSpace` 提供线性、指数等映射。[作者机构论文页](https://researchportalplus.anu.edu.au/en/publications/non-linear-continuous-action-spaces-forreinforcement-learning-int/) [冻结实现](https://github.com/RL4H/RL4T1D/blob/ea9e1723b1d0961474887af3de2f683c941881bd/utils/control_space.py)

**重要区分。** 非线性映射是改变同一物理动作集合中的探索密度；扩大 basal 上限或加入 bolus 是改变控制权限。必须分别消融。若把动作从 anchor±0.25改成更宽范围后效果提高，不能全部归因于算法或世界模型。

**建议任务顺序。** 先在现有任务检查 residual 的饱和频率、实际执行差额、策略熵及梯度；如存在控制权限瓶颈，再另立“带实时餐食申报的 basal+meal-bolus”研究合同。后者可设两种有单位的输出：持续 5 min 的基础率 U/h，与申报餐时才允许的 bolus U；保持当前瞬时申报时间，绝不自动给予未来餐食。

**公平设计。** 所有方法都可优化同样的 basal 与 bolus 上下限、相同 CGM/申报记录、同样的 IOB 规则、同样设备限制；外部 `meal/CR×0.8/1/1.2` 自动 bolus 在新任务中不能再叠加，否则双重给药。0.8/1/1.2若继续用于研究，须明确是申报误差、处方误差还是剂量倍率，所有方法语义一致。原 P03/D05/D06 以“原算法+预先冻结的相同 bolus wrapper”进入新任务，另保留原 basal-only 表，不把两表合成一个排名。

**可观测性。** CR、CF、basal schedule 等只能来自合同允许的资料；不能从虚拟患者隐藏 `u2ss` 或真实胰岛素敏感性得到更准的处方。新动作合同需要重新采样 world 数据，否则原来固定外部 bolus 的训练分布不能证明反事实 bolus 效果。

**最小消融。** 相同物理 bounds 的线性映射 vs 非线性映射；之后才比较“固定 meal bolus”与“所有方法均可控制 meal bolus”的两张任务表。双头动作可先只在餐时开放 bolus，以限制搜索空间；完全无餐食申报的总输注控制另作 FCL 任务。

### M5：生理状态估计 + 多步 MPC，建立真正有竞争力的模型控制对照

**一手证据与代码。** Hauser、Jørgensen、Peuscher，2025，*An Open-Source Browser-Based Nonlinear Model Predictive Controller for Type 1 Diabetes*，DOI 10.1016/j.ifacol.2025.06.025，作者实现为 Hovorka NMPC 加 EKF，已整合到 LT1。[作者机构原文](https://backend.orbit.dtu.dk/ws/portalfiles/portal/409060115/1-s2.0-S2405896325003295-main.pdf) [冻结控制器](https://github.com/hpeuscher/loopinsight1/blob/9a6092065bd821f88691cd8d03b0550acd665573/src/core/controllers/MPC_Hovorka2004.ts) [论文实验脚本](https://github.com/hpeuscher/loopinsight1/tree/9a6092065bd821f88691cd8d03b0550acd665573/examples/MPC/IFAC_EDT_2025_Hauser)

**另一可用作者实现。** McGill Diabetes Lab `MPController.m` 使用基于 Bergman 的模型与 Kalman 状态估计；源码第 392–395 行设定 4 h 预测、2.5 h 控制且 MPC 部分仅操作 basal；第 426–427 行让输入惩罚随 IOB 增加。它另有 meal bolus 逻辑，必须按当前任务拆开适配。[冻结源码](https://github.com/McGillDiabetesLab/artificial-pancreas-simulator/blob/3521a51e2709f954fb321e7a1fb1aeecb4b65c80/library/controllers/%40MPController/MPController.m)

**建议用途。** 两个仓库均核验为 MIT。优先选择一个完成单位/观测/动作/失效适配、给与合理开发调参预算，作为 learned-world MPC 的强对照；不必同时移植两整套模拟器。生理模型状态可以由 CGM 和实际输注经 observer 估计，但不得直接读取受测模拟器隐藏状态。把已知模型参数直接复制进 MPC 会形成 oracle 对照，应另列，不能当正常基线。

**world+RL 的后续研究假设。** 同一 constrained MPC 下比较生理模型与生理+学习残差模型；再比较无 RL 与 RL 给候选序列/成本参数。这样可查清收益来自模型、规划还是策略。该组合是本项目提出的研究方案，不是上述论文已经证明的结论。

**最小消融及局限。** physics-MPC vs learned-world-MPC；同一 learned world 的 MPC-only vs RL+MPC。LT1 默认 sampling 为15 min，适配本项目必须统一到5 min并单独核验；不能通过换成 LT1 植物或简化噪声获得优势。McGill 原代码在无解时给0，这种研究实现的失败分支不能被包装成临床安全策略，须按本项目失败规范记录。

## 3. 作者项目、许可和可复用范围

下列 commit 由本次 GitHub `commits?per_page=1` 查询取得；未克隆、未连接 GPU 或其它 SSH 主机。链接固定到所读版本，避免后续默认分支变化。

| 项目 | 核验版本 | 许可 | 本次核验与适配边界 |
|---|---|---|---|
| [G2P2C](https://github.com/RL4H/G2P2C/tree/37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70) | `37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70` | MIT，读取 LICENSE.txt | 阅读 model/parameters/worker/pump；直接对应 world+RL，须修正统一动作映射与观测、评分适配 |
| [RL4T1D](https://github.com/RL4H/RL4T1D/tree/ea9e1723b1d0961474887af3de2f683c941881bd) | `ea9e1723b1d0961474887af3de2f683c941881bd` | MIT，读取 LICENSE.txt | G2P2C 后继工程；阅读 ControlSpace；README 的 BB 与 RL 信息权限不同，不能复制原表当公平比较 |
| [GluCoEnv](https://github.com/RL4H/GluCoEnv/tree/f78e11aa66acef162938597c58226d0a56270cbb) | `f78e11aa66acef162938597c58226d0a56270cbb` | MIT，读取 LICENSE.txt | PyTorch GPU 向量化；阅读 env/env.py/core.py 与 action文档；性能与数值等价未在此核验 |
| [OpenAPS oref0](https://github.com/openaps/oref0/tree/88cf032aa74ff25f69464a7d9cd601ee3940c0b3) | `88cf032aa74ff25f69464a7d9cd601ee3940c0b3` | MIT，读取 LICENSE.txt | IOB计算、低糖预测/恢复源代码；其 SMB 开关必须与任务权限一致 |
| [McGill MPC](https://github.com/McGillDiabetesLab/artificial-pancreas-simulator/tree/3521a51e2709f954fb321e7a1fb1aeecb4b65c80) | `3521a51e2709f954fb321e7a1fb1aeecb4b65c80` | GitHub license元数据及页面为MIT | MATLAB作者实现；读取MPController.m；适配同一被控对象，不换植物后对分数 |
| [LT1 NMPC](https://github.com/hpeuscher/loopinsight1/tree/9a6092065bd821f88691cd8d03b0550acd665573) | `9a6092065bd821f88691cd8d03b0550acd665573` | 根页面/MPC源码为MIT | TypeScript、EKF、多步MPC；阅读控制器与实验入口，执行效率未知 |
| [AIML4Diabetes](https://github.com/girtel/AIML4Diabetes/tree/560286bac9a098b60c2b8d3b21b1e7e2b4e73fd1) | `560286bac9a098b60c2b8d3b21b1e7e2b4e73fd1` | GitHub license元数据为MIT | 2022论文的SAC/PPO备选对照；本次只核验关联与仓库，未深读策略实现 |
| [PPO-AP-Controller](https://github.com/YanfengZhao-UKM/PPO-AP-Controller/tree/15de1f0d1adff3de73c0a64c643956eba84317fb) | `15de1f0d1adff3de73c0a64c643956eba84317fb` | 本次API为null，递归树未见LICENSE | 2025论文有代码，不能等同为已许可开源；不纳入首批代码复用 |

GluCoEnv 适合作为4090加速候选而非立即替换评分环境。其`env.py`提供CGM observation，同时在`info`中含真实BG、meal announcement、time2meal；wrapper必须白名单化，不能把整个`info`喂给模型。其极端BG终止/自动reset也须转化为本项目完整轨迹与失败证据，不能让重置后的正常值稀释失败前后低糖。[环境源码](https://github.com/RL4H/GluCoEnv/blob/f78e11aa66acef162938597c58226d0a56270cbb/glucoenv/env/env.py)

## 4. 不可直接照搬的作者默认设置

1. **隐藏参数初始化。** G2P2C `Pump.__init__` 从虚拟患者表读取`u2ss`及`BW`计算basal，而非从获准处方输入。适配时需用统一已知资料替代，记录由此引入的偏离，不能把这一默认设定作为本项目合法输入。[源码第22–27行](https://github.com/RL4H/G2P2C/blob/37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70/utils/pumpAction.py#L22)
2. **未来信息字段。** 作者worker传递`remaining_time`、`future_carb`给state-space构造器；默认announcement flags关闭并不免除本项目逐字段审查。原变量名存在不代表已证明泄露，应核对实际输出张量和调用路径。[源码第95–97行](https://github.com/RL4H/G2P2C/blob/37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70/agents/g2p2c/worker.py#L95)
3. **奖励、评分与终止不同。** 作者worker使用CGM构建日志和TIR，并在CGM≤40或≥600等条件结束episode；本项目要求真实BG独立评分和失败/未知尾部。外部代码原表成绩不得与当前表直接拼接。[源码第105–117行](https://github.com/RL4H/G2P2C/blob/37f7b7e161e9c1e7b702c8eebdc5cc7eb4f81d70/agents/g2p2c/worker.py#L105)
4. **动作单位及执行时间。** U/min、U/h、每区间U必须有明确转换；加入bolus时与basal历史分别记账，再合成真实输注。获准等价单位转换不等于获准把速率变成瞬时注射；预测历史必须写实际执行量而非actor未裁剪输出。
5. **模型选择与基线配置。** 原作者的单患者训练、精确餐食申报、处方资料、试验长度、噪声和终止条件，与当前共享模型可能不同。报告必须标明“作者实现适配”，用共同合同下的重跑值比较，且不得降低基线调参预算制造优势。

## 5. 单seed训练、开发与确认建议

以下是**待主任务冻结的实验设计**，不是已执行计划或已通过门槛。

### 5.1 先做机制检查，再选择任务

- 从训练/开发集选取已记录的低糖前窗口，按原因分为：meal bolus后、无餐/夜间、传感误差、暂停后高糖。标签只用于离线诊断，不能作为在线oracle输入。
- 从同一仿真状态分叉，保持未来扰动/噪声一致，比较当前动作、在允许范围内的最低输注、提前减量、恢复基础率。分叉状态只能给模拟器使用，策略仍只见可观测历史。
- 记录“最小允许动作仍低糖”的比例与剩余低糖负担；新basal+bolus任务仅在合同冻结后重新训练、重新评价全方法。事后取消已执行bolus不是合法动作，不得当可用方案。
- 审核CGM/insulin/meal的对齐；同一实际action从泵适配器到world输入再到评分日志需可追踪。记录饱和比例、actor与anchor差值、裁剪前后差值及KL/entropy，区分学不动、权限不足与被下游覆盖。

### 5.2 世界模型数据与闭环学习

- 训练seed固定260915；场景随机数与策略初始化seed分开登记。训练、开发、最终确认场景清单和哈希冻结；新增场景种子不包装成多训练seed。
- world数据包含合理控制器、不同合法输注、已知餐食误差与暂停/恢复段，保证有动作支持；先查低糖稀有窗、长胰岛素尾部与高糖平台覆盖。优先重采样不足事件，不先盲目加网络深度。
- 用同一模型容量和真实交互预算比较model-free与model-based；想象步数、model update次数、controller优化调用数另计。推荐先一次机制短跑测吞吐，再冻结总真实transition数和GPU小时上限，不能因某候选欠佳临时多训。
- 开发集选模型时同时看多步误差、低糖召回/误报、最低值偏差、候选排序、动作效应的时序与幅度以及闭环TBR/TAR；不能仅凭RMSE下降放行。
- 最少保留：旧线性world；新world无规划；新world+规划；同一world/MPC无RL；完整world+RL。若计算预算只容许少数训练臂，先用离线分叉检查淘汰动作无效模型，再决定训练，不删掉关键归因消融。

### 5.3 确认比较与共同任务

| 维度 | 必须冻结的共同合同 |
|---|---|
| 被控对象 | 同一模拟器实现、数值积分设置、传感器/泵模型、患者集合及初态 |
| 可观测输入 | 同一CGM历史、已执行用药记录、已发生/实时申报餐食、同一获准处方资料；不含未来餐食、真实BG、患者ID或隐藏状态 |
| 动作 | 同一5min更新，U/h基础率与允许时刻的U bolus分开；同一上下限、量化、延迟与超时规则 |
| 外部干预 | meal bolus、低糖救援、断泵恢复等对所有方法相同；任何救援均计次数、用量与时点 |
| 开发资源 | 训练/开发数据范围及调参次数合理匹配；报告每方法实际计算开销 |
| 评分 | 原control_metrics保持；真实BG计算全部指标；提前结束/无解/丢步保持失败及未知尾部，不删除患者 |

确认场景至少包含标称餐食、餐食申报误差/延迟、固定处方误配、夜间及重复餐后恢复段；只使用当前模拟器确实支持且核验过的扰动。已曝光虚拟患者的新场景只能称场景稳健性，不称新患者泛化。

候选冻结后，统一跑P03/D05/D06、现有外部基线、适配并调优的MPC、适配G2P2C及选中方法。若改动作任务，所有方法重新适配和评价，旧确认结果单独保留。不要在最终确认结果上选择权重、暂停阈值或训练时长。

### 5.4 指标和最低归因要求

- 优先报告TBR70、TBR54、严重低糖持续时间、失败/覆盖率；同时报告TIR、TAR180/TAR250、SD/CV、LBGI/HBGI、日总量、basal/bolus分量、动作变化和暂停/恢复统计。所有相互相关指标都展示，但不当作独立胜利次数。
- 原联合不退步标准原样重报；可额外给预先规定的临床/研究意义界值，但不事后修改原门槛。若低糖改善而TAR显著恶化，应明确为trade-off，不能称目标达成。
- 配对单元以患者汇总差异为主，场景为患者内重复，避免把5min采样点当独立样本；报告患者分布和最差患者。单seed置信区间只反映场景/患者抽样，不估计训练seed稳定性。
- 新world必须胜过旧world在控制相关检验上的表现；完整方法还须胜过相同守护机制但无RL/无学习world的版本，才能把改进归因于world+RL。仅安全壳生效、actor仍贴anchor，不是RL突破证据。

## 6. 已核验边界与未完成项

已核验：五个机制的可追溯一手依据；主要作者仓库及版本；G2P2C动作条件模型和硬编码映射；作者默认输入/评分与本项目的差异；可复用MPC/IOB代码位置及许可证状态。

未完成：作者环境安装和运行、MPC适配质量、GPU吞吐、GluCoEnv与当前环境数值一致性、任何新训练与确认、任何低糖或高糖收益。Zone-MPC 2023未定位到可核验作者完整代码。PPO-AP-Controller虽公开但本次未见明确代码许可。

检索访问限制：部分PMC正文直连触发访问检查，欧洲PMC XML端点返回500；相关方法描述通过搜索工具返回的一手论文正文段落核验，未声称完整读过全部补充材料。DTU原文链接的一个域名路径返回403，另一作者机构入口及论文摘要可访问；代码功能以冻结源码为准。未使用二手转载证明方法或性能。

后续执行前必须将最终选中的任务、基线适配、预算、阈值、消融和确认集另行冻结；本文件只提供可评审的研究提案，不授予产品自动给药或临床使用权限。
