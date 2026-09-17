# RL-DITR 方法与固定源码审计

审计日期：2026-09-15。只读检查固定作者源码和已保存论文全文，没有调用模型、AutoDL 或额外网络。作者源码 commit：`5080fdbe43979f9b8a9616fbc120aae4ac25842a`。下文源码路径均相对本目录 `vendor/RL-DITR/`；论文行号对应 `../Loop数据集/训练管线_v1/文献/RL-DITR_全文.txt`。

## 1. 结论与本轮建议

**应按论文的方法结构实现 Loop 适配基线，不能直接运行作者默认训练命令并把结果称为完整复现。** 固定源码含可直接确认的接口错误、损失 mask 错误、回报计算错误以及论文与代码不一致。保留原源码作证据，适配代码单独记录选择。

- 保留两阶段：动作条件患者模型 `fR/fT/fP` → 历史经验、模型生成经验、动作监督共同训练策略 → 多步候选规划。
- Loop 使用全量已冻结训练划分，food/exercise 只加特征、mask、记录年龄，不引入存在性筛选；没有记录不等于真实没有进食/运动。
- 泵基础率保持连续 `U/h`，不能复用作者整数注射动作 `1–40 U`。连续概率分布的 log density 替代离散动作 log probability，是必要的动作合同适配，应标明。
- 模型评价优先回答“预测比简单基线好吗、动作响应可靠吗、未来事件未知时还能成立吗”。仅在训练患者模型中提高奖励，不能证明策略有效，更不能直接作为 Harness 的合格个体化给药来源。
- 交接文件中的 Gate A/B 是策略阶段的前置条件。若患者模型不通过，不应用大量策略训练或超参搜索掩盖这一失败；最终 HTML 应展示患者模型所有尝试和阻断原因。

## 2. 网络映射

| 模块 | 论文 | 固定源码 | 适配选择 |
|---|---|---|---|
| `fR` | 将截至决策时点的历史映射为状态；3层 Transformer、隐藏256、8头；最后一个隐藏向量为初始状态。论文1343–1351、1480–1487 | `patient_model.py:48–50,59–69` 把观察、前一动作embedding、当前option embedding拼接；`baseline.py:87–144` 为有因果attention mask的Transformer Encoder，位置编码先拼接再线性映射 | 72×22历史保持因果性。连续基础率通过浮点投影；保留历史动作与当前候选动作的时间分界，不把候选动作放进初始state |
| `fT` | `(s_t,a_t)→s_{t+1},r_t`，3层Transformer。论文1352–1358、1485–1487 | `patient_model.py:53,57,77–87`：Transformer Decoder，query为state/action/option，memory为初始编码序列；reward是当前state/action/option的独立MLP输出。`baseline.py:177–205`同时给target和memory施加因果mask | 必须条件于连续基础率和动作时长。多步展开只由初始历史与受控未来情景递推；不能每步喂真实未来state |
| `fP` | 下一状态预测GLU与WTR。论文1358–1361、1396–1400 | `patient_model.py:52–55,140–150`：value、reward、GLU、aux四个头；GLU和aux取next_state | GLU回归+WTR分类是主体。独立产品预测模型F不等于这里的内部fP |
| policy/value | 论文称三层MLP、隐藏256。论文1487–1489 | `rlsl_model.py:70–74`、`patient_model.py:52–55`调用 `MLP(D,[D],out)`；`baseline.py:248–258`可见实际只有两层Linear（一个隐藏层） | 如选论文三层Linear须明确“按论文，区别于公开代码”；如选公开代码两层也须说明。不能同时宣称完全一致 |

作者源码患者模型dropout默认0.25（`patient_model.py:27`），策略默认0.5（`rlsl_model.py:29`）；论文为0.4。作者公开源码包含两个不同的规划模型类，不能把 `ts_v4.py` 与 `rlsl_model.py` 当成相同实现。

## 3. 患者模型目标：shift、mask、梯度

### 3.1 论文目标

论文1388–1404：`L_patient = μ L_T + L_P,GLU + L_P,WTR`。

- `L_T` 对到达同一真实时点的不同初始时间/不同展开深度的潜在状态做MSE；要求 `T_i+k_i=T_j+k_j`。
- GLU为MSE，WTR为分类交叉熵。
- `fR/fT/fP`联合优化，原文没有规定stop-gradient teacher；不能把停止梯度版本写成原文要求。

### 3.2 公开代码的真实目标

`ts/pl_module/patient_module.py:49–92` 为 `L_glu+L_aux+L_value+L_reward+μL_state`，默认 `μ=0.05`（14行）。多出的value/reward损失与论文展示的患者目标不一致。state差异是 `sqrt(diff²+1e-8).mean`，近似L1，而非论文MSE。

`patient_model.py:130–154` 在每个初始时点同时沿历史动作展开，`state_list[k,t]` 表示“从t起始展开k步的state”；GLU/value/aux头在next_state上，reward在当前state+action上。

**正向shift本身并不是错位。** `patient_module.py:69` 的 `roll(pred_state[:,k],+k)[t]` 是 `pred_state[k,t-k]`，和 `pred_state[0,t]` 都到达t，只须排除前k个循环回卷位置。举例：初始编码`[0,1,2,3,4,5]`、正确一步预测`[1,2,3,4,5,6]`，正向roll后`[6,1,2,3,4,5]`，t≥1处严格对齐。

**真正严重问题是mask反了。** `ts_dataset.py:314–324`明确padding=False为真实数据、True为填充；`patient_module.py:67–74`却直接 `mask0=padding`，最终只选择填充位置。完整非padding序列的state损失有效计数为0，模型没有接受真实状态一致性监督。相同问题出现在 `run.py:257–267`。

**两端均未detach。** `pred_state[:,0]`和移位后的预测state都在计算图内；梯度可进入表示与动力学。不是“固定teacher自蒸馏”。对齐后的两条路径不是同一张量，但若只优化一致性可塌缩；GLU/WTR锚定才提供可识别的监督。建议首版按论文使用双向MSE，记录各项量级与state方差；stop-gradient只作为明确命名的后续消融，不能悄悄引入。

### 3.3 标签和边界缺陷

1. `ts_dataset.py:386–408`已将GLU/aux/reward/return左移一格，代表动作后的目标；action保持当前时点（410–413）。`base.py:99–106`再按展开步左移标签，这个方向合理。
2. `patient_module.py:39–46`以及`run.py:216–223`的`step_mask`屏蔽的是头部，但左移标签的循环回卷发生在尾部。`base.py:87–95`产生另一个`stack_mask`，训练调用拿到后并未传入loss，而且其mask循环也累积了不规则移位。多步时尾部可能监督到序列开头的标签。
3. `ts_dataset.py:416`用当前`glu`构造`mask_reward`，而reward标签来自下一格；有缺失时目标有效性可能错位。应直接用每个目标时点的观测mask。
4. `patient_module.py:51–53`的value loss不含有效padding/真实return完整性检查；文件末端把剩余return归零并不能代表患者终止。
5. `run.py:174`患者模型默认`n_step=1`，所以即使修复mask仍不产生任何多步一致性项。适配实现须显式设定展开长度，并报告不同长度的真实可用样本数。

建议使用显式切片索引与 `valid_origin & valid_target & same_episode & no_gap`，不依赖roll处理边界。保持所有5分钟合格起点参与一步损失；仅多步损失使用确实存在的后续标签mask，不因一条样本没有60分钟尾部就删除其一步训练资格。

## 4. 策略目标、value与回报

### 4.1 论文方法

论文1411–1453明确：

```text
L_RL1 = -Σ R_logged(t) log π(a_logged(t)|s_logged(t))
L_RL2 = -Σ R_model(t)  log π(a_sampled(t)|s_model(t))
L_SL  = 历史动作监督
L     = L_RL1 + ε1 L_RL2 + ε2 L_SL
R(t)  = Σ γ^i r(t+i)
```

论文1451–1453又称训练时使用可学习`V(s)`代替R以稳定训练，但没有完整说明价值拟合/两项分别替代方式。不要把这种描述自动扩写为PPO、优势函数Actor–Critic或DPO。

历史动作来自行为策略，不是当前π采样。没有行为概率/重要性修正时，`L_RL1`应诚实描述为**回报加权的历史动作学习**；它不是标准无偏on-policy策略梯度。若权重是负回报，它会降低历史动作概率，所以监督权重与动作支持范围很重要。模型内`L_RL2`只有真实由当前策略采样时才是该学习模型中的on-policy训练，不能称真实患者on-policy证据。若用仅依赖s的V替代所有采样动作回报，理想期望下 `E[V(s)∇logπ(a|s)]=0`，因此必须说明具体实现，不能凭一句“稳定训练”认为梯度具有正确动作信用分配。

### 4.2 固定源码实际做了什么

| 项 | 源码事实 | 问题/选择 |
|---|---|---|
| 历史 `L_RL1` | `run.py:234–235,283,300–306`只有普通动作交叉熵与模拟logprob×return；没有历史动作logprob×历史return | 公开训练路径缺论文独立LRL1。若适配补齐，应明确按论文补齐 |
| `L_SL` | `rlsl_model.py:170–184`历史展开中未收集policy_train；`192–209`在策略生成state上收集；`run.py:229–235`却对齐历史动作标签 | 除第一步外，生成state不再是历史state，相同历史动作不是该新state上的专家标签。建议L_SL在历史state上计算，不在偏离历史的模拟state上强制复制未来日志动作 |
| 采样 | `run.py:312`传sample=True；`rlsl_model.py:194–200`尊重参数 | 可产生真实分类采样。另`ts_v4.py:220`强制sample=False，变成argmax，不能视为等价on-policy采样 |
| return | `rlsl_model.py:206,213–217`先累加reward logits；`run.py:302`最后support_to_scalar | 数学错误：softmax不是线性映射。必须每一步先解码标量再折扣累加 |
| detach | reward在`rlsl_model.py:206`detach，return在`run.py:303`再次detach | 阻止通过reward权重直接训练患者模型；不等于整个患者模型固定，须另外冻结参数并保持eval模式 |
| 截断/裁剪 | `run.py:304`将整个最终RL loss裁剪到[-1,1] | 超界时整项梯度变0；这不是梯度norm clipping，也不是PPO ratio clip |
| 默认权重 | `run.py:187`默认loss_rl_weight=0 | 默认即便跑通也可能只有动作监督，不能以rlsl类名证明RL训练发生 |
| value | 患者value用历史累计回报监督；`rlsl_model.py:71`复用其value头；模拟return未加value bootstrap | 是行为数据价值近似，不能自动视为当前策略的Vπ或用其为新动作作无偏背书 |

本地标准库数值检查：每步reward logits `[-2,0,2]`、γ=0.9，两步正确标量回报为`1.6167804752`，作者先相加logits再解码得`0.9771509119`。此检查仅证明代数差异，不是训练结果。

Loop首版建议选择并记录：历史完整有限时域return作为LRL1权重、冻结患者模型产生的有限时域标量return作为LRL2权重、历史state上的连续动作负对数似然作为LSL。保留论文三项，不添加未授权算法创新。若拟合有限时域value供规划，标注其行为策略来源与截断时域；不能把资格边界当作生理终止，也不能把未知尾部当作零真实回报。

## 5. Beam search与超参数

论文1454–1468：从策略中生成K步候选；每步保留B条最高价值轨迹；目标是 `Σ_{i=0}^{K-1} γ^i r_i + γ^K V(s_K)`。原任务K=7约为一天的七个时段，不等于本项目七个5分钟格。

源码 `ts/models/agent.py:158–209`：每步每条beam取策略top-B动作（不是完整枚举），option=0时强制动作0；用reward/value排序；最后返回第1条beam。`top_p`参数实际未启用（161–162行注释），不可报告为已使用nucleus pruning。

`agent.py:196,209`递推消除上一步value之后，末端保留约`γ^(K-1)V(s_K)`，而论文为`γ^K V(s_K)`；terminal value权重差一个γ。此外代码默认beam=5（99行），`arm.py:49`默认2，论文B=10。适配时直接实现显式累积奖励与尾部value，避免复杂减法错位，并用一个两步玩具例验算排名。

连续U/h不能沿用categorical topk。可以从连续策略分布取固定分位点/有限次样本作为每步候选，保留原始浮点动作并在beam中比较；这是**候选规划搜索**，不是把已观测动作静默量化成注射类别。必须记录候选生成规则与支持范围。

| 超参数 | 论文1480–1501 | 源码默认/示例 | 本轮应如何记录 |
|---|---|---|---|
| fR/fT | 各3层，hidden256，8头 | 相同主维度，FFN默认2048 | 初始基线可按论文，缩小/增大均记成新实验 |
| MLP | 三层，hidden256 | 一隐藏层、两Linear | 显式选择 |
| batch/历史长度 | 32、padding128 | DataModule 32，CLI默认8；max128 | Loop历史72×22是已冻结数据适配 |
| epoch/LR | 100、1e-3 | CLI默认10、1e-3；README示例100、5e-4 | 验证集选模型，保留实际epochs与early-stop |
| optimizer/decay | Adam、1e-4 | `base.py:60–62` AdamW，未传weight_decay | 不应把默认AdamW行为写成论文Adam配置 |
| dropout | 0.4 | patient0.25，planning0.5 | 初始基线可0.4；各次改动独立记录 |
| γ | 0.9每原文时段 | 0.9每源码步 | 沿用Loop合同每小时0.9：Δ分钟折扣`0.9^(Δ/60)` |
| μ/ε | μ0.1，ε1.0；对ε1/ε2具体分别值表述有限 | patient μ0.05，RL权重0 | 记录采用值与论文说明的不确定性 |
| beam | 10 | 2/5按入口不同 | 首先验证候选搜索确实改变动作；耗时和收益一起报告 |

## 6. 可直接确认的其他运行问题

- `rlsl_model.py:76`访问 `self.pm.dynamic_func`，而患者类字段是 `dynamics_func`（`patient_model.py:57`）。
- `rlsl_model.py:54–55`在pm=None时传参顺序/数量不匹配当前PatientModel构造函数（`patient_model.py:26–29`）；默认pm=None同时pm_freeze=True先触发52行断言。
- `rlsl_model.py:115–116` initial_inference遗漏action_prev/option，位置参数错误；131行recurrent调用也未正确传state_0。
- 冻结PM后，`rlsl_model.py:67–68`重新新建action/option embedding，而不是复用PM训练过的embedding；冻结encoder/dynamics所见输入分布随新embedding变化。即便修好拼写也不能假定这是完整固定环境。
- `run.py:236–241`默认loss_joint=False分支未定义loss_value/glu_mask，却在287/293行引用，可能直接UnboundLocalError。
- `baseline.py:199–200` dynamics=True分支引用未定义x。
- `run.py:70`把valid、test、other合并为ds_test。Loop必须继续使用独立validation与sealed_test；不能继承此测试口径。
- `ts_dataset.py:171,410–411` n_action=40且clip到39，与论文动作1–40存在端点差异；Loop连续基础率不使用该处理。

上述为静态代码确认；没有声称完整安装旧依赖并跑通作者训练。论文与源码的差异应进入最终HTML“方法适配与已修复问题”，不能混入实验成绩表伪装成跑过的实验。

## 7. food/exercise与未来事件泄漏

1. **历史已记录特征**：截至初始决策时点的记录可用于回顾性患者模型，保留mask与记录年龄。严格只用train拟合归一化。未证明录入时效前，不能把事件发生时间上的可得性等同于真实部署录入时间上的可得性。
2. **多步监督目标**：真实未来CGM和其编码可以只用于GLU/WTR/一致性loss；只要没有送入预测路径，就不属于预测输入泄漏。对一致性target使用编码器反向传播本身也不是未来输入泄漏，但要保证预测支路不能attend到target序列。
3. **未来action/外源事件条件预测**：使用日志后续basal、bolus、food、exercise来评估“给定已发生动作/事件时的患者模型预测”可以作回顾性诊断，但须清楚标为oracle/已知未来条件。该指标不能充当策略部署能力。
4. **策略规划**：必须只从初始历史出发。未知未来餐食/bolus/exercise用独立情景机制；不能每步从真实未来state更新这些列，也不能默认它们永远为0。没有可信情景机制时，策略效益结论应保持阻断。
5. **源码注意点**：`rlsl_model.py:209`滚动读取后续option，原论文场景可能是预先指定用药方案；Loop若把未来food/exercise放在option等位置，会引入未来信息。decoder的历史memory应固定在初始历史，不能在多步时滑入真实未来观测。

建议最小因果性检查：保持初始历史和候选计划固定，任意扰动t之后的真实CGM/food/exercise/bolus，策略输出与无oracle开环预测必须逐元素不变；允许监督loss/目标指标改变。另测跨episode、尾部padding、mask=0位置的极大噪声不影响有效位置loss。

## 8. HTML评价结构与可用性结论标准

建议一行一个可复核run，按“数据/合同 → 患者模型 → 动作响应 → 策略/规划 → 独立评价”分章节。每个run保留config hash、患者/样本数、seed、起止时间、实际epoch、模型大小、checkpoint、选择准则、失败原因；失败/取消实验同样收录。

- **预测**：同一验证起点比较last-value、线性外推和患者模型；5/15/30/60分钟MAE/RMSE、WTR以及按患者汇总。不同期限标签覆盖不同，要给各自样本数，不能只比较不同人群上的总均值。
- **动作响应**：固定历史、可比未来情景扫描候选U/h，画GLU变化/预测收益/不确定性；单步预测好不证明反事实成立。不能仅凭模型“方向通常正确”宣称医学有效。
- **稀疏模态**：按food/exercise记录可用性分组报告误差与样本占比；只作分层评价，不删无记录患者。必要消融使用相同患者/起点，避免把数据筛选改善误记为特征改善。
- **策略**：若Gate A/B未通过，展示“未启动/被阻断”与具体证据。若通过并开展策略实验，分别列LSL-only、三项目标、不同权重/搜索参数；验证选模型，封存组一次最终评价。绝不因策略在本身学习环境中得分高就授予合格来源身份。
- **Harness最终判断**：最多先得到“真实训练过的研究候选模型”；可作为产品RL部分仍需有效动作合同、独立评价、模型版本/证据登记、适用性/OOD/弃权、Core受控worker/MCP及双模型发布闸门。训练完成、fixture通过、HTTP可调用均不能替代这些验收。

本审计自身不提供新模型效果结论，也没有训练结果。是否可用应由本轮真实实验与闸门结果决定。
