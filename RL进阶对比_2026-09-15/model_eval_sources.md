# RL 改进与独立仿真评价：一手来源核查

核查日期：2026-09-15。本文是调研和可实施建议，不是已完成实验。保留 Loop 全部 225 名训练患者和 1,653,421 个训练起点；food/exercise 的稀疏程度不作为删样本条件。当前模型源码已核查：`RL训练_2026-09-15/model.py` 为 22 维历史编码及黑盒动作条件解码。

## 1. 建议主线：H²NCM 思路的 Loop 动作动力学 + 保守离线策略

**优先改变“动作怎样进入血糖动力学”，再训练策略。** 不以强迫某个剂量方向比例通过作为研究成功；方向先验可以参与训练，但最终依据应是独立环境中的真实闭环轨迹。

### 一手论文依据

- **ICML 2024，Hybrid² Neural ODE Causal Modeling and an Application to Glycemic Response**，Zou、Levine、Zaharieva、Johari、Fox。论文明确区分预测误差与干预效应，使用历史编码器、机理依赖图或 ODE 解码器，以及预测损失和干预排序损失。其原任务是 T1DEXI 运动后的血糖响应，并非 Loop 基础输注策略。应称为“基于 H²NCM 的 Loop 适配”，不能把其临床/数据结论搬到本项目。[会议页](https://proceedings.mlr.press/v235/zou24b.html)、[全文及公式 5–8、附录 E.4](https://arxiv.org/html/2402.17233v2)、[作者仓库](https://github.com/bobjz/H2NCM)。
- **NeurIPS 2020，MOPO**：模型生成转移应附加不确定性惩罚，避免策略利用分布外模型错误；模型奖励不等于真实环境收益。[会议原文](https://proceedings.neurips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html)。
- **NeurIPS 2021，COMBO**：深度模型不确定性也可能失准，可在模型产生的分布外状态动作上做保守价值正则。说明“集成模型彼此同意”本身并非真实性证明。[会议原文](https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html)。

### 推荐首先实现的患者模型：有向图 MNODE

以下是结合作者结构和本项目数据约束提出的**具体适配设计**，不是宣称已复现作者全部实验：

1. 继续输入 `72×22` 历史值、mask、age 与时间信息。所有原 eligible 起点逐轮遍历；未来目标使用已有有效 mask。不给没有 food/exercise 的样本赋零训练权重。
2. 历史编码器输出 7 维潜状态；第 0 维显式设为当前标准化 CGM，其余为学习的初态。避免将未来 CGM 或 simulator 私有 state 用于初始化。
3. 用 7 个独立小 MLP 代替黑盒 Transformer 动作解码器。每个 MLP 只接收指定父节点；每个 5 分钟格同步更新 `s_next[i] = s[i] + f_i(parents, allowed_inputs)`。这是离散化学习动力学，潜状态和增量不能解释为已标定生理物理量。
4. insulin 首节点只接收候选胰岛素输入，再通过中间节点影响 glucose，避免候选动作直接进入任意血糖残差。carb、activity 各有独立输入通道。MLP 层数/宽度可调，依赖图须作为方法版本固定并做消融。

作者代码的具体邻接表是：

| 节点 | 状态父节点 | 外部输入 |
|---|---|---|
| 0，glucose | 0,3,4,5,6 | 无 |
| 1，insulin 通路起点 | 1 | insulin |
| 2，insulin 中间节点 | 1,2 | 无 |
| 3，insulin 中间节点 | 2,3 | 无 |
| 4，activity 通路起点 | 4 | 原作者 heart rate、steps；Loop 应显式改成可用运动特征 |
| 5，activity 中间节点 | 4,5 | 无 |
| 6，carb 通路 | 6 | carb |

核查位置：[MNODE.py 第 33–39 行附近](https://github.com/bobjz/H2NCM/blob/7d2e119febf742621bea63f21a654d49d1d04e21/MNODE.py#L33)、[MNODE_model.py 的 DAG_RNN.forward 和 MNODE_LSTM.forward](https://github.com/bobjz/H2NCM/blob/7d2e119febf742621bea63f21a654d49d1d04e21/MNODE_model.py)。正文附录称简化状态数与脚本存在差异，报告应以实际实现 7 节点表为准，避免说逐字复现。

作者仓库固定提交 `7d2e119febf742621bea63f21a654d49d1d04e21`，本次 GitHub 元数据 `license=null` 且根目录未见 LICENSE。因此建议根据论文独立实现，不把作者代码复制进交付软件。下载到 `research_sources/` 的内容仅供本次核查。

### 可直接实现的排序损失

对同一个历史 `h`，固定相同未来共同干预情景 `c`，构造三个不同基础率序列 `a_low, a_mid, a_high`。定义每条轨迹的分数 `q_j = mean_t(g_hat(h,a_j,c))`；对于递增胰岛素且其他条件相同的先验，最高平均血糖的目标类别为 low。

```python
# pred: [batch, 3_interventions, horizon], valid: [batch, horizon]
q = (pred * valid[:, None, :]).sum(-1) / valid.sum(-1)[:, None].clamp_min(1)
rank_loss = cross_entropy(phi * q, target_low_index)
pred_loss = masked_mse(factual_prediction / glucose_scale,
                       factual_target / glucose_scale, factual_valid)
loss = (1 - alpha) * pred_loss + alpha * rank_loss
```

来源是原文公式 7–8；[utils.py 第 130–159 行](https://github.com/bobjz/H2NCM/blob/7d2e119febf742621bea63f21a654d49d1d04e21/utils.py#L130) 实际先 softmax、取 log 后再做 CE。直接用 logits 做 CE 可避免其数值截断；不要照搬很大的温度，使用验证稳定性确定 `phi`。MSE 单位归一化后再解释 alpha，不把同样 alpha 在不同单位下当成相同权衡。

**关键边界：**排序标签来自外部先验，并非 Loop 的真实反事实标签。小扰动、不同持续时长、不同当前血糖/IOB 分层都要观察；训练后排序检查只能说明先验一致性。不要强制 5 分钟立即出现降糖；应纳入延迟，评价 30/60/120 分钟及独立模拟的效应大小。若把 horizon 延长，保留全部起点、用变长有效目标 mask，不能要求所有样本都具有 120 分钟连续记录而削减原样本量。

### 为什么不首选直接照抄作者的全机理 LP

作者 `LP_reduced_model.py` 在归一化变量上学习 ODE 参数，且将一些皮下动力学约简；不应把这些学习参数视为公开患者的真实生理参数。MNODE 路线先验证依赖图价值，较容易保持完整样本与现有规范。若后续采用物理单位 ODE，必须显式整理状态单位、五分钟积分方式、泵输入转换和数值积分稳定性；**不能改变独立 simglucose 患者参数以使本方法得分提高。**

## 2. 未知未来 food / bolus / exercise 怎么处理

这是方法是否可用于决策的必要定义，不能仅靠增加历史特征解决。

- **历史条件预测表**：可以给所有方法相同的、事后记录的 future basal / bolus / food，并标注 retrospective conditional prediction。未记录的食物仍是缺失；不得称为真实零摄入。这张表不能宣称在线预测能力。
- **实际决策/仿真表**：策略只获得决策时已知资料。未来餐食、bolus 和运动不能读取日志/Scenario 真值；若已宣布计划，需所有方法同时得到相同信息。
- **患者模型 rollout**：未来外生事件需来自只用训练资料学习的情景分布，或明确“未宣布事件未知”的情景边界。不能固定所有未来 event=0 却用真实未来含事件的结果估计策略收益。任何模拟采样必须保留来源标签，不伪装成真实 Loop 记录。
- **首个公平闭环任务**：成人基础输注控制，设定统一的餐时 bolus 处理协议。可有“所有控制器共用同一个餐时 bolus 模块”的 hybrid closed-loop 组；也可有“所有控制器无 bolus”的 fully closed-loop 压力组。两组分别报告，不能让一个方法享有 bolus、另一个没有。
- 不允许 RL 把增大 basal 偷换为单次 bolus。policy 输出是 U/h 的基础率，持续 5 分钟，原始/已转换/实际交付剂量分别记账。

## 3. simglucose 官方源核查

当前固定提交：`a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc`（本次 API 已核查）。[官方仓库](https://github.com/jxx123/simglucose/tree/a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc)。这是公开研究仿真器；不能把原 UVA/Padova 的监管描述转嫁给这个 Python 包或本策略。

| 项 | 源码事实及实施要求 |
|---|---|
| 患者 | `adult#001` 至 `adult#010`；另外各 10 名 child/adolescent。本任务只使用成人。 |
| CGM | 默认 Gym 是 Dexcom，采样 **3 分钟**；GuardianRT 是 **5 分钟**；Navigator 是 1 分钟。要与 Loop 对齐，明确选 5 分钟环境配置。 |
| glucose 单位 | 传感器/环境为 mg/dL；Loop 为 mmol/L。转换函数版本固定，不能将两者混在同一 risk 函数。 |
| action 单位 | `Action.basal` 和 `Action.bolus` 都是 **U/min**。policy U/h 除以 60 后传入；每格实际总量为交付 U/min×5。 |
| pump | `pump.basal`/`pump.bolus` 还执行自身量化和上下限。必须记录 requested vs delivered，并在训练/验证使用同一映射。 |
| 默认动作空间 | Gym 直接取泵 CSV 的 `max_basal`，不是本项目可直接采用的 U/h 上限。官方 issue #71 对上限单位提出未解决质疑。应自定义项目 U/h 动作界限，不能把默认 high=30 当 30 U/h 后再直接传。 |
| 观测泄漏 | obs 只有 CGM，但 info 含 `patient_state`、`bg` 等。policy 不得读这些真状态；可以留给评价器算指标。 |
| meal | CustomScenario 的数值时间是**小时**，事件 amount 是克；核心 env 按分钟消费，返回 meal 是最近一格平均值。不要将时刻 30 误当 30 分钟。 |
| exercise | 原生 Scenario 只有 meal，标准 patient action 为 insulin/CHO；没有本轮可声称已验证的运动模型。闭环结果不支持运动疗效结论；exercise 输入可保持缺失并另做缺失鲁棒性。 |
| reward | 默认 `risk_diff` 是前后风险差，长轨迹累加容易近似望远镜抵消。建议研究策略使用每步负风险/明确非对称低血糖代价，并独立报告 TIR/TBR，不把变更奖励后的数值与旧奖励混比。 |
| 结束条件 | 核心 env 的 done 是 BG<10 或 >600；不是完整研究验收定义。记录提前结束率，固定研究时长，不能删去失败后的风险时段以美化 TIR。 |

具体源文件：[env.py](https://github.com/jxx123/simglucose/blob/a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc/simglucose/simulation/env.py)、[pump.py](https://github.com/jxx123/simglucose/blob/a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc/simglucose/actuator/pump.py)、[sensor_params.csv](https://github.com/jxx123/simglucose/blob/a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc/simglucose/params/sensor_params.csv)、[scenario.py](https://github.com/jxx123/simglucose/blob/a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc/simglucose/simulation/scenario.py)、[BBController](https://github.com/jxx123/simglucose/blob/a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc/simglucose/controller/basal_bolus_ctrller.py)、[issue #71](https://github.com/jxx123/simglucose/issues/71)。

### Python 版本结论

固定提交的 [setup.py](https://github.com/jxx123/simglucose/blob/a6f6777586c860457a2a1e6cce6d0bd3fd33d1cc/setup.py) 指定 `numpy>=1.25.0`、`scipy>=1.11.0`、`gym==0.9.4`、`gymnasium~=0.29.1`。NumPy 1.25 官方声明支持 Python 3.9–3.11；因此**不能在 Python 3.8 按声明依赖正常安装**。[NumPy 官方说明](https://numpy.org/doc/stable/release/1.25.0-notes.html)。

可行工程方案：现有 Python 3.11 `.venv` 跑 simulator，Python 3.8 native venv 跑 GPU policy；使用有 schema 的本地 IPC 交换 obs/action，避免引入第二套大体积 GPU Torch。初次先验证固定动作轨迹、相同 seed 重复轨迹、5 分钟时间增量与剂量守恒，再跑批量策略。这里仅核查依赖声明，尚未在远端执行安装验收。

## 4. 研究验收：相对基线 + 独立闭环，不设不可能门槛

以下是**本项目建议预先冻结的实验协议**，不是论文或临床法规规定：

1. 不要求每个 horizon、每个患者、每项指标全部第一，也不把一条人为方向比例设成最终否决条件。患者模型方向检查用于排错；最终策略主要看独立轨迹。
2. 仿真训练/调参患者、验证患者、封存评价患者分开。可先做 6/2/2 成人划分和多餐食种子作为工程阶段；论文最终尽量做外层患者交叉验证，让 10 位成人都至少一次只作为测试患者。仅 2 个封存成人不能支撑强泛化结论。
3. Loop-only 训练的策略在 simglucose 的零样本转移单列；若需要 simulator 数据继续训练或校准，另列 adaptation 组，所有 ReBRAC/FQL 等比较策略获得相同 simulator 数据量与调参预算。仍保留全部真实 Loop 样本和单独的 Loop 结果，不能用纯 simulator 成功代替 Loop 方法成功。
4. 相同患者、初始条件、餐食时间/克数、传感器噪声种子、已知信息、动作界限、bolus 协议、运行时长，成对比较。训练 reward 可变化，评价指标冻结。
5. 报告 TIR 70–180 mg/dL、TBR<70、TBR<54、TAR>180、TAR>250、平均风险、日均胰岛素、提前终止率及最差患者。主要结论要求 TIR 或风险有稳定改善，同时低血糖指标没有有意义恶化；相对非劣界值在测试前冻结，不能看到结果后改。
6. 可把成人 TIR>70%、TBR<70<4%、TBR<54<1% 当作参考列，不将其伪装为本次短期模拟的临床准入证据。[国际共识原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6973648/)。
7. 不把同一患者重叠窗口当独立样本计算漂亮 CI；Loop 以患者聚合，sim 以患者/情景配对层级计算 bootstrap CI。至少保留多个训练 seed，完整输出所有 seed 与失败。
8. 两张主表：①患者动力学预测 MAE/RMSE/风险区误差/参数/速度；②真正策略的独立闭环 TIR/TBR/风险/用量/失败。ReBRAC、FQL 是策略，不应与预测器的 MSE 排在一张含义不清的排名里。可在论文总表内分 panel。
9. 测试策略确实使用本研究患者模型：有/无模型、纯 BC、alpha=0、混合排序、无 food/exercise 的消融中只移除特征、保持样本数量。否则无法说明患者模型改进对最终控制的贡献。
10. Harness 研究接入最终另需 input/action/model provenance、worker 独立执行与失败弃权的工程验证；通过仿真不扩大人群或改变 Core 发布规则。

## 5. 本次核查保留资料

`research_sources/` 保存本次官方源码快照和 commit/API 元数据，便于核对上述行文。未连接 AutoDL、未改变模型训练代码、未修改独立模拟器参数。此文全部方案仍需主任务实际实现和验证。
