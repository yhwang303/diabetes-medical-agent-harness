# RL-DITR Loop 实现独立审阅

日期：2026-09-15。范围：`model.py`、`data.py`、`pack_data.py`、`train.py`、`experiment_protocol.md`。只读审阅，并核对现有v2管线、配置、测试源码；未改训练代码、未操作AutoDL、未执行额外训练。行号对应本次读取版本。

## 结论

**没有发现必须立即取消A01的确定性张量、动作/标签错位或padding损失错误。A01可继续作为回顾性、给定日志基础率动作的患者模型拟合实验。** 这不代表A01已经学到可干预的患者环境；真正阻断策略/产品放行的是动作内生性、未来共同干预缺失、部署可得性和独立评价。当前研究screen即使全部通过，也不能解除这些阻断。

作者代码审计中最严重的padding反用、尾部roll回卷、latent自比疑点，当前适配实现已经避免。GLU/WTR/一致性三项基本符合所声明的论文方法适配；没有伪装成已经训练策略、value或reward头。

## 一、优先处理的问题

### P1：30/60分钟验证是“已知未来日志基础率”的条件拟合，不能称普通未来预测

证据：`data.py:18–21`取起点之后整段日志action和CGM目标；`train.py:34–35`直接将这段真实未来action用于rollout。当前未来CGM没有直接作为模型输入，但未来基础率来自会随未来CGM反馈改变动作的Loop控制器。因此这些动作本身携带未来闭环状态的信息。

这是训练动作条件患者模型时常见的历史转移组织方式，不是简单的代码泄漏bug；但评估解释必须正确：

- `last-value`和`linear`只见初始CGM历史，模型还见真实未来基础率；因此“胜过两基线”只是宽松条件拟合screen，不能证明部署可用的60分钟预测，也不能证明更改剂量后的反事实准确。
- 初始历史固定后，`a_{t+1:t+11}`不是用户在t时已知的计划。不能把作者“没有未来真实state输入”的声明延伸成“预测只用了t时可得信息”。
- 应在HTML每个相关指标附近标注“给定真实未来基础率序列的回顾性条件预测”。`gate_prediction`不得改名为“临床/因果/部署预测验收通过”。

建议补一个诊断而不删除任何训练样本：在**日志未来确实恒定基础率**的匹配验证起点上比较同一计划条件下的预测；同时报告无条件历史基线、未来action置换/消融诊断。它们能分辨拟合信号来自哪里，仍不能消除观察数据因果混杂。

### P1：当前fT遗漏未来共同干预，不能作为已定义好的策略环境

证据：`model.py:26–36`只有初始history、未来basal和经过时间；`pack_data.py:25`保存未来bolus但只在`train.py:45–48`分层指标使用。未来food/exercise/bolus既不作为受控情景输入，也没有独立生成机制。

初始历史加入food/exercise是必要改进，但它只降低已记录过去事件的混杂。历史未来CGM由未来basal、bolus、餐食、运动等共同产生，模型目前学习的是在行为数据中边缘化这些事件后的统计条件关系。不能称为“未来没有进食/运动”的环境，也不能将恒定候选basal的响应直接当作其干预效果。

此外，`train.py:18–19`一致性target编码包含到s+k为止的真实food/exercise/bolus历史，而预测latent没有这些未来外源信息；该不可预测成分可能令一致性成为相互冲突的监督，或鼓励encoder抹去模态信息。**这不是未来输入泄漏，也不是两端同一tensor的自比较；是可预测性/可识别性限制。** 若一致性高或模态效益消失，应作为解释因素，不能先将责任全部归于优化器。

继续保留这些稀疏患者/样本。下一步若做策略，需要显式定义未知未来事件情景及记录可得性，或在范围明确的诊断任务上止步，不能把缺失事件当0。

### P2：评估命令会覆写原训练来源证明

证据：`train.py:72–79`在检查`--eval-only`之前就重写`config.json`和`provenance.json`。后者重新写当前source hashes、GPU和start_time。若之后修改评价代码再复评，同一个run的来源记录会变成复评时版本，丢失原训练起点/代码信息。

这不改变A01已经发生的参数更新，但影响用户要求“所有尝试、训练方式、结果均可审核”。建议训练provenance只创建一次；复评写独立`evaluation_provenance`并引用checkpoint哈希，不覆盖。重复同名run也存在history追加、checkpoint覆盖而无新run身份的问题；每次尝试应使用新run名或明确恢复协议。

### P2：动作响应screen包括未验证支持域的60分钟外推

证据：`train.py:50–54`取每位患者样本前16个，不检查该起点真实轨迹是否有12步；用初始action±0.5构造low，再加1 U/h，clip至[0,20]。即使全局筛选范围在20以内，也不代表该患者/该state有相邻动作支持。

当前screen数值阈值被协议正确称作“宽松研究异常检测”，返回字段也明确不是counterfactual validation，这一点是好的。需要继续保持：

- ≥80%负响应和中位下降0.02–3 mmol/L只是人工约定的方向/幅度screen。加入单调结构也可能轻易通过，所以不能成为因果证据。
- 应报告可验证60分钟支持的样本数、初始剂量分布、是否超患者历史动作支持、低/高初始CGM分层；响应样本不足60分钟时标为模型外推诊断。
- 当前代码仅将60分钟方向/幅度纳入gate；5分钟均值和逐样本30分钟delta有记录，但没有“生理延迟验证”。不可在HTML写“延迟关卡通过”。
- 选中样本是随机子集排序后最早的16个，而非直接均匀16个起点；这会偏向每位患者较早时段（`train.py:31,50–51`）。若更改抽样，应另记screen版本，避免混比旧新数字。

## 二、已核对且合理的实现

| 检查 | 结论与证据 |
|---|---|
| 所有合格起点保留 | `pack_data.py:15,20–24`从原index取全部starts，每个起点长度至少1；没有food/exercise存在性门槛。长展开不完整不删除一步样本 |
| 历史窗口时间对齐 | `data.py:12–15`为 `[s-71,s]`；v2 `encode_history`是同一区间，22维顺序一致：5值、5mask、5年龄、5年龄已知、2时间 |
| action/GLU标签 | `data.py:18–21`第h项action取s+h，目标取s+h+1；与基础率作用于接下来5分钟的合同一致 |
| episode连续性 | `pack_data.py:21–24`valid持续累积，同episode且每步allowed；一旦失效不会恢复，不会跨缺口拼接 |
| loss mask | `train.py:14–17`GLU与WTR使用有效future mask及有效token分母；没有作者的头尾回卷；padding target只为数值占位 |
| 一致性对齐 | `data.py:24–25`选真实k∈[1,length]，target history到s+k，tensor索引存k-1；`train.py:19`取第k步rollout latent，实际到达同一时间 |
| 一致性梯度 | 两边保留梯度，没有未声明stop-gradient；默认是MSE，μ0.1，与协议说明一致 |
| 初始编码因果性 | `model.py:23–25`encoder上三角mask；decoder memory只含截至起点的历史。decoder读取全部该history合理，不需要把初始history再次截短 |
| 多步autoregressive | `model.py:30–35`每一步将前一步z和当前候选action构造新query；prefix query有因果mask，没有每步将真实未来state灌入rollout |
| 模态mask和归一化 | `pack_data.py:17–19`等价于v2管线，使用train-only统计；占位0只在标准化tensor中，mask保持缺失语义 |
| 归一化常数 | `model.py:29`与`train.py:36`硬编码CGM均值/scale目前与v2 `normalization.json`完全一致；不是当前bug，但后续统计变化会静默错配，应绑定到模型配置 |
| 基线输入 | 线性趋势用最近30分钟7格真实观测mask，当前CGM恒定基线与target单位都是mmol/L；列5是正确CGM observed标记 |
| 模态分层索引 | 列8/9分别是carb/exercise observed，当前“无记录历史”的判断正确；未来bolus组是有记录，不是已证明无record组实际用药为0 |
| 封存集 | 当前pack只train/validation，train只加载这两组；没有看到sealed_test进入调参的代码 |

## 三、论文一致性与必须保持的描述

- 当前两Linear层MLP已明确选择公开代码而非论文三层；3层256宽8头encoder/decoder与公开主维度一致。
- 当前encoder没有复用作者的独立previous-action embedding，而是通过22维历史中的基础量输入动作历史；连续action投影仅用于transition。decoder采用从72格历史末端开始新增rollout query，与作者“全时点并行k步”实现不同。这是结构/计算适配，不是字节级原样复现。
- GLU/WTR+latent MSE目标对应论文患者主目标，未照搬源码多出来的reward/value头，这个选择在协议已声明。尚未实现`L_RL1/L_RL2/L_SL`，当前训练不能叫“策略已经训练完成”。
- 原论文是不同疾病/用药/时间粒度场景。任何A01结果都只是Loop研究适配的结果；不会复现原文临床效果。
- `residual=True`会给预测加当前CGM。这是有明确归纳偏置的结构改动，应单独命名实验；A01 absolute与后续residual之间的差异不能只归为“调学习率”。

## 四、验证指标与记录仍需注意

1. **validation就是validation。** `final_evaluation.json`由同一validation产生（`train.py:113`）；反复试超参后，扩大validation样本可以稳定描述，但不能作为独立泛化/封存结论。
2. **统计口径不同。** 全局RMSE按有效窗口池化，患者bootstrap按每患者MAE差均匀汇总，仅覆盖模型对last-value（`train.py:58–62`）。目前不能写“相对两个基线的RMSE改善均有95%置信区间”。长时域每患者保留数不同，且已知资格筛选偏差必须保留说明。
3. **WTR accuracy有限。** 没有低血糖召回率、混淆矩阵或概率校准。大量正常范围样本可能使accuracy好看；不能用它替代风险识别评价。
4. **优化选择只看30+60分钟RMSE。** `train.py:96,99`不使用剂量响应选择best。若最佳预测checkpoint方向失败，应诚实报告“当前选择模型失败”，不能翻看所有checkpoint后静默挑一个方向通过的模型并沿用原选择协议。
5. **配置不一致。** protocol第16行写batch128，A01实际config为256；属于合理容量选择但要更新记录。A01 reason写完整epoch愿望，而max_seconds可能截断；以completion实际full_epochs/samples_seen为准。
6. **部分epoch可能患者覆盖有限。** 患者轮转保证足够步数时覆盖全部患者，短烟测只访问30名的现有completion已诚实记录。不能概括“所有实验都已训练225人”。
7. **恢复不等于精确断点继续。** `train.py:75–76`只加载best模型，不恢复optimizer、step和Data迭代器；这是warm-start新实验。checkpoint未保存CUDA/Python/Data generator完整随机状态。若将来用resume，应写明warm-start，不能声称完全复现连续训练轨迹。
8. **packed完整性。** manifest绑定源合同/normalizer/audit哈希，但没有每份NPZ哈希；`Data`读入也不复验文件哈希。已存在跨本地/远端hash检查可补充证明本次传输，但报告应保存具体检查产物，而不是只依赖manifest来源字段。
9. **已写测试的范围。** `test_model.py:17–19`证明改变target tensor不影响rollout，因为target不在参数列表中；它没有证明pack阶段的历史完全不依赖未来记录，也没有处理未来反馈action的间接信息。需要分别描述管线时间因果测试、网络target隔离测试、科学可得性假设。

## 五、建议的下一步顺序

1. 允许A01完成，保留其“回顾性给定未来日志基础率的患者模型拟合”身份；记录真实训练覆盖、有效多步样本、失败/成功screen。
2. 先修正报告口径和provenance覆写问题，再开展下一组预先说明的优化/结构实验。保持所有原始A01结果，不用新评价覆盖旧指标。
3. 重点分析多步误差、实际bolus记录分层、稀疏模态分层、动作支持和响应，而不是只追总loss下降。记录未知未来事件限制。
4. 若screen失败，停止策略阶段并报告具体失败。若screen通过，仍须未来情景/外部独立环境/动作支持与弃权机制，才能讨论研究策略训练与Harness接入。

本次审阅没有发现确定的“训练数值结果整体作废”bug；当前最大风险是把**有未来日志action的统计拟合**误读为**可干预、可部署的患者动力学**。现协议后半部分的谨慎声明应保留，并在具体结果表内再次说明。

## 审阅快照 SHA-256

```text
model.py d27a00f0107dfe1bf7cb4340707bf4005f77bcb1cb7190f8a50267014ad6c6f8
data.py 4a02d704f07563aa1ba6202502b60260a6b76ce1054e6377c73573f17fed9a46
pack_data.py f1ae02edda215d4b7c5fa65f1602d38b9fca4b11f8595aa63eb8921ac7007157
train.py 0b8724eb338aaae778729b5f42d8cd4ddf43c7679149c89242d90e5d6ec58c74
experiment_protocol.md 5dab5e3cc7c3174f507fca5a3e79cde96b770c0d4fdc759451849d7cc2b070fb
```
