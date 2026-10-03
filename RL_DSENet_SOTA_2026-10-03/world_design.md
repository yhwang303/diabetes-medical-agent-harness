# WorldModelV2：最小动作条件概率模型候选

日期：2026-10-03。状态：本地实现候选，仅做语法和静态核验；尚未在 Torch/CUDA 运行、训练或取得任何新结果。未修改旧模型、评分器和产品 Harness。

## 路线评审

支持按 A→B→C 分开归因：A 先以宽基础输注动作、真实仿真回报 PPO 检查可达性能；B 用同一份可观测历史与宽动作仿真数据学概率 world，再比较风险规划/actor；C 最后加 world 辅助 PPO。A/B/C 均维持 basal-only，不能提前扩成 basal-bolus。

十名成人都已经曝光，可以在独立新训练场景使用全十人，但最后只能称 known-patient、新场景评价。训练 seed 只有 260915；本模块不设置或重置 RNG、不做 ensemble，调用者在构造前统一设置种子。A 的数据、更新量和新动作合同须冻结并提供给相应基线；C 的新增辅助监督必须单列消融。

B 不宜一开始复刻完整 Dreamer、切到 JAX，或仅冻结 H02 后换另一种 MLP。这里采用现有 Torch 2.0 可用的 GRU、Conv1d、LayerNorm、Linear、softplus；保留既有 Mamba 1.2.2 的真实 DSENet 加载路径，不增加新的运行依赖。高频规划只执行一次 DSENet/历史编码，然后向量化评价多个候选计划。

## 模块和实际 DSENet 来源

文件：`world_model_v2.py`。

```python
import json
import torch
from world_model_v2 import WorldModelV2, load_frozen_dsenet

torch.manual_seed(260915)
# 传入已核 SHA 的真实 P03 best.pt；函数没有默认权重或替代模型。
forecast = load_frozen_dsenet(p03_checkpoint_path)
normalization = json.loads(normalization_path.read_text())
world = WorldModelV2(forecast, normalization).cuda()
cache = world.encode(history, anchor_u_h)
outputs = world.predict(cache, actions_u_h)
```

`load_frozen_dsenet` 只读加载旧 `RL_DSENet_2026-09-17/forecast_model.py` 的 `Forecast`、checkpoint 的真实模型配置和权重，strict state-dict 匹配；模型未使用 H02。既有 `selected_version.json` 的 P03 路径是 `results/P03_g12s3_l2s1/best.pt`，最终仍由主执行器核验实际文件和 SHA 后传入。旧 Forecast 自己读原归一化配置，新 history normalizer 也须来自同一份冻结 `normalization.json`；不要另拟归一化而继续喂旧 P03。

forecast 参数冻结，每次 encode 执行其真实前向，`train()` 后也保持 forecast.eval()，防止旧 dropout/BatchNorm 被改变。其 48 点输出既进入全局特征投影，也直接成为前 48 个点的 residual baseline；不可能通过占位器跳过计算。是否对最终成绩有帮助仍须有/无其特征的公平消融，接入本身不是增益证明。

CGM 单位使用旧研究链一致的 `mg/dL = mmol/L × 18`，不在新旧对比里擅自改成 18.0182。基础输注历史第 1 列是前 5 分钟 U，总量×12 后才是 U/h；未来动作从入口到输出均保留 U/h，不量化、不裁剪。

## 精确接口

### `encode(history, anchor_u_h) -> cache`

- history：`float32 (N,72,22)`；列 0:5 为五通道旧 z-score，5:10 为 observed，10:15 为 age_log_hours，15:20 为 age_known，20:22 为时间特征。五通道依次为 CGM、实际已送基础量、已记录 bolus、已记录碳水、运动事件。
- anchor_u_h：`float32 (N,)`，非负基础率上下文。它不是下一步动作、未来标签或患者 ID；必须按新行动合同由截至决策时资料计算。
- 两者同 device、有限值。没有任何已观测 CGM 时拒绝。未观测值先清零，仅作为张量占位；其 masks/age/age_known 始终保留。
- cache 含可训练 `context`、anchor、最后实际观测 CGM、真实 DSENet 4h 输出和 logged_exposure_summary。不包含未来 BG、餐食、bolus 或模拟器生理状态。
- 新 GRU 读取物理缩放后的五通道及全部缺失/时间信息，梯度可更新；与旧冻结 H02 附加 residual 不同。
- 为提供简单长时剂量记忆，另计算 30/90/180/360 分钟四种指数核的已记录 basal、bolus、carbs 加权和，并带各自加权记录覆盖率。这些是 **logged-exposure 特征**，不是医学 IOB/COB。无记录不能推出没有实际给药/进食；6h 之前的信息未知，不补成已知零。
- 推理同历史/同权重可复用 cache；训练必须每次更新重建，不能把旧计算图或旧编码缓存跨 optimizer.step 复用。

### `predict(cache, actions_u_h) -> dict`

- actions_u_h：`float32 (N,A,H)`，`A>=1`、`1<=H<=72`。候选未来 basal 是计划条件，允许输入；未来真实结果不是条件。支持 wide actions，最大值/变化率/授权边界由外层冻结合同限制；模块只拒绝非有限或负值，不静默改变动作。
- index k 的动作表示随后第 k 个 5 分钟区间的实际基础输注，CGM index k 为该区间末端。数据生成器必须保证此对齐；有执行器限幅时须存 actual actions。
- 未来输入包括绝对动作、相对 anchor 的动作、固定 72 点时间坐标、DSENet 本时点预测和可用标记。七层门控因果膨胀卷积，dilation=1,2,4,8,16,32,64，kernel=3；受野 255 点覆盖当前允许的全部 72 点。
- 所有卷积只左填充，LayerNorm 只在每个时间点的 channel 上计算。第 k 个 CGM 预测只能依赖 `a[:k+1]`；不使用未来时间的 BatchNorm 或全序列 attention。全局 context 只依赖决策前历史。
- 48 点后 DSENet availability=0，baseline 明确改用最后观测 CGM，扩展由新网络学习。它不是 6h DSENet 成绩，不能将复制/外推伪称原 P03 输出。需要观察第 48/49 点的连续性与 6h 验证误差，防止边界伪迹。

返回值：

| key | shape | 语义 |
|---|---|---|
| cgm_quantiles_mgdl | N,A,H,Q | 有序边际 CGM 分位点，默认 Q=7 |
| cgm_median_mgdl | N,A,H | .5 分位点，不冒充条件均值 |
| quantile_levels | Q | .05,.1,.25,.5,.75,.9,.95 |
| bg_event_logits | N,A,3 | 请求时域内 BG 事件互斥类别 logits |
| bg_event_probabilities | N,A,3 | 无 BG<70；有 BG<70但无BG<54；有BG<54 |
| p_bg_below70 | N,A | 整段至少一次 BG<70 的概率 |
| p_bg_below54 | N,A | 整段至少一次 BG<54 的概率 |
| dsenet_available | H | 当前时间点有无原 P03 预测 |

分位数用中位数及两侧正间隔参数化，结构上不交叉；这不是概率校准保证。`quantiles=(.5,)` 可直接创建单点结构消融，保留同历史/动作网络和同成本头。事件概率来自独立分类头，非阈值化中位数；其三个互斥类别确保 `P54<=P70`。整段事件头可依赖整段候选动作，不能把它当逐时刻风险。

这些是边际分位点，**不能把不同时间相同分位数连线称为一条联合采样轨迹，也不能据此计算已校准整段 CVaR**。整段风险先用单独事件头解决；若之后确需路径 CVaR，需要另学时序联合分布并验证。未来未报告进餐/bolus 不输入，训练输出相当于按训练外生事件过程边际化；过程变化仍有分布偏移风险。

## 建议损失与样本单位

实现本轮仅包含模型；以下是训练器需要冻结的目标，不是已经执行的 loss/评分改动。

1. **分位数预测**：真实未来 CGM `y` 以 mg/dL 提供，missing mask 单列；`pinball_tau(e)=max(tau*e,(tau−1)*e)`，`e=y−q_tau`。按有效时间和 Q 求平均，最后按 history/arm 分组平均，避免一个多分支起点变成多个“独立患者”。用 `/100` 固定量纲归一化，不为危险例事后改权。
2. **单点消融**：Q=.5 时 pinball 就是中位数绝对误差；若需 MSE 点模型，另标目标变化，不能称其与分位数模型只差输出宽度。
3. **低糖事件监督**：仿真训练奖励/成本标签从真实 BG 得到请求 H 内 `none/70-only/54`，对独立 event logits 用交叉熵。不得拿未来真实 BG 作任何 encode/predict 输入，也不得用 CGM 事件冒充 BG 标签。发生率改变的富集采样需重要性修正或在自然发生率开发集重校准。
4. **完整标签**：H 不完整时不把未知尾部当“无事件”。最简训练先只给完整 H 记录事件类别；若要利用截断后的已观察阳性，需要专门的部分标签损失，不能直接赋 70-only 或 none。
5. **主组合**：`L = L_pinball/100 + lambda_event * L_event`。lambda 由主执行器在训练/开发预算内预先冻结，不从确认集选择。本模块没有设安全成本阈值、reward 参数或修改旧控制指标。
6. **成对动作效应**：可追加中位数差分对照，但保持 `lambda_pair=0` 的基准；仅差分好仍可绝对低糖漏报。先用 literature_world_model.md 的 reference/effect 2×2 分解决定是否需要该项，不默认加复杂损失。

重用 A 的轨迹训练 B 可以检查自然访问状态；仅普通 A rollout 无法保证足够的同状态动作差异。若排序不可信，需要从训练专用快照建立 wide-basal 配对分支，分支共用外生过程但未来外生量不输入模型。同一快照全部分支必须处于同一个分区。一次6h分支的72点不是72名独立患者。

## 对规划和联合 PPO 的边界

B 首先使用固定 finite candidates 做 event-risk 与任务收益排序，和点模型相同候选相比。概率尚未校准时不根据“p很小”宣称安全。风险事件头目前不含低糖持续时长/恢复代价，完整闭环仍需旧独立指标检查，不能拿事件概率代替全部风险。

C 可把新状态/分位数/事件预测作为 PPO 的辅助表示或额外训练目标，但真实仿真奖励保留为主要可检验回报；不能让 world 自己给自己评分。必须保留 A、B、C 相同数据/预算范围的清晰消融。只保留真实 BG reward/cost 标注边界，不改变 Core 最终控制和发布规则。

## 远端运行前必须核验的行为

本地静态通过不代表这些行为已通过。主执行器在既有 CUDA 环境运行：

- 真 P03 checkpoint SHA、state-dict、48 点输出与旧入口逐值一致；保存/加载新 world 仍一致；forecast 无梯度、无训练态漂移。
- 随机真实合法历史在 N/A/H=不同值、H=1/48/49/72 时 shape 和数值有限；Q=1 与默认7均工作。
- 改 `actions[:,:,j:]`，所有 `CGM[:,:,:j]` 逐值保持（容忍仅必要浮点尾差）；同样验证长短 query 的前缀预测一致。整段风险无需满足该前缀断言。
- 相同单样本单候选与批处理一致，无跨患者/候选归一化；非零 action 梯度、history encoder 梯度、forecast frozen 检查。
- 改未观测占位值不改变结果；“观测到0”与“缺失”保留不同的掩码和覆盖特征；只有缺失 CGM 时拒绝。
- 有序分位点、概率和为1、P54<=P70；动作/anchor 单位，最后已送区间与下一待执行区间索引匹配。
- 实测 encode 和不同 A/H 的 predict 延迟、峰值显存；先小批训练过拟合检查绝对轨迹/动作变化，再正式固定预算。不要将过拟合成功称独立模型验证。
- 15/30/60/120/240/360 分钟、低糖富集和自然样本分开评估；后48点扩展、宽动作分布外行为、严重事件和恢复尾部必须独立检查。

结构先验包括固定暴露记忆核、因果局部卷积、DSENet residual baseline、有序分位点与嵌套事件概率；它们都不是已验证生理定律或临床保证。新模块不是完整随机生成式 RSSM，也不声称精确复现 PETS/Dreamer/SafeDreamer/TD-MPC2。

`agent_proposed_status: implemented_static_checks_pending_cuda`。
