# 本轮冻结实现独立审查

审查日期：2026-10-03。范围：WorldModelV2、world 训练器、world 特征 PPO、共用评价入口、观测生理特征及自然/配对数据链。第1–5节保留初审源码快照，后续到达的修复与运行证据按第6–7节更新，以更新结论为准。没有连接远端、运行训练/模型推理/仿真，未改已绑定源码、IQL 六文件、旧评分器或原始轨迹。本文件是独立审查，不是成绩验收或 SOTA 批准。

**未发现足以要求废弃本轮 world 训练的直接未来标签泄漏、五分钟错位或剂量单位错误。初审发现的两项正式评价门禁缺口，现已由团队修复并完成本地独立复核，见第7节；已有结果的实际身份仍须逐manifest核对。科学限制中最重要的是自然验证没有严重低糖阳性、配对严重低糖排序很好但绝对风险/最低值仍有明显误差、闭环行为动作与干预动作的条件分布不同，以及 MPC/PPO 候选计划不完全相同。**

## 1. 明确实现问题：正式评价的候选资格没有完整强制执行

### B1：完成烟测训练的 world / world-PPO 可以进入 `smoke=false` 的全开发评价

- 位置：`evaluate_candidates.py:316–323` 对 `world` 调用 `inspect_provenance` 后丢弃返回的训练配置；没有读取 `allow_smoke`。`world_control_worker.py:82–90` 允许完成其声明预算且 SHA 正确的 smoke world，这是加载器用于工程检查的合理行为，但不能由此认定正式预算完成。
- `evaluate_candidates.py:324–350` 对 world-PPO 允许 `run_mode=smoke` 的有效配置，并用该配置自己的 `allow_world_smoke_budget=true` 判断 world。没有要求评价的 `--smoke` 与训练烟测状态一致。wide PPO 分支 `307–313` 也未建立此资格绑定；新 IQL 分支已有独立烟测检查。
- `run` 的 `manifest.smoke` 来自命令行，见 `450–456`；最终 `summary.smoke` 同样如此，见 `728–729`。因此单看 `smoke=false`、60 条完整轨迹或 summary.completed，无法证明被评估的是正式训练候选。
- 本次实际执行了**纯内存 metadata fixture**：从 AST 单独提取当前 `dependencies`，将 `inspect_provenance` 的返回配置设为 `budget_override=true, steps=4`，调用 `kind=world, allow_smoke=false`，仍成功构造 worker 命令。未读虚构 checkpoint、未载 Torch/模拟器；此验证证明缺少门禁，不证明当前实际运行已经误用烟测权重。
- 最小处理：最终候选清单独立绑定 checkpoint/config/provenance/completion SHA，并要求 world `budget_override=false`、world-PPO `run_mode=formal`、实际训练预算符合登记表。若后续修评价入口，只在该入口显式拒绝“烟测权重 + 非烟测评价”，新增评价版本记录；不要修改本轮已绑定训练源码或伪造旧 provenance。

### B2：正式 PPO 的四个预定开发检查点只是配置，评价入口接受所有迭代

- 位置：`train_world_policy.py:42–45` 冻结 `development_checkpoints=[8,16,32,40]`；`evaluate_candidates.py:332–338` 只检查 `1 <= iteration <= iterations`，没有检查成员资格。wide PPO 分支 `307–313` 也没有此限制。故“只选择四个检查点”不能单由入口代码保证。
- 影响：如果读取额外迭代的开发成绩再选择，会扩大已声明选择预算。没有证据表明当前已发生这一行为；本地没有完整远端运行目录可审计。
- 最小处理：主执行器按预先冻结的四个迭代调度，最终列出**全部实际评价过的 checkpoint SHA/iteration/split**并核对日志；如出现额外开发评价，应如实列为额外探索，不能藏入四点预算。后续入口可加显式 formal 白名单，不需要改训练器。

上述两项应在最终完整方法表和 confirmation 选型前排除。`summarize_panels.py:178–209` 严格核对完整随访与评分，但不认证模型训练预算；“Complete”描述随访完整，不是正式模型资格。

## 2. 已确认的数据、时间和信息边界

| 核对项 | 源码证据与结论 |
|---|---|
| 模拟器隐状态与部署输入 | 旧 `RL_DSENet_公平低糖_2026-09-21/ppo_env.py:23–35` 在环境内部使用 patient、CR、nominal 和餐表初始化/实施固定外源 bolus；发送的消息虽含 nominal，新 runner 只从 history 解码 anchor。`evaluate_candidates.py:82–107,622–635` 给 PPO/world/IQL 的请求只有 `op/history/anchors`；world-PPO `149–152` 严格拒绝额外字段。未发现 hidden/ID/BG/未来餐表进入模型。 |
| 真实给药与历史量纲 | `ppo_env.py:34–40` 将 U/h ÷60 送泵，以实际 basal U/min ×5 写入历史，原始行保留 requested 与 delivered U/h；bolus U ÷5 送环境，再按实际泵量 ×5 记录。`History.append` 在旧 `RL进阶对比_2026-09-15/observable_history.py:11–18` 用 CGM÷18、前五分钟胰岛素总量及 mask/age 形成 72×22。新 physiology 只逆变换已观测值（`physiologic_features.py:97–117`）。 |
| 状态→动作→标签 | 决策 t=360 的 state 是 t=5…360；动作作用于随后五分钟，记录为 t=365。`prepare_windows.py:18–38` 用 history[i]、future=rows[i+1:…]，动作和 BG/CGM 均来自同一未来行；`paired_collect.py:65–78` 同样先执行再取末点。无把当前点当动作后标签的问题。 |
| 配对干预 | `paired_collect.py:61–100` 每臂从同一个环境快照 deepcopy，arm0 是六小时 anchor hold；其他臂前 60/120 分钟为 0/.5/1.5/2 anchor，之后回 anchor。共享未来 meal/bolus 是**标签生成中的共同外生条件**；这些仅写 `label_only_*`，训练器只读 `ARRAY_KEYS`。smoke 有重复 arm0 全路径一致和同动作单步原环境一致检查（`80–92,104–113`）。arm0 不能描述为“继续原 exploration 轨迹”；base 后续实际运行的是 `101` 的 exploration。 |
| mask 与未知尾 | 自然/配对均保留前缀 mask。world loader 拒绝空前缀或 false→true mask（`train_world_v2.py:106–108`）；回归只用 mask，事件 CE 只用完整72点（`133–156`）。未观察尾部没有被当作无事件。完整事件统计也排除截尾（`203–242`）。 |
| world 因果时间结构 | `world_model_v2.py:39–51,161–180` 逐时间 channel norm、仅左 padding 的卷积保证点 k 的 CGM 不看 k 之后的动作；事件头针对整六小时，可以使用整候选计划。全局 context 只来自当前历史。`checks/world_mechanics.json` 已保存 future-prefix 与 masked-placeholder 误差0，batch/single最大差约3.05e−5；这些证明工程性质，不证明动作效果正确。 |
| DSENet 真接入 | `world_model_v2.py:129–141` 使用真实冻结 Forecast，48点乘18转换成 mg/dL；`154–159` 第49点起 baseline 改为最后已观测 CGM并带 availability。没有把原 DSENet 伪称为72点模型。`train_world_v2.py:378–382,442–444` 核验归一化及冻结状态。 |
| PPO 真终止与时限截断 | `train_world_policy.py:164–178` 技术失败中止；只有 native_environment_done 标 terminal。继承 `ppo_wide_worker.py:180–200` 每个 action 对应一个 post-action BG，真终止 bootstrap=0、末步−100；时限用 final_history 的 value bootstrap，GAE 在每条 episode 内计算。world features `no_grad`，world 冻结，未用预测 BG 给 PPO 自评分。 |
| train/val/development/confirmation | `train_world_v2.py:48–76` 在读数组前查 split/job seeds，NPZ逐文件 SHA校验；`train_world_policy.py:42–60` 禁止非训练 seed。评价 `evaluate_candidates.py:59–76` 只允许 development/world_validation，确认集另需冻结入口。本次核对正式 world provenance 的全部文件 job seeds 与协议一致，未见交叉。world_validation 被用于每500步选 best，属于模型开发数据，不能再称独立封存测试。 |

额外做了本地内存人工序列检查：864 行重建 history 与在线 History（含 t0）在首决策点逐元素相同；t360→t365 对齐；单 episode 264 origins、241完整窗口、最后窗口3个有效点；只改变 BG 标签不改变 history/actions。此为机制检查，未产生或冒充真实患者/仿真证据。

## 3. 科学限制，不能误写成明确代码 bug

### L1：自然验证不能估计严重低糖 FNR；配对排序也不是 FNR

已取回 `checks/world_quantile_r1_provenance.json` 的数据身份如下，计数为窗口/干预臂，不是独立事件或患者：

| 数据 | 文件数 | origin数 | 完整臂窗口 | 截尾臂窗口 | 完整 BG<54 阳性 | 场景 seeds |
|---|---:|---:|---:|---:|---:|---|
| natural train | 120 | 31,680 | 28,920 | 2,760 | 69 | 103001–103004 |
| paired train | 60 | 480 | 4,320 | 0 | 74 | 103001–103002 |
| natural world_validation | 60 | 15,840 | 14,460 | 1,380 | 0 | 103011–103012 |
| paired world_validation | 60 | 480 | 4,320 | 0 | 55 | 103011–103012 |

自然验证 **BG<54 阳性=0** 已由保存的 provenance 核实；主执行器另报告 BG<70 阳性214，以及500步 paired BG54排序380 eligible pairs。本次没有该500步预测 NPZ/汇总原件，不把这两个数字写成独立重算结果。380 是同起点内事件标签不同的臂对数，不是380个严重事件或独立受试者。

`train_world_v2.py:166–167,229–242` 和独立 `diagnose_world.py:124–142` 均在无阳性时返回 FNR=null，代码没有写成0，正确。自然集可报告负例上的 Brier、FPR、预测风险分布，但即使很小也不能支持“严重低糖零漏报”“严重风险校准完成”。CGM<54 点也不能补作真实 BG<54 阳性，传感器量和真实 BG 是不同标签。

paired验证有55个阳性臂窗口，因此可独立报告该人为干预分布的 TP/FN/FNR、Brier、最低值偏高程度和排序。不要将其发生率与自然样本混合校准；不要把同患者、同 origin 多臂或重叠自然窗口当独立样本。将来要估计自然严重低糖 FNR，需要另行事先冻结、具有阳性覆盖的独立资料；不能反复改 seeds 直到确认集出现有利结果。

### L2：自然行为轨迹中的未来动作是反馈动作，不等于预先设定的干预

`controller_baselines.py:112–145` 的 exploration 使用后续可见 CGM/ROC，低糖 override 随时间改变动作。`prepare_windows.py:27–36` 又把未来实际动作串当条件输入 world。因此自然数据中的未来动作可能携带未来中间观测的线索；部署的固定候选串没有这些反馈线索。点预测是动作前缀因果并不能自动把这种观察条件关系变成干预效果。

这不是“部署偷偷输入未来 BG”的代码泄漏；当前显式同快照多臂干预和 delta loss 正是在补充动作效果证据。但 natural/paired 1:1 group目标（`train_world_v2.py:160–163,415–425`）不能据此声称所有 wide动作、所有状态都获得无混杂因果识别。应分别看 paired delta/排序与绝对低糖误差，并最终看闭环真实 BG。自然 MAE好不能单独证明 planner可靠。

### L3：PPO、MPC 和配对训练的动作计划不同，滚动执行也不兑现整计划

paired/MPC为hold及4倍率×2时长；world-PPO则为9倍率均保持60分钟、余下回anchor（`ppo_world_worker.py:32–36`）。PPO中的.25/.75/1.25/1.75并无该确切配对训练臂，属于插值而非直接对照证据。两种控制都只执行首5分钟，之后重规划；六小时事件头对应的是原假设计划，不是实际未来闭环事件概率。

所以 MPC vs PPO 不能解释成“只改变 optimizer”；概率特征 vs旧wide也同时改变编码表示、数据/训练预算及输入维度。可称不同系统候选；单机制收益需相应消融。`world_policy_design.md` 已如实说明主要差别，应保留。

### L4：有序边际分位数、BG事件头和风险代价均不构成安全保证

非交叉来自结构（`world_model_v2.py:172–181`），不是校准结果；边际q05下包络不是轨迹最小值的5%分位数。MPC期望风险是7点中点权重积分、20–600 mg/dL裁剪及恒定端尾假设（`world_control_worker.py:103–158`）；不能称联合轨迹分布/CVaR。预测风险主要作用于CGM，而PPO回报/主评价用真实BG；两者虽同Kovatchev方向也不是同一随机量。

三分类事件是任意一次阈值穿越，没有持续时间、恢复时间或严重低糖120分钟事件头。完整表仍必须使用原评分器报告持续严重事件、失败和coverage。point只训练中位数pinball（Q=.5等于0.5×MAE），不是MSE均值预测；point/quantile的自身组合loss不可直接横比为优劣。训练 provenance已注明该边界。

### L5：数据覆盖有限、截尾排除有代价；当前不能称生理机制已识别

自然验证1,380个尾部不足72点的窗口被排除事件CE/完整校准；已观察阳性的截尾窗口也未用于CE。这样避免假阴性，但不能证明未见尾部安全；对未来 native终止资料尤其要单列截尾阳性。paired当前4,320臂均完整，不能外推其遇到终止的性能。`diagnose_world.py:140–142,197–235` 已区分完整与已观察前缀，保持这一口径。

IOB为固定300分钟/75分钟峰值曲线估计、COB为180分钟线性代理，net basal IOB相对observed anchor（`physiologic_features.py:48–50,79–94,136–173`）；不读取患者真实吸收、ISF/CR。controller的1800规则、0.5 basal fraction及3.6碳水效应是先验假设（`controller_baselines.py:21–39,79–90`），不是官方OpenAPS/临床controller精确复现。无bolus记录仍非真实无bolus。有限历史/固定先验偏差属于该基线能力边界，不是隐藏信息泄漏。

### L6：比较预算、精度及外部方法适配不能被“同输入范围”掩盖

P03是额外预训练先验；world使用120自然轨迹加480×9配对分支，再训练PPO。旧wide依赖P03+D05/H02；IQL使用不同表示与冻结数据预算；旧8方法保留旧权重后投影到新动作范围。相同history接口/仿真jobs不能使预训练、数据、优化和动作支持完全匹配。优于失败/不适配的旧方法不是强SOTA证据；应同时报告hold、physiology、原D06、适配的新IQL与各自边界。

新world/C-PPO明确关闭matmul和cuDNN TF32并记录（`train_world_v2.py:359–361`、`world_control_worker.py:186–189`、`ppo_world_worker.py:113–117`）；A诊断 `ppo_wide_worker.py:75–80` 未设或记录两项TF32，`train_wide.py:35–36` 没有保存ready内容。因此现有本地证据不足以声称A/B/C所有精度开关一致。此项不要求修改已绑定A源码或重训；若无原运行证据，表中如实注明未确认，而不是把参数dtype=float32推成完整精度等价。

### L7：开发诊断、模型校准和最终主张需分层

world_validation每500步被用于best选择（`train_world_v2.py:436–460`），在同一集合回看校准是开发诊断。point与quantile应核对实际训练provenance中的四个数据manifest/NPZ hashes，而不仅模板路径；本次仅有quantile正式provenance可独立核对。保存预测NPZ未直接绑定checkpoint SHA，独立诊断已披露这一身份链限制；若需强绑定，可另行只读重算冻结权重预测，但本次没有执行。

闭环主表应保持患者内平均→10患者间mean±sampleSD，BG主表、CGM次表；患者间SD/患者cluster bootstrap不是训练seed稳定性。`summarize_panels.py:178–209,248–295` 已禁止失败前缀参与完整方法比较，并将原joint与family描述分开。仍要由最终报告保留全部方法失败、未知尾、TIR上下界，且先处理B1/B2的候选资格。已曝光十名成人的新场景、单训练seed及有限适配基线均不支持普遍SOTA、新患者泛化或临床安全。

## 4. 最有价值的后续验证，按优先级

1. **严重低糖识别是否真实可用。** 对预先明确的完成训练step（如固定4000，及按既定规则选出的best分别标明）独立重算自然/配对诊断。自然BG54保持FNR=null；paired55阳性单列固定0.5阈值混淆矩阵、最低CGM偏高、BG54异臂排序分母；同时按origin/患者展示失败集中情况。不要根据该诊断事后改阈值再称独立验证。
2. **动作效果与计划适用性。** 使用现有paired验证的完整共同支持核对delta、最低/终点排序和事件排序，检查第48/49步预测边界；将PPO新增中间倍率明确列为未直接配对验证。只有动作梯度非零、平均MAE降低不算通过。若需要新增干预，先声明为开发补充及用途，不能占用封存场景调参。
3. **资格冻结后的实际闭环收益。** 列出正式权重/预算/SHA和全部预定开发checkpoint，统一jobs/meal/factor比较；满足完整随访后才做患者配对差与置信区间，原joint和family判定并列。选型冻结后再生成confirmation。任何失败方法仍进入Observed表，不能用幸存前缀制造多数指标领先。

## 5. 本次核验与源码快照

- 审查范围内的源代码、原History/ppo_env/control_metrics、研究合同及设计文档已逐段对照；未引入评分改动。
- 正式quantile provenance所有绑定源码在本地重新SHA核对，**0项漂移**；train/validation文件job seeds均属于预定分区。此为已保存元数据核验，不冒称已对远端NPZ或正在运行的全部实验重新哈希。
- 纯内存B1门禁复现、History/窗口时序/BG隔离检查已通过；没有Torch、simulator或网络运行。已有CUDA机制结果来自保留的检查JSON，作为他方运行证据引用，不声称由本审查重新执行。
- 初审时本地没有本轮完整 `results/` / `cache/`，所以未重新执行正式4000步推理、实际开发候选枚举或最终闭环评分；后补取得的正式检查JSON复核见第6节。

以下SHA固定本报告对应的版本；后续评价入口如经另行修复，需追加新的审查证据，不覆盖本快照：

| 文件（均在本研究目录） | SHA256 |
|---|---|
| world_model_v2.py | ccc8ddf2b628c11ae544b43cebe31015864e024677cfcecfa10da6ce383378db |
| train_world_v2.py | 230393aa25089442f719fc356e59ae2f21cf5b153b04d1423910760ffcf57bec |
| ppo_world_worker.py | c949d01d679baf58a95c231692975164890989861c9e17f891fd197028d0957e |
| train_world_policy.py | 3da8a35d089c8320ab46be28dd4103e2d0c4e2fb2bff4ea92663bf1e7ff95cf4 |
| world_control_worker.py | 88a40896b6e46b75b3a3612932831cc360a755cb36fbb3c5de800957c6ffffcb |
| evaluate_candidates.py | 30e303cf58874a6a958e7314ec3c7a16b90ee33549ef51aa76de8068b879982c |
| controller_baselines.py | e1ee43f85edd38f5f65ad5638315b60226708933281271a06fcc58eba0cb36e2 |
| physiologic_features.py | 0c0f89a6bbe8f8a8dc6a494f7457ad743d74069ac96f81eb6a8ee6496530b59c |
| paired_collect.py | c350557a654fc27f724770fb965ff4fa3a6abe12f72e6eacbe1b4c52163c1440 |
| prepare_windows.py | 398326ad5b64c9551a0b7c846459f237cd3e5cceaf5ebf22fc36486ff69b0cf2 |
| collect.py | 9428a84a7b77659987eaa5d9964e90860da8ba6d889ff6f5ba8663fcce8fafc6 |
| ppo_wide_worker.py | 282e7f58a36f94f6e13b6a84b258c3c3903662f49cedd3346708f012f7ee18eb |
| protocol.json | 7f59e99eb5bd76fb430c0e6f488f75436315a2fffa3c415e7205e4f52a369b36 |
| diagnose_world.py | b816e6a851c554d3eaff0e2dfb7b931b82db18385e7839ff03ec7bb52d81c198 |
| summarize_panels.py | 82c7a11db9d3316af1cfd9584e6175c594262127831ccad61ac77f16c32c6438 |

独立审查状态：源码审查与本地无模型机制核对完成；训练/推理/医学有效性未由本审查验收。B1/B2需在最终候选资格审核时关闭；其余限制必须进入结果解释，不能以表格获胜自动消除。

## 6. 同轮后补证据复核：正式 world 诊断

主执行器在初审后取回正式 best 的全量预测重算绑定和独立诊断。本审查只读这些真实产物，未自行重跑推理。已核对：两者均完成4000步、非budget override；point按原world_validation规则选step1000，quantile选step3000；两种模型的natural/paired验证文件顺序与逐文件SHA相同。每份diagnostics的prediction SHA、标签manifest SHA和training provenance SHA均与对应prediction_binding一致。

两份binding均记录全量重算而非抽样，CGM和三类概率全部元素最大误差0、模型加载前后state digest相同。这关闭了第L7项关于这两份best保存预测与权重之间**缺少独立数值身份核对**的待办；它只证明身份与数值一致，不证明精度/校准/控制效果。

| 已保存best诊断 | point step1000 | quantile step3000 |
|---|---:|---:|
| natural CGM median MAE，mg/dL | 19.1172 | 17.7014 |
| natural BG<70，固定概率0.5阈值FN/阳性 | 155/214（72.43%） | 129/214（60.28%） |
| natural BG<54 FNR | null，0阳性 | null，0阳性 |
| paired BG<54，固定概率0.5阈值FN/阳性 | 41/55（74.55%） | 18/55（32.73%） |
| paired BG<54 概率异臂排序 | 376/380（98.95%） | 379/380（99.74%） |
| paired BG<54阳性臂的median最低CGM偏差，mg/dL | +28.4279 | +16.9294 |
| 上述55臂中median最低CGM偏高比例 | 55/55 | 53/55 |
| paired CGM delta MAE，mg/dL | 5.1921 | 4.1349 |
| paired CGM endpoint排序 | 16893/17277（97.78%） | 16832/17277（97.42%） |

这些实际结果直接说明：quantile在部分指标有改善，但**异臂排序接近100%仍不等于绝对风险识别或最低值偏高问题解决**。自然BG54的FNR继续不可评；不能把point/quantile自然BG54的低Brier或零FN计数写成零漏报。0.5是预设诊断阈值，MPC用连续概率惩罚、PPO用概率特征，表中FNR也不能直接冒充控制器临床漏报率。上述指标仍来自用于选best的world_validation，仅支持开发诊断。

此外，`checks/panels_development_ppo08_r1.json` 中 `dev_ppo08_r1` 与 `dev_hold_r1` 都是60例完整随访，BG汇总逐字段相同，已本地核实。主执行器另报告47,520动作距anchor最大5.53e−8；本审查没有该决策原件，保留其证据层级。不能把PPO08的结果解释为超越hold；后续softmax均值部署若实施应另标候选/消融，不覆盖原argmax成绩。

| 新取回证据 | SHA256 |
|---|---|
| checks/world_point_r1_best_prediction_binding.json | ba4c82ec2863dd1c9cd9ae0aefa599f97e1e16888751fec0db2ba1c3678d14a2 |
| checks/world_quantile_r1_best_prediction_binding.json | 86842de2e260f3f91a3b0609cb1d76e38bcc762318947a39f7650c26ac0075a7 |
| checks/world_point_r1_best_diagnostics.json | e98c0925b0c9e452b2cd732b11da0cbfba5566970315f75324a6c4c942b6b88c |
| checks/world_quantile_r1_best_diagnostics.json | 2f0b5c077051800324ee665a79654b71d49544b060cb00f751064d366c5259be |

## 7. B1/B2 修复复核与最终状态

修复由负责评价入口的团队成员实施，本审查未改实现。新稳定 `evaluate_candidates.py` SHA256：`b06e78609653a9d24bc4214d2b827b985d6694904e0012dbbee6f88a6a1691dd`。第1节行号和第5节eval SHA专指修复前版本，保留原发现证据。

- 新 `policy_evaluation_binding`（`270–335`）要求同run配置与provenance一致；wide PPO按完整模板识别formal/smoke，world-PPO读显式run_mode；正式评价仅允许8/16/32/40。绑定已经完成的history连续前缀、目标迭代checkpoint路径与SHA；不因训练器随后继续追加日志改变该绑定。
- 新 `world_evaluation_binding`（`338–350`）要求正式world实际/configured均4000步且budget_override=false；smoke权重只能经显式smoke路径。`dependencies` 的world/world-PPO分支均调用此门禁；PPO/world-PPO均调用policy门禁。
- runner保存history前缀快照、绑定写入manifest/provenance，并对ready中的iteration及world训练预算再核对（`627–638,677–699`）。失败不会成为正式模型资格。
- 本审查实际独立运行9项纯内存AST函数检查：formal4000通过；smoke4在formal路径拒绝、显式smoke通过；4000但override=true拒绝；world-PPO正式08通过、09拒绝、SHA错拒绝；smoke01正式拒绝、显式smoke通过。未加载Torch/模拟器、未创建结果文件。
- 团队独立检查 `checks/evaluation_mode_gates.json` 为73项通过，包含错误ready迭代在仿真启动前失败、history快照以及受保护源码未变；SHA256 `58719950a0576675a2d91b5d40b4b97f43562f7d78679d279a456b6c689e1ff0`。该产物绑定上述稳定eval SHA。另保留IQL门禁58项回归检查 `checks/evaluation_mode_iql_regression.json`。

**关闭状态：B1/B2在稳定源码及本地metadata机制层已关闭；没有新增要求重训的实现阻断项。** 本次未部署/运行远端入口，因此正式大表仍须核对各评价manifest实际使用的版本、权重资格、固定checkpoint和完整jobs。合法旧结果不因评价器升级自动失效；反之，也不能凭修复代码存在就替缺失证据的旧结果补资格。

**仍阻断的主张：自然严重低糖零漏报/完成校准、配对排序等于低糖风险已解决、PPO08超越hold、严格单因素机制归因、普遍SOTA或临床安全。** 这些是证据边界，不阻断继续已授权的world-PPO固定预算训练和按冻结协议的闭环验证。审查文件交由主执行器登记工作记录；不改变产品Harness状态。
