# Frozen WorldModelV2 驱动的真实奖励 PPO

这是新的模型辅助 PPO 候选。它在每次五分钟决策时，用完成训练且冻结的 WorldModelV2 预测九条六小时基础输注计划，再把动作条件下的预测摘要输入新 actor。PPO 的回报、GAE、value target 和更新均来自原仿真环境的真实转移；不生成想象训练轨迹，不通过世界模型反传 actor 梯度，也不联合更新 world。工程实现不证明控制收益或临床安全。

## 与现有候选的差别

- `ppo_wide_worker.Trainer.act/update` 原样继承，只重写初始化、特征和请求字段门禁。没有调用旧 Trainer 初始化，因此不实例化 D05/H02，不载入 D06/旧 actor。导入旧 worker 只是复用已测 PPO/GAE 源码；旧 `ppo_wide_worker.py`、`train_wide.py` 不改。
- 相比 wide PPO，新 actor 的直接输入包括新 world 的非线性动作响应、低端分位数和独立 BG 事件概率。旧 wide 特征依赖旧 reference，但不对九个候选动作作新 world 预测。
- 相比新 MPC，动作由经过真实 BG 奖励训练的 actor 选择，不用风险代价 argmin。MPC 的配对数据臂是 0/.5/1.5/2 倍、60/120 分钟；本 PPO 的动作菜单固定为 0/.25/.5/.75/1/1.25/1.5/1.75/2 倍，各持续 60 分钟。因此二者的候选计划并非完全相同，不能将其比较描述为只差优化器的严格单因素消融。
- 相比 G2P2C 式联合辅助训练，这是较小的第一步：新 world 提供冻结动作响应特征，actor/critic 独立学习；未实现联合 world loss 或 world 迭代更新。

## 信息、计划与 195 维特征

推理只接受实际过去 `history[N,72,22]` 和 warmup 已交付基础率 `anchors[N]`。不传患者 ID、nominal hidden physiology、真实 BG、未来餐食或未来 bolus。训练结束的 BG 序列仅传给 PPO `update` 的奖励计算。历史缺失记录按既有 mask/age 表示，physiology 估计不宣称缺失事件等于无事件。

对每个 anchor，每次构造 `actions[N,9,72]`：前 12 个五分钟间隔为 `clip(anchor × multiplier, 0, 20)` U/h，后 60 个间隔为 anchor。模型调用一次 `encode(history,anchor)`，再一次向量化 `predict(cache,actions)`。执行 actor 所选倍率的第一个五分钟动作，下一次用新历史重算。六小时内后续计划是用于比较的假设，不是承诺执行计划。

固定顺序为：

| 范围（从 0 开始） | 项目 | 维数 |
|---|---|---:|
| 0–63 | 新 world 的 context | 64 |
| 64–111 | 真正冻结 P03 的原始四小时 DSENet 预测，mg/dL ÷ 100 | 48 |
| 112–139 | 原 `physiologic_features` 28 项，顺序原样保留 | 28 |
| 140 | anchor U/h ÷ 2 | 1 |
| 141–194 | 按倍率递增顺序，每条计划六项摘要 | 54 |

每条计划六项顺序：median 全窗口最小值 ÷ 100；median 六小时终点 ÷ 100；q05 全窗口最小值 ÷ 100；完整六小时任意 BG<70 的概率；任意 BG<54 的概率；预期平均加权 Kovatchev 风险 ÷ 10。

风险特征使用 `f(g)=1.509[(ln g)^1.084−5.381]`、`10f²`，低风险分支乘 2、高分支乘 1，与训练 reward 权重方向相同。仅预测风险特征将 glucose 裁剪到 20–600 mg/dL；真实 BG reward 继续采用继承的原 wide 定义，不另加裁剪。分位数积分权重由相邻 levels 中点划分 `[0,1]`，边端常值延拓。例如七个 levels 的权重为 `[.075,.1,.2,.25,.2,.1,.075]`。每一步加权求和，再对 72 步取平均。此为边缘分布数值近似；没有声称 joint trajectory、CVaR 或概率已经校准。

point world 的唯一 level 为 .5：q05 通道明确使用 median，风险为 median 轨迹风险；独立 BG 事件头仍保留。它不伪造 0.05 分位数。相同特征维度允许 point/quantile 配置对比，但两个 world 本身需用各自训练完成的权重。

所有 world/DSENet 参数 `eval/requires_grad_(False)`，features 全部在 `no_grad` 中计算。buffer 保存冻结特征，PPO optimizer 只包含新 actor/critic。初始 actor 的最后层权重为零、bias 为既有 log prior，因此两者初始动作分布相同；不同输入维度意味着前层参数不逐元素相同。

## 训练与冻结依赖

正式训练：seed 260915；10 个已曝光虚拟成人，每轮各两条三天轨迹，共 40×20 条；场景 seeds 103201–103280。与 wide 相同 actor/critic 128×2 Tanh、actor lr .0003、value lr .001、gamma .997、GAE lambda .99、clip .2、entropy .01、4 epochs、batch 512、gradient clip .5，无 teacher KL。每五分钟 reward 为 `−(HBG risk + 2×LBG risk)/(12×10)`；真实 native termination 末步另减 100，行政时限截断用当前 value bootstrap。技术失败中止并记录，不转成可学习 terminal penalty。

`world_control_worker.load_world` 是共享严格入口：只接受训练完成文件绑定的 best/last；检查 checkpoint→training provenance→source→P03/normalizer，检查 checkpoint 内嵌 frozen tensors，保持 world CUDA/eval/frozen。PPO 再核对 runner 固定的 checkpoint/provenance/completion SHA。正式配置拒绝 `budget_override=True`；smoke 配置允许**已经完成其显式 smoke budget**的 world，仅用于工程检查。两者都拒绝尚在运行的 best。默认路径是 `results/world_quantile_r1/best.pt`，不存在或未完成会直接失败；没有等待中途 best 或自动回退。

运行目录中的 `config.json` 是解析绝对路径和三项 SHA 后的有效配置，worker 必须读此配置，不直接读模板。`provenance.json` 保留 world 训练数据 manifest/NPZ hashes（引用其完整训练 provenance，不再读训练数组）、所有已用源码及独立 scorer SHA。`source/` 复制源码，`worker_ready.json` 保存 GPU/精度/冻结特征绑定。PPO checkpoint 沿用 wide 结构：`config, provenance, model, value, opt, iteration, simulator_transitions, rng_python, rng_numpy, rng_cpu, rng_cuda`。world 参数不复制进每个 PPO checkpoint，由固定依赖路径和 SHA 绑定。

runner 只生成训练/smoke场景；development/confirmation/world_validation seeds 均在 gate 拒绝。它不选择 actor；只存 40 个迭代 checkpoint，预先指定 8/16/32/40 后续作 development 对比。confirmation 必须另按冻结协议执行。`completion.json` 仅标记候选训练完成，不是收益验收。`trajectories/` 保留原 scorer 的原始轨迹，`history.jsonl` 保留更新统计，`failure.json`/`worker_stderr.log` 保留失败。新目录禁止覆盖；runner 暂不实现中断续跑。

Torch 线程 2、FP32、无 AMP；禁 `cuda.matmul.allow_tf32` 和 `cudnn.allow_tf32`，禁 flash/memory-efficient SDP，并禁 cuDNN benchmark；`cudnn.deterministic=False`，保持已测卷积吞吐，不声称逐位可复现。原仿真进程仍由旧 `ppo_env.environment_worker` 执行，meal-bolus/泵/传感器/评分源码不改。训练 actor 只看到 observed anchor；runner 不使用环境消息中的 nominal。

## 运行与验收

在项目根目录的研究主机运行。主 runner 必须使用 `.venv/bin/python`（Python 3.11 / NumPy 2.4.6，原仿真与评分环境）；它内部另启 `.venv-native/bin/python` GPU worker（原 Torch 2.0 / Mamba 1.2.2 环境）。不要把主 runner 放入 native 环境。本轮未连接或启动远端任务。

```bash
.venv/bin/python RL_DSENet_SOTA_2026-10-03/train_world_policy.py \
  --config RL_DSENet_SOTA_2026-10-03/configs/ppo_world_smoke.json \
  --name PPO_world_smoke_r1 \
  --world-checkpoint results/world_quantile_smoke_r3/best.pt

.venv/bin/python RL_DSENet_SOTA_2026-10-03/train_world_policy.py \
  --config RL_DSENet_SOTA_2026-10-03/configs/ppo_world_risk.json \
  --name PPO_world_risk_r1 \
  --world-checkpoint results/world_quantile_r1/best.pt
```

上例 checkpoint 名为显式占位/默认名字，以实际已经完成的运行目录为准。point 对照应另建配置，将 `world_variant` 改为 `point`、绑定已完成 point checkpoint，并使用新输出目录；不能把 quantile 声明配到 point 权重。现有两个配置仅支持正式/烟测固定预算，不自动放宽其场景门禁。

本地已通过两份源码 AST/编译、runner CLI help、28 项有效/无效配置及 seed/SHA/path 门禁检查，并核对 wide 的全部既有超参相同、195 维声明与未调用旧初始化；旧 wide worker/runner SHA 未变化。本机没有 NumPy/Torch，尚不能声称特征张量和 PPO 更新已在本机运行。CUDA 仍需核验：195 维；九条计划首 12 步与后 60 步正确；point fallback；单条/批量一致性；概率嵌套；world 梯度全部为空；每条训练 BG 数量等于 buffered actions；实际 policy/value 参数发生更新且 world 权重不变；写出 checkpoint 后严格重载得到同样 greedy action。必须保留失败记录，不能把以上待跑检查写成通过。
