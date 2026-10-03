# IQL_wide：新成人基础输注任务对照

这是从头训练的公开仿真任务适配 IQL，不依赖 DSENet 权重。训练集、表示、预算与当前 DSENet/world 候选仍有差异；相同评价任务不能替代全面公平性说明，也不构成官方本任务 SOTA 或临床证据。

## 已冻结设计

| 项目 | 定义 |
|---|---|
| 训练数据 | `natural_train_r1` 全部 120 条完整训练轨迹，加 `PPO_wide_risk` 前 8 个已完成迭代、每轮 20 条，共 160 条；合计 280 episodes |
| 数据选择 | 数量在开发集结果选择前冻结。读取 PPO 日志前 8 行并校验对应 checkpoint SHA；不要求等待第 40 轮，不使用第 9 轮以后轨迹 |
| 输入 | 展平 `72×22` 可观测历史 1584 维 + 原 `physiologic_features.features` 的 28 维 + observed warmup anchor / 5，共 1613 维 |
| 动作坐标 | `a_normalized = delivered_basal_u_h / anchor - 1`；[-1,1]。评估 `min(20, anchor×(actor_mean+1))` U/h，每 5 分钟执行 |
| anchor | 从 t360 的真实 warmup-delivered basal 标准化观测反解；不是隐藏 nominal、生理参数或患者 ID |
| reward | BG Kovatchev 高风险权重 1、低风险权重 2：`-(high+2×low)/120`；真实 native terminal 最后一步额外 -100；不使用平坦低糖 status_score |
| bootstrap | `r + .997×(1-terminal)×V(actual_next_history)`；时间限制结束仍保留真实最终 history 的 V |
| 模型 | 独立 actor、V、双 Q，各 2×256 ReLU；actor 均值 tanh，state-independent log_std clamp[-5,2]，初始 log_std=0；PyTorch 默认层初始化 |
| IQL | expectile .7；actor 权重 `min(exp(3×(min(targetQ1,targetQ2)-V)),100)`；双 Q 平方误差之和；target τ=.005 |
| 优化 | actor/Q/V Adam lr=3e-4；actor cosine decay 按固定 20,000 总步数，Q/V lr 不衰减；batch256，uniform transition sampling with replacement |
| seed / 预算 | 一个训练 seed260915；从头训练固定 20,000 updates = 5,120,000 sample visits；没有按开发结果挑 checkpoint |
| 保存 | 5k、10k、20k；正式评估 worker 只接受最终 `policy_020000.pt`。checkpoint 含模型、全部 optimizer、scheduler、Python/NumPy sampler/legacy RNG、Torch CPU/CUDA RNG |

自然轨迹的时间长度是 4,320 分钟，前 360 分钟只作 warmup，完整轨迹可供 792 个训练 transitions。PPO 真终止轨迹保留其实际长度和终止记录；技术失败、缺失 BG/CGM、异常时序、warmup 后无有效 transition 均拒收。不能将技术错误伪装成生理终止，也不能补齐未观测未来。

## 一手方法核验

已阅读作者 [implicit_q_learning 固定 commit 09d700248117881a75cb21f0adb95c6c8a694cb2](https://github.com/ikostrikov/implicit_q_learning/tree/09d700248117881a75cb21f0adb95c6c8a694cb2) 的本地存档：

- [learner.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/source_audit/iql/learner.py)：V → actor → Q → target Q EMA；actor 采用本步新 V 和更新前 target Q，cosine actor learning rate。
- [actor.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/source_audit/iql/actor.py)、[critic.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/source_audit/iql/critic.py)：加权 Gaussian log likelihood、expectile 回归和带 bootstrap mask 的 Q target。
- [policy.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/source_audit/iql/policy.py)：所选 `tanh_squash_distribution=False` 是“均值经 tanh 的普通 Gaussian”，不能加入 tanh 分布 Jacobian，也不能把加权 likelihood 换成确定性 MSE 后仍称同一 actor 目标。
- [mujoco_config.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/source_audit/iql/configs/mujoco_config.py)：2×256、expectile .7、temperature3、lr3e-4、τ=.005；本任务改 discount、输入、动作单位、奖励及数据。PyTorch 默认初始化不同于官方 JAX 实现，明确作为实现适配。

也对照了旧 [revision/iql_algorithm.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/iql_algorithm.py)。旧代码为缺未来的 Loop 样本设置 q_valid；本次只接纳有真实下一观测的完整 transitions，含真实终止最后一步，因此改用明确 terminal mask，不复制旧版本的无终止 Q target。旧文件不变。

运行依赖使用新研究目录 `vendor/iql/` 的 7 个作者档案（actor/critic/learner/policy、configs/mujoco_config、upstream.json、LICENSE）。由主任务从以上已核验本地档案等字节复制，保持各文件相同 SHA256 和固定 commit；prepare 的 `UPSTREAM` 指向这个新副本，不要求向服务器旧研究目录写文件。副本的所有哈希进入 replay/source provenance。

## Replay 与泵量化

[prepare_iql_replay.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/prepare_iql_replay.py) 从原始 records 调用原 `History`，按 minute=5,10,… 重建每个事件的数值、mask、age、known。一个 transition 的 history 是 t5..t360，动作是随后区间实际 delivered basal，目标为 t365 BG，next_history 是 t10..t365。最后一个真实状态保留，索引不会跨 episode。输入没有 BG、job metadata、CR、隐藏变量或未来餐次。

存储为紧凑 frames20、physiology28、window-end start 索引和 transition 数组，避免重复保存 72 帧窗口。`episodes.json` 保留 raw 路径/SHA、job 元数据、frames/transitions 起止、anchor、终止/时间限制、量化与边界微夹记录。患者/场景 ID 仅在审计元数据中。训练和评估共享 `encode_observations`，预计算 physiology 绑定同一源码和 normalizer 哈希。

泵源码由主任务从原仿真环境只读核验：Insulet `U2PMOL=6000`、`inc_basal=.05`，对应 U/h 量化步长 .0005、最大舍入误差 .00025。绑定配置中的 pump.py SHA `54dc4ec98f34ae600dacd1edd2e43fc2701512eb0824ce33acb07f3f756a9719` 与 pump_params.csv SHA `9b92646519f5501869626164f98e59cdc9303318958df78d7a670de1d023df6f`；prepare 会真实读取这些本地原环境文件核对，不导入或运行模拟器。

每步都验证 `delivered = round(requested/60×6000/.05)×.05/6000×60`，数值核验 atol1e-8 U/h。请求只允许 1e-6 U/h 浮点边界误差；泵舍入容差 .0002501 U/h。如果真实 delivered 导致 normalized ratio 微超 [-1,1]，仅在该物理容差内夹取，并逐条保存原动作、请求、夹取后动作、target minute。大投影直接拒收；归一化后 float32 的普通表示舍入与这一显式边界夹取分开。

数据 manifest 包含所有选中 raw SHA、自然采集 manifest/summary、PPO config/provenance/归档源码、8 个 PPO checkpoint SHA、日志前 8 行的独立哈希、所有 replay 数组、episode metadata、配置、feature/history/scorer/pump/作者源码及 license 哈希。训练再次核对 replay 数组和 source/feature 哈希；运行输出另存源码副本与完整数据 provenance。原 raw、评分器及训练旧权重均只读。

## 使用接口与验证状态

在项目根目录运行：prepare 用 `.venv/bin/python`（Python 3.11、原 scorer/NumPy）；train/worker 用 `.venv-native/bin/python`（Python 3.8、CUDA Torch）。旧评分的逐字段核验要求原 NumPy 运行口径。配置必须是绝对路径。源码已作 Python 3.8 语法审查，不使用 `Path.is_relative_to`。

```text
.venv/bin/python RL_DSENet_SOTA_2026-10-03/prepare_iql_replay.py --config /absolute/project/RL_DSENet_SOTA_2026-10-03/configs/iql_wide.json
.venv-native/bin/python RL_DSENet_SOTA_2026-10-03/train_iql_wide.py --config /absolute/project/RL_DSENet_SOTA_2026-10-03/configs/iql_wide.json --steps 4 --name IQL_wide_smoke
.venv-native/bin/python RL_DSENet_SOTA_2026-10-03/train_iql_wide.py --config /absolute/project/RL_DSENet_SOTA_2026-10-03/configs/iql_wide.json
.venv-native/bin/python RL_DSENet_SOTA_2026-10-03/iql_wide_worker.py --checkpoint /absolute/project/RL_DSENet_SOTA_2026-10-03/results/IQL_wide/policy_020000.pt
```

默认 device=cuda；显式 `--device cpu` 可作受控检查，没有隐式 fallback。smoke 只允许恰好 4 steps 和独立名字，写 `smoke_step4.pt`，其评估需显式 `--allow-smoke`，不会标为正式训练完成。

native train/worker 在导入 NumPy/Torch 前明确覆盖 `OMP_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`；训练与推理禁用 matmul/cudnn TF32，使用 FP32，runtime 写入 provenance/ready。worker 逐项核对 checkpoint 内嵌 config/provenance 与同目录 JSON，并在 ready 返回这两个文件的 SHA256。

worker stdin 接受 `{"op":"evaluate","history": [...],"anchors":[...]}`，返回 `actions` 与 `actions_u_h`（同一 U/h 列表）、`normalized_actions`、`global_cap_applied`、`batch_seconds`。确定性评估不抽样、不更新模型/缓冲。`op=close` 退出。新 evaluator 的集成由另一 agent 负责，本文不声称已接通。

本机已完成：4 个 Python 文件 AST/compile；利用旧真实 records 的只读内存副本验证 1613 维编码、预计算与即时 feature 一致、t360→t365 对齐、episode 边界、实际泵公式、时间限制/真终止、技术失败拒收、大投影拒收、BG 修改只影响 reward 不影响输入。另对 10 个 BG 点及 terminal penalty 与现有 PPO `risk_components` 函数逐值核对完全一致；合成微边界 fixture 的 normalized 超界约 1.406e-8 被明确记录并夹取。没有修改原轨迹。

**尚未完成：**正式 280 例 replay 生成、Torch 有限损失/梯度/参数更新检查、远端 smoke、20k 正式训练、完整新场景评估。本机无 Torch/simglucose，未安装、未连接远端、未进行训练或模型推理。本文件不提供任何 IQL 成绩。
