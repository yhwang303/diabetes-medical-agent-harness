# 条件性 planner BC 工具：已实现，未激活

本次只新增 `planner_bc_common.py`、`prepare_planner_bc.py`、`train_planner_bc.py`、`check_planner_bc.py` 和本文。没有修改协议、world、PPO worker/runner、evaluator 或队列，没有运行真实数据准备、BC、PPO、仿真或开发集评估。遵守 [条件性设计](planner_actor_followup.md)：等待当前 world-PPO 完成固定预算和固定 checkpoint 面板；它若满足原目标，就不启动本组实验。root 已报告 wide-PPO40 改善、现有 world-PPO 后期尚待判断；这些不由本工具重新解释成提前训练理由。

另据 root 后续结果，median-only/expected/risk 三种同 quantile world 的 MPC 都已有零 TBR54，因此不能把“消除 <54”独归于事件代价或概率积分。teacher 仍按已选设计固定 `risk`，不据这些结果进行新一轮模式搜索。

## 工具的实际路径

**准备程序**只接受已经完成 4000 步的正式 quantile world 和该 world provenance 原先绑定的 natural train pack。它先校验 world/P03/normalization/完成状态/source SHA，再核对 pack manifest、NPZ 文件顺序与 SHA、原 collector manifest/summary、原始 episode SHA 和全部 120 个 patient×seed×bolus 组合。固定场景只有 103001–103004，额外 PPO train 场景也不接受。smoke、world_validation、development、confirmation 在 NPZ 数组读取前被拒绝。

原始 JSON 的 job、状态和完整度只用于来源核对/审计；history 重建只使用每个起点之前的 minute、CGM、实际 basal、已记录 bolus/meal。现成 pack 只取 `history/anchor_u_h/origin_minute` 三个数组，与原 `History` 逐起点精确核对；不访问 future CGM/BG、future actions、mask 或 label-only 未来字段。全体已有合法 origin 都保留，不能按未来低糖或终止删样本。原 pack 自身要求至少存在一条后续记录，其现有末端边界不在本工具中扩大。native 提前终止的合法历史照常保留，技术失败拒绝。

同一个原 `ppo_world_worker.Trainer` 加载同一冻结 world，原样调用 `features(history,anchor)` 得到 195 维。teacher 另复用 `world_control_worker.candidate_plans/score_predictions(mode='risk')`；没有复制或重写风险公式。九条 60/120 分钟 teacher 计划先选原 argmin，再以 `[4,0,2,6,8,0,2,6,8]` 映射原 actor 类别；逐元素验证 cap 后实际首动作相同。195 维里的计划仍是 actor 的九条 60 分钟计划，未换成 teacher 的计划。准备过程校验 world、actor、value state SHA 前后相同，原 PPO optimizer/buffer/iteration/transition 均未动。

cache 每个来源 episode 一个 NPZ，保存冻结 features、teacher plan/action index、实际动作、九个 teacher cost、anchor 和 origin；文件顺序、job、source raw/pack SHA、fit/internal-check 分区和完整来源轨迹标记进入 manifest。完整来源轨迹在本工具里明确指 raw `failure_reason=None` 且最后 minute 等于该 job 的总时长；它只影响内部检查“三条完整来源轨迹”证据门槛，不改变 fit 权重或剔除截尾样本。

**BC 程序**只在 103001–103003 的缓存 features 上更新 actor；103004 的全部 episode 只做训练内部克隆检查。三遍全量、batch512、Adam3e-4、clip0.5、label smoothing0.05 和 seed260915 均为常量，没有改预算或挑中间 epoch 的 CLI。actor/critic 结构和初值来自原 Trainer；world/P03 和 value 不更新，只有独立 BC optimizer 更新 actor。使用原 Trainer 的真实 state hash 和 runtime/provenance 核对缓存的生成模型，拒绝换模型使用旧缓存。

内部检查保存完整 logits/概率/实际动作、origin 和原文件顺序索引，以及逐轨迹报告。门槛按 **cap 后 U/h 动作一致**计算，绝对容差 1e-6 U/h；类别 index agreement/recall 另报，避免在大 anchor clipping 别名下把同一个实际动作误判为不同给药。要求总体 ≥90%、teacher 非 hold 子集 ≥80%、非 hold ≥100 个状态且覆盖 ≥3 条完整来源轨迹。浮点 NaN、缺少非 hold 证据和全 hold 假阳性不能过门槛。这个检查不是独立 world 验证或医学标准，重叠窗口不独立。

三遍结束无论门槛是否通过，都保留 `bc_final_candidate.pt`、BC optimizer 状态、`bc_history.jsonl`、`internal_check.json/npz` 和实际 BC provenance。门槛通过且原 Trainer 真重载验证成功，才另外导出字节相同的 `bc_init.pt`；失败则保留候选而不导出初始化别名，退出码 2。技术错误写 `failure.json` 并停止。所有 cache/run 目录及文件排他创建，失败后也不覆盖。

## 原模型加载兼容与尚未开放的部署入口

导出 state 保留原字段：`config/provenance/model/value/opt/iteration/simulator_transitions/rng_python/rng_numpy/rng_cpu/rng_cuda`。`iteration=0`、`simulator_transitions=0` 是事实；原 PPO actor/value Adam 均为空，BC optimizer 另存 `bc_optimizer_state`，不可当作 PPO optimizer 导入。`bc_initialization` 明确记录 `behavior_cloning_only`、epoch/update 数、实际 state SHA、来源和 gate，`true_RL_training_performed=False`。

`config.json` 是原 Trainer 能解析的后续 PPO **目标配置/模型合同**（含正式 40 轮设置、世界模型绑定与输出目录），不是已完成 40 轮的声明。实际训练方法与预算在 BC provenance/completion 中明确为 BC 三轮、PPO 零轮。state 内的原 `Trainer.provenance` 为了严格加载保持原 schema；其中 controller family 名称不能用来把 BC 行叫作 RL。

训练代码会实际执行 `Trainer(same_BC_run_config, bc_final_candidate.pt)`，核对重新加载后的 world/actor/value 全部 state SHA 完全相同，全部 103004 cached feature 的 actor logits 逐元素误差 ≤1e-6，并验证 iteration/count 为零、原 PPO optimizers/buffer 为空。它复位并保存同一个 seed260915 的交接 RNG。**这个 CUDA 重载检查已写入流程，尚未实际运行，不是当前验证成果。**

现行 evaluator 要求完整正式 PPO history 前缀及 8/16/32/40，因此会且应该拒绝此 iteration0 BC 文件。不得伪造 `policy_iter08.pt`、history 行、PPO completion 或真实 transition。`train_world_policy.py` 也没有接受 BC 初始化的入口，当前两个命令都不会启动它。

如后续另行授权激活，最小接入路线是：

1. 为 **BC-only** 加独立 method/worker 与 eval gate，核对 BC completion、gate、cache/source/world/P03/config/checkpoint SHA 和 `iteration=0/count=0`；部署仍为原 features 加 actor argmax，无 teacher 重排或接管，统一走原模拟器/评分。需要显式展示为 BC，不能复用 PPO 名称绕过门禁。本次未改 evaluator。
2. 为 **BC+RL** 建新 runner/worker，先按新 run config 创建原 Trainer，再严格检查导出 BC artifact，导入相同 actor tensor；保留全新原 critic/Adam、空 buffer、iteration0，并记录 BC 父 checkpoint/state SHA。原 checkpoint config 包含旧 BC 输出路径，因此**不能**直接将其交给 `Trainer(new_run_config, checkpoint)` 假装是原地恢复；应由新 adapter 显式完成已审计的 actor 初始化。参数合同、world、特征及原 Trainer provenance 必须一致，只允许声明的新输出路径/名字变化。随后才允许原 40×20 真 BG PPO，另行冻结来源，不使用 teacher reward 或 BC 样本充当 PPO rollout。
3. BC-only 固定终点和 BC+RL 的 8/16/32/40 分开评估；BC+RL 无超越 BC-only 的有效闭环增量，就不能声称 RL 收益。新 method 的评估/confirmation 授权和证据绑定由主任务继续控制。

## 命令与当前验证

以下准备/训练命令只供**将来主任务判断需要并授权激活后**使用；本次未运行。两者均使用 native Py3.8/Torch2.0 CUDA 环境，没有模拟器主进程，不能误说已完成原 `.venv` 仿真评估。使用实际绝对世界权重/数据路径和全新名字：

```bash
.venv-native/bin/python RL_DSENet_SOTA_2026-10-03/prepare_planner_bc.py \
  --natural-train /absolute/project/RL_DSENet_SOTA_2026-10-03/cache/natural_train_windows_r1 \
  --world-checkpoint /absolute/project/RL_DSENet_SOTA_2026-10-03/results/world_quantile_r1/best.pt \
  --name PPO_world_bcinit_r1

.venv-native/bin/python RL_DSENet_SOTA_2026-10-03/train_planner_bc.py \
  --cache /absolute/project/RL_DSENet_SOTA_2026-10-03/cache/PPO_world_bcinit_r1
```

运行后 cache 在 `cache/PPO_world_bcinit_r1`，BC 产物在 `results/PPO_world_bcinit_r1`。不会自动触发 PPO、评估、confirmation 或队列，不支持 smoke 改预算、重采样、恢复覆盖或自动重试。

本地已运行 `check_planner_bc.py`：66 项纯 NumPy/metadata fixture 检查通过，涵盖真实计划构造函数的 60/120 分钟映射、大 anchor clipping、既有 teacher 评分、train 数据门禁、世界训练文件顺序、cache SHA、固定 gate 边界、忠实 actor/塌缩 actor 假数据，以及四份 Python3.8 AST。prepare/train CLI help 通过。本检查没有导入 Torch，没有读取真实训练数据，没有梯度更新、真实模型调用、仿真或远端操作。尚未验证真实数据全链路、CUDA 梯度或原 Trainer 实际重载；后续不能把这些待验项写成通过。

可以单独复跑无训练机制检查：`python check_planner_bc.py`（需 NumPy，路径在研究目录内）；可选 `--output` 写一个项目内全新 JSON，否则只输出 stdout，不留下伪装为正式结果的 fixture 目录。
