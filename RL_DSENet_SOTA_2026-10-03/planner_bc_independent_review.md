# 条件性 planner BC 工具独立审查

日期：2026-10-03。审查角色：独立只读审查。范围为 `planner_bc_common.py`、`prepare_planner_bc.py`、`train_planner_bc.py` 及必要的原计划、特征、checkpoint 接口；本次仅新增本文。

**结论：在本次源码审查和下述无模型机制检查范围内，未发现需要阻断准备阶段的实现错误。候选仍未激活；这不是对真实 BC 训练、闭环效果或后续实验启动的批准。** 数据准备、CUDA 梯度与原 Trainer 实际重载尚未运行，不能将已写入代码的检查当成已经通过的结果。

## 审阅源码快照

写入本文前重新读取以下文件字节，三份核心源码仍与独立审查时一致。

| 文件 | SHA256 |
|---|---|
| [planner_bc_common.py](planner_bc_common.py) | `fde885e892056f049785e7caa1e79d8d9c068e85ba4eb75e4401fc72a97c164e` |
| [prepare_planner_bc.py](prepare_planner_bc.py) | `bb81b391e9df5a8a33c73acab39d6ab2b5a929aebb2a635598013f460b7e4828` |
| [train_planner_bc.py](train_planner_bc.py) | `42ee042565a3251edbfb2d9f90d6923ae022096203a100853bc54ae09b3e742b` |
| [check_planner_bc.py](check_planner_bc.py) | `7789ea76a61e9edd35418f8ee1fe04a8eddf4827aadba22ccc0bdc85ac3ad25d` |

以下行号对应这组源码。原机制参照为 [paired_collect.py](paired_collect.py)、[ppo_world_worker.py](ppo_world_worker.py)、[world_control_worker.py](world_control_worker.py)、[prepare_windows.py](prepare_windows.py)；条件性研究边界见 [planner_actor_followup.md](planner_actor_followup.md)，工具状态与未来命令见 [planner_bc_tooling.md](planner_bc_tooling.md)。

## 已核对的实现事实

| 检查项 | 源码证据与结论 |
|---|---|
| train 来源与覆盖 | `planner_bc_common.py:72–159` 要求既有 world 绑定的 natural train pack、完整患者×103001–103004×bolus factor 组合、固定文件顺序，并核对 pack、raw、collector、协议和模型依赖 SHA。正式 quantile world 必须完成 4000 步且没有 smoke 预算覆盖。不是从 development、world_validation 或 confirmation 选示教状态。 |
| 历史时序与截尾起点 | `planner_bc_common.py:162–195` 仅取 NPZ 的 `history/anchor_u_h/origin_minute`，从 raw 的既有 CGM、实际 basal、已记录 meal/bolus 重建每个历史前缀并逐项比较。合法起点精确采用原 pack 的 `range(71, len(rows)-1, stride)`，没有按未来 BG、低糖事件或 mask 删样本。它继承原 pack 至少存在下一条记录的末端边界，不额外补最后一个无后续记录的状态。native 提前终止的合法历史保留；技术失败拒绝。 |
| anchor 与单位 | `planner_bc_common.py:183–191` 使用 warmup 末条记录的实际 basal U/h；向原 `History` 传入每 5 分钟的 basal 量时除以 12，与既有 pack 一致。输入精确比较不依赖隐藏状态或患者 ID。 |
| 计划到首动作映射 | `planner_bc_common.py:198–208` 明确映射 `[4,0,2,6,8,0,2,6,8]`，并核对 0–20 U/h clipping 后的实际首动作完全一致。九个 teacher 计划仅提供五种首动作标签；不能把九个计划编号直接当作九个 actor 类别。 |
| 195 维与计划时长 | `prepare_planner_bc.py:45–59` 先调用原 `Trainer.features`，保持 actor 的九条 60 分钟干预计划；另用原 teacher 九条 60/120 分钟计划生成 hard winner 标签。`ppo_world_worker.py:129–147` 的特征顺序仍为 context64、P03预测48、physiology28、anchor1、响应摘要54。没有把 teacher 选择结果作为部署输入。 |
| world/P03 冻结 | `planner_bc_common.py:100–159` 继承 world/P03/normalization 来源验证；`prepare_planner_bc.py:73–80` 核对 world、actor、value 状态前后 SHA，空 buffer、零 iteration/transition、空 PPO optimizer 及无梯度。`train_planner_bc.py:113–155` 核对实际加载状态与缓存来源一致，且只允许 actor 改变。 |
| 固定 BC 预算与内部划分 | `planner_bc_common.py:10–21` 固定三轮、batch512、Adam3e-4、label smoothing0.05 和 seed260915。`train_planner_bc.py:118–159` 仅用 103001–103003 更新 actor，完整 103004 场景只在最终权重上做一次内部检查，不选择中间 epoch。 |
| 防止 hold 高占比假阳性 | `planner_bc_common.py:211–222`、`train_planner_bc.py:59–95` 按 cap 后实际 U/h 动作计算 agreement，另报类别指标；要求 overall≥90%、非 hold≥80%、非 hold 状态≥100 且来自≥3 条完整来源轨迹。无非 hold 或 NaN 不能过关。完整轨迹标记仅作用于内部证据门槛，不筛掉 fit 样本。 |
| iteration 0 与实际重载流程 | `train_planner_bc.py:160–198` 保存真实 `iteration=0/simulator_transitions=0`、独立 BC optimizer 状态和交接 RNG。代码安排由原 Trainer 使用同一 config 重载候选，核对全部模型状态 SHA、全部 103004 logits 和空 PPO 状态。只有该检查成功且内部 gate 通过，才复制字节相同的 `bc_init.pt`。未伪造第 8 轮或 PPO transition。 |

## 本审查实际执行的检查

独立检查使用本地 bundled Python/NumPy，以 `PYTHONDONTWRITEBYTECODE=1` 运行内存脚本。仅从 `paired_collect.py` 的 AST 提取真实 `plans` 函数；未导入整个 collector、未启动模拟器。

1. 用 `np.linspace(0.0001, 20, 10001, dtype=np.float32)` 生成 10,001 个合法 anchor；对全部九个计划逐一调用原 `map_teacher`，核对映射类别和 clipping 后实际首动作，共 **90,009 个组合通过**。这些是合成数值组合，不是患者样本或模型预测。
2. 对上述四个 Python 文件执行 `ast.parse(..., feature_version=(3,8))`，**四份 Python 3.8 语法检查通过**。这只验证语法，不证明 Python 3.8 环境中的所有依赖/API/CUDA 操作可运行。
3. 断言 `torch` 未进入 `sys.modules`。检查未写文件、未读取真实训练数组，也未执行模型推理、梯度更新、训练、仿真或远端操作。

实际 stdout 为：

```json
{"review_check":"passed","mapping_cases":90009,"python38_ast_files":4,"torch_imported":false,"files_written":false,"real_data_model_training_or_simulation":false}
```

实现代理另报告 `check_planner_bc.py` 的 66 项 NumPy/metadata fixture 通过。本审查读取了该检查源码，但未重新运行这组 fixture，不将代理报告混计为本审查独立执行的测试数量。当前没有真实 cache、CUDA 梯度或实际 Trainer 重载通过的独立证据。

## 科学限制与尚未接入部分

- **103004 是 BC actor 的训练内部检查。** 冻结 teacher world 已经接触过这组 natural train 场景；不能称为独立 world 验证、独立患者泛化或闭环效果验证，相邻历史窗口也不独立。
- **首动作克隆不是完整计划克隆。** teacher 比较 60/120 分钟计划，但 actor 的 195 维只显式包含自身 60 分钟计划的响应。该信息差能否在固定表示和三轮预算下被学到，需要真实 gate 检查；不能预设无损提取。195 维部署仍需 world 预测，不能据此宣称推理提速。
- **离线 agreement 不证明真实收益。** 即使内部 gate 通过，也没有证明 BC-only 保留 teacher 的闭环性能，更没有证明后续 RL 有增益。只有保留 BC-only 对照并取得真实配对闭环证据，才能分析 BC+RL 的额外贡献。
- **当前正式 evaluator 应继续拒绝 BC iteration 0。** 现有接口要求正式 PPO history 和固定 8/16/32/40 checkpoint。未来 BC-only 需要独立 method/worker 与门禁，核对 BC completion、内部 gate、真实重载记录及 cache/source/world/P03/config/checkpoint 绑定；实际计数保持零，部署为原特征加 actor argmax，不由 teacher 重排或接管。
- **未来 actor-init adapter 尚未实现。** 原 `train_world_policy.py` 没有 BC 初始化入口，原 Trainer 严格要求 checkpoint config/provenance 相同。将 BC checkpoint 直接交给另一个输出目录的新 config 会被拒绝。若后续激活，应先在新 run 中创建原 Trainer，验证 BC 父产物及同一参数/特征/world 合同后显式导入 actor；保留全新 critic、PPO optimizer、空 buffer 和 iteration0，并记录父 checkpoint/state SHA。不能以改名为 `policy_iter08.pt` 或伪造 history 绕过检查。

这些接入缺口不构成当前条件性准备工具的源码错误，但在真正运行 BC-only 评估或 BC+RL 前必须完成对应接入与实际验证。本次没有修改训练、评分、协议、evaluator 或队列，也没有据此激活候选。
