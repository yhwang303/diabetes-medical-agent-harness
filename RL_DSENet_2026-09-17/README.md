# DSENet × RL-DITR / Loop 研究实验

当前版本入口：`revision/论文正文对比报告.html`；固定版本指纹见`selected_version.json`。原`训练审核报告.html`保留为训练过程记录。这是公开数据/公开仿真研究，不是 Harness 产品接入或真实患者给药模型验收。

## 数据与模型

- DSENet 上游固定为 `560712e7d8e01f25a5dad3262da786f4a0ae52de`，MIT 许可随 `dsenet/` 保留；双流、双向 GateMamba、LoRE、Router、Fusion 不变。
- Loop 患者先划分；225 人训练、55 人验证。每 epoch 保留原 1,653,421 训练起点。稀疏食物/运动不导致删样本；DSENet 使用 CGM，外部冻结历史编码器保留多模态上下文。
- 历史72点、短流36点、预测48点，每点5分钟。四个 patch 超参数搜索见 `checks/patch_selection.json`；按固定验证分数选择 P03：global length12/stride3，local length2/stride1。3 epoch 初筛，不能称全局最优。
- `world_model.py`：冻结选中 DSENet 与已有 H02 历史编码器；旧血糖/reward/value头不参与。参考适配器用实际执行参考动作的分支监督，响应核拟合相同状态的动作差效应。
- `policy_bounded.py`：4小时7候选计划，围绕一次读取的实际观测基础率 ±0.25 U/h；每5分钟只执行首步。该范围是研究动作空间，不是医学安全界限。
- RL 目标为世界模型回报的精确有限候选期望，减0.05 KL正则；不是仅MSE，也不是原论文作者临床任务代码逐位复现。所有新训练仅 seed260915。

## 运行环境

远端项目根 `/root/autodl-tmp/diabetes-agent`，RTX4090，Python3.8 / Torch2.0.0+cu118 / Mamba1.2.2。训练及推理用 `.venv-native/bin/python`；官方 simglucose 用 `.venv/bin/python`。Mamba 官方 wheel 保存在 `dependencies/`，版本记录在 `checks/environment_and_sources.json`。本地 Apple Silicon 不作为该 CUDA 模型的性能运行环境。

## 复现顺序

训练输出目录均 `exist_ok=False`，避免覆盖历史证据；重复实验应显式新命名，不盲目重跑已有阶段。

1. `prepare_forecast_masks.py` 从冻结患者索引生成事实 CGM 标签 mask；`check_forecast.py` 做真实CUDA/缺失/归一化检查。
2. `train_forecast.py --name <新名字> --model-config configs/P03_g12s3_l2s1.json`：3完整epoch。所有搜索配置及结果都保留，不只保存最好的一组。
3. `train_world.py --name <新world> --forecast results/<预测运行>/best.pt`；随后 `evaluate_world.py --name <新world>` 和 `check_policy.py --world <新world>`。
4. `train_policy_bounded.py --world <新world> --name <新policy>`：固定world、完整Loop一轮。`config.json` 绑定world路径/hash，world配置进一步绑定forecast/hash。
5. `evaluate_control.py` 支持相同仿真合同下的 hold、planner、actor、beam、legacy、ditr。新的planner/actor/beam必须使用 `--bounded-reference`；planner传world.pt，actor/beam传policy.pt。
6. 最终评测前 `freeze_final.py` 冻结依赖、代码及场景；`run_final_controls.py --queue baselines|ditr|ours` 只复用冻结权重。Loop sealed_test由独立 `prepare_final_forecast.py` 和 `evaluate_final_forecast.py` 在冻结后打开，不允许训练loader访问。
7. `analyze_control.py` 重新计算每条原始轨迹、校验方法间场景匹配、患者配对bootstrap，导出 CSV/LaTeX。`build_report.py` 生成离线 HTML；报告数字直接读JSON，不手工填数。

## 推理合同

`bounded_policy_worker.py --checkpoint results/D06_selected_policy/policy.pt --mode actor` 为持久 JSON 行 worker。输入 `history` 是 `[batch,72,22]` 的原冻结 Loop 特征，`anchor_u_h` 是每个病例预热结束最后实际基础率组成的数组。不能将未来BG、未来餐食或隐藏患者参数放进请求。输出 `actions_u_h` 是基础输注率，不能解释为一次注射剂量。

模型需要新目录的 world/forecast 权重、原 H02 context checkpoint、原 normalization 及代码依赖；不能只复制policy.pt就称完整可运行。最终依赖见 `configs/final_freeze.json`。

## 证据边界

- 旧 baseline 没有重训，只核验原权重并按统一新场景重新评价。
- 新模型有额外仿真监督及不同动作限制；跨系统表不等于纯算法同预算消融。同world的planner/actor/beam才是对应的策略对照。
- 10位虚拟成人以前已曝光。最终控制是冻结的新场景评价，不是未见患者外推；32位Loop留出患者只证明预测泛化，不能证明给药效果。
- TIR终止后给未知结局上下界，TBR仅观测阶段，不伪造血糖尾段。区间以患者为簇，不包含训练seed随机性。
- D03连续重定中心导致严重低糖，完整保留。D04修订和P03选型均在最终评价前完成，不用最终结果回调模型。

论文表位于 `paper_tables/`，LaTeX采用常见table/table*及booktabs格式（主文需加载 `booktabs`）。所有限定需连同表注一起保留；不能仅复制较好数值而删去提前终止、监督差异、单seed或个体风险。
