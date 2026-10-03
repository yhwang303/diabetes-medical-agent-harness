# 评价入口与同权重均值部署（2026-10-03）

`ppo`、`world_ppo`、`ppo_mean`、`world_ppo_mean` 必须传**原训练运行目录的 `config.json`**，与 `policy_iterNN.pt` 在同一目录。`configs/ppo_wide_risk.json` 等是训练模板，不能再作为评价配置。入口核对运行配置、训练 provenance、源代码、目标迭代为止的完整 `history.jsonl` 前缀和检查点 SHA；训练可继续追加后续记录，无需等满 40 轮。

原 `ppo`、`world_ppo` 继续执行 argmax。新增 `ppo_mean`、`world_ppo_mean` 使用同一已训练 checkpoint，先取得原 actor 的九个概率，再逐项计算 `min(20,anchor×倍率)` 的实际动作，最后执行概率加权均值。它们是同权重部署方式消融，**不是新增外部算法、不是重新训练，也不是原预注册项**。提出背景是根任务已观察到 PPO08 development 全部 47,520 个动作与 hold 一致；“argmax 抹去有用概率变化”仍只是待检验假设。

正式评价沿用原 family 的完整门禁：PPO 只允许正式训练配置及预定第 8/16/32/40 轮，world 依赖须完成 4000 步且 `budget_override=false`，confirmation 仍拒绝。工程 smoke 可以使用正式权重；smoke 权重只允许显式 `--smoke`。下例从项目根目录运行，主入口使用仿真环境 `.venv/bin/python`，内部自动以 `.venv-native` 启动 `ppo_mean_worker.py --family wide`：

```bash
.venv/bin/python RL_DSENet_SOTA_2026-10-03/evaluate_candidates.py \
  --kind ppo_mean --name SMOKE_PPO16_mean_r1 --smoke \
  --checkpoint "$PWD/RL_DSENet_SOTA_2026-10-03/results/PPO_wide_risk/policy_iter16.pt" \
  --config "$PWD/RL_DSENet_SOTA_2026-10-03/results/PPO_wide_risk/config.json"
```

正式 development 使用另一个新输出名并去掉 `--smoke`。`world_ppo_mean` 同理，传它自己的原 world-PPO 运行配置和检查点；入口自动选择 `--family world`。不得把 wide 权重或未完成 world 依赖换进该路由。

每个新 mean 运行独立保存 manifest、源码、训练历史前缀和决策审计，不覆盖原 argmax 面板。审计包含九个概率、九个实际动作、执行均值、同状态 argmax、hold 概率、前两名概率差、熵及均值与 argmax 的差。评价器独立重算概率和、动作网格、加权均值及这些统计；先平均倍率再限幅、冒用 argmax 动作、source/iteration 不符均拒绝。原 family 编码器将 anchor 转为 FP32，审计按该实际编码精度还原网格，softmax 和均值统计使用 FP64。

本地证据见 [mean_evaluation_mechanics.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/checks/mean_evaluation_mechanics.json)：47 项新增机制检查、73 项底层门禁检查、58 项 IQL 回归检查通过。它们使用元数据 fixture 和人工 logits，不代表真实 checkpoint 推理或控制收益。真实 CUDA 与两条完整 smoke、正式 development 由根任务另行执行，结果须与同 checkpoint 的 argmax 配对解释。

新规则与限制详见 [ppo_mean_design.md](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/ppo_mean_design.md)。均值部署没有新增刹车、温度、探索噪声或 reward；在非线性动态系统中，离散策略的期望结局不等于执行平均动作的结局，收益必须由真实闭环评价确认。
