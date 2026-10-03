# 同权重 PPO 概率均值部署消融

这是在完整 PPO08 development 结果已经曝光后提出的新机制实验，**不是原预注册项**。root 提供的观测是 60 条轨迹、每条 792 个部署动作，共 47,520 个动作及全部 BG 指标与 hold 一致。本批只读源码，并未在本地重新核验全部原始轨迹或已训练 actor 的逐状态概率；“argmax 抹去了有用概率变化”保留为假设，不能由 hold 一致性直接证明。

源码事实是：wide 和 world PPO 都用 `trainer.policy`（128×2 Tanh 后接 9-logit 线性层），没有另一个隐藏 actor/head。原 `act(evaluate=True)` 用 `logits.argmax(-1)`，采样仅用于训练。二者的特征入口均为 `trainer.features(history,anchors)`，倍率 tensor 均为 `[0,.25,.5,.75,1,1.25,1.5,1.75,2]`。world 的 195 维特征仍会逐次重算九条动作计划；wide 的 334 维特征原样保留。

## 新部署规则

对于同一个实际历史和 anchor，取得原 actor 的 logits，并令 `p=softmax(logits)`。先按原离散动作的实际映射计算 `a_k=min(20,anchor×ratio_k)`，再执行连续基础率 `a_mean=Σ p_k a_k`，单位 U/h。这里没有温度、阈值、探索噪声、教师、低糖刹车或新 reward。

必须先对每个动作做原 20 U/h cap，再平均。比如 anchor=20、0 倍和 2 倍各 0.5 概率时，实际动作网格是 0 和 20，执行均值应为 10 U/h；先平均倍率再 cap 会得到 20 U/h，这是不同规则。当前成人较小 anchor 下二者可能相等，不能因此省略大 anchor 边界。`expected_ratio`、`raw_ratio_mean_rate` 只作审计，不作为执行值。

logits 和特征仍采用原 family 的真实 FP32 前向及 backend flags；wrapper 不额外修改 wide 的 TF32 状态，也不改 world 的既有禁 TF32 设置。九个 logits 传到 CPU 后用稳定 softmax 和 Python float64 累加，降低概率均值的舍入误差。只允许≤1e-12 U/h 的浮点边界修正并记录；不是新增临床保护逻辑。返回的概率为这些原 logits 的 FP64 softmax，不声称和原 `Categorical` 的 FP32 概率逐位相等。

原 argmax 相同并不要求均值相同。例如 hold 概率都是 .6，0 倍/2 倍分别为 .3/.1 或 .1/.3，argmax 都是 hold，未触及 cap 时期望倍率分别为 .8 和 1.2。但如果尾部概率对称，均值也可能仍是 hold；没有真实概率记录前不能认定是哪种情况。均值也可能抵消一个有用的多峰动作分布。

在非线性、有时滞和状态依赖的系统中，`E[f(a)] ≠ f(E[a])`。随机执行离散策略的期望结局、确定性执行概率均值的结局和 argmax 策略的结局通常不同。PPO 训练并未直接优化这个连续均值部署规则；固定同一 checkpoint 并不使两种部署等价。必须在相同可观测信息、泵、餐时 bolus、场景和 scorer 下分别运行真实仿真闭环。

## 文件与接口

仅新增 `ppo_mean_worker.py` 和本说明；原 wide/world worker、训练脚本及权重不改。worker 支持：

```text
.venv-native/bin/python <R>/ppo_mean_worker.py --family wide|world \
  --config <absolute PPO run>/config.json \
  --checkpoint <absolute PPO run>/policy_iter08.pt
```

JSON 请求只接受 `{"op":"evaluate","history":...,"anchors":...}`。任何 act、update、close、indices、BG、患者身份或未来信息字段均拒绝；关闭 stdin/EOF 退出。不会调用原 Trainer 的 `act` 或 `update`，不会抽样动作，也不会写训练 buffer。

加载时直接调用指定 family 的原 `Trainer(config,checkpoint)`，因此 config、训练 provenance、模型依赖及 state_dict 校验保持原样。额外核对同 run 的 config/provenance 和完整 `history.jsonl` 前缀：迭代必须连续，最后一条明确绑定该 checkpoint 的绝对路径和 SHA。只冻结目标迭代之前的前缀，后续训练可以继续追加。加载后再次核验该前缀和文件 hashes，并核对原 actor 的输入维度、9 输出 head 和倍率 tensor。

原 Trainer 初始化为了严格恢复会暂时构造并加载 optimizer 和 RNG 状态；wrapper 随即删除 optimizer 与空 buffer，冻结 policy/value/world。后续仅计算 features→policy→softmax→动作均值，没有训练入口。ready 保留原 `provenance`，另给 family、iteration、checkpoint/config/provenance/history-prefix SHA、新/原 worker SHA、原 backend 状态和新部署规则。父 eval gate 可据此继续严格绑定；wrapper 不导入 Python 3.11 的 evaluator，也不自行决定正式数据资格。

每次返回以下批量审计字段：

- `actions` 和 `action_mean`：同一最终连续 U/h 动作，供执行器读取。
- `probabilities`、`actual_action_grid_u_h`：九个概率及已逐项 cap 的执行网格。
- `argmax_indices`、`argmax_actions_u_h`、`argmax_probabilities`：原规则在同一状态下的动作；索引只是最高概率，不称医学最优动作。
- `expected_ratio`、`raw_ratio_mean_rate`、`mean_minus_argmax_u_h`。
- `hold_probability`、`probability_margin_top2`、`entropy`、`arithmetic_boundary_adjustment_u_h`。

正式比较仍只允许原候选集合 8/16/32/40，由 eval gate 校验完整正式配置和 checkpoint 前缀，不能因为看见某次均值消融结果再扩大 checkpoint 搜索。smoke 的预算与数据资格也由 gate 明示。新增部署规则须独立方法名、manifest、源码 SHA、审计行和结果目录，不能覆盖原 argmax 面板或事后改称原预注册方法。

合理解释是“同一个训练权重的部署方式消融”。wide 若改善不能称 world 贡献；world 家族若改善也必须与同 checkpoint 的 argmax、以及 wide 的相同部署规则分别比较，才能讨论关联。一次新部署闭环改善本身不能证明 world 特征的独立因果贡献。

## 本批验收边界

本地检查 Python 3.8 语法和 CLI，纯函数机制检查覆盖同 argmax 不同期望、单峰退化、概率/动作范围、大 anchor 先 cap 后平均、非法请求和非有限 logits。纯函数测试使用人工 logits，不等于真实 checkpoint 已加载或闭环收益已验证。root 负责原生 CUDA 权重加载及 eval 集成；本批不连接远端、不训练、不执行 development/confirmation 仿真。
