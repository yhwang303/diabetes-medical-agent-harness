# IQL 实现核验与任务适配

方法：Kostrikov, Nair and Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022。这是补充经典基线，不能把它称作2025/26新算法。

作者官方源：[implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning)，固定commit `09d700248117881a75cb21f0adb95c6c8a694cb2`。本地 `source_audit/iql/` 保存 `actor.py`、`critic.py`、`learner.py`、`policy.py`、`configs/mujoco_config.py` 与 MIT LICENSE。`iql_algorithm.py` 是最小PyTorch任务适配实现，不是声称逐字运行官方JAX。

保留的算法定义：

- 双Q取min；对target Q做expectile价值回归，expectile=0.7。
- 更新顺序严格按照作者 `learner._update_jit`：V → actor → Q → target-Q EMA。actor优势采用本步新V和更新前target Q。
- actor为均值经tanh的普通Gaussian，state-independent log_std，clamp(-5,2)；不是tanh变换后的分布。以数据动作Gaussian log likelihood训练，权重 `min(exp(3*(Qtarget-V)),100)`；推理使用确定性均值。
- Q目标为 `r + gamma*V(s')`，双Q平方误差求和；target EMA tau=0.005。Actor采用与原作者一致的cosine LR衰减，按本次固定50k总步数；Q/V固定Adam学习率。

共同设置/任务适配：

- 输入为既有Loop `rl_data.Replay` 的72×22=1584维历史，原训练split和标准化不变；输出[-1,1]映射0–20 U/h基础率。
- 隐藏层统一3×256；官方MuJoCo默认2×256。复用项目既有ReLU MLP和小输出层初始化；可使用原有 `initial_action_center` 使起始均值与其余baseline一致。这些容量/初始化变化必须在正文脚注或实现细节明确。
- b256、50k更新、seed260915；使用已有5min gamma和Loop奖励，不应用D4RL reward rescaling。配置默认actor/Q/V lr=3e-4。
- `q_valid` 为假的样本没有可用连续未来：仅排除其Q和V损失，**不伪造terminal**，全部训练起点仍进入Gaussian BC；无有效未来者actor权重设1，有效者采用IQL优势权重。该保留缺失未来样本的规则是Loop任务适配，不是原IQL处理完整D4RL transition的默认规则。
- 无额外模拟交互、无反事实世界模型、无新动作限制、无临床安全模块、无development/final结果选参。

验证分工：本地完成语法编译；实际CUDA有限损失/参数更新、无未来样本的BC更新及评估导出一致性由根任务统一运行。不能仅凭本说明标注训练或复现完成。
