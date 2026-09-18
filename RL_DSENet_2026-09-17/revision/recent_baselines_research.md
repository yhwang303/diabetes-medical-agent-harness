# 正文外部基线筛选与 LOM 实现依据（2026-09-17）

## 筛选结论

优先加入 **GFP（ICLR 2026）和 LOM（ICLR 2025）**，它们是真实外部方法，且分别补足 value-aware flow regularization 与 mode-selective imitation 两类策略学习机制。已核查会议原始出版页与作者源码；不是把本项目历史版本换名当作新论文。

| 方法 | 已核查正式出版 | 官方实现与固定提交 | 许可/依赖 | 本轮处理 |
|---|---|---|---|---|
| Guided Flow Policy (GFP) | [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html) | [Simple-Robotics/guided-flow-policy](https://github.com/Simple-Robotics/guided-flow-policy), `e60468e31066ed072e494be8c6cce14ac19f60e9` | MIT；JAX/Flax/Optax/Hydra，作者建议 Python 3.12 | 首选；可以保留算法，适配现有 PyTorch 数据合同 |
| Learning on One Mode (LOM) | [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/be62c4a943675195ff5a2a98d5b9724f-Abstract-Conference.html) | [MianchuWang/LOM](https://github.com/MianchuWang/LOM), `e7cf8a37b05adeb983af8c288abd06f1c37e9f40` | 仓库无 LICENSE；作者环境 Python 3.9 / PyTorch 2.3 / D4RL / MuJoCo | 按论文公式独立实现，不复制或整合无许可证源码 |
| MeanFlowQL | [AAAI 2026 proceedings, DOI 10.1609/aaai.v40i31.39885](https://ojs.aaai.org/index.php/AAAI/article/view/39885) | [HiccupRL/MeanFlowQL](https://github.com/HiccupRL/MeanFlowQL), `3fab100a40a1069e5e6f6fcd24e489a1998cc276` | MIT；JAX，核心 `agents/meanflowql.py` | 可复现候选，未选入本轮训练；不能写成复现失败 |
| Value Flows | [ICLR 2026 accepted paper](https://openreview.net/pdf/28c79ac88246f905c184aea9570b1b9c59092d18.pdf) | [chongyi-zheng/value-flows](https://github.com/chongyi-zheng/value-flows), `01833354f547ad842bdaa21d2b761a16069f9724` | MIT；JAX/Flax，作者环境 Python 3.10.13 | 具有额外分布价值模型及采样开销，暂不选入；未运行，不能声称不可复现 |

GFP 的 arXiv 首稿是 2025 年，但正式会议年份应写 **2026**；LOM 首稿是 2024 年，正式会议年份应写 **2025**。MeanFlowQL 是 **AAAI 2026**，不是 ICLR。以上日期均由正式会议页核实。

## GFP 忠实适配要点

已保存 MIT 源码审计快照 `source_audit/gfp/agents/gfp.py` 和配置 `config/agent/gfp.yaml`。

- 双 Q TD；目标动作为当前 one-step actor 的下一状态动作，target critic 计算 bootstrap；默认 `q_agg=mean`。
- Flow 行为拟合带价值权重：`w = sigmoid(lambda * (Q_target(s,a_data) - Q_target(s,a_actor)) / eta)`，其中 `lambda = stopgrad(1 / mean(abs(Q_online(s,a_actor))))`。
- Actor 目标：`alpha * E[(actor - flow_teacher)^2] - lambda * E[Q_online(s, actor)]`。权重和 teacher 不接收 actor-loss 梯度。
- Flow 目标：`E[w * ||v(s,x_t,t) - (a-z)||²]`，`x_t=(1-t)z+ta`；10 步 Euler 生成 teacher。
- 默认 `alpha=1`, `eta=1e-3`, normalized Q loss；原始流网络带 64 维时间编码。修改为原始标量 t 必须作为网络适配披露，不得称逐行作者实现。
- 原作者同步更新所有梯度后，再 Polyak 更新 target Q；不可先改 Q 再计算当前 actor loss 而称其完全等价。

原作者安装要求不等于 Loop 适配需要 MuJoCo 或新 CUDA。只在无数据/任务语义变化、梯度目标经过比对后，才可以用现有 PyTorch 环境报告“任务适配复现”。

## LOM：独立实现与论文对应

依据 [LOM 正式论文 Algorithm 1 / Section 5](https://proceedings.iclr.cc/paper_files/paper/2025/file/be62c4a943675195ff5a2a98d5b9724f-Paper-Conference.pdf)，实现位于 `lom_algorithm.py`，不导入下载的作者代码。

1. Eq. 15：state-conditioned Gaussian mixture 的负对数似然预训练；之后固定 GMM。
2. Eq. 16：behavior Q 使用数据实际后继动作 `a_next` 的 SARSA bootstrap。
3. Eq. 17：mode-Q 回归该 mode 采样动作的 Q 期望；每状态每 mode 一次 Monte Carlo 样本。
4. Algorithm 1：选择 mode-Q 最高的 mode，只从该 mode 抽取拟合动作。
5. Eq. 18：以 `min(exp(A / temperature), C)` 加权确定性 actor 的平方模仿误差；`A = Q(s,a_sample) - Q(s,mean_of_selected_mode)`。

最后一点严格按论文。作者公开代码实际上用当前 actor 动作的 Q 作为 advantage 基准，与论文所选 mode 均值的描述不同。此处不混称二者等价。

### 明确固定的任务适配

| 项目 | 本轮固定值 | 理由/边界 |
|---|---|---|
| seed | 260915 | 与既有训练约定一致 |
| GMM / RL 更新数 | 10,000 / 50,000 | 保留作者 200k:1M 的 1:5 阶段比例；与其他方法统一 50k TD 更新，另记 10k GMM 成本 |
| GMM | 512×2，10 mixtures | 保留作者 mixture 网络；M=10 在作者 full-replay 设置中使用，并非验证后挑选 |
| Q / mode-Q / actor | 256×3，无 LayerNorm | 统一隐藏规模；作者对应网络是 256×2，属于已披露容量适配 |
| Q ensemble | K=1 | 论文 Algorithm 1 和作者部分 full-replay 配置 |
| 温度 / 权重上限 | 0.2 / 50 | 等价于作者代码权重中的 multiplier 5；避免 beta 与 inverse-temperature 名称混淆 |
| next-action smoothing | std 0.2，clip 0.5 | 作者 full-replay 配置；仅用于 behavior-Q target，不改变真实动作记录 |
| GMM / mode learning rate | 1e-3 / 1e-3 | 作者设置 |
| Actor / Q learning rate | trainer 显式给定 | 统一实验配置；应在报告中记录实际值 |
| target update | tau 0.005，每两次 RL 更新 | 排除 GMM 步数来计数 |
| 状态/动作/奖励 | 既有 1584 维历史、单维 basal、既有奖励和 gamma | 不新增真实 BG、患者隐藏状态、未来餐或个体 ID |
| 缺失后继 | `q_valid` 屏蔽 TD 和 mode 学习；actor 对有效行用 LOM 加权 mode 误差、无效行用实际动作 BC 误差，按全部行平均；所有 origin 仍参加 GMM | gap/censor 不是终止状态，不以零 bootstrap 制造目标；保留所有 origin 的行为学习，明确这是 Loop 缺失数据适配 |

GMM 各组 `log_sd` 固定截断于 `[-15,0]`，与作者用于密度拟合的范围一致，并一致用于所有抽样。mode-Q 期望回归的 Gaussian 抽样不截断（与作者该步骤一致）；选中 mode 供 actor 拟合的动作截断 `[-1,1]`。这保留了 Gaussian expectation 与 bounded action 的原有区别，不能声称 GMM 支持严格限制于物理动作域。

GMM 10k 是预先固定的训练预算，不能称已证实收敛。适配不等于复刻作者 D4RL 分数；共同 50k 更新也不等于各算法最优超参数或达到相同收敛程度。

### 检查入口和当前边界

`check_lom.py` 是合成机制检查：GMM-only 阶段；RL 阶段 GMM 冻结；actor/Q/mode 参数更新；动作形状与范围；缺失 TD 不受伪造目标影响；SARSA 对实际 next action 的依赖；最高 mode 的选择及对应均值。`py_compile` 已通过。本文写入时当地运行环境没有 Torch，机制脚本由主任务上传既有训练 runtime 执行；不得把语法通过写成 GPU 运行通过或有效性证据。

保存训练 checkpoint 时必须包含 `gmm_opt`, `mode_opt`，以及常规 `actor_opt`, `q_opt`。推理仅需 actor。

## 正文表边界

BC、TD3+BC、IQL、ReBRAC、FQL、LOM、GFP 是外部算法行；项目内部 H02/H03/Bound040/D01/D06/no-RL/planner 等属于消融或开发记录。RL-DITR 的临床任务与本项目不同，必须标注公开任务适配及所用实现，不能冒充原论文私有医院实验。

相同测试轨迹、相同观测和动作单位是必要条件，仍不足以证明同训练资源：当前 DSENet-world 使用额外模拟配对监督，且部署动作候选受锚点限制。正文必须注明这些差别，并通过另列统一训练资源/统一动作合同的对照才能作纯算法优势结论。此结论来自本地训练合同审计，不是引用外部论文的性能数字。
