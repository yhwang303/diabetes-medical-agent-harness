# 最终对比表：八个旧基线的文献身份

核验日期：2026-10-04。仅提供已实际评价方法的来源，均保留 **frozen+projection**。7 篇论文及作者代码身份由一手资料核对；BC 属于通用行为克隆，未指定单一原论文/作者仓库，因此对应字段为空。文献身份核验不等于原实验复现。

| 对比表方法 | 原论文准确标题与正式发表 | 作者代码 | 本项目适配边界 |
|---|---|---|---|
| BC frozen+projection | 无单一归属；通用行为克隆 | 无指定单一作者仓库 | 通用监督行为克隆，以确定性 MLP 和动作 MSE 拟合已记录基础率，不使用 RL reward。沿用 shared_bc 旧权重并作公共投影，不冠以某篇论文官方实现或独立强化学习算法。 |
| TD3+BC frozen+projection | [A Minimalist Approach to Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html) / NeurIPS 2021 | [作者仓库](https://github.com/sfujim/TD3_BC) | 按 TD3+BC 双 Q、延迟策略更新和 actor 行为正则适配基础输注；1584 维历史、3×256 网络及任务奖励/折扣为本项目设置。沿用旧权重后投影，非作者 D4RL 默认配置或分数复现。 |
| IQL frozen+projection | [Offline Reinforcement Learning with Implicit Q-Learning](https://openreview.net/forum?id=68n2s9ZJWF8) / ICLR 2022 | [作者仓库](https://github.com/ikostrikov/implicit_q_learning) | 旧 IQL 的 PyTorch 任务适配，保留 expectile V、双 Q 与优势加权 Gaussian 似然，评估取均值。此处为 shared_iql 冻结权重加投影，不能与新 IQL_wide 混同，也不是作者 JAX/原 D4RL 复现。 |
| ReBRAC frozen+projection | [Revisiting the Minimalist Approach to Offline Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/26cce1e512793f2072fd27c391e04652-Abstract-Conference.html) / NeurIPS 2023 | [作者仓库](https://github.com/tinkoff-ai/ReBRAC) | PyTorch 适配保留 actor/Q 双重行为正则，采用已修正更新顺序的 shared_rebrac 旧权重；正则、学习率、噪声与初始化已有任务适配。部署再投影，不能称原生或充分调优的整个 ReBRAC 算法族成绩。 |
| FQL frozen+projection | [Flow Q-Learning](https://proceedings.mlr.press/v267/park25f.html) / ICML 2025 | [作者仓库](https://github.com/seohongpark/fql) | 流匹配行为模型、单步 actor 蒸馏/Q 优化的 PyTorch 适配，使用 source-order 修正后的 shared_fql 旧权重。部署带 Gaussian noise 的单步 actor 后投影，非作者 JAX/原基准完整复现。 |
| LOM frozen+projection | [Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/be62c4a943675195ff5a2a98d5b9724f-Abstract-Conference.html) / ICLR 2025 | [作者仓库](https://github.com/MianchuWang/LOM) | 依论文独立实现 GMM、SARSA behavior-Q、mode-Q 与单 mode 加权模仿；固定作者快照未发现 LICENSE，未纳入其源码。优势基准取所选 mode 均值，区别于所审作者代码的 actor 基准。沿用 corrected 旧权重并投影。 |
| GFP frozen+projection | [Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html) / ICLR 2026 | [作者仓库](https://github.com/Simple-Robotics/guided-flow-policy) | 价值引导 flow 拟合与单步 actor 蒸馏的 PyTorch 任务适配；保留价值权重、时间编码及目标更新逻辑，网络/学习率采用项目预算。Gaussian noise actor 后投影，非作者 JAX/原 benchmark 复现。 |
| RL-DITR* frozen+projection | [Optimized glycemic control of type 2 diabetes with reinforcement learning: a proof-of-concept trial](https://www.nature.com/articles/s41591-023-02552-9) / Nature Medicine 2023 | [作者仓库](https://github.com/rlditr23/RL-DITR) | *表示公开成人 T1D 连续基础输注适配：患者模型/策略/价值/beam 思路改为 5 分钟连续 U/h，区别于原住院 T2D 给药任务。旧 R03 categorical 继续训练权重与原 beam worker，首动作在搜索后投影；非范围内重新规划或原临床试验复现。 |

八行沿用 9/21 最终权重。七个 shared actor 保持历史输入及绝对 0..20 U/h 映射，再显式投影到 `0..min(20,2×已观测 warmup anchor)`。RL-DITR 在 beam 搜索后投影首动作；FQL/GFP 保留原逐 episode 外生噪声流。不能标 native，也不能用冻结实例的失败代表整个算法族的上限。

共享数据继续训练为七方法追加 20k 更新，LOM 另有 5k GMM；RL-DITR 追加 2k patient 与 2k policy。此前训练、奖励、容量和表示仍有差异；共同评价不自动成为充分调优或等预算 SOTA 比较。旧 IQL 与本轮 IQL_wide 是不同训练实例。

IQL 的 OpenReview 本次为验证页，正式年份由 [ICLR 2022 官方目录](https://iclr.cc/Downloads/2022) 与 [作者原稿](https://arxiv.org/abs/2110.06169)核对。RL-DITR 的 Nature 页面本次跳转受限，题名、卷页、DOI 与代码关联由 [同文 PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10579102/)直接核验。LOM/GFP 采用正式会议 2025/2026，不改写为首稿年份。

LOM 按论文独立实现，优势基准与所审作者代码有已披露差别。RL-DITR 的公开训练器与论文差异已审查，未复现医院队列或临床验证。FQL 实际旧配置 `normalize_q_loss=false`，不能把早期来源文档中的建议写成已采用参数。

结构化 JSON 包含 `methods` 列表和 `retained_*` ID、正式论文/作者仓库链接、历史作者审阅 commit、本地算法证据 SHA 与旧权重/worker 引用。权重 SHA 是对冻结清单的引用；本文不更新实验绑定，也不登记开发或确认成绩。

[paper_comparison_sources.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/paper_comparison_sources.json)
