# 冻结外部基线接入计划与状态（2026-10-03 更新）

当前状态：**8 个保留基线已接入 `evaluate_candidates.py --kind retained --method <方法>`，统一标为 frozen+projection。** 本机真实权重 SHA256 与 9/21 最终清单一致；源码、worker、配置及 provenance 的只读预检通过。58 项本地机制检查已通过，覆盖原 RNG 循环、投影审计、失败与未知尾部。根任务随后报告：服务器上 8 种 `dependencies` 的只读 SHA 预检亦全部通过。该远端状态来自根任务回报，本轮没有连接服务器；不把本机哈希核验写成远端核验。

配置见 [retained_methods.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/configs/retained_methods.json)，实现见 [evaluate_candidates.py](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/evaluate_candidates.py)，机制证据见 [retained_evaluation_mechanics.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/checks/retained_evaluation_mechanics.json)。当前没有在本文登记这 8 个方法的新开发集成绩。共同可观测信息与共同动作上限，不等于同一网络输入、训练目标或原论文复现。

时间说明：本文初版是同日较早的只读核查，彼时尚未接入。下文保留其来源审计和拟议追加对照，并逐项标明已落实或仍待实施。最新方法与运行状态见 [method_registry.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_SOTA_2026-10-03/method_registry.json)；实现、smoke、正式训练和控制评价分别登记。

冻结依据：[final_manifest.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/configs/final_manifest.json)，SHA256 `bd2884f7103e5ed913ed5e9e799622be22d2ece1e205291bf79ea3b12e6218ec`。以下路径均为最终 9/21 权重，不是最早 Loop-only 导出或中间失败版本。

| 方法 | 真实冻结权重 | 本次复核 SHA256 |
|---|---|---|
| BC | [shared_bc/last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/shared_bc/last.pt) | `1a1cab603f8aed8df8121b8b7a8983cd568dbeaac6d32c34cca2e84c434872a1` |
| TD3+BC | [shared_td3bc/last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/shared_td3bc/last.pt) | `0f2735dfb0f16184e4bba8ded6eec6467c42ee5f713a127410554df04c89a507` |
| IQL | [shared_iql/last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/shared_iql/last.pt) | `f0ebb6763b4ffac949a71ec00afcfa4b555ff506ee9b1ff75d1bbd52d9b2dc37` |
| ReBRAC | [shared_rebrac/last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/shared_rebrac/last.pt) | `a27c670ffeabeee8b68c2a610769d78756e5505d8babfa7cb90ba8dcacf40c00` |
| FQL | [shared_fql/last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/shared_fql/last.pt) | `1a624bc8f6f883a7614e9b6ce3426a9342c20c53e2b7ed24cc07418dd9834e71` |
| LOM | [shared_lom/last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/shared_lom/last.pt) | `4ed59c51a9db055a890d02275d48592a8e5f3dd7b45269c92edb4af5a854f1fe` |
| GFP | [shared_gfp/last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/shared_gfp/last.pt) | `84d1c901b7fb4804a8e2ee8a3d58c1cb09255b50d4ce1fb0cd105a5e117e9e76` |
| RL-DITR* | [ditr_shared_r03_seed260915/policy_last.pt](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/results/ditr_shared_r03_seed260915/policy_last.pt) | `19a1e810c4eb6d11bd6f5c34fed556694656c6add7884363285125e1f4f315d9` |

7 个 `shared_*` 都是已有 Loop 权重的继续训练：每批 128 Loop + 128 仿真记录，额外 20,000 updates；LOM 是额外 5,000 GMM + 20,000 RL。训练 seed 均为 **260915**，仿真循环取样流另用 260916，不能称为第二训练 seed。RL-DITR* 从 R03 递归 categorical 权重继续 2,000 patient + 2,000 policy updates，每批 64 Loop + 64 仿真，seed 260915。配置、初始化权重及其哈希见 [shared_methods.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/configs/shared_methods.json)、[train_shared.py:24](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/train_shared.py:24)、[ditr_shared.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/configs/ditr_shared.json)。这些不是同预算从头训练；FQL/ReBRAC/LOM 已使用 source-order/phase 修正后的初始化，不能把早期修正再算成新方案。

## 部署入口、信息和动作

最终旧入口为 [evaluate_final.py:66](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/evaluate_final.py:66)：前 7 个 `mode=external, shared_worker=true`，RL-DITR* 为 `mode=ditr`，实际 worker 命令也已与 `results/C_native_<key>/manifest.json` 核对。

| 方法 | 网络实际输入与部署方式 | 原生动作（每 5 分钟请求一次） |
|---|---|---|
| BC | `72×22` 展平 1584；确定性 tanh actor | `10×(actor+1)` U/h，范围 0..20 |
| TD3+BC | 同上；确定性 actor | 同上 |
| IQL | 同上；训练 Gaussian actor，评估使用均值，忽略传入 noise | 同上 |
| ReBRAC | 同上；确定性 actor；此冻结配置没有 `anchor_context` | 同上 |
| FQL | 展平历史 + 1 维外生 Gaussian noise；蒸馏 actor，推理不运行 flow 积分 | actor 输出先 clamp 到 [-1,1]，再映射 0..20 U/h |
| LOM | 展平历史；确定性 actor，推理不从 GMM 采样 | tanh 输出映射 0..20 U/h |
| GFP | 展平历史 + 1 维外生 Gaussian noise；蒸馏 actor | actor clamp 到 [-1,1]，映射 0..20 U/h |
| RL-DITR* | 原 `72×22` 序列 → Transformer patient model → deterministic Gaussian-quantile beam | 12 步 × 5 分钟，beam 10；候选 clamp 0..20，执行最佳计划第一个动作 |

共同 history 来自 [observable_history.py:9](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL进阶对比_2026-09-15/observable_history.py:9)：5 个标准化观测值（CGM、前 5 分钟实际 basal、已记录 bolus、已记录 carbs、exercise），相应 mask/age/known，以及 2 个时间特征。模拟器 exercise 未提供，保持缺失。真实 BG、CR、生理参数、未来餐次及患者 ID 不进入网络。前 7 个及 RL-DITR* 都**没有持久 warmup anchor 网络输入**；投影器可以合法使用已观测 anchor，但不能把这写成 actor 已获得 anchor。

前 7 个实际部署：[shared_worker.py:10](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/shared_worker.py:10)，SHA256 `23498816ec315f960bfcb4877eac32070a86964b6ed85e133b4b188b0edc73b4`。每个病例的独立 RNG 为 `default_rng(scenario_seed+20000)`；FQL/GFP 每次消费一个 noise，其余 5 个方法虽收到 noise 但 `act` 忽略。`case_keys` 仅索引 RNG，`scenario_seeds` 仅初始化 RNG，不拼入状态。新 adapter 已保留固定逐病例随机流，使用不含患者语义的 opaque handle 管理流；原 RNG 循环在 batch 重排、病例退出后的流一致性已作机制核验。没有改成全局按 batch 消费，也没有把 noise 固定为 0。

动作实现证据：[rl_algorithms.py:62](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL进阶对比_2026-09-15/rl_algorithms.py:62)、[td3bc_algorithm.py:22](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/td3bc_algorithm.py:22)、[iql_algorithm.py:46](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/iql_algorithm.py:46)、[lom_corrected.py:77](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/lom_corrected.py:77)、[gfp_algorithm.py:39](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/revision/gfp_algorithm.py:39)。RL-DITR* 入口 [policy_worker.py:27](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DITR创新_2026-09-16/policy_worker.py:27)，SHA256 `8179e74fe541d879d2b1adb8bf8f4cd6965ccdd48a5f1cf9f2304af1ff7afb68`；beam 默认值见 [ditr_model.py:135](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DITR创新_2026-09-16/ditr_model.py:135)。该部署无需随机 action seed。

**命名陷阱：**[legacy_policy_worker.py:18](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/legacy_policy_worker.py:18) 加载 `NumpyPolicy`，不是上述最终 `shared_*` 权重入口；当前新 runner 的 `--kind legacy` 则特指 D06，并且冻结选取其 checkpoint。三者不能混用。

## 新任务中的接入（原计划与当前落实状态）

1. **已接入。** 8 个冻结方法使用独立 `retained` kind，保留原构造器、训练配置、权重哈希、history normalizer 和原部署采样。`dependencies(...,method=...)` 校验原最终 manifest、真实权重、worker/算法源码及旧配置/provenance；七方法路由原 `shared_worker.py`，RL-DITR* 路由原 `policy_worker.py --mode beam`。不使用 legacy NumPy 导出，`--kind legacy` 仍只指 D06。
2. **已实现公共投影。** 原网络输出保持真实单位 `u_raw=10×(a+1)`，RL-DITR* 取原 beam 第一动作。公共新任务动作显式定义 `u_requested=clip(u_raw,0,min(20,2×observed_warmup_anchor))`。**这是 frozen method + common projection，不能标 native。** 没有用 `anchor×(a+1)` 替换旧映射；旧权重按绝对 U/h 训练，那个替换是另一种策略，不是等价单位转换。
3. **已实现审计。** `decisions.jsonl` 保存 raw/requested/anchor/投影标记/差值；按区间结束分钟与原始轨迹连接，轨迹保留原泵 delivered，并增加相应投影字段。每病例及总体 summary 报告请求决策的投影率、平均/最大幅度；原评分器的 pump changed fraction 仍独立报告。`project_retained_actions` 执行显式投影，`validate_actions` 继续只作合同验证。技术失败和未知尾部保留。RL-DITR* 投影发生在原 beam 后，故计划内预测与实执行可能不一致；不能称为边界内重新规划。
4. 旧 D06 保持原 checkpoint `348caaf3d3474d34cd9091dfa90aeba31305499d2e34d98279999bfcfdb16913` 和 anchor±0.25 的 7 个计划、执行首动作：[policy_bounded.py:9](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/policy_bounded.py:9)、[bounded_policy_worker.py:33](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_2026-09-17/bounded_policy_worker.py:33)。不能扩宽它再称冻结 D06。旧 60 例 observed anchor 范围实读为 0.888..1.7245 U/h；其窄支持在公共范围内，新病例仍应逐例检查，不能对任意小 anchor 假定包含关系。
5. **统一入口已落实，评价成绩待实际运行证据。** 全部方法使用同一新 60-job 开发清单、实际历史更新、固定餐时 bolus 规则与 unchanged scorer；原生 0..20 的旧 9/21 表只作历史附表，不能直接混入新场景数值排名。新开发/确认种子不重用 92111/92112；模型参数保持冻结。此 runner 继续拒绝 confirmation，须另由最终冻结协议放行。接入及远端哈希预检成功不等于这些基线已完成新控制评价。

## 哪些对照有价值，哪些结果不能成为强证据

旧 9/21 每个方法各 60 例；以下是直接读取保存的 summary，未重新评估。native/brake17 都是旧方案名，brake17 是额外控制器，不是权重修复。

| 方法 | native 提前终止 | brake17 提前终止 |
|---|---:|---:|
| BC | 0/60 | 0/60 |
| TD3+BC | 10/60 | 0/60 |
| IQL | 0/60 | 0/60 |
| ReBRAC | 60/60 | 7/60 |
| FQL | 2/60 | 4/60 |
| LOM | 12/60 | 12/60 |
| GFP | 10/60 | 12/60 |
| RL-DITR* | 5/60 | 1/60 |

证据目录为 `RL_DSENet_公平低糖_2026-09-21/results/C_{native,brake17}_{bc,td3bc,iql,rebrac,fql,lom,gfp,ditr}/summary.json`；完整汇总见 [analysis/final.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/analysis/final.json)。失败方法的低糖/高糖等只是 observed prefix，不能与完整随访方法排名；例如 LOM 原表 TBR70=0 同时覆盖率仅 85.3851%、TAR250=27.5468%，不能视为低糖胜利。反过来，大幅胜过 ReBRAC 的失败适配实例也不是胜过整个算法族的证明。BC、IQL 原生完整随访且有较强 TIR，应保留，不能只挑最差基线展示。

以下为初版提出的追加工作，保留其论证；2026-10-03 当前进展另标在括号内，不预设原因已经证实：

- **无需重训的保留版本对照（已接入、远端哈希预检通过，新八基线成绩未登记）。**8 个 frozen+projection 与 hold、当前 physiology、D06 用新场景同表；失败、投影率、支持范围并列。它能回答新完整系统是否优于已保留部署，不能分离投影、输入表示、数据和算法贡献。若 RL-DITR* 投影频繁，再单列“候选在每一步进入模型前限制到公共范围”的重新规划对照；该项仍为拟议，未实施，不需训练但改变搜索策略，必须另名、另存配置。
- **若需要同任务训练的强基线，优先新增一个 IQL/BC 对照，而不是靠失败行撑结论。**已有 actor 只有 1584 维 history，warmup anchor 在超过 6 小时后不再由窗口保证保留；当前新模型显式有 anchor 和 28 个可观测生理特征。若让基线也消费这些特征并直接输出 0..2 anchor，需要新输入层/动作参数化与训练，不能给冻结权重多拼特征。旧 [shared_methods.json](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/configs/shared_methods.json) 没有 anchor_context。优先以同一新 train 数据、同一信息、明确固定预算训练一个“任务适配 IQL + BC”对照，保留原 8 行；这只检验公平信息/动作表示，不自动复现原论文。
- **新 reward 或严重尾部收益的归因需单独控制。**[prepare_shared_replay.py:95](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DSENet_公平低糖_2026-09-21/prepare_shared_replay.py:95) 的公共 TD reward 是下一点 CGM 的旧 `status_score/12`，[ditr_model.py:31](/Users/george/Desktop/糖尿病医疗Agent_强约束Harness方案/RL_DITR创新_2026-09-16/ditr_model.py:31) 在 <70 时固定 -1；6 个共享 RL 方法实际取该 reward，不取文件中另存的 true-BG/ppo_reward；BC 只做动作模仿，不使用 reward。公共 replay 有 392,832 origins、5,704 episodes，记录的 genuine terminal transitions 为 **0**；最后一步因缺 next action 不进公共 TD。不能声称这些旧权重已用当前连续 BG asymmetric-risk 或严重终止训练。只有当新训练数据确实含终止时，才需为 Q 方法分别实现“真终止 bootstrap=0、行政截断保留 bootstrap”；现有 TD3BC/IQL/FQL/ReBRAC/LOM/GFP 的目标代码不消费 `bootstrap_valid`，不能直接继承到有终止的新 replay。旧训练没有真终止，故这不是已证明导致旧失败的 bug。新增风险训练应按独立消融命名，不能无披露替换旧源方法目标。

最新补充：新 IQL_wide 已实现 1613 维输入（1584 历史 + 28 生理特征 + anchor）、连续 0..2 anchor 动作及固定 20,000 更新配置；其 replay 预设为 120 条自然 train 加 wide PPO 前 8 轮 160 条 train，固定最终 checkpoint，不按开发结果选步数。当前已接入 `--kind iql_wide` 并通过独立 58 项评价机制检查。根任务最新回报：280 例、221,760 transitions 的 replay 已生成，CUDA 四步训练及冻结推理检查通过，正式 20,000 更新训练已启动；实际闭环评价 smoke 尚待完成。本轮未独立连接服务器复核。独立 BC_wide 与 DITR 候选内重规划仍为提案，不能与已有实现混写。

这些追加对照不要求立即重训全部基线。论文来源、公开 Loop 成人基础输注改编、训练预算、原始目标/表示差异、单 seed 和已暴露虚拟患者边界应保留；最终大表只支持具体冻结实验范围内的结论。
