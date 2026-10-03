# IQL 优势权重只读诊断

本批只新增诊断与机制检查，不修改冻结 IQL、replay、worker 或 evaluator。目标是检验奖励尺度假说，不重新选择模型、调温度、训练 BC 或产生新闭环成绩。固定 20k IQL 基线保留。

## 输入与运行

在已有完整训练 replay、原始依赖和正式权重的项目环境运行；以下 `PROJECT` 是实际项目根目录的占位符：

```text
PROJECT/.venv-native/bin/python -u PROJECT/RL_DSENet_SOTA_2026-10-03/diagnose_iql_weights.py \
  --checkpoint PROJECT/RL_DSENet_SOTA_2026-10-03/results/IQL_wide/policy_020000.pt \
  --output checks/iql_weights_final_training_r1.json \
  --device cuda --batch-size 512
```

`--checkpoint` 必须为绝对路径，且只接受非 smoke、完成固定 20k 的 `policy_020000.pt`。配置来自同目录 `config.json`，同时核对冻结模板；没有任意 config 覆盖、重新选步数或样本子集参数。`--output` 相对于研究目录，必须是 checks 下新 JSON；既有文件拒绝覆盖，运行异常保存 `technical_failure`。CUDA 不可用时直接失败；CPU 只能显式指定，运行口径写入结果。

反序列化前核对 completion 对最终权重的 SHA、config/provenance、训练 manifest、episodes、全部 replay 数组和声明的源码依赖。检查 checkpoint 内嵌 config/provenance 与独立 JSON 相同，并验证未使用 development/confirmation、数据来源为冻结的 120 natural + 160 PPO-first8。原始 raw 文件的 SHA 清单须与训练 provenance 一致；不重新加载全部 raw，因为本次计算直接消费的是逐文件重新验 SHA 的冻结 replay 数组。前后再次核对所有输入/源码字节，诊断脚本自身也绑定。

## 统计定义

按 replay 原始索引顺序遍历全部训练 transitions，不随机抽样。每批只编码历史、observed anchor 和原生理特征；将**已记录动作**拼入状态输入 target Q。真实 BG、来源和 episode ID 只用于结果分层，不进入模型。

```text
A = min(final_target_Q1(s,a), final_target_Q2(s,a)) − final_V(s)
z = 3A
w = exp(min(z, log(100)))
```

三者均按原实现的 FP32 运算产生，在内存保留完整三列向量，再用 float64 汇总。训练日志中的优势使用当时更新后的 V、更新前的 target Q；这里使用最终保存的两者，是固定模型诊断，不是回放最后一次 optimizer 更新。不能要求其均值精确等于日志最后一行。

输出每列 p0/p1/p5/p25/p50/p75/p95/p99/p100（全向量、linear 插值，无 reservoir/近似 sketch）、均值、总体标准差，以及 A>0 比例、权重为零数量、上限达到率、ESS 和 top 1% 权重质量。

- 上限达到定义为 `z >= float32(log(100))`。FP32 的 `exp(log(100))` 可略大于 100；保留真实运算值，统计不再次静默夹取。
- `ESS=(Σw)²/Σw²`，同时报告 ESS/N。它反映权重集中程度，不是对相关历史/episode 调整后的统计独立样本数。
- top 1% 使用 `ceil(0.01×N)` 个最大权重，占该分层权重总和的比例。空分层明确 count=0、分位/比例为 null；若全部权重下溢为零，ESS 和质量比例为 null。
- 来源分层只连接实际存储的 `episode_id.npy` 与 `episodes.json.source`；没有则明确 unavailable。动作分层使用实际 `action.npy` 归一化坐标的四个固定区间；BG 分层只使用已存 `bg_mg_dl.npy` 的后动作标签，区间为 `<54`、`[54,70)`、`[70,180]`、`>180`。缺失 BG 不用 CGM 猜测。三类为各自边际分层，不声称因果归因。

不创建 optimizer、没有 step/backward/采样或梯度对照。模型以 eval、requires_grad=False、inference_mode 运行，前后比较全部 state_dict 字节；模型初始化包在 Torch RNG 保存/恢复作用域内，并核对 Python、NumPy、Torch CPU/CUDA RNG 未变。输出明确这些不变性，不保存新权重。

## 本地验证与下一步证据

`check_iql_weight_diagnostics.py` 的 r2 机械证据为 `checks/iql_weight_diagnostic_mechanics_r2.json`，42 项通过：含统计量、分层边界/缺失处理、配置/来源/权重/数组不一致拒收、正式与 smoke 分离、内嵌 checkpoint 元数据、失败留存和覆盖拒绝。r1 保留为前一诊断源码版本的历史检查，不替代 r2。

本地未加载真实 Torch checkpoint 或完整 replay，也未连接远端；上述是 fixture/源码机制检查。实际全量输出尚未产生。执行成功后，应首先核对 transitions=221760、episodes=280、逐来源计数及 `all_bound_artifacts_unchanged=true`，再解释权重分布；任何异常保留原输出并另开版本化诊断，不改 20k 主结果。

诊断源码 SHA：`e268434cd3afbd4b1edb6c735a5d25600e81b09ab1a156227dd0cc7882ddd235`。机制证据包含检查脚本自身 SHA、冻结源码 SHA，以及“未执行推理/训练/远端”的明确标记。
