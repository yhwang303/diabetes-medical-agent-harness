# WorldModelV2 独立验证诊断

`diagnose_world.py` 读取已经保存的预测和 `world_validation` 标签，独立重算指标并写新的 JSON。它不加载 Torch、world 或 checkpoint，不执行新推理、重训、后验校准、阈值优化或 checkpoint 选择。每次必须明确给出一个已经保存的 `--step`，不会读取 selection.json 或自动挑选 best。point 与 quantile 分别运行；两者比较前应核对输出中的验证 manifest 和 NPZ hashes 相同。

## 输入与绑定

训练器的实际产物为：

- `validation/natural_stepXXXXXXX.npz`：`cgm_quantiles_mgdl[N,1,72,Q]`、`bg_event_probabilities[N,1,3]`、`quantile_levels[Q]`、`sample_index[N]`。
- `validation/paired_stepXXXXXXX.npz`：相同字段，arm 轴为 9，同一 origin 的九臂不能拆开。
- `validation/stepXXXXXXX.json`：训练器自己的摘要，供独立重算后的交叉核对。
- `provenance.json`：验证数据 manifest SHA、逐文件 SHA、按文件名排序的读取顺序与组数、训练 config 和 protocol SHA。
- `completion.json`：训练确已完成其记录预算；烟测预算可诊断，但在输出里保持 `smoke_budget_override=true`，不能写成正式训练证据。

预测 NPZ 本身没有标签。脚本按 `provenance.data.natural_validation/paired_validation.file_order` 重建 CGM、BG 与 mask，只打开绑定的这两个验证数据目录。下载位置可以改变，原远端路径只作 provenance 记录；本地 manifest、文件 SHA、逐文件 job/order/count 必须与训练记录完全匹配。`sample_index` 必须正好是 `0..N-1`；不会按猜测重新排列。所读附加字段只有 `origin_minute`，用于确认样本来源范围，不读取 history、动作、未来餐食/bolus 或任何训练数组。

脚本先检查 `split=world_validation`、完成状态、72 步 horizon、患者名单、bolus factor 和场景 seeds 103011/103012，再打开标签。train、development、confirmation、未知 seed 和路径逃逸全部拒绝。运行目录、下载数据目录及输出必须在本研究目录内。只读取训练 provenance 中的训练数据元信息，不打开其中 train 路径；协议内其他 split 的名称不是数据读取。

每个报告保存脚本、protocol、completion、provenance、验证摘要、预测 NPZ、标签 manifest/NPZ 的 SHA。重新计算的 CGM MAE/RMSE、窗口计数、自然事件 Brier、配对 delta MAE 和最低 CGM 排序必须与训练器摘要在浮点容差内一致，否则不写结果 JSON。容差为 `rtol=1e-5, atol=1e-4`，应覆盖 FP32/FP64 累积差异，不代表统计等价。

现有预测 NPZ 没有嵌入 checkpoint SHA，原训练器也没有另存预测文件 SHA。因此脚本记录本次收到的文件 SHA，加上 run/step/标签绑定和摘要一致性，**不声称已经建立密码学级别的预测→权重身份链**；若需要更强验证，应另行用冻结 checkpoint 重算预测。此脚本不会偷偷做该推理或改变既有训练产物。

## 指标与分母

自然轨迹与配对干预分开报告。前者是现有采集策略产生的验证分布，不自动代表临床人群发生率；后者是人为干预臂的分布，不能合并后把 Brier/校准率称为自然概率校准。

| 项目 | 计算范围与解释 |
|---|---|
| CGM median MAE/RMSE/bias | 仅 mask=true 的传感器 CGM 点；bias 为预测减观测。另报观测 CGM<70/<54 的点误差。 |
| 每个分位的误差与 pinball | 报 MAE/RMSE/bias、pinball 和 `P(target≤qτ)`；偏尾分位的 MAE 不是判断其预测中位数质量的标准。 |
| 中心区间 | .05–.95、.1–.9、.25–.75 的逐点覆盖、平均/中位宽度；point 模型返回空区间表，不能写成 0% 覆盖。 |
| BG70/BG54 事件 | 全部 72 步均观测的 arm windows，标签为窗口内任意真实 BG 严格小于阈值；Brier、正/负数量、预测均值、发生率、固定 10 箱校准、0.5 阈值 TP/TN/FP/FN/FNR/FPR。无阳性时 FNR=null；不写 0。 |
| 截尾事件 | 全部排除完整窗口 Brier/校准/FNR，包括已经观察到低糖的截尾窗口；另列已观察阳性和“尚未观察到事件、尾部未知”的数量。后者绝不标成阴性。 |
| 最低 CGM 偏差 | 完整窗口与截尾已观察前缀分别算 `min(prediction)−min(observed CGM)`，两侧使用同一已观察时间支持。分别在全部、BG<70/<54、CGM<70/<54 子集报告。正偏差代表预测最低 CGM 偏高。 |
| 低端分位最小值 | 仅诊断各时间点边缘低分位的下包络偏差；`min_t qτ(t)` 不是轨迹最小值的 τ 分位数。 |
| 配对响应差值 | 每个非 hold 臂减同 origin 的 hold 臂，在共同 mask 上报告 delta MAE/RMSE/bias；另给与训练损失一致的 group-macro mean-arm MAE，以及两臂都完整时的六小时 endpoint/minimum delta 误差。 |
| 配对排序 | 仅九臂均完整的 origin，枚举 36 个 arm pairs，报告最低/终点 CGM 排序、BG70/BG54 事件概率排序；真实差值≤1e-6 的 ties 排除，预测完全相同算未排对；分母为 0 时 accuracy=null。 |
| 完整/截尾覆盖 | origin、arm window、完整/截尾 arm 和 group 数、计划点/观察点/未知尾点、每臂已观察步数、配对共同支持比例及完整 reference-arm pairs。mask 不能单独说明截尾原因，脚本不推断为 native termination。 |

CGM label 与 BG event 不同：前者是传感器时间序列，后者是仿真真实血糖的窗口事件。同一个窗口可能真实 BG<54 而传感器 CGM 最低值仍≥54，反之也可能；不能用 CGM 的最小值命中率替代独立 BG 事件头的漏报率。

这些计数的单位是窗口/干预臂/时间点。自然滑窗重叠、同患者重复场景以及同 origin 的配对臂均不独立；不提供把所有窗口当独立样本的置信区间、显著性或有效样本量。已有十个虚拟成人均已曝光，不宣称新患者泛化。任何 marginal coverage 都不是整条轨迹联合覆盖，更不是 CVaR 或闭环低糖风险保证。

## 待下载文件与运行

目前本地只有 trainer smoke 的汇总 JSON，没有真实预测/标签 NPZ；本批没有运行真实模型诊断。下载某个完成 run 的 `provenance.json`、`completion.json` 和选定 step 的三个 validation 文件，以及与 provenance 绑定的两个验证数据目录（manifest + 所列全部 NPZ）即可；不需要下载 train 数组或权重。全部保持在本研究目录内。

示例以最终 4000 步作为显式诊断点，不表示脚本选中了它：

```bash
.venv/bin/python RL_DSENet_SOTA_2026-10-03/diagnose_world.py \
  --run RL_DSENet_SOTA_2026-10-03/results/world_quantile_r1 \
  --step 4000 \
  --natural-data RL_DSENet_SOTA_2026-10-03/cache/natural_validation_windows_r1 \
  --paired-data RL_DSENet_SOTA_2026-10-03/cache/paired_world_validation_r1 \
  --output RL_DSENet_SOTA_2026-10-03/checks/world_quantile_r1_step4000_diagnostics.json
```

point run 用其真实目录另跑并换输出名；对应 step 必须已保存。已有输出拒绝覆盖。上述路径均为预期名字，以实际下载且 hash 匹配的产物为准。

机制测试命令：`python diagnose_world.py --self-test`，只用内存中的人工数组与压缩 NPZ 字节，不创建假 run 或模型结果文件。已使用 Codex 自带 Python / NumPy 2.3.5 实际通过：CGM/BG 语义分离、完整事件分母、无阳性分母、截尾处理、padding 不影响观测指标、point 区间缺省、配对共享偏差消除、正/反/并列排序、空分母、0/.5/1 分箱边界、七类错误 split/seed 拒绝、四类预测格式/顺序/概率错误拒绝及 JSON 无 NaN。测试输出明确 `synthetic_mechanism_tests_only`、`real_saved_predictions_loaded=false`。没有安装/改变项目环境，没有连接远端。
