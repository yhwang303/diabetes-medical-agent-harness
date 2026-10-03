# 阶段报告构建约定

`build_report.py` 使用 Python 标准库，只读取已经审计的 JSON、方法登记表和身份文件，输出独立离线 HTML、CSV 与构建清单。不导入评分器、数值模型或模拟器，不读取原始轨迹来重算分数，不生成或选择新权重，不写旧研究文件。`method_registry.json` 只提供方法定义；其中滞后的 runtime/status 快照不用于推断完成情况。

## 输入与校验

- `--panel checks/panels_*.json` 可重复，接收 development 或 world_validation 的已审计控制面板；`--confirmation` 必须显式提供 confirmation 面板，两者不能混用。每个包内所有方法要求相同完整 job、餐表、factor、split。非 smoke 包要求 protocol 的完整患者×场景×factor 矩阵。多个包独立显示，不跨不同场景混合排行。
- 校验 schema、原评分器固定 SHA、`scoring_verified_exact`、protocol 与汇总器 SHA、原始病例 SHA 格式及 manifest/checkpoint 身份。构建器只读原评分器字节校验 SHA，不执行它。JSON 拒收 NaN、Infinity、重复键。
- 从病例分数重新核对患者内均值、患者间 sample SD、覆盖、未知尾和患者配对差。stdlib 与 NumPy 聚合允许 `rtol=1e-10, atol=1e-9` 的算术舍入差；这不改变上游 raw/score 的**逐字段精确一致**条件。
- `--world DIAGNOSTICS PREDICTION_BINDING` 可重复，要求保存预测身份、完整 natural/paired 重算、逐元素检查、数据文件/顺序/场景一致、正式训练预算完成、事件分母和 FNR 自洽。只读 `checks` 文件。自然 BG<54 无阳性必须是 `FNR=null`，不得写成零漏报。
- 可选 `--actions checks/actions_*.json` 要求实际请求逐点关联成功，并绑定输入面板 summary/manifest SHA、病例和决策数。其正文只陈述观测动作，不作 argmax 的因果解释。
- `--summary-file` 或 `--summary-json` 提供可审查解释、显式方法映射与显示分组。输入中的解释不是生成器自动科研判断；HTML 会转义文本，保留一份 `报告解读.json`。构建期间来源 JSON 字节若变化则拒绝继续。

摘要 schema 示例（面板映射必须恰好覆盖全部输入）：

```json
{
  "schema": 1,
  "paragraphs": ["研究仍在进行，独立确认尚未运行，当前没有最终模型结论。"],
  "panels": {
    "dev_hold_r1": {
      "method_id": "hold", "label": "观测基础率 Hold", "group": "comparison"
    }
  },
  "equal_bg": []
}
```

`group` 仅允许 `comparison` / `internal_ablation`；内部消融必须填写 `deployment_variant`，保留原算法 `method_id`。PPO08、PPO16 属于同算法的预定检查点，不能据行数声称不同算法数。mean/argmax 消融同理。旧外部方法登记为 frozen+projection，明确公开动作范围投影不是作者原生部署。`equal_bg: [{"left": "dev_ppo08_r1", "right": "dev_hold_r1"}]` 只有完整 BG 汇总、逐患者均值及 jobs 逐字段一致才通过。

## 输出口径

HTML 中文、无外部字体或脚本、横向滚动、固定首列/表头。主表使用真实 BG，CGM 作为折叠次表。数值全部由 JSON 生成，不在模板手抄。已完成、非 smoke 方法才可标记本表数值最优；精确相同值并列加粗，显示舍入不改变比较。加粗不代表显著性、SOTA 或晋升。

不完整方法显示 **Observed**，保留失败数、覆盖、TIR 上下界、未知尾、缺失信号及事件计数，不参与完整方法最优值。无观测显示“—（无观测）”，不能补零。事件合计、未知尾分钟与信号缺失分钟是病例总数；患者均值采用先患者内平均再患者间 mean±sample SD（ddof=1）。血糖 SD 指标仍使用原单轨迹评分器 ddof=0。±不是多训练 seed 稳定性或置信区间。

配对差取候选减参考，读取已审计的 patient-cluster bootstrap CI，不在报告构建时重采样。原联合不退步和 TIR/低糖/高糖/波动 family 判断分开；原始 JSON 中的“多数指标改善”不会自动变成优胜结论。失败前缀仅作 Observed 差值描述。

World 表突出固定 0.5 阈值严重低糖 FN/阳性与自然无阳性；异臂排序、绝对风险分类、最低 CGM 偏差分别显示。相互重叠窗口和同源多臂不解释为独立患者。best 来自 validation，所以不是独立确认；全量预测重算不是临床或性能晋升证明。

输出：`阶段研究报告.html`、`指标长表.csv`、`患者配对差.csv`、`病例审计.csv`、`报告解读.json`、`核查结果.json`、`构建清单.json` 与 `sources/`。CSV 保留 JSON 的数值精度，HTML 通常显示两位小数。病例 CSV 保存每例 failure/coverage/unknown-tail、原轨迹路径与 SHA；轻量报告不复制原始大轨迹。所有输入快照和除 manifest 自身外的输出均列 SHA。来源链接可以离线打开。

`--output` 必须是研究目录内**尚不存在**的新目录；默认 `delivery`，拒绝覆盖。`--status interim` 醒目标记“研究进行中”，没有确认包时显示“确认未运行”。`--status final` 必须显式输入非 smoke、精确评分审核的确认包，并在摘要中加入 `confirmation_review`：`reviewed: true`、非空 `reviewer`、覆盖每个报告确认面板的 `panel_manifest_sha256` 映射。此记录是报告内容审核，不是启动确认实验的许可，不替代独立冻结/运行闸门；即便 final 也不自动晋升或声称 SOTA。尚未生成正式 confirmation 包，本轮仅验拒绝路径。

## 当前构建与复用 CLI

首份 `delivery/阶段研究报告.html` 包含早期 6 面板（360 病例），保留为不可覆盖快照；已核查 108 个主表数值单元、234 行指标 CSV、25 行配对差。后续使用新目录名。下面以最新 10 面板（含真实失败）为例，从上一份可审查摘要追加明确映射；没有联网或模型执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import json, pathlib, subprocess, sys
r = pathlib.Path('RL_DSENet_SOTA_2026-10-03').resolve()
summary = json.loads((r/'delivery/报告解读.json').read_text())
for name, method, label in [
    ('dev_ppo16_r1', 'ppo_wide', 'Wide PPO · 第 16 轮（argmax）'),
    ('dev_retained_rebrac_r1', 'retained_rebrac', 'ReBRAC · 冻结 + 公共范围投影'),
    ('dev_retained_lom_r1', 'retained_lom', 'LOM · 冻结 + 公共范围投影'),
    ('dev_retained_gfp_r1', 'retained_gfp', 'GFP · 冻结 + 公共范围投影'),
]:
    summary['panels'][name] = dict(method_id=method, label=label, group='comparison')
cmd = [sys.executable, str(r/'build_report.py'),
       '--panel', 'checks/panels_development_ppo16_partial_baselines_r1.json',
       '--summary-json', json.dumps(summary, ensure_ascii=False),
       '--actions', 'checks/actions_dev_ppo08_r1.json',
       '--actions', 'checks/actions_dev_ppo16_r1.json',
       '--status', 'interim', '--output', 'delivery_interim_r3']
for variant in ('point', 'quantile'):
    cmd += ['--world', 'checks/world_'+variant+'_r1_best_diagnostics.json',
            'checks/world_'+variant+'_r1_best_prediction_binding.json']
subprocess.run(cmd, check=True)
PY
```

运行位置为项目根目录。路径含中文没有外部依赖；`--summary-file /absolute/path.json` 可替代内联 JSON，便于单独审核后重建。

## 验证与边界

已完成语法检查、6 面板及两个正式 world 的真实输入核验；15 项拒绝/边界检查覆盖非法 JSON、缺失确认、评分器或审核标志错误、覆盖/场景/聚合/完整性冲突、自然无阳性写零 FNR、局部 world 绑定、登记方法错配、Observed 不加粗/不补零、HTML 文本转义。

构建时再次解析实际 HTML 的数值单元、粗体标记、导航及离线链接，逐字符串回读无损 CSV；检查失败会停止，不声称构建成功。视觉检查仍由根任务在浏览器完成；生成器不会把机器解析通过写成视觉验收。报告遵循研究合同：单 seed、公开仿真、已知虚拟成人，不作真实患者安全证明。原 raw-score 审计真实性依赖已验证的上游检查 JSON，生成器不是第二个评分器。
