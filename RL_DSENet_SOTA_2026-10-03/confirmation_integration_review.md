# Confirmation 整合独立审查

审查对象为当前本地源码与已有机制证据，未连接远端、未 freeze、未生成真实确认 jobs 或轨迹。结论：既有 gate 的冻结/授权设计可以保留，但尚不能直接接当前 evaluator→summarizer→report；下列问题是正式确认前的整合阻断，不否定已完成的 development 分数。

## 审查快照

| 文件 | SHA256 |
|---|---|
| confirmation_gate.py | a2046889db04d1de52655b6d137c0f1723309a6d401513a4df9e21a59ea50362 |
| confirmation_plan.md | 6bea27a77523c85040656af6ccf212a6b0844f11d692d601cac74b266a7962b8 |
| evaluate_candidates.py | 74249da6749a3b1f6f93867b5185645955efe61dcdd17aaf91dc0c6b211da6ec |
| summarize_panels.py | 82c7a11db9d3316af1cfd9584e6175c594262127831ccad61ac77f16c32c6438 |
| build_report.py | 394590c1e975a7d1691032d0329417484c12dc2b9116947841ec924124d0a02a |

以下行号均指此快照；后续修复不能抹去该审查依据。已有 gate 的 39 项检查是临时 fixture 状态机证据，不是 runner 整合或确认结果。

## 阻断项与最小接法

**B1：运行目录、jobs 及 provenance 契约尚未接通。** Gate 在 `confirmation_gate.py:251–271` 生成已冻结方法的 60 jobs，并先创建 `confirmation/<plan SHA>/runs/<id>/ticket.json`。当前 evaluator `:600` 调 `make_jobs`，后者 `:61–77` 拒绝 confirmation；`:614–616` 又固定创建全新的 `results/<name>`。普通 CLI 同样在 `:956` 拒绝确认。不能用 monkeypatch 或 `exist_ok=True` 绕过。最小接法是普通 `run` 保留原 jobs/新目录校验，抽出共同执行主体，新增只收 ticket 的受控入口；后者验证唯一 ticket、只允许 ticket/start 的预留目录、再调用相同 loop。初始和最终 manifest 均须 `confirmation=true,tuning_split=false`，并补齐 gate 要求的 freeze/ticket/protocol/normalizer/worker/config/source SHA。特别注意原 `:689–753` 会重新赋值 source/provenance，不能把 gate/runner 的冻结 source 集合覆盖丢掉。

**B2：唯一授权不等于一次执行。** `validate_ticket(:274–286)` 只核授权事件，未拒绝 finished；重复调用仍能得到相同有效 ticket。`authorize` 与 `record_result` 分别防重复授权、重复结束，但没有防同 ticket 被两个 runner 同时启动。未来入口必须在输出目录用排他创建的一次性 start 标记消费执行权，并核已有 finished/其他产物；首次失败也不能删除标记后重跑。无需新增调度框架。

**B3：gate 接受的结果并不一定能通过原汇总器。** `check_result(:289–325)` 只对 summary/raw 做 scorer 重算，没有要求 manifest 存在、没有核 manifest SHA、raw 的 evaluation provenance 或 minute/warmup 连续性。现有测试 `checks/check_confirmation_gate.py:113–122` 使用单行 `minute=365,warmup=false` 并被 gate 接受；但原汇总器 `summarize_panels.py:162–170` 要求第 i 行为 `(i+1)*5`、前 72 行 warmup，会拒绝。成功 finished 的 `result_sha256` 在 gate `:311–324` 也未覆盖 manifest。最小修复：确认结果必须有原格式 manifest；复用未改动的 `load_panel(folder,name)` 执行原 scorer、时序、raw/summary 逐字段检查，再补 gate 的 frozen/ticket/source/模型绑定。将 manifest 和真实 raw/summary 等输出 SHA 绑定到 finished。不要手写 `scoring_verified_exact=true` 或复制一个 fixture passed。

**B4：确认输出无法从现有汇总 CLI 进入报告。** `summarize_panels.py:301–315` 只接受 results 下单目录名；gate 的 runs 无法传入。底层 `load_panel(folder,name)` 已支持项目内其他目录，且 `:123` 允许 confirmation。最小接法为新的确认汇总薄适配器：只收 freeze 和显式参考 ID，从冻结集合派生每个 runs 路径，验证全部授权/结束事件及输出 SHA，再调用原 `load_panel/compare`；保存新的 `checks/panels_confirmation_*.json`。不要移动/复制确认输出冒充 development，也不改原汇总器。原因之一是 `build_report.py:200–201` 将既有开发包绑定到原汇总器 SHA；改它会令现有已审核包全部失配。

**B5：final 报告只核了输入集合，未核整个冻结比较集合。** `build_report.py:321–330` 只要求输入确认面板具有审核标志并被人工 review 映射覆盖；一个完整冻结集合的子集也可能满足它。当前 panels schema 没有 freeze/ticket/finished 信息。最小接法：新确认适配器输出额外的授权审计字段，包含冻结全部 method ID、逐 ticket、finished、manifest/summary/output SHA 及真实 source 身份；报告确认分支核精确集合，不能只核传入的面板。人工 review 字段仍只表示报告解释审核，不能代替真实 score/gate 验证。

**B6（若选择 mean）：报告登记兼容尚未实现。** 当前 registry 的 `ppo_wide`/`ppo_world_quantile` entrypoint 是 `ppo`/`world_ppo`；report `:308` 要求严格等于 panel controller，所以 `ppo_mean`/`world_ppo_mean` 即便标为 internal_ablation 仍被拒。mean 应归同算法，显式核 underlying kind、部署规则及同 checkpoint 身份，不能新增“外部算法”来规避此判断。此项只在实际选入 mean 时阻断。

## 可以复用、不应改变的部分

- Gate 在冻结时要求完整训练：PPO 满 40 轮后才选 08/16/32/40；World/IQL 依赖既有 completed budget 校验。`authorize` 才调用 `scenario` 生成确认餐表；读计划或写 freeze 不提前生成确认餐次。
- 普通 evaluator 当前 `save(:638–674)` 保存真实前缀；异常 `:891–913` 请求活着的环境返回真实已观测记录；`finally(:929–943)` 为全部未完 jobs 写空记录、显式失败与 unknown tail。原生提前终止不是技术崩溃，但仍是失败/不完整随访，不能列入完整方法排名。
- 真正进程被 kill 或磁盘不可写时，任何 Python finally 都不能保证写齐文件。此时 gate 的基础设施结束事件必须保留“未知 60 例/证据不足”；不应伪造可审核完整分数或 final 成功。能写出的正常异常路径应保留全部 60 raw。
- 既有原 scorer 不变，BG 主表、CGM 次表、患者均值和 SD、Observed/TIR bounds/未知尾规则不变。技术失败若确有全部 60 条正确评分与 raw，可作为 Observed 完整报告记录，不能标成完整有效控制；缺 manifest/raw 或时序不合法则不能获 score audit passed。
- 预曝光扫描不能证明没有未记录的外部运行，当前文档已准确披露；无需为本次整合再扩张审批框架。

## 最小落地顺序与验收

1. 先本地完善 runner/一次 start/result schema；冻结包含这些源文件的最终 SHA。正在运行的 development 不热替换源码，待根任务统一切换版本。
2. 临时 fixture 验证普通 CLI 仍拒绝 confirmation、预留输出目录不覆盖、同 ticket 第二次或并发启动被拒、startup/loop/cleanup 异常保留 60 raw、已 finished 不重跑、时序/manifest/provenance 错误拒收；不启动模型或仿真。
3. 确认汇总适配器从 frozen runs 真实调用原 scorer 与 `load_panel`，记录完整集合与 finishes；以同原始开发证据核对原接口行为不变。已有 development panels 继续通过旧 SHA 契约。
4. 报告验证完整 frozen 集合及逐面板 audit；失败记录仍显示 Observed。确认资料不存在时 final 继续拒绝，不能把机制测试当确认完成。

以上是整合方案，不构成 freeze 或确认运行许可。根任务随后授权修复与本地机制验证；本审查快照保留，修复证据另行追加。

## 本地修复与接入结果（保留上方原审查）

根任务授权后已实现下述最小接入，**未上传远端、未冻结真实方法、未生成真实确认 jobs/轨迹**。原 `confirmation_plan.md` 的“待接入”描述保留为设计时点记录；本节说明当前本地状态，不替代根任务的后续研究选择。

- `evaluate_candidates.py:597/620/649`：普通 `run` 的参数、jobs、results 目录校验保留；原运行循环抽到 `_execute`。新增 `run_confirmation(ticket_path)`，只由新 `evaluate_confirmation.py --ticket` 进入；参数全部来自 ticket，允许初始化的目录内容仅 ticket/start。source 集合包含 gate、确认入口、未改汇总模块、新适配器和报告器，并逐字段等于冻结来源；运行产生的 history-prefix 等文件作为额外 runtime artifact 保留。确认 flags/provenance 不会被后续依赖初始化覆盖。
- `confirmation_gate.py:290/306`：`start_run` 在原 gate 锁内排他写 start 和 started 事件；已开始/结束或已有产物不可再启动。`validate_ticket` 保持只读可用，finished 后仍可供审核读取；不等于允许再次执行。`check_result` 强制唯一 start、固定 manifest、confirmation flags、完整 sources/artifacts、raw provenance，并调用原 `summarize_panels.load_panel` 检查时间轴和逐字段分数；finished 绑定全部已有输出文件。基础设施无 summary 时保留显式 unknown，不产生 score passed。
- `summarize_confirmation.py:100`：仅收 freeze、冻结参考 ID 和新输出名；从冻结的**全部**方法派生 runs，要求各有唯一 authorized/started/finished。重算当前原始结果并逐文件等于 finished SHA 后，真实调用原 `load_panel/compare` 生成分数与统计。没有改原汇总器、没有手工写分数审核结论。
- 新包沿用原 schema=1 的数值主体；`provenance.script_sha256` 明确代表未修改的聚合模块，另列真实 entrypoint SHA。新增 `confirmation_audit` 携带完整冻结方法映射与 freeze/seal/ticket/start/events/manifest 的原始 JSON 字节快照及 SHA。`summarize_confirmation.validate_bundle(:34)` 只用标准库，验证精确集合、事件和输出身份，供离线报告复用。
- `build_report.py:212`：确认输入必须通过上述授权审计，省略任一冻结方法会拒收。技术失败但 60 raw/score 齐全时可生成 Observed 行，final 仅表示含已审核确认资料，不写控制成功。无 summary、缺文件、时序错误或未结束方法无法获得这类完整审核包。mean 在内部消融组沿用原方法 ID；确认 mean 额外校验冻结 manifest 的 underlying kind/部署规则/非新增外部算法标志。普通开发 mean 仅接受已知 mean→原算法映射及明确 internal_ablation 标签。
- 报告通用提示已说明同算法检查点/部署沿用同一 ID、行数不等于算法数、加粗及相关指标计数只是描述。原 12 面板 checks 继续通过，原汇总器 SHA 未变化。

### 机械证据

| 新证据 | 实际检查 | 边界 |
|---|---:|---|
| checks/confirmation_gate_integrated_mechanics.json | 40 | 原状态机回归，加一次性 start；旧 39 项证据文件未覆盖 |
| checks/confirmation_integration_mechanics.json | 48 | 临时目录 4 方法、240 条 synthetic raw；运行真实共享 loop 和原 scorer，模型/环境传输均为显式 fixture |
| checks/evaluation_mode_confirmation_regression.json | 73 | 既有 formal/smoke/ready 门禁回归；旧 evaluation_mode_gates.json 未覆盖 |

48 项包括启动、动作循环、worker cleanup 三类失败各保留 60 raw；cleanup 在完整轨迹之后失败时，保留真实 100% coverage 与原患者分数，整个方法仍为 Observed，不伪造患者失败。也检查同 ticket 重启拒绝、finished 后只读可查、manifest 缺失、重哈希后的 minute/warmup 错误、raw 冻结 provenance 变更、未结束/冻结子集拒收、技术失败不加粗、普通 CLI 不接确认且不创建目录、普通开发 raw 原评分器一致。独立标准库进程导入报告审计模块没有加载 NumPy/Torch。以上均不是真实仿真、GPU 推理或研究 confirmation。

最终实现 SHA：

| 文件 | SHA256 |
|---|---|
| confirmation_gate.py | 9d750958931e422babd1fdf5999ae1aed1b55c2b09b7c56f4ca4899fc39f6fc5 |
| evaluate_candidates.py | fa135911d85ad29ebe06268144ea19646aa3f97800a2f169927e0be76017947f |
| evaluate_confirmation.py | 4823e61bd64cfed3b72b5f94d12bc7f6850d2f851d36211f0a297bd101f5b239 |
| summarize_confirmation.py | 5b0a5f5562bd17b4756d5436a66982ffb261b1eb9169a46a0a8892b2453af398 |
| build_report.py | f40245cc228207cdbe39762d3423c5a1cd90f9d1871d0ea7c5c6c1c3b55bc9e9 |
| summarize_panels.py（未改） | 82c7a11db9d3316af1cfd9584e6175c594262127831ccad61ac77f16c32c6438 |
| control_metrics.py（未改） | 0a6d44298d942bc09764f15e7ef45a71d3afbc7ba63cad3dae5425f0b22b8cfd |

### 后续 CLI（本轮未执行）

待全部 development 队列完成，再由根任务统一上传新版；不要热替换当前远端 SHA `74249…`。先完成真实方法选择和源/权重冻结清单，再 validate/freeze/authorize；首次正式运行必须满足满 40 轮等原门禁。原评分任务使用 `.venv/bin/python`（Python 3.11 原 NumPy/scorer runtime），模型 worker 仍走既有 native GPU 路由。

```bash
.venv/bin/python RL_DSENet_SOTA_2026-10-03/confirmation_gate.py validate --plan <已审查计划.json>
.venv/bin/python RL_DSENet_SOTA_2026-10-03/confirmation_gate.py freeze --plan <已审查计划.json>
.venv/bin/python RL_DSENet_SOTA_2026-10-03/confirmation_gate.py authorize --freeze <freeze.json> --method-id <固定ID>
.venv/bin/python RL_DSENet_SOTA_2026-10-03/evaluate_confirmation.py --ticket <ticket.json>
.venv/bin/python RL_DSENet_SOTA_2026-10-03/summarize_confirmation.py --freeze <freeze.json> --reference <冻结参考ID> --output checks/panels_confirmation_r1.json
.venv/bin/python RL_DSENet_SOTA_2026-10-03/build_report.py --panel checks/<已审计开发包.json> --confirmation checks/panels_confirmation_r1.json --summary-file <报告解释审核.json> --status final --output <全新报告目录>
```

`evaluate_confirmation` 自动登记 finished，调用者不再手动重复 finish；错误保留后仍返回非零，不能据此自动重跑。全部 frozen 方法未结束或任一结果缺真实可审计 raw 时，确认汇总拒绝生成“全部通过”的包。报告解释的 reviewer/逐 manifest review 是人工解释审核字段，不是 score passed 的替代品。
