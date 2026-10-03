# 方法 registry 状态追加说明

日期：2026-10-04（Asia/Shanghai）。本批只更新 `method_registry.json`，新增本说明；未联网、连接远端、训练、推理、重新评分或生成确认资料。

registry 新增根字段 `current_status_snapshot`。当前运行状态以此字段为准；原 `snapshot_date=2026-10-03`、全部方法定义、status、runtime_status_reports 和所有历史 evidence/source SHA 原样保留。旧的 running/smoke-only 文本是历史快照，不再代表当前状态；历史源码 SHA 没有被替换成当前文件 SHA，也没有被冒充为本批新验证。

| 当前状态 | 本地证据 |
|---|---|
| 25 行 development，1,500 条 raw 均存在并经原评分器逐字段 exact 核验 | `checks/panels_development_25_rows_r1.json` |
| 19 行随访完整；6 行旧方法随访不完整，共 44 次 `native_environment_done`，未知尾部保留 | 同一面板包；6 行为 ReBRAC、LOM、GFP、TD3+BC、FQL、DITR frozen+projection |
| wide PPO 和 quantile-world PPO 均完成 40 轮；8/16/32/40 开发面板均完成 | `completed_wide`、`completed_world_ppo` 的归档索引、本地 proof，以及归档 completion/history/config |
| 新 IQL 固定最终 20k 已完成；point/quantile world 均完成 4,000 步 | `completed_models` 的归档索引、本地 proof 和归档 completion |
| 计划 18 个 confirmation 方案，尚未冻结或生成 jobs/轨迹 | 主任务本次明确状态；已有 gated entry 和本地机制验证不等于真实确认已执行 |
| planner BC 工具已实现并作本地审查，仍未激活 | `planner_bc_tooling.md`、`planner_bc_independent_review.md`；不与 registry 中另一个 matched-replay BC 提案混同 |

同一算法的多个检查点/部署消融分别保留 panel ID，并映射原 method ID；25 行不等于 25 个独立算法。PPO16 mean 仅记为已有权重的事后部署消融，不补称原预注册或新外部方法。完成的 world PPO 是 quantile-world 版本，不能连带将原 point-world PPO 提案标成已训练。

本次重验三份归档整体 SHA，并从本地归档内存读取 completion/config/history、逐成员验 SHA；没有解包。全部 40 行 PPO history 顺序及 8/16/32/40 checkpoint SHA 与对应开发面板一致。1,500 条 raw 的 exact 分数核验沿用已存在的审计包，本批不宣称重新执行评分器。新增状态还区分“运行结束”与“完整随访”，不把 6 行不完整面板改写成完整成功。

状态字段记录了全部证据路径/SHA、各面板 manifest/summary/checkpoint SHA 和训练归档成员 SHA。更新前 registry SHA 为 `e040931c53a6861046e2e3abe8e168c4820aaa13fd01e84f255f4bfc73992f9e`；程序核对移除新增字段后的对象与原对象完全一致。主要开发包 SHA 为 `13a849dc0785ceb54368c788e470b5b2024978fef70de35b883dd0a64e4f5bb4`。

本批不作成绩选型、confirmation 授权或 SOTA/临床有效性判断，不激活 BC 工具，不改变冻结对照或评分口径。
