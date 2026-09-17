# 真实审核报告版本绑定与失效专项

2026-09-10：本项完成。完整回归 **303 passed**，新增 **34 项**绑定与失效测试；真实 SDK/Flash 双角色审核及基于真实审核记录的隔离账本检查通过。报告与预测/RL 仍为合成工程 fixture，不是临床验证。

## 修复的实际缺口

修复前的三个测试均失败，证据为 `runtime/review-binding-before.txt`：

1. 命中缓存时只返回 verdict，没有重新校验完整审核记录；篡改 producer 未被发现。
2. 改动提示词内容但不修改版本标签时，旧审核仍可用于发布。
3. 已接受审核 job 的 input_digest 被改动后，发布仍能通过。

现在缓存读取、发布及已发布报告读取共用完整审核校验；登记前也重新构造当前请求，复核绑定、执行器和 job 输入摘要。真实 SDK 审核仍先暂存，完整成功并且绑定有效后才由 Core 登记。

## 冻结内容与绑定链

Core 创建真实审核 run 时，为 medical/ethics 分别持久保存 `review_configs` 快照及 canonical SHA-256 摘要：

- 角色、执行器身份与版本、模型 ID、SDK/CLI 版本和固定官方端点。
- 系统提示、任务提示、两个工具的描述及 schema。
- 无内置工具/子任务/项目设置、权限策略版本、thinking 关闭、最大轮数和输出上限。

实际提示词、工具描述/schema、模型与端点来自运行时定义，修改内容会改变摘要，即使人工版本标签没有变。SDK worker 启动前核对宿主发来的摘要，并检查实际 SDK 版本及 CLI 初始化版本；其结果元数据回报同一配置摘要。权限执行逻辑等程序语义仍由受信代码与显式版本管理负责，配置摘要不是整个程序的可复现构建证明。

每份真实审核包含由 Core 赋予的 `binding`：run_id、case_id、snapshot_id、draft_id、revision、role、config_hash。reviewer 不能填写或修改这些身份字段；它只能返回固定 ReviewOutput。审核 job 的 input_digest 覆盖报告正文、报告 hash、证据 hash 和完整绑定。登记后的审核记录也有独立摘要。

读取时检查：记录完整性和 schema、列与正文一致、当前草稿/修订/证据、角色与注册执行器、配置快照与当前配置一致、来源与版本、对应 job 已接受及其输入摘要。报告正文完全相同但新建了修订，仍必须重新获得两份审核。

## 失效与旧数据

| 情形 | 已验证行为 |
| --- | --- |
| 提示词、任务提示、工具描述、模型、SDK/CLI、端点或输出限制变化 | 缓存、发布和已发布读取返回 REVIEW_CONFIG_CHANGED |
| 缺失/损坏配置快照 | REVIEW_CONFIG_MISSING / REVIEW_CONFIG_INTEGRITY |
| 审核正文、schema、角色/绑定、job 输入摘要被改动 | REVIEW_INTEGRITY / REVIEW_INVALID |
| 更换快照、撤销 reviewer/规则/模板、取消、过期 | 旧审核不能支持发布或继续读取 |
| 同文新修订 | REVIEW_REQUIRED；旧审核保留，不挪用到新修订 |
| SDK 暂存后配置变化 | 作业被阻断，worker/CLI 进程组回收，审核登记数为零 |
| 进程重启，配置和依赖保持有效 | 缓存审核正常复核，无新增模型请求 |

迁移仅新增 `review_configs` 表与可空 `reviews.digest` 列。旧 fixture 审核保留原合同，通过既有身份/证据/job 校验，已验证仍可读取和工程发布。旧真实审核没有当时冻结的配置快照，不能用今天的配置反向补造历史；记录完整保留，但发布资格被拒，需要新建 run 并重新审核。

上一轮真实 SQLite 数据库通过 backup 复制后迁移，原有病例、快照、run、job、产物、草稿、审核、事件和 outbox 的原有列逐项一致。该迁移检查将验证时钟固定在原 run 过期前，仅用于单独验证缺失配置的拒绝原因，未改原库或延长实际任务有效期。

## 实际验证记录

- 全量 303 passed，1 条既有 Starlette/AnyIO 弃用提示，59.92 秒。新增专项 34 passed，6.54 秒；compileall、git diff --check 通过。
- 两个真实 SDK 会话调用官方 API 共 **7 次**，请求/响应均为 deepseek-flash，两个角色均返回 pass。Core 快照摘要、审核绑定与 worker 回报摘要逐项一致。
- worker 用时约 7.025/3.774 秒，峰值 RSS 约 334/344 MiB，均 exit 0 并回收。持久预算预留 **0.0386172 美元**；这是保守请求准入预留，不是实际账单。
- 主联调账本保持 5 jobs、2 artifacts、1 draft、2 reviews、0 releases。只在独立副本中验证工程发布/读取；改提示词后旧读取被拒，同文新修订被拒，重算摘要的错角色审核也被拒。
- 重启、篡改和迁移检查明确禁止调用 reviewer.execute，预算文件前后字节一致，未追加付费请求。
- 上一轮真实记录保留：1 case、1 snapshot、1 run、5 jobs、2 artifacts、1 draft、2 reviews、27 events、27 outbox；原数据库未操作。

证据：`runtime/review-binding/result.json`、`binding-result.json`、`harness.sqlite3` 和四个检查副本；专项日志 `runtime/review-binding-tests.txt`；全量 `runtime/test-output.txt` / `test-results.xml`。源码与文档不包含凭据。

```bash
# 离线测试，不使用真实凭据/官方 API
.venv/bin/pytest -q tests/test_review_binding.py

# 新验证环境中显式调用真实模型；已有证据时脚本拒绝隐式重跑
.venv/bin/python scripts/review_binding_smoke.py --live
```

## 验收边界

配置摘要固定的是本地可观察配置及 provider 返回的模型 ID；DeepSeek 的服务别名不提供不可变远端权重快照，因此不能据此证明云端权重从未升级。两个角色使用同一 Flash 基础模型，不视为统计独立审核。数据库摘要用于完整性与绑定检查，不抵御掌控受信服务/SQLite 的管理员全面改写。

本次没有接通桌面 Agent 入口，也没有完成产品全通道发布限制验收；真实预测/RL、医学规则及临床审核有效性继续保留未完成状态。
