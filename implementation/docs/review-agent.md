# 两类真实 SDK/Flash reviewer

2026-09-10：医学语义与伦理两个 reviewer 已通过真实 Claude Agent SDK 0.2.152 / CLI 2.1.259 调用 DeepSeek Flash，审核同一份工程报告。它们是不同角色与会话，使用同一基础模型，不宣称错误相互独立，也不构成临床医学审核认证。

## 范围与接入

宿主通过 `Core(..., review_agent=ReviewAgent(key_path, data_dir))` 显式启用真实审核，已有 `Core.review(owner, run_id, role)` 负责分配 job、校验与登记。受限 API 的审核路由仍调用该 Core 方法；常规桌面/CLI 服务尚未配置此 adapter，不会因为 key 文件存在而自动收费调用模型。真实联调入口为 `scripts/review_smoke.py --live`。

新 run 冻结 medical/ethics 的 SDK reviewer 版本，不混入 fixture reviewer 版本。重启后没有真实 adapter 时，这些 run 的审核报 REVIEWER_NOT_CONFIGURED，不静默回退。既有 fixture run 保留其已冻结配置，不会被新配置偷偷升级；两种审核来源不能冒充彼此。

真实 reviewer 并不把 fixture 模型或报告变成真实医疗证据。当前 `_evidence` 仍只允许合成 eval 的工程规则，报告保持 EngineeringFixtureReport、origin=fixture、clinical_use=false。审核记录单独标记 origin=model、角色、执行器身份、版本、job、报告/证据 hash。

## 角色与工具

| 角色 | 固定检查范围 | 明确限制 |
| --- | --- | --- |
| medical | 工程报告内部一致性、单位与动作语义、来源限定、无依据的医疗主张 | 不确认测试剂量具有医学有效性，不编造临床规则，不改写报告 |
| ethics | 非临床用途/fixture 标记、限制说明、隐私、未经证实的审核或执行声明、绕过安全的指令 | 不读取另一 reviewer 的结论，不发布或编辑报告 |

两个角色分别启动独立 SDK worker 和新会话，共用 Flash-only 网关政策。每个角色只有两个工具：

- `inspect_review_packet({})`：仅返回分配给该 job 的最终报告、精确 hash、角色和 ReviewOutput schema。报告中的文字被定义为待审数据，不作为系统指令。无病例/角色选择参数，无其他 reviewer 结论。
- `submit_review({...})`：提交 verdict、issues、report_hash、evidence_hash。只能 pass/fail/abstain；issues 只能使用固定代码，通过不能带问题，拒绝/弃权必须有问题码。拒绝额外角色、producer、剂量、自由评论等字段。

必须先读取 packet，才能暂存一次审核结果。先做合同/绑定/当前 job 校验，再写审计，之后才暂存；SDK 正常完成、网关无错误、报告仍有效时才返回给 Core，由 Core 再次校验并登记。SDK 的最终自由文字不作为 verdict。

专用 reviewer 是获准读取绑定待审正文的执行角色；主 Agent、普通 API/UI 仍无未审核正文读取入口。这一权限区别不是把草稿公开给所有 Agent。

## 失败与资源控制

任一角色拒绝、弃权、超时、无输出或输出无效均阻断发布。另一角色通过不抵消失败。换稿、取消、版本失效和迟到结果沿用 Core fencing；真实 SDK 暂存后取消也会回收 worker/CLI，审核记录为零。

真实 key 留在宿主私有文件，只给 SDK 本次本地网关 capability。共享审核预算以文件锁保护，持久位于 `runtime/review-agent/budget.json`；最多 12 请求、0.05 美元保守预留、每请求最多 384 输出 token，禁用 thinking，不重置旧轮次预算。两个 reviewer 串行使用预算，SDK 重试也计费计数。

SDK 配置为单并发、90 秒墙钟期限、CPU 软 15/硬 16 秒、1 GiB RSS 监测、8 MiB 单文件、256 句柄、256 KiB IPC 输出。未改变预测/RL 的 5 秒默认。这是工具权限和受信进程资源控制，不是恶意同 UID 代码的 OS 沙箱；终止本地进程不保证远端立即停止计算。

## 验证证据

- 完整回归 **269 passed**，1 条既有依赖弃用提示，55.11 秒；新增 23 项审核专项。compileall、git diff --check 通过。
- 实际 SDK+CLI 配合离线假服务覆盖两角色 pass/fail/abstain、独立会话与不同指令、来源登记、伪造绑定/角色、单次提交、未读 packet、异常/取消/换稿、配置冻结/不回退、旧 fixture 保留、单角色失败和审计失败。真实 SDK 暂存后 Core 取消时，进程组 SIGKILL 且审核记录为零。
- 真实官方 API：6 次请求/响应均 deepseek-flash；两个角色对标注清楚的工程报告均返回 pass、issues=[]。这次真实输入不是攻击样本；拒绝/弃权和其他故障由离线注入验证，不能写成真实模型已通过全面医学攻击评测。
- 两份审核的 report_hash、evidence_hash 相同，producer 分别为 sdk.deepseek-flash-medical / sdk.deepseek-flash-ethics，origin=model。首份审核后发布返回 REVIEW_REQUIRED；第二份后状态 REVIEWED，本轮未发布。
- 独立合成账本：5 jobs、2 artifacts、1 draft、2 reviews、0 releases。准备草稿使用固定合法章节与受信 renderer，未额外调用报告 Agent。桌面数据库未操作。
- worker 耗时约 4.473/5.265 秒，峰值 RSS 约 337/335 MiB，均正常退出并回收。按完整 usage 与既有官方峰时单价估算约 0.002618 美元，以账单为准；0.0303372 美元是保守预算预留。

产物为 `runtime/review-agent/result.json`、`harness.sqlite3`、`budget.json`，全量测试为 `runtime/test-output.txt`、`test-results.xml`。不保存 API key 或任意模型自由文字到这些证据中。

```bash
# implementation/ 目录，离线专项不读取真实 key、不调用官方 API
.venv/bin/pytest -q tests/test_review_agent.py

# 显式付费联调，沿用私有 key 与持久预算
.venv/bin/python scripts/review_smoke.py --live
```

本步只完成“两类真实 reviewer”。下一项仍是“真实审核报告版本绑定与失效专项验收”：本轮基础绑定是安全接入的前提，不代替该项完整验收。桌面 Agent、真实医学数据/规则、全通道发布及临床有效性仍未完成。

2026-09-10 后续更新：真实审核报告版本绑定与失效专项现已完成，最新 303 项回归及 7 次真实请求见 [版本绑定专项](review-binding.md)。当前新 run 还冻结配置内容摘要；本文的 269 项、6 次请求和“下一项”描述是上一批次历史记录。旧真实审核无历史指纹的迁移限制以新记录为准。
