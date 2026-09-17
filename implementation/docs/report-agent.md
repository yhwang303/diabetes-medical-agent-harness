# 受限报告 Agent 与 renderer 联调

2026-09-10：真实 Claude Agent SDK + DeepSeek Flash 已生成受限章节提案，并由 Core 从 fixture 权威产物渲染工程草稿。此项验收不代表真实医疗报告、真实审核或桌面 Agent 入口完成。

## 执行入口和工具

`ReportAgent.generate(owner, run_id)` 是宿主侧入口，供后续业务层调用；当前真实验收入口为 `scripts/report_smoke.py --live`。它复用已有 Core，而非另建一个由 Agent 决定有效性的结果库。验收脚本为避免影响桌面数据，创建独立的合成病例账本。

| 工具 | 模型可提交的参数 | 返回内容 |
| --- | --- | --- |
| inspect_report_contract | 空对象 | 当前证据摘要、模板版本和 Proposal schema；无原始病例数据、动作值或凭据 |
| inspect_report_status | 空对象 | 当前 job 状态、是否暂存提案、仍需审核；无报告正文 |
| submit_report_proposal | sections 数组 | 合法时仅返回 PROPOSAL_STAGED；错误仅返回固定错误码 |

三个必需章节为 forecast、policy、limitations，必须各出现一次；允许选择顺序，不允许增加、删除或重复章节。提案不接受文字、剂量、频率、诊断、HTML、病例 ID、证据 hash 或 capability。SDK schema 和 Core 的 Proposal 校验分别执行；即使绕过 SDK 直接请求工具端点，宿主仍会拒绝非法输入。

这是在既有三项 SDK 烟测工具之外新增的三项报告工具，按任务 profile 授权。报告任务不获得预测、审核、发布或子 Agent 权限；不是把六个工具全部默认开放。

## 提案暂存与最终提交

1. 宿主调用 Core.create_draft_job，先验证当前双模型证据和工程规则。单次草稿凭据留在宿主，不交给模型或 SDK worker。
2. SDK 在独立注册的 report worker 中运行；每次工具调用及执行期间持续复核当前 job、run、证据和期限。
3. 合法提案在审计成功后先暂存于宿主内存；此时数据库仍为 DRAFT_PENDING，没有草稿。
4. 只有 SDK 返回正常完成、模型网关无失败且确有暂存提案，宿主才调用 Core.submit_proposal。
5. Core 在提交事务中再次检查凭据、job、当前证据，然后由受信 renderer 读取原始预测/RL 产物生成草稿并计算 hash。
6. 对调用方仅返回 draft_id、report_hash、DRAFT_READY。SDK 的自然语言、流式文本和未审核报告正文不作为返回值。

超时、崩溃、SDK 错误、取消、换稿、快照变更、审计失败均不提交暂存内容。即使在最终校验后、提交事务前取消，Core 仍会拒绝。新任务/取消状态优先于旧任务的迟到失败，沿用原 job fencing 规则。

模型负责合同范围内的章节提案，renderer 的模板、数值来源和医学语义不由模型修改。这里保留 EngineeringFixtureReport、origin=fixture、clinical_use=false；真实 LLM 参与不能把占位数据升级成真实预测或临床依据。

## 权限、费用与审计

报告 Agent 复用 Flash 固定上游地址、请求和响应模型白名单、384-token 输出上限及禁用 thinking。`runtime/report-agent/budget.json` 单独持久记录本轮报告预算，最多 12 请求、0.05 美元保守预留，不清空旧 SDK 烟测预算。宿主以文件锁避免同一预算被并发使用。真实 key 使用已有私有文件，SDK 仅拿到短期本地网关 capability。

报告任务使用既有 SDK 资源配置：单 worker、90 秒墙钟、CPU 软 15 秒/硬 16 秒、1 GiB RSS 监测、单文件 8 MiB、256 句柄、256 KiB 输出。该配置是受信执行代码的资源控制和工具权限，不是抵御恶意同 UID 进程的 OS 沙箱。

Core 记录报告工具接受/拒绝、job、worker 生命周期、SDK 完成时的身份/版本/模型/session，以及实际 draft_created。审计不写原始工具参数或自由文字。SDK hook/usage 证据保留在私有验收记录中，尚未接通外部 Observation。

## 验证

最终完整回归 246 passed，1 条既有依赖弃用提示，33.98 秒；本步新增 24 项报告专项和 1 项 worker 并发关闭回归。Python compileall 与 git diff --check 通过。

离线新增报告专项包括：实际 SDK 的 schema 拒绝、Core 的重复章节拒绝、修正后成功、模型自由文字不进入报告、精确产物取值、剂量/频率/诊断/HTML/伪造绑定、缺证据/越权、提案重放、失败后不提交、审计失败、换稿/换快照/取消及最终提交前取消。另用实际 SDK+CLI 验证暂存后的 Core 取消会终止整个 worker 进程组且不生成草稿。

真实 Flash 联调证据：`runtime/report-agent/result.json`、`harness.sqlite3`、`budget.json`。本轮 3 次模型请求均请求/响应 deepseek-flash；Agent 调用了全部三个报告工具，最终状态 DRAFT_READY。账本为 3 jobs、2 artifacts、1 draft、0 reviews、0 releases。报告 hash、数值与来源逐项核对通过，发布仍返回 REVIEW_REQUIRED。

实际 SDK worker 约 13.631 秒，峰值 RSS 约 304 MiB，退出 0 并回收。按完整 usage 及既有官方峰时单价估算本轮约 0.000529 美元，以 DeepSeek 账单为准；0.0119748 美元是预算预留值，不是已扣费用。

```bash
# implementation/ 目录；离线，不读取真实 key 或调用官方 API
.venv/bin/pytest -q tests/test_report_agent.py

# 显式真实调用；使用已有私有 key 和持久费用预算
.venv/bin/python scripts/report_smoke.py --live
```

本步没有连接真实 reviewer、发布模型生成的报告、修改医学规则或接通桌面 Agent。下一具体 TODO 为“两类真实 reviewer”。
