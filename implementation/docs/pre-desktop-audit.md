# 桌面壳之前 TODO 的逐项核验

日期：2026-09-09。范围：总体方案第 7 节阶段 0/1；不推进桌面业务、真实模型或 LLM。下列“通过”仅指已运行的公开工程验证，用户审核仍待完成。

| TODO | 实际实现与补齐 | 可复核验证 |
| --- | --- | --- |
| 占位入口、依赖、窗口与范围 | `adapters.py`、`contracts/README.md`；两个确定性替身、空真实注册表、依赖锁 | fixture 完整 HTTP 报告；research 返回 MODEL_NOT_CONFIGURED；`test_research_never_falls_back` |
| 快照、预测、RL、错误合同 | `contracts.py`；新增请求/产物/ErrorResponse、公开版本化 schema；策略状态与原因互斥 | `test_invalid_snapshot`、`test_strict_json`、`test_candidate_cannot_claim_failure`、`test_contract_export_matches_runtime` |
| 真实/仿真/fixture 标记 | 数据 source 与执行器 origin 分层；执行前匹配 mode，非 synthetic 不运行 fixture | `test_simulation_data_cannot_be_relabelled_as_fixture_by_mode`、`test_executor_origin_must_match_run_before_call` |
| 项目输出路径隔离 | `paths.py`、CLI、Store；解析主文件/锁/令牌/sidecar 路径；既有脚本指定项目缓存和构建根 | `test_database_cannot_write_outside_project`、`test_database_sidecar_symlink_cannot_escape`、CLI 路径/凭据测试；smoke 产物位于 runtime/smoke |
| 作业/事务/产物库 | SQLite WAL，冻结快照，run/job/attempt，原子状态、事件与 outbox，私有报告库 | 完整链路与重启、审计失败回滚、并发发布、outbox 恢复测试 |
| 来源和依赖校验 | Core 注册表授信；输入/父预测/作业/版本复核；受控模板、证据 hash、双审核绑定 | 原有父项/跨 run/capability/版本/改稿测试；新增 artifact job/input 损坏检测；无客户端登记或强制放行 API |
| 最小 CLI | `case-create`、`run-start`、`execute`、`status`、`cancel`、`events`、`report`，均调用同一 API；demo 完整链路 | `scripts/smoke.py` 实际起服务/子进程操作，覆盖默认 research、幂等、取消后再执行被拒和重启读取一致 |
| 取消/重试/fencing | 有限 attempt、过期、取消、输入替换、版本撤销；服务独占恢复未决作业 | 超时/迟到/并发/旧稿/过期/取消测试；HTTP 并发取消；真实重启 smoke |
| 发布 API | 唯一 release，事务内复核并审计；读取再验有效性，状态/事件不泄漏草稿动作 | 双审核/缺任一模型/改稿/撤销/过期/取消/并发发布/审计失败回滚测试；真实 CLI 发布读取 |
| 补充：认证与拒绝审计 | 非 ASCII Bearer 固定 401；schema 拒绝也入审计；无入参操作拒绝客户端伪造正文；405 保留 Allow | `test_non_ascii_credential_is_fixed_rejection`、审计断线测试、`test_no_input_actions_reject_client_facts`、传输错误合同测试 |

## 本轮实测

- 初始基线：75 passed，2 条既有依赖弃用提示，见 `runtime/audit-baseline.xml`。
- 修复后的完整回归：109 passed，2 条既有提示，见 `runtime/test-results.xml`、`runtime/test-output.txt`。
- 独立只读代码复核与中间版本公开回归：98 passed；发现的 Allow 响应头回归已修复并加入最终回归。
- 真实 HTTP/CLI 链路：见 `runtime/smoke/smoke-result.json`；其中的 “real_http_service” 仅表示真正启动了本地服务进程，真实模型标记始终为 false。报告与事件为工程替身证据，见同目录导出。

复现命令（在 implementation 目录）：

```bash
.venv/bin/pytest -q --junitxml=runtime/test-results.xml
.venv/bin/python scripts/smoke.py
```

## 清单修订及尚未完成

原阶段 0 把“真实预测器确认”和“一份真实推理可复现”列为桌面前提，与用户要求先空置两模型冲突。本轮拆成已实现的工程合同和 **0R 真实预测核验**；0R 保持未勾选，接入前必须完成，没有把 fixture 当成真实模型验收。阶段 2 的真实预测 adapter、阶段 5 的真实 RL 仍未勾选。

计划中的真实 LLM 审核、医学规则、独立 worker 隔离、MCP、RAG、Observation 实接、UI 全渠道约束和完整产品对抗测试均不能由这些工程测试代替。后续 TODO 补充了真实执行器隔离验收；本轮只建立闸门，没有启动该实现。

保留约定：每项有实际实现和验证再勾选；发现遗漏/错误据实补充；改动即时记入根目录工作记录。当前代码与数据库均属于可信服务，不保证管理员改写后安全，也不代表临床安全证明。
