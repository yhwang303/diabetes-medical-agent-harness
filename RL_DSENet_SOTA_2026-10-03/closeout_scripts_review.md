# 收尾产物脚本独立审查

日期：2026-10-04。范围仅为 `backup_confirmation.py` 与 `build_closeout_overview.py` 的本地只读审查及隔离机制检查。正式 18 方法确认仍在运行，本审查未读取确认分数，不代表结果或最终产物验收。

## 版本与结论

| 文件 | 当前 SHA256 | 审查状态 |
| --- | --- | --- |
| `backup_confirmation.py` | `622395df430047a6298db5faa7180860c0a808e810c42a2209f6487e135d3eb0` | 原阻断已关闭；未发现新增收尾阻断 |
| `build_closeout_overview.py` | `cda20f33ec695eb3bd4d1493cc699b35f37bdf43cfd900f2635d088717c0d67a` | 逻辑审查版本之后仅追加论文表 CSS；详见下文 |

归档脚本最初只验证审计包的自含快照，再对当前目录建立新哈希清单；这不足以保证归档的 raw、summary 和元数据仍是已评分版本。现第 39–46 行逐一核对当前元数据与审计快照 SHA，并核对每个已审计 finished 事件的全部 `result_sha256`；第 95–96 行再次要求全部已评分输出进入归档条目且哈希相同。原证据绑定缺口已关闭。归档后的逐成员核验仍是独立步骤，不能由这次源码审查代替。

总览脚本已检查构建清单中的 builder、scorer、protocol SHA，核对完整报告输出哈希，并原样复制确认比较 section。该 section 保留 BG 主表、Observed／失败状态、粗体、覆盖、未知尾、剂量补表与 CGM 次表；横向 CSV 明确导出页面的两位小数展示值，精确指标 CSV 仍另行链接。冻结 builder 的主表为 18 列，与解析约束一致。

此前逻辑审查及 Python 3.8 AST 检查对应的总览 SHA 为 `bdec991906609071c38b756c6fede8bbeacdee2200a2ada4a40be4a54a0f4122`。之后仅新增第 55–61 行 `#papers` 的固定列宽、换行和长文本折行 CSS，避免论文说明撑宽；不改变读取、数值、SHA 判断或确认表段复制。本次记录重读了这段 CSS 并取得当前文件 SHA，按要求未重复运行检查。此前 AST 通过记录不冒充对当前 CSS 修订版重新执行的结果。

## 已执行的 4 项隔离检查

临时目录位于本项目 `checks/` 下，运行结束后已删除。检查实际调用归档函数的新增字节绑定分支；为隔离该分支，`validate_bundle` 使用 stub，未构造或提交真实 final 审计包。

| 检查 | 实际结果 |
| --- | --- |
| 改动 raw 字节 | `AssertionError: Scored output changed`，归档创建前拒绝 |
| 删除 summary 文件 | 对该 summary 抛出 `FileNotFoundError`，归档创建前拒绝 |
| 改动 metadata 字节 | `AssertionError: Audited metadata changed`，归档创建前拒绝 |
| 保持所有绑定字节不变 | 通过新增绑定检查，在故意缺失的 freeze 边界抛出 `FileNotFoundError`；未创建归档 |

4/4 达到上述机制预期；最后一项只证明可通过新增绑定检查，不是完整归档成功。两个当时受审脚本的 Python 3.8 AST 检查通过；未运行 Torch、模型、仿真、训练或真实确认。

## 尚未执行的真实验收

未运行真实归档、真实归档成员核验、final 报告／总览构建或最终页面视觉验收；未修改冻结训练、评测、评分、协议及模型文件。完整确认包到齐后，仍须以真实审计包执行归档并逐成员核验，再检查最终总览与冻结 builder 报告、CSV 和来源清单是否一致。本审查不推出成绩、选型成功、SOTA 或临床有效性结论。
