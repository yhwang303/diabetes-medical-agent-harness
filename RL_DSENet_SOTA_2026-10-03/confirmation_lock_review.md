# Confirmation gate 并发锁修复与验证

日期：2026-10-04。根任务指出：原 `.gate.lock` 使用 `O_EXCL`，遇到另一方法正在完成依赖核验/评分登记时会立即抛 `FileExistsError`。正常的两个方法并发结束或启动/结束交叉，因而可能被当成执行错误。此时尚未进行真实确认冻结，本次按授权仅修改 gate 锁获取，新增本地检查与本文。

## 最小修改

[confirmation_gate.py](confirmation_gate.py) 新增 `time`，`lock(directory, timeout_seconds=30.)` 改为在 monotonic 期限内获取同一个排他文件锁：仅捕获 `FileExistsError`，每次最多等待 50 ms；默认最多等待 30 秒。其他文件系统错误立即传播。调用者及锁内的 validate/start/finish 操作均未修改。

成功取得锁后，仍由原持有者记录 PID 并在退出时删除自己的锁。超时抛出包含路径的 `TimeoutError`，保留既有锁；不会判断 PID 是否存活、自动删除陈旧锁或清理已有输出。进程被强制结束后留下的锁仍需按原恢复边界处理。

**等待只重试锁获取，不重试 ticket、方法、评分实验或确认运行。** 同 ticket 第二次开始、重复结束、重新授权及覆盖输出仍由原门禁拒绝。该 30 秒限制约束锁获取等待，不限制持锁核验本身的时长；超过期限仍是明确错误，不伪造完成事件。

## 实际本地验证

| 新证据 | 检查数 | 验证范围 |
|---|---:|---|
| [confirmation_lock_mechanics_r1.json](checks/confirmation_lock_mechanics_r1.json) | 25 | 真实 OS `spawn` 多进程竞争、等待释放后获得锁、共享计数确认临界区无重叠；活持有者下超时、陈旧锁超时均保留字节和 inode；正常/异常退出释放自身锁；非竞争错误不重试；默认期限 30 秒与 Python3.8 AST。 |
| [confirmation_gate_lock_regression_r1.json](checks/confirmation_gate_lock_regression_r1.json) | 40 | 原 gate 状态机与一次性 start 回归：冻结集合、来源变更、重复授权/结束、失败记录、满额训练门禁等。 |
| [confirmation_integration_lock_regression_r1.json](checks/confirmation_integration_lock_regression_r1.json) | 48 | 原整合回归：临时 fixture 的共享 runner loop、4 方法/240 条 synthetic raw、原 scorer、startup/loop/cleanup 失败各保留60条、finished 后只读核验、普通 CLI/development 门禁及完整集合报告检查。 |

新锁检查使用进程共享占用计数判定互斥，不比较不同进程的 monotonic 时间原点；等待时长只在各自进程内测量。最初检查草稿的跨进程时间戳断言被替换为该共享状态断言后，最终 25 项通过；没有据草稿失败修改生产锁机制。

40/48 项回归使用未修改的既有检查源码。执行包装只将它们最后的检查结果写到上表的新 JSON 名称，临时 fixture 行为不变；原 `confirmation_gate_integrated_mechanics.json` 与 `confirmation_integration_mechanics.json` 保留。检查保护的源码/配置/已有 JSON 在测试期间未变。历史 40 项输出保留了原测试自带的旧 limitation 字符串；当前整合状态以 [整合审查](confirmation_integration_review.md) 及本次 48 项真实检查范围为准。

这些测试只在 `checks` 下排他新建的临时目录中使用 synthetic 元数据，未 freeze/authorize 真实研究选择，未生成真实确认场景/轨迹，未导入 Torch/simglucose，未训练或连接远端。机制通过不构成控制效果、模型晋升或临床安全证明。

## SHA 与改动边界

| 文件 | SHA256 |
|---|---|
| `confirmation_gate.py`（修复前） | `9d750958931e422babd1fdf5999ae1aed1b55c2b09b7c56f4ca4899fc39f6fc5` |
| `confirmation_gate.py`（修复后） | `262908efbeb6e444782706fdc844b3e1477bb1904becc7a237bdf51725851559` |
| `checks/check_confirmation_lock.py` | `997c97b07bb59f257a91c5befbc2181e542e0d5c1c9cdf0ea1b1a543641ce5c4` |
| `checks/confirmation_lock_mechanics_r1.json` | `5745d8943937bb96f71c8c6c2aabf4ee2102e8aae35da71e04374a0efc9a3464` |
| `checks/confirmation_gate_lock_regression_r1.json` | `0b212788dbd835bb4426204677aca13d7a9aae01e0bf55fdc6fbdd7d77247a1a` |
| `checks/confirmation_integration_lock_regression_r1.json` | `fb667b36ade840f363c768fe9c12995027a3c38cc8a42a7e9a35c0f8b542d2f1` |

本次未修改 evaluator、runner、原汇总器、报告器、权重、训练或原评分器。复核 evaluator SHA 仍为 `fa135911d85ad29ebe06268144ea19646aa3f97800a2f169927e0be76017947f`，原汇总器仍为 `82c7a11db9d3316af1cfd9584e6175c594262127831ccad61ac77f16c32c6438`，原 scorer 仍为 `0a6d44298d942bc09764f15e7ef45a71d3afbc7ba63cad3dae5425f0b22b8cfd`。根任务须上传并绑定新的 gate SHA 后，再按既有流程作真实冻结；本次不执行这些步骤。

可用标准库 Python 在本地复跑锁检查，必须使用全新输出名：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 RL_DSENet_SOTA_2026-10-03/checks/check_confirmation_lock.py \
  --output RL_DSENet_SOTA_2026-10-03/checks/confirmation_lock_mechanics_<新名称>.json
```

## 同批后续：事件原子发布与最终版本

根任务随后指出 `events()` 在锁外读取 `events/*.json`，原 `append_event` 直接写可被 glob 发现的最终 JSON，因此读者可能读到部分内容。上方 SHA `262908…` 是仅完成锁等待的中间版本，保留其检查历史；**本批最终 gate SHA 为 `8e25df56e5177f022d2b69ced63fbe736d06c04303eceffa39d1bd7d94c30d44`。**

`append_event` 的调用者仍持原 gate 锁。事件先排他写入同目录带 UUID 的非 `.json` 临时文件；写完并关闭后，用 `os.link(temporary, final_numbered_json)` 原子、无覆盖地发布，再删除临时文件。编号仍按原规则计算；最终编号存在时直接拒绝，不替换、不跳号重试。写入或 link 失败保留临时字节，不自动清除陈旧临时文件。只读 `events()` 继续只 glob `*.json`，不改变 event schema、ticket 或科学实验行为。这里保证并发读者的完整发布可见性，不宣称新增了断电持久性或自动崩溃恢复。

[check_confirmation_events.py](checks/check_confirmation_events.py) 实跑了两个独立 spawn 写者和一个锁外读者。写者通过真实 `append_event` 发布 12 个事件，并故意分块缓慢写临时文件。读者实际读取 **323 个完整快照**，其中 **236 次读取时存在正在写的临时文件**，没有 JSON 解析错误或残缺 payload，最终 12 个事件编号连续且各出现一次。另验证未完成临时文件被忽略且保留、NaN 写入失败不发布、最终编号碰撞不覆盖并保留临时文件。18 项全部通过，无真实研究事件。

最终版本又重新通过 25 项锁检查、40 项 gate 回归与 48 项整合回归；新输出如下，前一轮结果均保留：

| 文件 | SHA256 |
|---|---|
| `confirmation_gate.py` | `8e25df56e5177f022d2b69ced63fbe736d06c04303eceffa39d1bd7d94c30d44` |
| `checks/check_confirmation_events.py` | `0867763ecabbbd3c6334fc88c044a92ec6f73a9fc52b0b8197c3ca0a5c644483` |
| `checks/confirmation_event_atomic_mechanics_r1.json`（18 项） | `66f0315e2310a31e019001421268fe512798083d8c608d10168544dcbf6932e3` |
| `checks/confirmation_lock_atomic_mechanics_r1.json`（25 项） | `9945a36cf89a12761c96366bfd3fe91092053d141aa9aa6eac82c7c3a6ee066e` |
| `checks/confirmation_gate_atomic_regression_r1.json`（40 项） | `d13b3ec351e46de99382673d9ec21861066c7a65517c16d9f06da2b8291cfa7e` |
| `checks/confirmation_integration_atomic_regression_r1.json`（48 项） | `cc82535efbb5088df1122916294bb54e4c4ea9d63c966827ef5d508870c84d35` |

## 薄调度器只读审查

审阅根任务新增的 [execute_confirmation_set.py](execute_confirmation_set.py)，SHA `76ef01587bbc60f2bc9072c23a06058d5dbaa96bfce561efa664c8988a3244d5`，Python3.8 AST 通过；未修改或执行该脚本，未创建实际队列、freeze 或 ticket。

未发现阻断项：第 18–30 行先核真实 freeze、拒绝已有事件与已有审计目录，且在任何子进程产生结果前按冻结顺序授权全部方法；第 31–42 行只调用绑定 runner，参数只有既有 ticket，使用当前 Python runtime；第 46–58 行对无 finished 的子进程明确登记基础设施失败，保留 exit/log/ticket SHA，不自动重跑；第 62–69 行最多两方法并发，完整集合固定，不按分数改顺序或删行，非零退出/登记错误使队列非零退出。

`fixed_set_executed/all_methods_reported` 只描述调度与登记，不能解释为全部控制完整或研究目标通过。子进程尚未 start 就退出时，fallback 只能保留明确未知/失败；原确认汇总仍按真实 start/manifest/raw 证据决定是否可完整审核。整个协调进程被 kill 或磁盘不可写的恢复并未实现，不能宣称能自动补跑；它也没有自动恢复入口。门禁、模型、评分与一次性 runner 合同保持原约束。本审查不代替其将来的真实运行验收。
