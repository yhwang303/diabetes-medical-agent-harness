# 独立执行进程：实现与验收

2026-09-10：生产 Core 的预测、策略和两类审核均通过 `WorkerRunner` 启动独立 Python 解释器。Core 保存快照、作业、产物与审计，worker 只收受控 JSON 请求并返回未受信结果。Core 原有来源、版本、输入摘要、父预测、当前作业、审核绑定与发布校验继续执行。

## 执行边界

- 固定注册表解析执行器身份/版本/来源；不能从 HTTP 或桌面传任意模块、Python 对象、命令或模型结果。替换为未注册 Python callable 会失败，不会回退到 Core 内执行。
- 使用独立解释器、`-I -B`、独立进程组、关闭无关文件句柄、最小环境；不继承 Core 的 token/API key 环境或数据库句柄。工作目录和 TMPDIR 为项目 `runtime/workers/`，stderr 丢弃，不把模型异常文本写到日志。
- 输入/输出是有长度上限的 JSON；输出仍需通过原有模型合同。输出过大、无效传输、进程崩溃、限制无法设置或监测不可用均失败关闭。
- 作业期限包含启动开销。执行时复核当前作业、快照、版本、有效期与取消状态；失效后杀死进程组并回收 leader。审核换稿同样会 fence 旧作业。
- 进程正常完成也清理同组普通后代；Core.close 终止所有活动 worker。Core 被 SIGKILL 时，生存管道写端关闭，worker 监测线程杀死自身进程组。这里没有完成阶段 8 的“App 强制退出后 Core 服务孤儿回收”TODO；该生命周期范围仍单列未完成。

## 当前工程资源配置

| 资源 | 限制与实际机制 |
| --- | --- |
| 并发 | Core 内最多 4 个活动 worker；无无界排队，超出返回 EXECUTOR_BUSY |
| 墙钟时间 | 默认每作业 5 秒，期限到达即 SIGKILL 并 wait 回收 |
| CPU | POSIX RLIMIT_CPU：每进程软 2 秒、硬 3 秒；普通后代继承限制 |
| 内存 | 进程组 RSS 合计阈值 256 MiB；通过系统 ps 约每 50 ms 采样，另有 ps 调用耗时，超限终止 |
| 单个文件 | RLIMIT_FSIZE 1 MiB；不是整个目录的总磁盘配额 |
| 打开文件句柄 | RLIMIT_NOFILE 64 |
| core dump | RLIMIT_CORE 为 0 |
| JSON 输入/输出 | 各不超过 128 KiB；父进程非阻塞读取并限制输出缓冲 |

内存是采样后终止，短暂峰值可以超过阈值；审计中的 peak_rss_bytes 是采样观测峰值，不保证捕获瞬时最高点。CPU/句柄/单文件限额为进程级，不能宣称为进程组累计 CPU 或总磁盘限额。当前在本机 macOS arm64 验证，尚无跨平台分发验收。

这是固定可信执行代码的故障/资源隔离，不是恶意代码的完整 OS 沙箱：进程仍为同一 OS 用户，没有独立文件系统/网络身份限制；不以普通进程组清理宣称抵御主动 setsid 逃逸。未来开放任意代码或文件/网络工具仍须独立沙箱与权限验收。未来 SDK/API 的本地进程终止也不等于远端服务已停止计算。

这些默认资源适用于当前工程执行器；真实 Agent SDK/LLM/模型接入时必须按真实负载验证受信配置、子进程、凭据注入和工具权限，不能直接把本轮 fixture 验收当作真实 Agent 已接入。

## 验证证据

2026-09-10 报告 Agent 联调补充：并发 close()/任务 finally 的终止操作现由同一锁认领，成功回收后移出 active 表，避免对已回收进程组重复 killpg。全量回归曾真实触发 macOS PermissionError，现有确定性专项模拟重复信号拒绝；未通过忽略权限错误绕过清理。当前 25 项 worker 专项与 24 项报告专项共 49 项通过，完整回归 246 项通过。原 200 项记录为当时阶段证据。

- 全量 200 项通过，2 条既有依赖弃用提示；`runtime/test-results.xml`、`runtime/test-output.txt`。
- 新增 24 项进程专项：超时、CPU/内存/输出超限、崩溃、协议错误、OS 文件/句柄限制、最小环境、并发预算与关闭、取消/换快照/撤销、父进程 SIGKILL 与普通后代终止、监测/审计失败关闭，以及 8 个 RL 场景在生产进程链上过闸。测试故障程序只在 tests 下，不进入生产注册表。
- 旧测试保留任意函数与线程事件故障注入，其 ThreadRunner 仅在 `tests/thread_runner.py`，不由服务导入。不能把所有 200 项笼统称为独立进程测试；生产链另由专项和 HTTP/native 验证。
- 实际 HTTP/CLI 双模型、两类审核、发布、取消与重启一致性通过：`runtime/worker-smoke-result.json`，对应 `runtime/smoke/` 的产物与事件。
- 原生实点：正常对照两次 worker 分别退出 0；超时策略 PID 35957 的 PPID 为 Core 35905，独立 PGID 为 35957，约 5.001 秒后 exit_code -9，随后 PID 消失、策略未登记、发布仍阻断。证据在 `runtime/desktop/worker-verification.json`、`worker-build.log`。
- 原有 2 个病例及 9 个旧任务/17 个旧作业/11 份旧产物逐字段保留；本轮原生仅新增正常与超时两个任务，没有审核或发布。前后数据见 `runtime/desktop/worker-before.json` 和核验 JSON。
