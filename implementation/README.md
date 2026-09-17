# Medical Research Harness · 0.1.0

2026-09-11 批次 26：P04 受控 RL MCP 完成。真实 SDK 通过 request_rl / inspect_rl 请求/查询 Core 管理的独立 policy worker；父预测由 Core 装配，新增独立 rl 授权与父 SDK/数值子 job/配置/预测摘要绑定，下游只消费 SDK 完整成功的结果。全量 604 项通过（新增 56 项）；实际双 SDK/HTTP/worker 与故障回收通过，上游均离线 mock，模型继续 fixture。见 [RL MCP](docs/rl-mcp.md)。设置已显示 RL 工具定义与入口状态，原自检仍仅预测；HTTP 业务入口默认关闭，P05 主循环未完成。下一项 P06，待审核。以下历史状态以最新工作记录为准。

当前可运行的是医疗研究 Agent 的控制核心、HTTP API 和 CLI，另已完成 [原生桌面病例导入与时间线](app/README.md)、预测/RL fixture 演示，以及 [真实 Agent SDK + DeepSeek Flash 兼容烟测](docs/sdk-flash-smoke.md)。桌面通过受限原生桥接连接 Core，已接入受限报告 Agent 任务，完整报告双审/发布业务仍未完成。真实预测模型、真实 RL 模型均未接入。`eval` 模式使用显式测试替身验证闸门；默认 `research` 模式在模型未配置处阻断。

持续工作记录位于项目根目录 `../工作记录.md`；后续开发遵守 `../AGENTS.md`。

## 快速运行

在本 `implementation/` 目录执行。已为当前 Mac 建好 `.venv`。

```bash
# 一条命令完成：启动独立服务、CLI 演示、研究模式阻断、停服重启核对，最后停止服务
.venv/bin/python scripts/smoke.py

# 执行对抗与功能回归，临时数据保留在 runtime/ 下
.venv/bin/pytest -q --junitxml=runtime/test-results.xml
```

检查结果：`runtime/smoke/smoke-result.json`；完整工程报告：`runtime/smoke/fixture-report.json`；事件：`runtime/smoke/events.json`。这些都是 synthetic/fixture 工程证据，不是临床数据。重复 smoke 会创建新 run，旧数据库记录保留，最新导出文件覆盖上一次导出。

如果需要持续运行服务：

```bash
# 终端一：只有显式开关才允许工程 fixture
.venv/bin/medical-harness serve --enable-fixtures

# 终端二：CLI 调用上述服务，不直接写数据库
.venv/bin/medical-harness demo
# 也可逐项调用；run-start 默认 research，eval 必须显式指定
.venv/bin/medical-harness case-create examples/synthetic-case.json
.venv/bin/medical-harness run-start CASE_ID --mode eval --idempotency-key MY_UNIQUE_KEY
.venv/bin/medical-harness execute RUN_ID
.venv/bin/medical-harness status RUN_ID
.venv/bin/medical-harness cancel RUN_ID
.venv/bin/medical-harness events RUN_ID
.venv/bin/medical-harness report RELEASE_ID
```

默认监听 `127.0.0.1:8787`。访问令牌自动保存在 `runtime/operator.token`（0600），不打印到日志；不要把此令牌交给 Agent。Agent 只领取 Core 指派的单作业 capability。服务默认不启用浏览器跨域请求或 Swagger UI；`/openapi.json` 提供接口描述。桌面已通过受控后端桥接调用病例 API，没有放宽 Origin 策略。

默认安全启动方式：`.venv/bin/medical-harness serve`。这种配置拒绝创建 eval run；真实模型注册表为空，研究流程不会退回 fixture。

重新建立环境（Python 3.12+，使用本地已安装 Python）：

```bash
uv venv --python python3.12 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r requirements.lock
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
```

如果新环境未包含 setuptools，先按 pyproject 的构建要求安装 setuptools，再执行 editable 安装。`requirements.lock` 锁定本次运行与测试依赖，不包含个人路径、令牌或模型权重。

## 两个模型如何占位

| 执行器 | 当前行为 | 保证边界 |
| --- | --- | --- |
| 预测 fixture | 将最后一个观测值重复为六个测试点，显式声明 uncertainty unavailable | 只验证预测合同，不评价预测精度 |
| RL fixture | 固定工程动作 0.01 U/min、5 分钟，标为 fixture_only | 不是推荐剂量，不训练、不模拟、不执行 |
| medical/ethics reviewer fixture | 检查固定报告的模式和警示；分别签发绑定版本的作业结果 | 不是 LLM、专家或临床审核 |
| 真实模型注册表 | 空 | research 模式返回 MODEL_NOT_CONFIGURED |

Fixture 只接 synthetic 病例。未来真实模型必须经受信启动配置注册，固定执行器身份、版本、输入/输出合同和适用性规则，不能仅删除 fixture 标签就上线；当前研究规则/真实审核也未配置。

固定合同与适配器说明见 [contracts/README.md](contracts/README.md)，JSON Schema 见 `contracts/harness-engineering-v1.json` 和 `GET /contracts`。historical/simulation/synthetic 数据声明与 model/simulator/fixture 执行器来源分开；simulation 当前仅可冻结，不运行仿真。

工程规则中 `0.05 U/min` 是检测拒绝路径的任意测试上界，**不是医学阈值**。失败与弃权不会被映射为动作零；合法零值仍是 candidate。被拒绝后的流程终止于阻断，无继续沿用上次动作、给药设备接口或仿真器调用。

## 结构与信任边界

```text
App/CLI/未来 MCP → 带身份的 HTTP API → Core
                                     ├─ 冻结快照 / run / job / attempt
                                     ├─ 受信执行器 → 严格合同 → 权威产物
                                     ├─ 工程安全检查 / 依赖 manifest
受限 Agent ← 一次性草稿 capability ────┤
受限 Agent → 章节选择提案 ──────────────┤
                                     ├─ 固定 renderer → 最终报告 hash
                                     ├─ medical + ethics 审核作业
                                     └─ 同事务复核、审计、ReleaseBundle
SQLite events + outbox → 受信 sink（尚未接实际 Observation）
```

| 文件 | 职责 |
| --- | --- |
| `contracts.py` | 严格 JSON、数据类型、时序、模型状态、提案和审核合同 |
| `store.py` | SQLite WAL、事务、持久事件与 outbox |
| `core.py` | 所有状态推进、来源认证上下文、依赖校验和发布权 |
| `adapters.py` | 受信执行器接口、隔离 fixture、空真实模型注册表 |
| `render.py` | 受控结论/动作/警示模板、固定技术说明（非医学 RAG） |
| `api.py` | 归属认证、作用域 capability、唯一报告读取入口 |
| `cli.py` | 本地服务启动、单实例锁及薄 HTTP 客户端 |

Agent 没有注册模型结果、提交 reviewer verdict、修改状态/规则/模板或强制发布的 API。提案只能选择 `forecast / policy / limitations` 的顺序；三个章节必须齐全，不能提交自由文本、HTML、动作数值或 `verified` 字段。代码、数据库和运行凭据属于受信服务；当前并不声称抵御同一 OS 用户执行任意 Python、修改源码/SQLite 或控制管理员账号。

## 闸门与生命周期

1. **输入**：支持一种固定成人 T1D 工程合同，5 分钟网格、显式时区、mg/dL、缺失掩码；不做隐式单位换算、插值或字段补全。输入变更创建新快照并使相关旧 run 失效。
2. **预测**：Core 创建带期限的 job，从冻结快照生成请求；身份/origin/version 由注册表赋予。输出无效、超时或缺失时不产生预测产物，不调用 RL。
3. **策略**：Core 组装预测和快照；验证输入摘要及精确父预测 hash。只接受 candidate 分支，abstain/unsupported/error 都不能生成可发布动作。
4. **安全与证据**：校验模式、输入、双模型及工程边界。依赖 manifest 含快照、模型、规则、模板及技术知识版本；下游和最终发布均重算验证。
5. **草稿**：Core 发出单次、限时、绑定证据 hash 的 capability。模板受信且不可自动进化；无原始草稿读取/流式输出 API。
6. **审核**：两个受信执行器分别审核同一最终报告及证据 hash。请求审核与“提交通过结论”是不同能力；客户端没有后者。修改稿件会 fence 旧作业，两轮修订即最多三版。
7. **发布**：同一 SQLite 写事务中重验当前快照、版本、期限、依赖、模板及两份审核，并写入 release、状态、事件和 outbox。审计失败导致整个发布回滚。
8. **读取**：再次验证当前有效性。撤销、取消、快照更新或过期后，API 不再提供该报告正文；权威历史记录保留于私有数据库。已经下载的 JSON 是存档副本，不能被远程收回，不代表当前仍有效。

正常状态：`SNAPSHOT_FROZEN → PREDICTION_RUNNING → PREDICTION_ACCEPTED → POLICY_RUNNING → POLICY_ACCEPTED → SAFETY_ACCEPTED → DRAFT_PENDING → DRAFT_READY → REVIEWED → RELEASED`。

故障状态：`UNAVAILABLE / BLOCKED / REVIEW_BLOCKED / CANCELLED / INVALIDATED`。`currently_valid` 只表示 run 的生命周期/版本尚未失效，**不表示模型就绪、策略安全或可以发布**；必须结合 state/reason。

每个模型阶段最多三次 attempt；只对暂时不可用状态允许继续尝试。超时结果不注册；取消/替换后的迟到结果被 fence。网络/执行器调用在数据库事务外进行，HTTP 等待在线程池中，不会占住事件循环阻止取消请求。服务启动持有独占文件锁，再 fence 未决作业以便恢复。

## API 最小流程

所有操作型 API 使用研究者 bearer；只有提交 proposal 使用本次 job capability。

| 方法/路径 | 作用 |
| --- | --- |
| `POST /cases` | 创建并冻结输入 |
| `PUT /cases/{case_id}/snapshot` | 更新输入、使旧依赖失效 |
| `POST /runs` | 创建 run，mode 默认 research，要求幂等键；可传 expected_snapshot_id 防止旧页面启动新快照任务；fixture_scenario 仅接受 eval 下的固定场景并冻结到任务 |
| `POST /runs/{run_id}/prediction` | 仅执行预测并返回状态元数据，正文必须为空 |
| `POST /runs/{run_id}/execute` | Core 依次执行双模型和工程安全检查 |
| `POST /runs/{run_id}/draft-jobs` | 建立有范围的提案作业 |
| `POST /proposal-jobs/{job_id}` | 用 capability 提交章节选择 |
| `POST /runs/{run_id}/reviews/{role}` | 请求 medical 或 ethics 受信审核，body 必须为空 |
| `POST /runs/{run_id}/release` | 请求发布，body 必须为空 |
| `GET /releases/{release_id}` | 读取当前有效 ReleaseBundle |
| `GET /runs/{run_id}`、`GET /runs/{run_id}/events` | 只读状态和受控审计元数据 |
| `POST /runs/{run_id}/cancel` | 取消并 fence 未决作业 |

## 当前验证与待接入项

2026-09-10 桌面 Agent 任务：**320 项回归通过**（新增 17 项）；原生受限报告 Agent 入口、异步状态、工具记录、幂等、取消及退出重开已验证。两个真实 SDK 任务共 6 次 Flash 请求生成待审工程草稿；取消任务 0 草稿，所有任务 0 审核/发布，原有数据保留。普通 CLI 默认关闭，开发 App 显式启用，启动/刷新不会调用模型。桌面双审调度、真实 reviewer 配置冻结及报告读取/导出留在下一项；当前范围见 [桌面 Agent](docs/desktop-agent.md)。以下为历史批次记录，当前状态以本段及总体 TODO 为准。

2026-09-10 真实审核版本绑定专项：全量 **303 项通过**（新增 34 项），真实双角色 **7 次 Flash 请求**验证通过。新 run 冻结角色提示词、工具 schema、模型/SDK/CLI 等配置内容摘要；审核精确绑定当前任务、快照、草稿修订、角色与作业输入，缓存/发布/读取均完整复核。独立副本验证配置变更拒绝、同文新修订重审及旧数据库无损迁移；旧真实记录缺少历史配置指纹，保留记录但需新建 run 重审。主联调库没有发布，桌面入口与全通道产品发布继续待办。证据、错误码及边界见 [审核版本绑定专项](docs/review-binding.md)。以下是历史批次记录，当前状态以本段及总体 TODO 为准。

2026-09-10 真实双 reviewer 步骤：269 项回归通过（新增 23 项）。医学语义/伦理两个独立 SDK 会话各自读取同一份受限工程报告，通过固定合同提交审核，Core 登记 model 来源；6 次真实 Flash 请求均通过工程范围审核，报告仍为 fixture，未执行发布。显式配置、工具权限、失败阻断、旧配置保留和证据见 [真实双 reviewer](docs/review-agent.md)。普通桌面/CLI 服务不自动启用收费审核，完整版本绑定专项和桌面入口仍待完成。

2026-09-10 报告 Agent 步骤：全量 246 项通过（新增 24 项报告专项、1 项 worker 并发关闭回归）。新增合同查询、状态查询、受限提案提交三个任务工具；SDK 完整成功后，Core 再次校验证据并由 renderer 生成工程草稿。真实 Flash 3 次请求通过，草稿数值与两份权威 fixture 产物逐项一致，未审核所以发布仍被拒。原始草稿和模型自由文字不返回 Agent/UI。入口、权限、暂存/取消语义及证据见 [受限报告 Agent](docs/report-agent.md)。真实 reviewer 和桌面 Agent 入口仍未完成。

2026-09-10 SDK/Flash 步骤：221 项回归通过（21 项新增专项），实际 claude-agent-sdk 0.2.152 / CLI 2.1.259 与官方 deepseek-flash 完成 8 次真实模型请求，覆盖工具成功、Core 拒绝发布、只读子任务和流式中断。真实 SDK/CLI 的卡住、取消与超时回收使用本地假服务验证。独立合成账本无正式报告/发布，原生数据库未改动。本步仅 CLI 兼容烟测；桌面 Agent 入口、真实报告和两类 reviewer 未完成。模型白名单、凭据和费用限制及复现方法见 [SDK 验收记录](docs/sdk-flash-smoke.md)。

2026-09-10 worker 步骤：200 项回归通过，其中新增 24 项独立进程专项。生产预测/策略/审核均经独立进程执行，超时/失效终止并回收，CPU/内存/文件/输出/并发限制已实测；真实 HTTP/CLI 和原生正常/超时验证通过。资源范围与限制见 [执行隔离验收](docs/worker-isolation.md)。Agent SDK 和真实 LLM 是后续必做项，本轮仍未接入。

2026-09-10 RL fixture 步骤：176 项回归通过；桌面 8 个固定场景全部实点验证，复用双模型/工程安全闸门，所有场景无审核或发布。超时实际等待约 5 秒，迟到策略未登记；旧库增量迁移保留原数据，场景与幂等/版本/任务持久绑定。产物在 `runtime/desktop/rl-demo-verification.json`；下一步真实 LLM 前须先完成独立 worker 隔离，真实预测和 RL 继续暂缓。

2026-09-10 预测占位步骤：154 项回归通过；桌面明确点击后经快照绑定的 eval 任务调用预测占位器，Core 登记产物。实际原生操作核对仅 1 个预测作业/产物，RL/审核/发布未执行；开发 App 显式启用 fixtures，CLI 默认关闭、research 不回退、非 synthetic 数据拒绝。详见 [桌面说明](app/README.md) 与 `runtime/desktop/prediction-verification.json`。

2026-09-10 证据卡步骤：139 项回归通过；新增 `GET /cases/{case_id}/evidence` 及对应只读原生桥接，六张卡复核当前证据/发布条件，失效或断线不继续显示旧通过状态。原生缺失/缺模型/断线恢复已验证，详见桌面 README。

2026-09-10：JSON 导入与原始时间线已接到原生桌面；回归更新为 127 项通过。新增 API 为 `POST /imports`、`GET /cases`、`GET /cases/{case_id}/timeline`，仅对归属病例开放原始输入。具体使用/进程与格式限制见 [桌面说明](app/README.md)。

2026-09-09 前置 TODO 补齐：109 项自动测试通过；真实 HTTP 服务和逐步 CLI/完整演示通过，服务重启后报告逐字段一致。逐项证据及暂缓项见 [前置核验](docs/pre-desktop-audit.md)。测试产物见 `runtime/test-results.xml`、`runtime/test-output.txt`。测试客户端存在两项依赖弃用提示，未隐藏。

覆盖：缺前置条件、伪造来源字段、重复键/非有限数、错输入摘要/父预测、真人错入 fixture、弃权、边界拒绝、超时重试、并发作业/发布/取消、限时与跨 run capability、审核缺失/弃权/篡改、审核中换稿、版本撤销、快照替换、过期读取、归属越权、审计失败回滚、outbox 故障恢复。零失败只说明这些测试通过，不是普遍安全证明。

尚未完成：真实预测/RL、真实数据的报告及生产会话管理、医学规则包与临床审核有效性、检索型 RAG、多模态解析、完整业务 MCP 服务、桌面双审核/发布业务连接、Observation 实际接收展示和自进化。SDK 内嵌的烟测、报告及审核工具按任务授权，不等于完整业务工具服务。renderer 和真实 LLM reviewer 的当前联调保留工程 fixture 范围；固定技术说明的版本绑定不等于 RAG 已实现。当前按方案 TODO 每轮只做一项，完成后等待用户审核；不要提前跨步。

`deliver_outbox` 提供可测试 sink 接缝，采用至少一次投递，sink 必须以 event id 去重；当前未实现网络 Bridge、自动重试调度及 HTTP200/业务失败识别。独立历史/教育/阻断报告目前通过状态错误返回，尚未实现各自的完整报告 schema。

受信执行器已改为独立 worker，生产无线程回退；默认 4 并发、5 秒墙钟期限、CPU/文件/句柄硬限制与 RSS 监测后终止，执行结果仍由 Core 登记。具体资源配置及 macOS 采样边界见 [执行隔离验收](docs/worker-isolation.md)。这不等于任意代码的 OS 沙箱。SDK 烟测使用经实际负载验证的独立资源配置，后续真实预测/RL 仍须分别验收。
