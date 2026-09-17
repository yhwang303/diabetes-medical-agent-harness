# Harness 工程合同 v1

固定合同文件为 `harness-engineering-v1.json`，由 `contracts.py` 的实际校验类型导出；`GET /contracts` 返回同一份结构。自动回归检查导出文件与运行时代码一致。JSON Schema 描述字段结构，跨字段、时间、来源和发布资格仍由 Core 确定性校验，不能仅凭客户端 schema 校验签发证据。

2026-09-11 P01a/P01b 增量加入 `BasalResearchInput`、`BasalAction`、带 profile hash 的请求/产物与工程用途配置。批次 21 新增 BasalRateAction/BasalRatePolicyOutput/BasalRateArtifactEnvelope 和 v2 用途配置，支持 U/min↔U/h 等价转换；原始输出保留，转换版本参与绑定，旧 v1 schema/配置不变。默认仍阻断新输入执行，仅显式开启的 synthetic eval 基础输注 fixture 可走完整证据链；真实模型配置为空。旧 `SnapshotInput`、fixture 请求及摘要不变。字段和来源边界见 [研究输入合同](../docs/basal-inputs.md)，动作/消费者/时效/兼容见 [动作和配置合同](../docs/basal-actions.md)。

P02a 新增 `DataPermissionRequest` 与 `data_policy` 清单，认证授权 API 和报告/双 reviewer 外发链已接通，旧 schema 保留。撤回不等于删除，生命周期验收留在 P02b/P16；详见 [数据与授权合同](../docs/data-policy.md)。

P04 新增 `RLJobQuery` 与独立 rl 外发用途，既有数值 schema/profile 不变。受控 SDK→数值子 job 与父预测依赖在 Core 强制绑定，后续消费重验父 SDK 完整成功及权限；详见 [RL MCP](../docs/rl-mcp.md)。

## 旧 fixture 适配器入口与适用范围（保持兼容）

| 项目 | 预测 fixture | 策略 fixture |
| --- | --- | --- |
| 入口 | `medical_harness.adapters.fixture_prediction` | `medical_harness.adapters.fixture_policy` |
| 版本 | `fixture-prediction-v1` | `fixture-policy-v1` |
| 运行依赖 | Python 3.12+；服务与校验依赖锁定于 `requirements.lock`；无权重和模型服务 | 同左；不调用训练器或仿真器 |
| 请求 | `PredictionRequest`：冻结快照、run/snapshot ID、输入摘要 | `PolicyRequest`：同一快照、Core 加载的完整预测 payload 与产物 hash |
| 窗口 | 6–288 个点；5 分钟等间隔；最后一点等于带时区的 decision_time；mg/dL；缺失掩码必须准确，存在缺失时不调用 | 固定引用已接受的六点预测；不自行取数或替换预测 |
| 输出 | 最后观测值重复六次；5 分钟间隔，未来 30 分钟；uncertainty=unavailable | 固定工程测试值，basal_rate，U/min，5 分钟；不执行动作 |
| 域与训练 | 仅 synthetic、adult_t1d 标签的工程合同；未训练，没有训练域或精度结论 | 未训练；没有药效模型、患者生理仿真或临床有效性结论 |

`PolicyOutput` 状态和原因固定对应：candidate/fixture_only、abstain/insufficient_input、unsupported/unsupported、error/model_error。只有 candidate 承载非空动作。合法的动作零与失败的 null 分开；矛盾状态/原因被拒绝。

模型输出不能提供身份、来源或“已验证”权限字段。Core 登记的 `ArtifactEnvelope` 绑定当前病例、快照、run、job、attempt、输入、父预测、执行器版本及接收时间。数据截止/窗口/来源从绑定快照读取，时效从绑定 run 与 job 读取；规则、模板与技术说明版本从 run 依赖清单读取。每次消费重算输入摘要、查验登记作业，发布时再核验全链路。

## 三类来源不会相互授信

| 标签所在层 | 含义 | 当前可执行路径 |
| --- | --- | --- |
| Snapshot source=synthetic | 人工合成工程输入的声明 | 仅显式开启 fixtures 的 eval 可运行替身 |
| Snapshot source=historical | 历史数据输入的声明 | 可校验/冻结，不能进入 fixture eval |
| Snapshot source=simulation | 仿真数据输入的声明 | 可校验/冻结，不能冒充 synthetic fixture 或真实病例 |
| Executor/artifact origin=fixture | Core 注册的工程替身 | 仅 eval；输出固定标为 EngineeringFixtureReport、clinical_use=false |
| Executor/artifact origin=model | 经批准的真实推理执行器 | 注册表为空，research 返回 MODEL_NOT_CONFIGURED |
| Executor/artifact origin=simulator | 仿真执行器 | 当前不配置、不运行；不能当作 model 或 fixture 放行 |

数据 source/source_ref 是导入者声明和引用，不是系统对真实数据来源的鉴定。实际数据集的许可、身份关联、原始文件定位及模型训练域核验仍需真实接入阶段完成。身份和 artifact origin 由 Core 注册表赋予，客户端不能通过上传字段获得相应权力。

## 真实模型待接入合同

真实预测/RL 按用户约定暂缓；当前没有可报告的真实入口、权重、窗口、训练域、实测吞吐或真实请求/响应。接入时必须单独确认这些资料，并版本化更新合同/适用性规则与对应测试。现有 6–288 点和固定动作窗口只是 fixture 合同，不能默认为真实模型参数。

独立 worker 的超时终止与资源限制已实现并通过真实进程验收，详见 [执行隔离](../docs/worker-isolation.md)。真实 SDK/LLM/模型的实际负载、工具权限、真实 reviewer 与医学规则仍需分别验收，不能用工程 fixture 代替；真实 LLM 联调时再向用户索取 key。

## 错误与文件边界

API 的领域、schema、认证和已处理传输拒绝返回 `ErrorResponse`：只有大写错误码，无输入回显、动作值或异常堆栈。固定拒绝有 HTTP 状态；已认证 schema/领域拒绝写入审计和 outbox，核心审计失败返回 503。未知进程/服务器故障不被解释为合法模型结果。

CLI、Store 将项目可控写入限制到 `implementation/`，解析符号链接并检查 SQLite sidecar；默认数据在 `runtime/`，令牌不输出，日志/测试/构建路径见 README 和 `scripts/desktop.sh`。这不构成对同一 OS 用户/管理员的文件系统沙箱承诺。
