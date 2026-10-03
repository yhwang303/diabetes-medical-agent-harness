# 一次确认冻结独立审查

日期：2026-10-04。审查范围截至本次本地核对完成时：已下载的 `freeze.json/seal.json`、冻结选择、源码与已有开发/训练归档身份。**未读取 confirmation 运行分数或轨迹，未执行模型、评分、授权或新确认；没有修改任何已绑定源码、配置、选择或冻结文件。** 本次只新增本文。确认全集完成和评分包到本地后的结果审查属于后续独立步骤。

**结论：冻结身份、18 行覆盖、源码/产物 SHA 与已审选择一致，未发现范围遗漏或 hash 冲突。** 本地独立字节复核覆盖全部 166 个去重绑定文件的身份：106 个平铺文件、59 个归档成员及随后补充的 1 个 H02 独立备份。该结论是字节身份核验，不是模型运行或确认结果验收；本审查不把根任务的远端运行回报冒充自己的运行结果。

## 冻结身份与用途

| 项目 | 本次核验 |
|---|---|
| plan SHA / 目录名 | `ba438ef52c7f7fc40950b3a3100dd20a4f9b2df9cdba41843e53fff2cf98152e`；按 gate 的 canonical JSON 规则重新计算一致 |
| freeze 文件 SHA | `0a6dbea2c401e41fe36336003b19d9b387b028073a70719858062228c030fc03`；与 seal 记录一致 |
| seal 文件 SHA | `6fb3a7263d529a581d020f6139f7d0f586d7edadc0ff3ac36f74cb7439bdfe52` |
| freeze 创建时间 | `2026-10-03T16:30:28.553423+00:00`，对应北京时间 2026-10-04 00:30:28 |
| selection | finalized=true、previously_generated=false、split=confirmation、smoke=false |
| 固定病例规格 | 60；按原 `case_specs(protocol)` 只重算规格元数据，没有生成餐次或运行 jobs |
| case-spec SHA | `3043fc3ad86c5464f70269cade233c53abbf808db2bd178a07d2f3ea9c02190d`，与 freeze/preflight 相同 |
| 显式选择说明 | `confirmation_selection_20261004.md`，SHA `724eb0b7553469fc489e86d8edbda69fdc909faab8857574c92e0273cdc11c7d` |
| 本地 preflight 记录 | `checks/confirmation_preflight_r1.json`，SHA `fa851d4de9580cc232bf8013709fe159d0caa45b645bd3f628f3d5a825b89c28`；其全部验证输出字段与 freeze 一致 |

协议仍是 seed260915、10 名既有虚拟成人、3 种 bolus factor、4320 min 总时长和 360 min warmup。确认规格使用固定 103901/103902，与协议列出的训练、world-validation、development、smoke 及 PPO 场景范围不重叠。冻结方法每行 batch-size=8。

freeze 的曝光审计保存了 108 份元数据 SHA，并明确 `independently_proves_no_unlogged_external_runs=false`。它支持已保留文件范围内的预曝光审计，不证明没有未记录的外部运行。本次不重新扫描正在运行中的确认结果来倒验“冻结前无曝光”。

## 固定比较集合

18 个 method ID 唯一，恰为 12 个 required comparators 与 6 个 selected internal 的不相交并集；与 `checks/confirmation_selected_panels_r1.json` 的映射精确一致。gate 的 6 个 internal ID 包含 4 个 MPC 机制行，不能读成 6 个 RL 算法。

| 作用 | 冻结 ID / 实际固定版本 |
|---|---|
| 主 RL 候选 | `ppo_world_quantile`：`PPO_world_risk_r1/policy_iter40.pt`，SHA `294df745dd5951a15b5125e874acd2895bafe4778f2b04eb293c7be0d9672a8e` |
| 第二 RL 候选 | `ppo_wide`：`PPO_wide_risk/policy_iter40.pt`，SHA `d9e1c70150b60f47becad77bef6aa312c7032dfb91c882fb4c8887692b869114` |
| 固定控制/项目旧部署 | `hold`、`physiology`、`d06_frozen` |
| 历史论文方法适配 | `retained_bc`、`retained_iql`、`retained_td3bc`、`retained_rebrac`、`retained_fql`、`retained_lom`、`retained_gfp`、`retained_ditr`；既定旧最终权重，全部保留 |
| 新任务训练参照 | `iql_wide`：固定 `policy_020000.pt`，SHA `705b08a4da2c35a8759f3c70331e1fc0e37e27710000ac271c06032c4abaaab5` |
| MPC 机制行 | `mpc_point_world`、`mpc_quantile_median`、`mpc_expected`、`mpc_event_risk` |

point MPC 使用 point world best（SHA `1f86f2881986660de0ab4f111be71173a7271f49ce59a1a614112edb8699fd30`）；其余三种 MPC 使用同一 quantile best（SHA `51ba03d242583e081e53e46fa738c3a55cc3456f14c67fb5b9a7f40b0f5cc1c5`）。quantile-median 与 point MPC 复用 `mode=point` 的控制配置，但绑定 world checkpoint 不同；不能把这两行误标成同一模型，或把任一 MPC 标成 RL。

18 行各自冻结的 development manifest/summary SHA 和 checkpoint SHA，与已核 25 行 development 汇总逐项吻合。其余开发 8/16/32 checkpoint 与 mean 消融继续留在开发表/附录，未进入确认不是删除负结果。BC 与未执行的 point-world PPO 没有被加入冻结集合。

## 源码与产物字节核验

冻结清单去重后有 70 个 source 路径、62 个 artifact 路径；加主绑定字段和 36 份 development manifest/summary 后，共 166 个唯一文件身份，没有同路径不同 hash 的冲突。

1. 当前本地平铺文件中，106 个实际文件重新计算 SHA 全部匹配，包括 68/70 source 和 40/62 artifact；没有把不存在的平铺路径写成已读取。
2. 对 `archives/development_evidence_r1.tar.gz` 整体重新计算 SHA，并顺序读取其中 59 个对应成员的字节，逐一与 freeze 的预期 hash 比较，全部一致；未解包、未改源文件、未读取确认结果。两份 simglucose pump 源文件由归档中的等 hash source 副本核实，因此全部 70 个冻结 source 的字节身份均已在本地直接或归档副本中核验。
3. 归档 SHA 为 `32d14d613af59e4dc02b6ec3c9325f898474797677f8cdcbed35f5f8da8a7443`；其成员 manifest `checks/development_evidence_backup_r1.json` SHA 为 `fa0dc86da0ef4ce4e7c420b6776e872893eaf59bfe644a63db73085d39a2fcfe`。匹配 source 副本证明字节身份，不证明本机已装好远端 Python/模拟器运行环境。
4. 初次扫描未找到旧 `RL_DITR创新_2026-09-16/results/H02_prefix_sim_factual/patient_best.pt`。根任务随后另存 `archives/legacy_H02_patient_best.pt`，本审查已独立读取全部 155,736,727 bytes，SHA 为 `18ecdc57a60fd656cfd239348dd5e0677c78744d15b75a12218febfe9719f012`，与 freeze 一致，没有反序列化模型。补充凭证 `checks/legacy_h02_local_backup_r1.json` SHA 为 `21e03804bf717c0cff0235c1e04467d473c916e4aae7055105927852e5b26bdc`。本次缺少本地字节副本的限制已关闭，旧目录和冻结绑定未改。

从已核归档成员读取的 completion 元数据还确认：wide-PPO 与 world-PPO 均满 40 轮，world-PPO 为 formal；IQL 固定 20,000 更新；point/quantile world 均 completed/configured=4000、budget_override=false，best step 分别为 1000/3000，选择集为 world_validation。这里只核验已完成产物与其元数据，没有加载 checkpoint 张量执行模型，也没有重跑训练。

以下六份共享源码存在于所有冻结方法的 source 集合，实际本地文件及本地 Git commit `2dc0a713e33c175b4835dcd38833ea75016120c6` 的 blob 均匹配：

| 文件 | 冻结 SHA256 |
|---|---|
| `confirmation_gate.py` | `3a8da7223ddce6dfc13e9767faa86c8735b638214d40d5276e0944bb4ee7c9ce` |
| `evaluate_candidates.py` | `fa135911d85ad29ebe06268144ea19646aa3f97800a2f169927e0be76017947f` |
| `evaluate_confirmation.py` | `4823e61bd64cfed3b72b5f94d12bc7f6850d2f851d36211f0a297bd101f5b239` |
| `summarize_panels.py` | `82c7a11db9d3316af1cfd9584e6175c594262127831ccad61ac77f16c32c6438` |
| `summarize_confirmation.py` | `5b0a5f5562bd17b4756d5436a66982ffb261b1eb9169a46a0a8892b2453af398` |
| `build_report.py` | `ed199d3c76489d2d026a5256af0add3d5e5786649eebc5d2f6a581b040ee129c` |

原 scorer SHA 仍为 `0a6d44298d942bc09764f15e7ef45a71d3afbc7ba63cad3dae5425f0b22b8cfd`。Git 已推送的状态来自根任务回报，本次没有连接远端核查 push。薄调度器 `execute_confirmation_set.py` 不属于这份逐方法 source 清单；它自行保存编排源码 SHA 的审计属于另一层，不能冒充 freeze 对该文件的绑定。实际运行方法、权重、配置和唯一 ticket 仍由上述已绑定 gate/runner 限定。

## 最终报告必须保留的区分

- **计划、执行、可审核分数、完整随访不是同一状态。** 18×60=1080 是计划病例数，不是已证明完成了1080条三天轨迹。后续须核固定全部18方法的 ticket/start/finished、manifest/summary/raw与原 scorer 精确审核；当前 freeze 有效或队列已启动不等于确认成功。缺失尾部不得补成零、hold或安全。
- **确认与开发分表。** 25 行 development 用于开发/选择，18 行 confirmation 用于固定新场景检验；不能合并样本当独立确认、从两表挑更优数值替换或删去失败行。失败部署只报告 Observed、coverage、unknown-tail、TIR上下界，不参与完整随访最优排名。
- **“最好”写清指标和集合。** 主候选 world40 是确认前已选模型；未来实际最低 risk、最高TIR或最低TBR的行可能不同。原评分器 risk=LBGI+HBGI可以如实列出，但不得据一个综合风险最低改写为全指标或全领域最优；HBGI+2LBGI也不能成为事后新门槛。不得在确认后将两条 PPO 中更好的一条伪称事前唯一候选。
- **原门槛与扩大目标并列。** 原严格联合标准继续原样报告；development 中 world40 的 HBGI 失败和 wide40 的通过不互相替代。任何确认判断均须来自届时真实数据，本文没有推断其方向。当前不新增创新、训练或调参，未达目标时保留未达成结论。
- **论文方法行不是论文原始任务分数。** retained8应标 frozen+projection，新 IQL 标本任务适配；D06保留原窄动作支持。相同可观测信息、动作上限和评分不等于相同数据、表示、优化预算或官方原生复现。不能把其他论文在不同患者/环境/动作任务上报告的数字直接混进这张统一评分大表。
- **机制与算法身份不混淆。** 同算法不同 checkpoint/部署仍是同一方法家族；4个MPC不是4个新增RL算法，mean不是新外部方法。world-PPO是冻结world特征加真实模拟器PPO，不是联合训练world/policy，也未使用想象rollout。低糖减少不能在没有对应证据时专归于事件项或概率分布。
- **统计与风险边界继续保留。** 主表真实BG、次表CGM；患者内先平均、患者间sample SD和配对患者区间，不把相关指标计数当独立成功或把十名患者当训练seed重复。持续BG<54≥120min事件按原定义报告；单seed、已曝光虚拟患者和world低糖校准限制不因确认运行而自动消失。

后续冻结报告器已经要求完整 frozen method 集合与逐文件确认审计，不能手工填 `passed` 绕过。若另外生成纯展示概览，应保留其来源与展示脚本身份，并逐格核对最终已审表；这不授权修改冻结评分、选择或原报告器。

本文只完成冻结身份与交付边界审查，不包含确认成绩结论，也不提出新训练或额外创新实验。
