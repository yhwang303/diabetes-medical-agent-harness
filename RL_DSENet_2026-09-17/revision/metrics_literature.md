# 正文评价指标与近期文献核验

核验日期：2026-09-17。用途：为统一设置的闭环控制主表选择指标；本文件不把他人论文数字搬进本项目实验表，也不宣称新基线已经复现。

## 1. 结论

TIR 和 TBR 是必要指标，但只有这两列不足以支持“控制模型优秀”。主表需要同时呈现控制收益、低糖/高糖代价、波动与失败；预测器的 RMSE/MAE 应单列预测实验。训练计划完成、优化已收敛、独立测试占优、临床可用是四件不同的事。单个 seed 的全部计划更新完成，只能支持第一件事。

## 2. 近期原论文到底使用什么

| 来源 | 可复核位置 | 该文使用的相关评价 | 对本项目的用途 |
|---|---|---|---|
| [Zhao et al., PLOS ONE 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0317662) | Experimental results，Table 2，Comparison with the baseline | TIR、TAR、TBR、risk index、极端高低血糖、治疗失败；30虚拟受试者、10日测试 | 说明单看TIR不完整；另须记录失败。其部分基线来自已有文献，本项目不能照搬这种跨设置数值比较。 |
| [PAINT，ECAI 2025](https://journals.sagepub.com/doi/10.3233/FAIA251067)；[2025原始稿](https://arxiv.org/html/2501.15972v1) | §4.1，Tables 1–2；原始稿§4、Appendix6.2 | Magni risk、目标血糖偏差、TIR、TBR、CoV；原始稿重复训练3 seeds | 风险/波动需和TIR一起看；Magni risk与本项目simglucose风险指数未必同公式，不可混称。原始稿与正式版场景重复次数不同，引用应注明版本。 |
| [GUIDE，arXiv 2026](https://arxiv.org/html/2604.00385v1) | §IV-D，Table IV；Table VI | TIR、TBR、TAR、CV；每人配对Wilcoxon并用Holm–Bonferroni校正21项比较；另评行为相似度 | 提供2026年依据，支持患者级配对统计；行为相似度不证明控制最优。此为预印本，不伪称已接收顶会。 |
| [RL-DITR，Nature Medicine 2023](https://www.nature.com/articles/s41591-023-02552-9) | Extended Data Fig.3；Methods中clinical endpoints | 开发时剂量MAE、PCC/R²、WIS及有效样本量；临床primary为平均每日指尖血糖，secondary有TIR/TAR/TBR/CV与低糖事件 | 原论文是住院T2D滴定；本项目是成人T1D基础输注。应写“方法适配”，不能宣称复现原临床试验。剂量拟合和WIS不能直接替代独立闭环。 |
| [DA-CMTL，npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-01994-4) | Performance evaluation，Fig.2，Supplementary Tables2–3 | 血糖预测RMSE、Clarke error grid、低糖检测敏感度；展示预测误差与事件检测的取舍 | 预测表应有按时距RMSE/MAE及低糖分层，不能把预测误差当RL控制收益。 |
| [Future-aware forecasting，Scientific Reports 2026](https://doi.org/10.1038/s41598-026-41787-7) | Experimental results，Table5 | RMSE、MAE、Clarke error grid | 预测器评价的近期补充依据。 |

## 3. 主表建议：可直接用于论文的列与含义

所有百分比均按预注册的连续时间口径计算。每位受试者先聚合相同场景，再对受试者等权汇总，避免长轨迹或多场景患者主导均值。

| 列 | 定义 / 方向 | 建议呈现位置 |
|---|---|---|
| Method / year / venue | 真实外部方法名、来源；适配说明写脚注 | 主表 |
| TIR ↑ | 70–180 mg/dL时间百分比 | 主表主要收益 |
| TBR70 ↓ | <70 mg/dL时间百分比，包含<54 | 主表主要安全 |
| TBR54 ↓ | <54 mg/dL时间百分比 | 主表主要安全 |
| TAR180 ↓ | >180 mg/dL时间百分比，包含>250 | 主表 |
| TAR250 ↓ | >250 mg/dL时间百分比 | 主表 |
| CV ↓ | 100×血糖标准差/均值，单位% | 主表 |
| LBGI / HBGI ↓ | 明确公式/代码版本的低/高血糖风险指数 | 主表宽版或同页副表 |
| Failed / total ↓ | 原生仿真终止次数及总场景数；和正常时间截断分开 | 主表必需 |
| Mean BG、SD | 单位mg/dL；平均值不能替代范围指标 | 扩展表 |
| Hypo events/day、最长事件、持续低糖事件 | 明确开始/恢复阈值；右删失标记 | 风险表 |
| Basal/bolus U/day、动作总变差 | 保留实际输注与请求动作区别 | 效率/控制行为表 |
| Inference latency、训练预算 | 固定硬件、warmup、批大小；数据与模拟调用量分列 | 计算成本表 |

依据：[国际TIR共识2019的Tables1–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC6973648/)列出范围、CV、LBGI/HBGI、事件等；[ADA 2026简版第6章](https://diabetesjournals.org/docm-care/article-pdf/doi/10.2337/doc26-a006/860548/docm26a006.pdf)仍给出成人TIR>70%、TBR70<4%、TBR54<1%、TAR180<25%、TAR250<5%。这些是临床参考目标，不是3日虚拟实验达到后即可获准临床使用的验收线。2019共识建议的CGM资料充足性是14日且有效数据≥70%，不能将本项目短仿真包装成满足这项临床随访要求。

事件定义可按[国际CGM共识2017](https://pedsendo.net/wp-content/uploads/2018/04/International-consensus-on-CGM.pdf)：低于阈值连续≥15分钟，恢复为≥70连续15分钟；<54连续≥120分钟为持续低糖。注意：<54属于level2生化低糖，不能仅由模拟血糖值判定需要他人协助的level3“严重低血糖”或临床DKA。2025年[成人T1D原始研究](https://diabetesjournals.org/care/article/48/2/273/157636/Limitations-in-Achieving-Glycemic-Targets-From-CGM)也同时检查TIR/TBR/CV与持续低糖，说明群体平均达标不排除个体风险。

以上表格布局与统计规则是针对本项目任务作出的研究设计建议，不是假称某一篇论文规定的唯一标准。不要为了列数把临床无法观察的指标填成0。

## 4. 不能省略的公平性说明

1. **同测试设置并不等于同训练信息。** 本模型有额外配对仿真监督、既有仿真训练history encoder；旧BC/FQL/ReBRAC仅Loop离线数据。若保留该差异，应按信息权限分组，并将整体系统比较称为“同一测试协议”；不能写“只换算法、公平证明算法优势”。纯算法结论必须使用相同训练信息。
2. **动作范围及附加规则必须公开。** 本模型以持久参考基础率周围±0.25 U/h、7个4h计划为候选；旧方法直接0–20 U/h。限幅/候选集合/安全模块本身能决定结果。提供统一动作适配的补充比较或共同支持域实验，保留原方法/完整系统结果，不能只给baseline加不匹配后处理后还无脚注沿用原名。
3. **Warmup和餐时bolus是共同系统的一部分。** 6h已知基础率warmup与共享餐时bolus会使任务容易；必须交代控制期是否排除warmup，避免固定warmup稀释方法间差异。
4. **终止轨迹不能只统计成功轨迹。** TIR未知尾部给出界限；观测部分TBR不能与完整轨迹值无说明直接排名。无终止不代表无低糖。
5. **个体风险不能被均值掩盖。** 建议报告每患者最大TBR及同时满足TIR/TBR阈值的患者比例；当前最差adult009的持续低糖必须保留。
6. **一seed边界。** 可给患者分层配对bootstrap CI，明确它条件于唯一训练seed，只表示受试者/场景变化，不能表示训练随机性。多方法显著性需预指定比较和多重校正。n是患者数，不是每个5min采样点，更不能以720条同受试者轨迹当独立患者。
7. **已曝光虚拟患者**只支持同虚拟队列的新场景表现，不能称未见患者泛化。论文真实数据预测封存患者和闭环虚拟受试者是不同证据。

## 5. 2025/26糖尿病专用候选的复现筛选

| 候选 | 官方公开材料核验 | 与本项目关系 | 建议 |
|---|---|---|---|
| Safe-enhanced PPO-AP, PLOS ONE 2025 | [作者代码](https://github.com/YanfengZhao-UKM/PPO-AP-Controller)确有PPO训练、evaluation、env、training_model目录 | 在线个体训练、CGM-only、全闭环连续总输注、外部安全模块；论文说明每个体需>4000 episodes | 可以做独立在线/领域参考组；若统一为离线基础率设置需改变方法，不能把修改后结果当忠实复现。 |
| PAINT, ECAI 2025 | 原始稿Appendix6.1说接受后公开；本次正式页无GitHub链接，作者公开GitHub仓库列表未找到PAINT；[2023旧offline-glucose](https://github.com/hemerson1/offline-glucose)不能冒充PAINT | 还依赖偏好标注/奖励模型和安全先验 | 暂不作为可即时复现行；不能把TD3+BC重命名为PAINT。这里只能说本次未核验到公开PAINT实现，不断言作者永未发布。 |
| GUIDE, arXiv 2026 | [作者代码](https://github.com/SamanKhamesian/GUIDE)有model、predictor、environment、replay_buffer、bolus_safeguard等；MIT | 动作包含进食与bolus时机/大小，是行为建议框架，非基础率专用新优化算法 | 当近期相关工作/指标依据；若移植其中CQL-BC，要命名CQL-BC而非凭2026框架将旧CQL称2026新算法。 |
| PRIMO-FRL, JMIR Diabetes2025 | [原文](https://diabetes.jmir.org/2025/1/e72874/)可查 | 在线联邦多目标框架 | 未进一步确认官方训练实现，暂不入主表。 |
| Personalized Type1 Management, JMIR2026 | [原文](https://diabetes.jmir.org/2026/1/e79195/)为DQN+自建模拟 | 离散bolus与预测任务混合，原文训练仅10episodes且承认并未充分收敛 | 不为凑2026年份优先加入。 |

因此，新增“新且易复现”的正文基线优先从**真正新增的2025/26通用离线连续控制算法**选择，并按相同Loop动作合同重跑；不是把最新应用论文里的旧PPO/CQL改个名字。主表只列外部方法和ours；H02/H04/C01、anchor-FQL、bound040、ours planner/beam等内部试验单独消融/开发记录。

## 6. 已做核验与限制

已实时查阅原论文、出版社/作者页面及公开GitHub文件列表；日期区分会议年和网页上线年。PAINT正式页显示ECAI2025卷、网页2026-08-25发布，不能把会议改成ECAI2026。代码列表可访问不等于已安装、跑通或复现实验；本子任务未修改训练/评分代码，也未启动训练。ADA2026 PDF搜索索引内容可读，但直接打开工具返回内部错误；所列阈值同时由可读国际共识原文支持。
