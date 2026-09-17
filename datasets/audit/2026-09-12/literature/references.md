# 文献与官方源码引用核对表

核对日期：2026-09-12。只列本轮实际查阅的一手来源。搜索结果只用于定位；报告事实来自原论文正文、作者正式摘要、药品标签或官方代码/文档。访问失败或仅摘要可见均单独标注。

| ID | 文献/资源与直接链接 | 支持的命题 | 证据范围与限制 |
|---|---|---|---|
| R01 | Wang 等，2023，RL-DITR，[DOI](https://doi.org/10.1038/s41591-023-02552-9) | 数据驱动 patient model、动作条件转移、监督＋策略梯度、beam search | 本地完整30页PDF，本轮重点11–12、16页。住院T2D注射；排除泵患者；不能直接证明成人T1D基础泵控制 |
| R02 | Fox 等，2020，[MLHC 原文 PDF](https://proceedings.mlr.press/v126/fox20a/fox20a.pdf) | GRU/SAC、4h历史、5min、部分可观测、多日最差情景 | 30虚拟对象；动作语义和本项目只基础率不完全一样；实际效果均in silico |
| R03 | Emerson 等，[原文 PDF](https://arxiv.org/pdf/2204.03376) | BCQ/CQL/TD3+BC 离线比较，训练支持度和行为探索重要 | 本地已保存33页PDF/文本；固定训练数据仍由模拟器产生，不是公开真实队列实验 |
| R04 | Hettiarachchi 等，2024，G2P2C，[正式论文](https://doi.org/10.1016/j.bspc.2023.105839)、[作者仓库](https://github.com/RL4H/G2P2C) | PPO＋模型学习＋短时规划；不声明餐食的T1D模拟控制 | 正式论文页可读，机构PDF403；20虚拟对象含成人和青少年，不能转写为纯成人临床结果 |
| R05 | End-to-end Offline RL for Glycemia Control，[原文 PDF](https://arxiv.org/pdf/2310.10312) | 真实AID日志、保守离线算法、IOB/COB/TDD、配套餐时与保护器 | 本地24页PDF/文本；2023预印本；商业DBLG1日志不是本项目可取得数据；使用模拟和OPE，不等同策略人体试验 |
| R06 | Learning control-ready forecasters for Blood Glucose Management，2024，[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11426357/) | CGM预测误差不足以保证控制；胰岛素与CHO纠缠、低支持度、动作响应学习 | 研究的主要干预验证用仿真；真实观察数据仅能提供部分代理证据；单调性是其假设而非任意闭环永久成立 |
| P01 | Hovorka 等，2004，[原文记录/摘要](https://pubmed.ncbi.nlm.nih.gov/15382830/) | 皮下吸收、胃肠吸收、Bayesian时间变化参数、模型预测控制 | 原实验主要空腹/夜间；不能据此声称已完整覆盖现实运动和多日扰动 |
| P02 | UVA/Padova S2013，[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4454102/) | 低血糖非线性、胰高血糖素及人群生成的版本更新 | 论文可公开访问，不等于本轮取得官方完整软件 |
| P03 | UVA/Padova S2017，[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5851236/) | 日内敏感性、黎明现象、吸收延迟、CGM描述更新 | 官方运行软件/许可/虚拟人群未在本轮取得；不虚构S2017可用性 |
| P04 | ReplayBG，2023，[PubMed原始摘要](https://pubmed.ncbi.nlm.nih.gov/37368794/)、[DOI](https://doi.org/10.1109/TBME.2023.3286856) | 真实CGM/CHO/胰岛素个体辨识＋替代疗法模拟；100虚拟患者五种修改及真实例子 | 机构全文直连403；主要反事实真值对照来自T1DS，不能声称真实人体同状态两种治疗真值 |
| P05 | ReplayBG 多日扩展，2024 DTM，[作者会议摘要](https://pmc.ncbi.nlm.nih.gov/articles/PMC11871574/) | multi-stomach、日内敏感性、跨日扩展、OhioT1DM应用 | 会议摘要证据；不是独立完整随机干预验证 |
| P06 | ReplayBG，[官方输入要求](https://gcappon.github.io/py_replay_bg/documentation/data_requirements.html)、[辨识说明](https://gcappon.github.io/py_replay_bg/documentation/twinning_procedure.html) | homogeneous grid、basal/bolus/CHO完整输入、≥6h/CGM缺失≤10%建议、体重、初始条件 | 接口要求不授权将未知记录补零；体重和初始状态不允许伪造；≥6h是经验建议而非临床准入线 |
| P07 | ReplayBG，[模型结构](https://gcappon.github.io/py_replay_bg/documentation/choosing_blueprint.html)、[重放](https://gcappon.github.io/py_replay_bg/documentation/replaying.html) | 皮下吸收、胃肠、血浆/间质与作用状态；替代basal/bolus可由DSS生成 | 单日blueprint需正确跨日状态接续；重放结果是模型产物不是实测 |
| P08 | ReplayBG Python，[固定代码版本](https://github.com/gcappon/py_replay_bg/tree/7b47b5c4b30eed8e3f6f5d40bb4756de154b0379) | 已有Python实现；事件与单位处理可溯源；GPL-3.0 | 本轮只读源码，未安装/运行；未确认全包完整运动功能，不能据exercise标志推断能力 |
| P09 | GluCoEnv，[固定README](https://github.com/RL4H/GluCoEnv/blob/f78e11aa66acef162938597c58226d0a56270cbb/README.md) | 作者明确说明基于simglucose/UVA-Padova2008，GPU/PyTorch实现 | 加速/新代码维护与新版生理机制不是一回事 |
| P10 | Deichmann 等，2023，[PLOS原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010289)、[官方代码](https://gitlab.com/csb.ethz/t1d-exercise-model) | 运动影响摄取/产生与敏感性；独立场景验证；开放Python | 个体化自由生活示例是儿童；成人运动验证不等于提供新的成人联合训练队列；本轮未运行 |
| P11 | FDA HUMALOG 标签，2025，[官方 PDF](https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/020563s214,205747s038lbl.pdf) | 作用/吸收时间因人、部位、运动变化；PK与PD不能混为即时效应 | 本轮采用其机制描述，不据此给用户或实际患者计算剂量/时长 |

## 固定版本与本地材料

- ReplayBG repository commit SHA：`7b47b5c4b30eed8e3f6f5d40bb4756de154b0379`。
- GluCoEnv repository commit SHA：`f78e11aa66acef162938597c58226d0a56270cbb`。
- `sources/RL-DITR.txt` 是用户提供本地PDF的文本提取，保留页码；不是新的论文来源。
- `sources/offline_emerson.pdf/.txt` 与 `sources/RL4T1D.pdf/.txt` 为实际读取的论文材料。
- `sources/ReplayBG-*.py/.md` 与 `sources/GluCoEnv-README.md` 为固定版本公开源码/文档快照，只用于审计，没有执行训练代码。
- `sources/*-tree.json` 为官方GitHub公开树清单；版本和文件hash见 `source_manifest.json`。
- 因403未成功保存ReplayBG/G2P2C机构PDF；报告明确披露，通过可读正式页面/摘要/作者文档核验，不记作已全文下载。
