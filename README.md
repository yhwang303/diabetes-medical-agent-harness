# Diabetes medical agent harness — selected DSENet-RL version

当前研究版本为 **P03 DSENet + D05世界模型 + D06有限候选RL策略**，单训练seed `260915`。当前正文比较结果为TIR **97.08 ± 4.50%**、TBR70 **1.55 ± 3.89%**、TBR54 **0.57 ± 1.79%**；±为10位虚拟患者之间的SD，不是训练seed SD。

- [版本指纹](RL_DSENet_2026-09-17/selected_version.json)：固定预测器、world、策略checkpoint的SHA256，以及global patch/stride 12/3、local 2/1。
- [完整训练与推理说明](RL_DSENet_2026-09-17/README.md)：Loop处理、DSENet、动作响应监督、策略训练、原始轨迹评价的执行顺序。
- [模型结构与训练状态](RL_DSENet_2026-09-17/revision/结构与训练状态.md)。
- [正文对比报告](RL_DSENet_2026-09-17/revision/论文正文对比报告.html)：8个已有外部方法、固定基础率参考和本方法；[论文表](RL_DSENet_2026-09-17/revision/paper_tables/main_comparison_compact.tex)。

## 代码范围

本次源代码快照固定上述模型，不启用额外低糖罚分或执行层暂停。既有仓库中的Loop预处理、H02历史编码器、配对仿真和外部基线代码作为依赖保留。该版本目录含实际训练、推理、评价和报告生成代码，以及对应配置与证据摘要；不把内部开发变体当成正文外部对照。

模型权重、患者级Loop数据、CUDA依赖二进制和凭据不放入Git。现成权重的准确SHA256在版本文件中；本地保留完整研究推理恢复包`RL_DSENet_2026-09-17/delivery/DSENet_RL_single_seed_model.tar.gz`。没有权重时须按文档训练，只有源码不等于已经在新机器恢复了数值结果。

## 证据边界

这些是公开数据/公开仿真的研究结果，不是患者给药或产品准入证明。本方法包含额外仿真监督（L+S），外部离线方法为Loop（L）；单seed、已曝光虚拟成人、TIR与低糖交换及adult009长时间低糖限制必须保留。详见正文报告。Core/Harness产品边界未因研究结果改变。
