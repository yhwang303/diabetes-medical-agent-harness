# Harness接口与真实数据/模型接入差距

2026-09-12仅源码审计，未修改或运行产品代码。下列结论来自当前真实文件，旧方案/历史测试不能覆盖这些差距。

| 当前源码事实 | 与这批真实数据的差距 | 后续应实现（本轮未实现） |
| --- | --- | --- |
| basal_inputs.py:ResearchSource.kind仅synthetic/simulator；BasalResearchInput.source仅synthetic/simulation | 不能将真实历史改标签成模拟输入 | 新增显式historical research资料合同，offline audit/replay与真正simulation闭环用途隔离 |
| ResearchTreatment.insulin=generic_simulation_insulin，regimen=basal_only | 五源大多是基础+bolus治疗；制剂异质 | 真实制剂/浓度/途径，therapy=basal_bolus、controlled_action=basal_rate分开表示；不扩大actor为bolus |
| ResearchInsulinHistory仅BasalDelivery actual_delivery，连续无缝区间 | 缺bolus、延长输注、补注、指令/递送证据等级、历史未知区间 | 版本化胰岛素暴露事件合同，保留原始量/单位/证据并受控重建；IOB过滤器与有效史要求绑定 |
| basal_actions.py:fixture最低6CGM点/25分钟史 | 这是占位工程门槛，不足以支持几小时作用的真实模型 | 从模型卡冻结历史长度、采样与缺失规则；当前拟6小时，需4/6/8小时敏感性；未知史不得置IOB=0 |
| basal_actions.py:BasalRateAction支持U/min、U/h及5分钟区间 | 可复用动作表示思想；当前仍generic insulin与fixture理由 | 新真实模型输出版本、absolute rate+duration+integral，abstain原因、支持度、依赖与阈值；不把积分称bolus |
| adapters.py:REAL_EXECUTORS为空；profile.real_model_configuration=None | 目前没有真实预测/RL执行器 | 固定权重/配置/预处理/状态估计/训练划分与哈希、worker执行、超时/失败/证据登记验收 |
| rl_agent.py/Core装配同run/snapshot父预测并绑定hash | 已有可复用受控架构，但非医学有效性 | 真实F+策略在相同快照生效；M/IOB估计若运行时需要必须成为版本化依赖；LLM不可填补数值 |

应保持Agent SDK→受控MCP→Core→独立worker→双模型证据→规则/审核/唯一发布→执行反馈→新快照。每5分钟数值闭环由受控执行器调度，LLM不成为泵控制的硬实时计算环节。研究仿真实际执行反馈必须绑定所选动作与新环境状态，不能把历史CGM原样当新动作后果。推荐被读/接受不等于已递送。

首版可完成的目标是研究模式模型接入和独立模拟闭环；当前源码和公开观察数据都不构成真人闭环设备或医学建议准入证据。本轮新增的是审计文档，不是Harness功能或真实模型集成。
