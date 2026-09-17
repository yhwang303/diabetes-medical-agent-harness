# 运行证据与限制

- A01 step2000出现非有限指标，严格JSON拒收；仅step1000权重已经保存，所以未能确定step2000根因。step1000梯度启用/禁用推理都有限且差1.91e-6。
- PyTorch官方有[快速attention路径mask回归报告](https://github.com/pytorch/pytorch/issues/107084)。这是排查线索，不能据此宣称本次原因已经确定。
- 后续采用sequence-first attention保持相同参数和数学结构、禁用相关快速路径，增加评价前保存及具体非有限现场。5项GPU检查通过。
- A02/A03均已训练一个完整epoch；全量validation复评使用固定随机顺序遍历所有起点，保证额外剂量诊断不是只看每人的最早片段。过程评价的样本集合固定，仍按旧固定抽样检查。
- 训练checkpoint含优化器，但当前配置resume仅加载权重作warm start，不是精确恢复GPU随机数/数据迭代位置；本轮各试验均从头初始化，未声称精确续训。
- 全量validation仍然是反复使用的开发集；不称sealed test、不称外部独立模型验证。

- 独立counterfactual_audit首版漏载权重，已将该输出另存local_dose_response_unloaded_INVALID.json并标明无效；修复后load_state_dict并逐张量验证，再重算。主train.py的训练/评价加载原本正确。A03启动时的辅助脚本源码快照保留该旧版本，最终诊断以当前修正脚本和输出的weights_loaded_and_exactly_verified=true为准。
- 输入诊断DIA01为完成后的相同起点对照，不重新训练；当前脚本显式核对checkpoint权重和SHA。该诊断不参加best checkpoint选择。
