# GFP / TD3+BC 实现与共同适配

GFP 正式来源为 ICLR 2026 Guided Flow Policy，作者仓库固定提交 e60468e31066ed072e494be8c6cce14ac19f60e9。保留：双Q平均；actor-reference二项softmax价值权重；权重来自target-Q，尺度来自online-Q绝对均值倒数；alpha=1、eta=.001；64维sin/cos时间编码；10步Euler teacher；one-step actor蒸馏与归一化Q目标；所有梯度从同一参数快照计算；target EMA使用作者实现的pre-optimizer critic。MIT许可证见licenses/GFP_LICENSE。

TD3+BC 为 NeurIPS 2021 原始论文更新：actor使用Q1，双Q最小值TD目标，无ReBRAC的next-action BC项、无critic LayerNorm；alpha=2.5，归一化噪声std .2、clip .5，actor及EMA每2步更新。公式与作者sfujim/TD3_BC逐项独立审阅，无新增模型指导。

两者均是Loop任务适配：1584维固定归一化历史；单维连续基础率映射[-1,1]；MLP3×256；actor lr1e-4、critic lr3e-4；50k更新、batch256、seed260915。以仅train得到的行为均值初始化输出bias；真实未来缺失不伪造成终止，TD和Q策略项使用共同q_valid交集，行为拟合保留全部起点。不是作者原benchmark配置/原分数复现，也没有相同收敛程度保证。GFP作者网络默认4×512、lr3e-4，与此处共同容量预算不同；时间编码没有删减。TD3+BC作者网络默认2×256，此处3×256属于共同容量适配。

共同动作投影只用于新增评估主表，不反向更改离线数据动作或训练目标；原生动作表同时保留。不能把外置限幅收益归给算法。没有最终评估场景调参。

验证：真实Loop64行、CUDA6次更新均有限且actor改变；保存/恢复动作输出精确一致；GFP在数据动作等于reference时权重精确0.5；另有独立源码/梯度审阅。以上证明实现机械性质，不证明闭环性能。
