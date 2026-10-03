# 真仿真回报的保守actor微调（确认未打开）

CGM连续brake17开发已出现低糖收益但TIR/高糖代价，因此另立机制假设：原actor主要依据漏报低糖的world回报学习，训练anchor又来自历史近似，与推理持久anchor不一致。直接在公开模拟器中采集真实动作后果与回报，训练时使用观察预热得到的同一持久anchor，可能更好保留高TIR同时提前减少有害动作。此假设需要闭环验证，不声称已证实根因。

P03 DSENet与D05 world完全冻结，原D06 actor作为唯一初始化，单seed260915。actor层不变，7计划映射到3种真正执行的第一步动作；中性组取max logit，与原argmax第一步严格等价，不强迫actor选择未来减量。采样仅为训练探索；评价仍用argmax，无外部brake。

固定8轮，每轮10已知成人×2病例，共20条3day/6h预热；仿真seed从930001开始，bolus条件轮换；并非未见患者训练。真实BG只用作训练reward/评价标签，不作为policy输入。CGM/实际输注历史、冻结DSENet上下文与预测、观察anchor是唯一输入。每5min reward=(-((BG-120)/60)^2-4max((70-BG)/16,0)^2)截断[-20,0]/12；真正终止额外-20/12，3day时间限不当终止，用最后观察状态value bootstrap。Reward选择为本项目研究选择，不冠以论文固定公式。

PPO超参固定gamma .997、GAE .99、clip .1、actor LR3e-5、value LR1e-3、teacher KL .05、entropy .002、4epoch、batch512；actor梯度范数≤.5，value≤1。新value头仅训练时用，不进入最终推理。

2/4/8轮checkpoint分别做917开发。只有按原联合严格标准通过者才可称达标候选；通过者优先TBR54、TBR70、SD，均失败保留失败并继续定位，不能用92111/92112确认来挑checkpoint。开发选择/训练额外仿真成本及信息必须披露。全baseline通用执行层比较与新actor消融单独呈现，不把两者收益混成新算法。
