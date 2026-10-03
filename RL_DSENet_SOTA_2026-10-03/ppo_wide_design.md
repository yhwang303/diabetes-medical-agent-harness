# A：宽基础率动作的真实回报 PPO 诊断

日期：2026-10-03。实现候选，尚未进行 CUDA 运行、训练或仿真。仅本目录 `ppo_wide_worker.py`；未改旧评分器、模型或产品 Harness。它不是最终完整 world+RL 方法。

## 任务与模型

保留 basal-only 5分钟任务，外部meal bolus及评价合同不变。动作索引0–8映射为 `[0,.25,.5,.75,1,1.25,1.5,1.75,2] × 实际预热末基础率anchor`，最终限制在0–20 U/h。当前速率可以走出旧±0.25支持；不得把旧worker的±0.26检查继续施加在这里。

冻结原P03/D05/H02及其真实权重，核对保留D06配置登记的D05 SHA，以及D05中登记的P03/H02 SHA。每次输入只包括过去72×22观测历史及预热anchor；`z256 + reference48/10 + rate/5 + anchor/5`共306维，追加同目录`physiologic_features.features(history,anchor)`输出的显式无量纲生理特征。该模块导出`FEATURE_NAMES`及`FEATURE_DIM`，输入维数从该接口取得，没有硬编码新增特征维数；特征名称及源码hash记录进checkpoint。BG、患者ID、未来餐食和隐藏状态不进入特征函数。

新actor与独立critic各为两层128宽Tanh MLP。actor9类，critic标量。原world不评分候选动作、不提供想象回报；PPO使用真实仿真BG回报。这是在共享冻结表示上检验更宽权限/更多可观测信息/较少保守约束能否改变控制行为的诊断；若改善，不能单独归因为世界模型更好。

冻结D05在宽动作诱导的新历史/当前速率上可能分布外；追加特征不自动修复reference。需在报告中披露，不能称旧world已验证宽空间预测。后续完整世界模型须独立检验动作效果和尾部。

## 初始化与优化

唯一seed260915。在冻结world装载后重新设置Python、NumPy、Torch种子，再创建新actor/critic，使新网络随机初始化不取决于旧网络构造消耗多少随机数。actor末层权重清零，bias=`log([.005,.01,.035,.15,.60,.15,.035,.01,.005])`，因此所有状态初始以60%选择anchor、相邻两档各15%，两端各0.5%。初始确定性策略正好是anchor；所有合法动作都有非零探索概率。没有复制旧actor，没有teacher KL，不把该prior作为后续正则限制。

默认actor lr3e-4，critic lr1e-3，PPO clip.2，gamma.997，GAE lambda.99，entropy coefficient.01，4个epoch，batch512，梯度范数.5。优势在本次rollout所有状态上标准化；按轨迹分别计算GAE，不跨病例串接。clip约束只是PPO更新规则，`approx_update_kl`是诊断量，不是teacher约束。

预算由根调度器写配置。建议固定40 iterations，每次20条3day轨迹，固定开发checkpoint8/16/32/40；worker每次update保存checkpoint并拒绝超过`iterations`。场景必须由根训练器按`protocol.json`冻结，worker不生成/挑选场景，也不读取确认结果。

## 回报与终止

首版只实现`reward_profile="risk"`，不宣称实现constrained PPO。对真实训练BG（mg/dL）定义：

`f=1.509*(log(max(BG,1))**1.084-5.381)`，`risk=10*f**2`；低侧分量`low=risk if f<0 else 0`，高侧`high=risk if f>=0 else 0`；每5分钟`reward=-(high+low_risk_weight*low)/(12*reward_scale)`，默认低侧权重2、reward_scale10。除12是5min→hour积分，除10只改变训练尺度。低侧指风险变换的低侧，不等同BG<70指示变量。源公式沿用旧`ditr_model.py`/`control_metrics.py`的Kovatchev变换；该非对称训练目标是本项目预先指定的研究适配，不是临床处方。未使用旧<70恒为−1的平台，也未裁剪低糖严重度。

独立输出原始high/low risk均值、BG<70及BG<54比例。这些没有乘低侧权重、没有加terminal penalty；只是训练样本诊断，不代替患者等权正式评分。当前没有cost critic、Lagrange multiplier或CMDP保证；后续约束策略应新设明确原始cost预算并另行验证。

`terminal=true`仅用于真实环境终止，bootstrap置0，最后一步额外扣固定`terminal_penalty`（默认100、训练回报单位，不再除12或reward_scale），减少通过提前终止逃避后续负奖励的倾向；该罚值不是安全保证。管理性3day时间截断必须`terminal=false`，用最终观测历史的critic值bootstrap。已给入终止步的真实BG仍参与risk。

技术失败/缺测不能作为正常terminal训练。若调用者提供`failure_reason`，worker只接受null或`native_environment_done`，其他失败直接终止worker且不写本次checkpoint；调用者仍须保存失败证据。最小协议只有terminal布尔值时，根调度器负责正确区分原因。奖励与old scorer独立，绝不改评分器的失败/未知尾部处理。

## 接口与配置

启动：`python ppo_wide_worker.py --config /绝对路径/config.json`。可加`--checkpoint /绝对路径/policy_iter08.pt`载入完整状态，用于确定性评价或续训；要求有效配置、特征/关键源码及冻结权重provenance完全一致，不允许默默改变特征后加载旧policy。

配置最小需要`name`，其它算法参数见源码DEFAULTS；可提供`output_dir`，否则为本目录`results/name`。输出必须位于本研究目录。即使传入output_dir，也保留name作为本实验标识。

```json
{
  "name": "A_wide_risk",
  "seed": 260915,
  "iterations": 40,
  "reward_profile": "risk",
  "low_risk_weight": 2.0,
  "reward_scale": 10.0,
  "terminal_penalty": 100.0,
  "actor_lr": 0.0003,
  "value_lr": 0.001,
  "gamma": 0.997,
  "gae_lambda": 0.99,
  "clip": 0.2,
  "entropy": 0.01,
  "epochs": 4,
  "batch_size": 512
}
```

每行JSON请求及响应：

- `op=act`：`history[N,72,22]`、`anchors[N]`、`indices[N]`，随机采样并按indices缓存动作前特征/类别/logp/value；返回`actions[N]`（U/h）、`action_indices`、`entropy`。同一批indices必须唯一。
- `op=evaluate`：只需要history及anchors，返回确定性argmax动作，不抽样、不改变buffer；因此可在训练状态内调用，但正式评测建议独立加载固定checkpoint。
- `op=update`：`outcomes`以indices的字符串形式为键，各含`bg[T]`（每动作之后的BG）、`terminal`、`history`（最终观测历史）、`anchor`。必须与本批全部缓存轨迹一一匹配。非terminal最终历史用于bootstrap。可加failure_reason。完成后清空buffer。
- `op=close`：结束，不隐式更新未提交buffer。

checkpoint保存`config`、`provenance`、`model`、`value`、两个`opt`、iteration、累计transitions，以及Python/NumPy/Torch CPU和全部CUDA RNG。旧模型不重复写入checkpoint，由hash绑定依赖。文件以exclusive create保存，已有路径拒绝覆盖；写盘失败留下的文件不得当完整checkpoint续训，外部调度器应保留失败并重新建独立结果目录。update返回checkpoint绝对路径与SHA。

## 验证与剩余工作

本机已通过源码AST解析及compile(source)语法检查，没有导入Torch、安装依赖或运行模型/仿真；静态核对正式/烟测两份配置、9档映射、初始prior和为1/anchor占0.6/各档非零。已阅读落盘的physiologic_features.py：FEATURE_NAMES为28项、FEATURE_DIM由其长度生成、features(history,anchor)返回np.float32[N,28]，故当前policy/critic输入为334维；checkpoint另登记normalization.json SHA以防特征尺度漂移。远端仍需根任务核验：真实权重恢复、physiologic_features维度与有限值、相同输入初始化动作、evaluate不改buffer、真实终止/截断bootstrap、小规模更新后参数变化且world冻结、checkpoint恢复/RNG一致、全协议场景与旧评分器不变。通过这些检查前不能称worker联调或训练完成。
