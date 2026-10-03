# 固定 checkpoint 开发集队列

`evaluate_checkpoints.py` 只用 Python 标准库，依次等待并评估指定的 PPO 8/16/32/40 子集，不选型、不改训练、不增加 seed，也不生成 confirmation。原 run 的 `config.json/provenance.json` 必须已经存在且是正式 40 轮配置。可以在训练仍进行时启动；只绑定目标轮已经完整写出的 history 前缀和 checkpoint SHA，不要求后续训练结束。

每 30 秒轮询，整个队列上限八小时（包含评估时间）。缺日志、日志暂时不增长或没有 completion 不等于失败；显式 `failure.json` 会停止，completion 明确完成但所需 checkpoint/完整前缀缺失也停止。超时会结束本队列启动的 evaluator 进程组，保留部分结果，不终止训练进程。

启动即记录所在机器的 evaluator、protocol、相关源码、配置与固定模型依赖的实际 SHA；没有硬编码某个本地 evaluator SHA。每次 launch 前及完成后复核。只启动原 `.venv/bin/python evaluate_candidates.py`，显式 `--split development`，绑定原 run/config 和绝对 checkpoint。真实模型 worker 仍由 evaluator 放入原 `.venv-native` 环境。原 evaluator 的完整正式配置、来源和 8/16/32/40 门禁继续执行。

输出名固定为 `<name-prefix><NN>_r1`。例如 `--name-prefix dev_ppo --iterations 32 40` 对应 `dev_ppo32_r1`、`dev_ppo40_r1`。任一计划结果目录已经存在即停止，不覆盖、不跳过、不自动换名或重试。审计目录以 `queue_<kind>_<run>_<prefix>_<iterations>` 命名，排他创建于 `checks/`，同一队列重复启动拒绝。新队列不是后台注册服务；root 应在其受控远端会话中启动并留存进程日志。

示例（将 run 名换成实际运行名称）：

```bash
python3 RL_DSENet_SOTA_2026-10-03/evaluate_checkpoints.py \
  --kind ppo --run PPO_wide_risk --iterations 32 40 \
  --name-prefix dev_ppo --batch-size 8

python3 RL_DSENet_SOTA_2026-10-03/evaluate_checkpoints.py \
  --kind world_ppo --run PPO_world_risk_r1 --iterations 16 32 40 \
  --name-prefix dev_world_ppo --batch-size 8
```

两条命令是独立串行队列，由 root 分别启动。每个队列保存启动 manifest、逐次等待/启动/完成事件、原 argv 命令、UTC 时间、exit code、stdout/stderr、每次 eval 的 manifest/summary/failure SHA 和最终队列 summary。检查返回非零、缺少完整 manifest/summary、`technical_failure` 或绑定漂移均会保留证据并停止。`failed_episode_count` 可以包含 native/environment 结局，单凭它不停止；只要原 evaluator 完整执行且没有技术失败，继续下一个固定 checkpoint。

本工具只调度原 argmax 的 `ppo`/`world_ppo` 开发面板；同权重 mean 部署消融仍是单独的方法和结果。队列完成不表示某 checkpoint 更好，也不授权 confirmation。没有真实训练或远端执行由本次实现发起。

本地验证：Python 3.8 AST 与 CLI 检查通过；45 项标准库临时 fixture/mock subprocess/clock 检查通过，覆盖两类正式来源格式、30 秒等待、部分 history、后续追加前缀稳定、串行两轮、原生失败继续、技术退出停止、已有结果拒绝、来源漂移、8 小时超时及非法轮次。首次模拟发现审计事件参数 `name` 重名，已修复后完整重跑。以上均为机制数据，不是真实模型或正式评估结果；所有临时 fixture 已清理。
