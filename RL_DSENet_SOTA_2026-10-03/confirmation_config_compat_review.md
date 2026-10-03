# 旧 development manifest 的空配置兼容检查

日期：2026-10-04。根任务的真实远端只读 preflight 发现：旧合法 `dev_legacy_r1`、`dev_hold_r1`、`dev_physiology_r1` manifest 没有 `config_sha256` 键，对应方法的 `method_config_sha(method)` 确实为 None。原方括号取值产生 KeyError，不能完成兼容核验。本审查未连接远端；上述发现来自根任务，本次独立验证其最小修复语义与回归。

根任务仅将 `confirmation_gate.development_evidence` 的一处比较改为：

```python
manifest.get('config_sha256') != method_config_sha(method)
```

**结论：修复可接受，targeted fixture 与相关回归全部通过。** 缺键仅在期望值为 None 时等价于显式 null；期望为非空 hash 时，缺键、null 或错 hash 仍拒绝。retained 的方法级 `config=None` 不等于其期望配置 hash 为 None：原 `method_config_sha` 从绑定的 `retained_methods.json` artifact 读取实际 hash，该路径也已覆盖。

没有修改旧 manifest，也没有省略其余源码、规则、旧权重或配置 artifact 的门禁。现有 `method_dependencies` 等验证保持不变；本次不是降低评分、训练完成或来源要求。

## 实际检查

新 [check_confirmation_config_compat.py](checks/check_confirmation_config_compat.py) 在 `checks` 下临时 fixture 目录调用真实 `development_evidence`，构造保持 hash、60 个 development job key 和 summary 绑定一致的 synthetic 元数据，只改变被测配置字段。它不是模型或真实轨迹评估。

| 检查 | 实际结果 |
|---|---|
| hold、physiology、legacy 的期望配置为 None | 三种均验证缺键接受、显式 None 接受、意外非空 hash 拒绝 |
| PPO 直接绑定的必需配置 | 缺键、None、错 hash 均拒绝；准确 hash 接受 |
| retained 从 artifact 取得的必需配置 | 缺键、None、错 hash 均拒绝；准确 hash 接受 |
| targeted 检查总计 | 25 项通过，含 Python3.8 AST、检查期间源码未变、无模型/模拟器导入 |
| 原 gate 回归 | 40 项通过，另存新证据，原检查源码与旧结果保留 |
| 原整合回归 | 48 项通过，另存新证据；真实共享 loop/原 scorer 仅作用于临时 synthetic 数据 |

另做逐字节差异范围核验：将当前 gate 中这一处 `.get` 逆替换为原方括号表达式，所得源码 SHA 精确等于上一原子发布版 `8e25df56e5177f022d2b69ced63fbe736d06c04303eceffa39d1bd7d94c30d44`。因此这批生产源码差异确为单行，不涉及原锁等待/事件发布逻辑。按任务范围没有重跑无关的 25 项锁和 18 项原子事件检查；它们的历史证据继续保留。

## 当前证据身份

| 文件 | SHA256 |
|---|---|
| `confirmation_gate.py` | `3a8da7223ddce6dfc13e9767faa86c8735b638214d40d5276e0944bb4ee7c9ce` |
| `checks/check_confirmation_config_compat.py` | `d4ca01bf659ce2da63349dc046a09f38f352c8760963b4e3c87c0f1ea8c251ed` |
| `checks/confirmation_config_compat_mechanics_r1.json` | `6fcc19da07c5344fef0d958c486fa1c54c29ec2e106428b0db0acbedc5f70c5b` |
| `checks/confirmation_gate_config_compat_regression_r1.json` | `f4adf93a0c7765734dc182d363c3690651651e4e2b0e77437159b9c5ad28392b` |
| `checks/confirmation_integration_config_compat_regression_r1.json` | `22304c2af33c840df1c3376558680942cec64cd938bb7c832dd7c7f5500f957b` |

本次审查代理只新增 targeted 检查、上述三份结果和本文，没有再修改 gate 或其他生产源码，没有改权重/训练/评分，没有 freeze/authorize 真实选择、生成确认 jobs、训练或连接远端。本地机制通过不冒充真实远端 18 方法 preflight、确认成绩或模型晋升；根任务后续须绑定当前 gate SHA。
