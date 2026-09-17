# RL训练与审核（2026-09-15）

打开 **训练审核报告.html** 查看数据、全部已执行实验、失败、配置、曲线与适用性结论。报告内的展开项提供每患者结果；原始JSON、方法审计与代码在本目录。

## 远端环境

项目路径：`/root/autodl-tmp/diabetes-agent`。通过项目现有AutoDL MCP管理，认证不放在本目录报告中。

- `.venv-native/bin/python`：Python3.8.10，继承镜像Torch2.0.0+cu118；numpy1.24.4、pandas2.0.3、pyarrow17.0.0安装在独立venv。
- 固定患者管线：`Loop数据集/训练管线_v2`；train225人/1,653,421个起点。
- 数据打包：`pack_data.py`，实际数组与权威管线核对见`results/packed_verification.json`。
- 权威原始患者表、管线v1及产品Harness均保留。

## 可复跑命令

从远端项目根目录：

```bash
.venv-native/bin/python RL训练_2026-09-15/test_model.py
.venv-native/bin/python RL训练_2026-09-15/train.py --config RL训练_2026-09-15/configs/A02.json
```

重训前复制配置并给 `name` 新名称；程序拒绝覆盖已有训练日志。每实验的确切执行代码在其`results/<name>/sources`；早期S00/A01源码在`source_snapshots/`，SHA与原provenance逐文件匹配。当前代码可能包含后来修复，复现某次原实验须用该次源码。

`--eval-only`加载相应`best.pt`，另外写评价provenance与`full_validation_evaluation.json`，不覆写原训练provenance。`resume`字段只作权重热启动，不是精确继续GPU随机数/数据位置；本轮各训练从头初始化。

## 报告更新

本机项目根目录：

```bash
Loop数据集/预处理_v1/.venv/bin/python RL训练_2026-09-15/build_report.py
```

HTML本身包含全部主要表格/曲线/配置，无需外部脚本；证据链接依赖本目录相对布局。下载的权重见`artifacts/manifest.json`（导出完成后生成）；完整优化器checkpoint另在远端`results/`保留。

## 用途边界

患者模型拟合、剂量响应、策略训练、独立仿真、Harness接入各自验收。这里的模型是回顾性研究产物，不是已准入的给药服务。food/exercise的稀疏mask不代表真实事件已全部记录；验证集不等于封存测试。以最终HTML当前结论为准。
