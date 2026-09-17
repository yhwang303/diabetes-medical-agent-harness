from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'07_真实数据集RL训练资格审计报告.md'
s=p.read_text().replace('datasets/audit/2026-09-12/loop/audit.md','Loop数据集/历史资料/2026-09-12审计/audit.md')
s=s.replace('解压文件位于 `datasets/extracted/{loop,dclp3,iobp2,uom,ohio}/`，原 ZIP 仍在 `datasets/raw/`；审计脚本、JSON、文献快照和分报告在 `datasets/audit/2026-09-12/`。','2026-09-14目录更新：Loop解压文件位于 `Loop数据集/原始数据/`，原ZIP在 `Loop数据集/官方压缩包/`，专属旧审计在 `Loop数据集/历史资料/2026-09-12审计/`；[全部文件/字段导航](Loop数据集/00_从这里开始.md)。其余来源仍位于 `datasets/extracted/{dclp3,iobp2,uom,ohio}/`、`datasets/raw/` 和 `datasets/audit/2026-09-12/`。')
p.write_text(s)
for name in ['06_RL真实联合数据集下载与构建方案.md','HANDOFF.md']:
 p=ROOT/name;s=p.read_text()
 if '> 2026-09-14目录更新：Loop专属' in s:continue
 end=s.index('\n')
 s=s[:end]+'\n\n> 2026-09-14目录更新：Loop专属原始数据和历史材料已迁至 [Loop数据集](Loop数据集/00_从这里开始.md)。本轮仅整理目录与逐文件/逐字段说明，未执行事件重建、资格划分或训练；下文历史目录示意不代表当前Loop存放位置。'+s[end:]
 p.write_text(s)
