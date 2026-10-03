"""Quarantine only byte-identical files from the initial parent-path upload."""
import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
names=['actor_worker.py','brake.py','check_mechanics.py','evaluate.py','run_development.py','实验合同.md']+['configs/'+n for n in ('brake17.json','brake30.json','confirmation.json','development.json','retained_weights.json','smoke.json')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=R/'checks/initial_transfer_quarantine';out.mkdir(exist_ok=False);records={}
for name in names:
 src=P/name;correct=R/name
 if not src.exists():continue
 assert sha(src)==sha(correct),name
 dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst));records[name]=sha(dst)
deps=json.loads((P/'RL_DSENet_局部风险_2026-09-21/configs/dependencies.json').read_text())
for name,value in deps.items():assert sha(P/name)==value,name
(R/'checks/transfer_recovery.json').write_text(json.dumps({'status':'passed','byte_identical_new_files_quarantined':records,'retained_dependencies_unchanged':len(deps),'initial_launch_had_no_simulation':'missing script path'},indent=2));print(len(records),len(deps))
