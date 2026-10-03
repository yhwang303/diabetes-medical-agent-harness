import subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
for method,mode,relative in [('fql','legacy','FQL_source_order_seed260915/export_0050000.npz'),('rebrac','legacy','ReBRAC_source_order_phase_seed260915/export_0050000.npz'),('lom','external','LOM_source_phase_seed260915/last.pt')]:
 subprocess.run([str(P/'.venv/bin/python'),str(R/'evaluate.py'),'--mode',mode,'--checkpoint',str(R/'results'/relative),'--common-limit','.25','--config',str(R/'configs/development.json'),'--name','S_order_'+method,'--batch-size','20'],check=True)
print('SOURCE_ORDER_EVALUATION_COMPLETE',flush=True)
