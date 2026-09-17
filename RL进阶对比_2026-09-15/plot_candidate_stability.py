"""Scientific paired trajectories documenting both successful and failed training seeds."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
cases=[('Reference','D09_reference_only_dev','Observed_reference_only','#333333'),('Residual seed260915','D07_anchor_rebrac_bc_dev','Anchor_ReBRAC','#14759d'),('Residual seed260916','D12_ours_seed260916_dev','Anchor_ReBRAC_s260916','#bd493a')]
for _,folder,_,_ in cases:assert (ROOT/'results'/folder/'completion.json').exists(),folder
fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True)
for row,patient in enumerate(range(1,5)):
 for label,folder,method,color in cases:
  t=json.loads((ROOT/f'results/{folder}/{method}_p{patient}_s3101_b1.0.json').read_text());records=t['records'];x=[r['minute']/60 for r in records];g=[r['bg_mg_dl'] for r in records];a=[r['delivered_basal_u_h'] for r in records]
  axes[row,0].plot(x,g,label=label,color=color,lw=1.3);axes[row,1].plot(x,a,label=label,color=color,lw=1.1)
  if t['metrics']['failed']:axes[row,0].plot(x[-1],g[-1],marker='x',color=color,ms=8);axes[row,0].axvline(x[-1],color=color,alpha=.25,ls=':')
 axes[row,0].axhspan(70,180,color='#c7e3ce',alpha=.4);axes[row,0].set_ylabel(f'Adult {patient:03d}\nBG (mg/dL)');axes[row,1].set_ylabel('Delivered basal (U/h)')
 for ax in axes[row]:ax.axvspan(0,6,color='#ddd',alpha=.4);ax.grid(alpha=.2);ax.set_xlim(0,72)
axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('Elapsed hours');axes[-1,1].set_xlabel('Elapsed hours');fig.suptitle('Reference-residual policy: training-seed stability check\nSame Loop dataset, 50k updates; development scenario3101, meal bolus factor1.0',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'candidate_seed_stability.png',dpi=150);fig.savefig(OUT/'candidate_seed_stability.svg');plt.close(fig)
print(OUT/'candidate_seed_stability.png')
