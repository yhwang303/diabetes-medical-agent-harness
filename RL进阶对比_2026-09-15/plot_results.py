"""Standalone scientific figures from actual paired traces; no smoothed or fabricated outcomes."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True)
colors={'nominal':'#333333','BC':'#2d8d63','ReBRAC':'#bc3a3a','FQL':'#356fbb'}
for row,patient in enumerate(range(1,5)):
 for method,color in colors.items():
  trace=json.loads((ROOT/f'results/D01_pilot50k_dev/{method}_p{patient}_s3101_b1.0.json').read_text());records=trace['records'];x=[r['minute']/60 for r in records];g=[r['bg_mg_dl'] for r in records];a=[r['delivered_basal_u_h'] for r in records]
  axes[row,0].plot(x,g,label=method,color=color,lw=1.3);axes[row,1].plot(x,a,label=method,color=color,lw=1.1)
  if trace['metrics']['failed']:axes[row,0].plot(x[-1],g[-1],marker='x',color=color,ms=8);axes[row,0].axvline(x[-1],color=color,alpha=.25,ls=':')
 axes[row,0].axhspan(70,180,color='#c7e3ce',alpha=.4);axes[row,0].axhline(70,color='#888',lw=.6);axes[row,0].axhline(180,color='#888',lw=.6)
 axes[row,0].set_ylabel(f'Adult {patient:03d}\nBG (mg/dL)');axes[row,1].set_ylabel('Delivered basal (U/h)')
 for ax in axes[row]:ax.axvspan(0,6,color='#ddd',alpha=.4);ax.grid(alpha=.2);ax.set_xlim(0,72)
axes[0,0].legend(ncol=4,loc='upper left',fontsize=9);axes[-1,0].set_xlabel('Elapsed hours');axes[-1,1].set_xlabel('Elapsed hours');fig.suptitle('Loop-only 50k pilot: paired independent simulator trajectories\nScenario seed 3101, common meal bolus factor 1.0; gray = shared warmup; x = early termination',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'paired_pilot_trajectories.png',dpi=150);fig.savefig(OUT/'paired_pilot_trajectories.svg');plt.close(fig)
print(OUT/'paired_pilot_trajectories.png')
