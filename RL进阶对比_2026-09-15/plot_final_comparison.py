"""All held-out patients, fixed representative scenario; no smoothing or outcome selection."""
import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
 d=ROOT/'results/T01_frozen_sealed_comparison'
 if not (d/'completion.json').exists():raise RuntimeError('Full sealed panel must complete before plotting')
 plan=json.loads((ROOT/'configs/T01_frozen_sealed_comparison.json').read_text());patients=plan['patients'];seed=plan['seeds'][0];factor=1.0
 methods={'nominal':('Fixed reference','#333333'),'ReBRAC_BC100':('ReBRAC','#377eb8'),'Bound040_noRL':('Same architecture, no RL','#e69f00'),'Ours_bound040':('Ours RL','#008050')}
 fig,axes=plt.subplots(len(patients),2,figsize=(14,16),sharex=True)
 for i,pid in enumerate(patients):
  for m,(label,color) in methods.items():
   row=json.loads((d/('%s_p%d_s%d_b%.1f.json'%(m,pid,seed,factor))).read_text());r=row['records'];t=np.array([x['minute']/60 for x in r]);bg=[x['bg_mg_dl'] for x in r];basal=[x['delivered_basal_u_h'] for x in r]
   axes[i,0].plot(t,bg,color=color,lw=1.15,label=label);axes[i,1].plot(t,basal,color=color,lw=1.0,label=label)
   if row['metrics']['failed']:
    axes[i,0].plot(t[-1],bg[-1],'x',color=color,ms=9);axes[i,1].plot(t[-1],basal[-1],'x',color=color,ms=9)
  axes[i,0].axhspan(70,180,color='#aaccaa',alpha=.15);axes[i,0].axhline(70,color='#999',ls=':',lw=.7);axes[i,0].axhline(180,color='#999',ls=':',lw=.7);axes[i,0].set_ylabel('Adult %03d\nBG (mg/dL)'%pid);axes[i,1].set_ylabel('Delivered basal (U/h)')
  for ax in axes[i]:ax.axvspan(0,6,color='#777',alpha=.1);ax.grid(alpha=.18);ax.set_xlim(0,72)
 axes[0,0].legend(ncol=2,fontsize=8);axes[0,1].legend(ncol=2,fontsize=8);axes[-1,0].set_xlabel('Hours');axes[-1,1].set_xlabel('Hours');fig.suptitle('Frozen held-out patients: scenario %d, bolus factor 1.0\nOne training seed260915,50k updates; raw trajectories; x marks termination'%seed,fontsize=13);fig.tight_layout(rect=(0,0,1,.96));out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'sealed_trajectories.png',dpi=160);fig.savefig(out/'sealed_trajectories.svg');plt.close(fig)
 # Single paired patient point represents average of9 scenarios; not9 independent patients.
 rows=json.loads((d/'summary.json').read_text());fig,ax=plt.subplots(figsize=(10,5));order=['nominal','BC','ReBRAC_BC100','FQL_alpha100','Anchor_FQL','Bound040_noRL','Ours_bound040']
 for p in patients:
  y=[np.mean([r['tir_pct'] for r in rows if r['method']==m and r['patient']==p]) for m in order];ax.plot(range(7),y,'o-',alpha=.6,lw=.8,label='Adult%03d'%p)
 ax.set_xticks(range(7),['Reference','BC','ReBRAC','FQL','Ref+FQL','No RL','Ours RL']);ax.set_ylabel('TIR70-180 (%)');ax.set_ylim(0,103);ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8);ax.set_title('Held-out comparison: each point is one patient averaged over9 paired scenarios');fig.tight_layout();fig.savefig(out/'sealed_patient_comparison.png',dpi=160);fig.savefig(out/'sealed_patient_comparison.svg');print('Created full held-out panel figures')
if __name__=='__main__':main()
