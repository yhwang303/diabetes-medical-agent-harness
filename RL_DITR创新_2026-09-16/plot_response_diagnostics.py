"""Keep mechanism metrics separate from independent control outcomes."""
import json,os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'.mpl_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def main():
 fig,axes=plt.subplots(2,2,figsize=(13,8));colors=['#747d87','#bd7733','#566db2','#218376']
 names=['R05_recursive_sim_stable_policy','H02_prefix_sim_factual','H03_prefix_sim_paired','C01_response_patient_composition'];labels=['R05','H02','H03','C01 response']
 ds=[read(ROOT/'checks'/(n+'_dynamic_actions.json'))['summary'] for n in names]
 for i,(d,c) in enumerate(zip(ds,colors)):
  axes[0,0].bar(i,d['final_direction_agreement']*100,color=c);axes[0,0].text(i,d['final_direction_agreement']*100+2,'%.1f'%(d['final_direction_agreement']*100),ha='center',fontsize=9)
  axes[0,1].bar(i,d['path_effect_mae_mg_dl'],color=c);axes[0,1].text(i,d['path_effect_mae_mg_dl']+.1,'%.2f'%d['path_effect_mae_mg_dl'],ha='center',fontsize=9)
 axes[0,0].set(title='Equal-dose timing direction (84 paired plans)',ylabel='Direction agreement (%)',ylim=(0,113),xticks=range(4),xticklabels=labels)
 axes[0,1].set(title='Dynamic intervention effect error',ylabel='Path effect MAE (mg/dL)',ylim=(0,9),xticks=range(4),xticklabels=labels)
 latent=read(ROOT/'checks/response_latent_development.json')['results'][1]['summary'];times=np.array([5,30,60,120,240])
 for key,label,color in [('H02_effect_mse','H02',colors[1]),('response_effect_mse','Response',colors[3]),('zero_mse','Zero response','#888888')]:axes[1,0].plot(times,[latent[str(t)][key] for t in times],marker='o',label=label,color=color)
 axes[1,0].set(title='Latent intervention error on dynamic plans',xlabel='Forecast horizon (minutes)',ylabel='MSE in frozen encoder coordinates',yscale='log');axes[1,0].legend(frameon=False)
 runs=['E20_H02_final_beam_dev','E22_H03_final_beam_dev','E36_C01_initial_policy_dev'];control=[]
 for name in runs:
  d=read(ROOT/'results'/name/'summary.json');assert d['status']=='completed' and len(d['episodes'])==12 and not any(e['metrics']['failed'] for e in d['episodes'])
  control.append([np.mean([e['metrics']['bg'][k] for e in d['episodes']]) for k in ['tir_observed_pct','tbr70_pct']])
 x=np.arange(3);axes[1,1].bar(x,[r[0] for r in control],color=colors[1:]);axes[1,1].set(title='Independent 1h control: no established gain',ylabel='BG TIR (%)',ylim=(0,105),xticks=x,xticklabels=['H02','H03','C01 response'])
 for i,r in enumerate(control):axes[1,1].text(i,r[0]-3,'%.2f%%\nTBR70 %.3f%%'%(r[0],r[1]),ha='center',va='top',color='white',fontsize=9)
 nominal=read(ROOT/'results/E16_fixed_basal_dev/summary.json')['episodes'];nominal_tir=np.mean([e['metrics']['bg']['tir_observed_pct'] for e in nominal])
 axes[1,1].axhline(nominal_tir,ls='--',color='#555555',label='Fixed nominal basal %.2f%%'%nominal_tir);axes[1,1].legend(loc='lower left',frameon=False,fontsize=9)
 for ax in axes.flat:ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('Response structure: mechanism improves; control must be checked separately',fontsize=14)
 fig.text(.5,.015,'Exposed development: 4 known virtual adults; 28 probe histories / 12 control trajectories. One training seed.\nP01/P03 use original constant-intervention training data; no new temporal training data. No unseen-patient or clinical claim.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.07,1,.96));out=ROOT/'figures';out.mkdir(exist_ok=True)
 for suffix in ['png','svg']:fig.savefig(out/('response_mechanism_and_control.'+suffix),dpi=160)
 print('response mechanism/control figures written')
if __name__=='__main__':main()
