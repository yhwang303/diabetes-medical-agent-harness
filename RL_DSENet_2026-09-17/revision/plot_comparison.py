"""Scientific figure from verified patient-level results, with paired cohort scope."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
a=json.loads((ROOT/'paper_tables/extended_analysis.json').read_text())
methods=a['panels']['main']['methods'];keys=list(methods)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(13,6.8),sharey=True,gridspec_kw={'width_ratios':[1.35,1]})
for ax,metric,title in zip(axes,('tir_lower_bound_pct','tbr70_pct'),('Time in range 70–180 mg/dL (%) ↑','Time below 70 mg/dL (%) ↓')):
 for y,key in enumerate(keys):
  r=methods[key];vals=np.array([p['bg'][metric] for p in r['patients'].values()],dtype=float)
  mu=vals.mean();rng=np.random.default_rng(260917);ci=np.percentile(vals[rng.integers(len(vals),size=(10000,len(vals)))].mean(1),[2.5,97.5])
  color='#147d68' if key=='ours' else '#6d829b'
  ax.barh(y,mu,height=.62,color=color,alpha=.88)
  ax.errorbar(mu,y,xerr=[[max(0,mu-ci[0])],[max(0,ci[1]-mu)]],fmt='none',ecolor='#283b4f',capsize=3,lw=1)
  ax.text(mu+.5 if metric.startswith('tir') else mu+.04,y-.23,f'{mu:.2f}',va='center',fontsize=9,color='#182d3d')
 ax.set_title(title,loc='left',fontsize=13,pad=12);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
 axes[0].set_xlim(0,110)
axes[0].set_yticks(range(len(keys)),[methods[k]['metadata']['label'] for k in keys]);axes[0].invert_yaxis()
fig.suptitle('Common action envelope: benefit and hypoglycemia together',x=.02,ha='left',fontsize=17,weight='bold')
fig.text(.02,.09,'Bars: patient-equal means; whiskers: 95% patient-bootstrap intervals (10 virtual adults × 6 scenarios).',fontsize=10)
fig.text(.02,.052,'One training seed. Previously exposed virtual adults. The proposed system has additional simulation supervision.',fontsize=10,color='#4f6579')
fig.subplots_adjust(left=.20,right=.96,top=.87,bottom=.18,wspace=.22)
for ext in ('png','pdf','svg'):fig.savefig(ROOT/'figures'/f'control_comparison.{ext}',dpi=180,facecolor='white')
