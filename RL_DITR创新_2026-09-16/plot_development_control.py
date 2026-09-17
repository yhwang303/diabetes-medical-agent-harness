"""Patient-level development results; never hide missing trajectory outcomes."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.mpl_cache')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 methods=[('E08_R03_final_beam_dev','R03 reference','#777777'),('E07_H01_final_beam_dev','H01 action-prefix','#0072B2'),('E16_fixed_basal_dev','Fixed basal reference','#D55E00')]
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True);maximum=0
 for index,(run,label,color) in enumerate(methods):
  episodes=json.loads((ROOT/'results'/run/'summary.json').read_text())['episodes']
  for j,patient in enumerate([1,2,3,4]):
   rows=[x['metrics']['bg'] for x in episodes if x['patient']==patient];lo=np.mean([x['tir_lower_bound_pct'] for x in rows]);hi=np.mean([x['tir_upper_bound_pct'] for x in rows]);low=np.mean([x['tbr70_pct'] for x in rows]);y=j+(index-1)*.2;maximum=max(maximum,low)
   axes[0].plot([lo,hi],[y,y],color=color,lw=2);axes[0].scatter([lo],[y],color=color,s=35,label=label if j==0 else None);axes[1].scatter([low],[y],color=color,s=35)
 axes[0].set_yticks(range(4),['Adult 001','Adult 002','Adult 003','Adult 004']);axes[0].invert_yaxis();axes[0].set_xlim(-2,102);axes[1].set_xlim(-.03,max(1,maximum*1.15));axes[0].set_xlabel('TIR over scheduled duration (%)');axes[1].set_xlabel('TBR <70 among observed intervals (%)')
 for ax in axes:ax.grid(axis='x',alpha=.25);ax.spines[['top','right']].set_visible(False)
 axes[0].set_title('Dots: TIR lower bounds; lines: missing-outcome bounds');axes[1].set_title('Read with coverage and failures, not as safety alone')
 fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,.91),frameon=False)
 fig.suptitle('Development only: 4 known adults, 3 matched scenarios each',y=.99)
 fig.text(.07,.035,'R03: 3/12 early terminations, all in Adult 004. H01 and fixed basal: complete coverage.\nLines are missing-outcome bounds, not confidence intervals. One training seed; this is not a final test.',fontsize=9)
 fig.tight_layout(rect=[0,.12,1,.81]);out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'development_patient_comparison.svg');fig.savefig(out/'development_patient_comparison.png',dpi=170);plt.close(fig);print('development figure written')

if __name__=='__main__':main()
