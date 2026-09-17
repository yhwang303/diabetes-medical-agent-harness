"""Publication-facing names and grouped tables; metrics remain exactly the audited rows."""
from pathlib import Path
NAMES={'BC':'Behavior cloning (BC)','ReBRAC_BC100':'ReBRAC (NeurIPS 2023)','FQL_alpha100':'FQL (ICML 2025)','Ours_bound040':'Ref-bounded ReBRAC (Ours, O07)','nominal':'Fixed basal reference','Anchor_FQL':'FQL + fixed reference','Bound040_noRL':'Ours w/o RL'}
GROUPS=[('A. Learned policy comparison',['BC','ReBRAC_BC100','FQL_alpha100','Ours_bound040']),('B. Non-learning reference',['nominal']),('C. Architecture controls / ablations',['Anchor_FQL','Bound040_noRL'])]
def write_paper_tex(paper,path,sealed):
 rows={x['method']:x for x in paper};n=6 if sealed else 4
 lines=[r'% Requires \usepackage{booktabs}. SD is across patients, not training seeds.',r'\begin{table*}[t]',r'\centering',r'\footnotesize',r'\caption{'+('Held-out' if sealed else 'Development')+r' closed-loop comparison after offline training on the same Loop data. Mean $\pm$ SD across '+str(n)+r' virtual adult patients, each averaged over nine paired scenarios.}',r'\label{tab:rl-'+('heldout' if sealed else 'dev')+r'}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}',r'\toprule',r'Method & TIR (\%) $\uparrow$ & TBR70 (\%) $\downarrow$ & TBR54 (\%) $\downarrow$ & Risk $\downarrow$ & Failures $\downarrow$ \\',r'\midrule']
 for title,methods in GROUPS:
  lines.append(r'\multicolumn{6}{l}{\textit{'+title+r'}} \\')
  for method in methods:
   r=rows[method];values=['%.2f $\\pm$ %.2f'%(r[k+'_mean'],r[k+'_patient_sd']) for k in ['tir_pct','tbr70_pct','tbr54_pct','mean_risk']]
   lines.append(NAMES[method]+' & '+' & '.join(values)+' & %d/%d \\\\'%(r['failures'],r['trajectories']))
  lines.append(r'\midrule')
 lines[-1]=r'\bottomrule';lines += [r'\end{tabular*}',r'\vspace{2pt}',r'\begin{minipage}{\textwidth}\scriptsize',r'All learned models use one training seed (260915) and 50,000 training iterations; optimizer schedules and network capacities follow each implementation. '+r'ReBRAC uses $\beta_\pi=100$ and FQL uses $\alpha=100$. ReBRAC/FQL are task-adapted implementations, not reproductions of the original benchmark results. '+r'All methods share meal boluses and a 6-hour nominal warm-up, excluded from scored time. O07 and the reference controls retain a scalar from the initial observed basal history; O07 and Ours w/o RL constrain actor outputs to $\pm0.4$ U/h around that reference. '+r'The fixed reference uses published simulator parameters. O07 received more development search; equal final iterations do not imply equal search or compute budgets. '+r'TIR: BG 70--180 mg/dL; TBR70/TBR54: BG below70/54 mg/dL. Risk is the prespecified glucose risk index. Terminal trajectories remain in the fixed evaluation denominator. '+r'Single-seed, simulator-only results do not establish SOTA or clinical safety.',r'\end{minipage}',r'\end{table*}']
 Path(path).write_text('\n'.join(lines)+'\n')
