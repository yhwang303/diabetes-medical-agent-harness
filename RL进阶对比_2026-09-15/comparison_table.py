"""All completed development evaluations, including failures and unequal-budget labels."""
from pathlib import Path
import json,csv,statistics,html
ROOT=Path(__file__).resolve().parent
METRICS=['tir_pct','tbr70_pct','tbr54_pct','mean_risk']

def collect():
 rows=[]
 for folder in sorted((ROOT/'results').glob('D*')):
  if not (folder/'completion.json').exists():continue
  provenance=json.loads((folder/'provenance.json').read_text());summary=json.loads((folder/'summary.json').read_text());contract=json.loads((ROOT/'simulation_contract_v1.json').read_text())
  spec=provenance['spec'];expected={(p,s,b) for p in spec['patients'] for s in spec['seeds'] for b in contract['bolus_factors']}
  for method in spec['policies']:
   rr=[r for r in summary if r['method']==method];keys=[(r['patient'],r['seed'],r['factor']) for r in rr]
   assert len(keys)==len(set(keys)) and set(keys)==expected,(folder.name,method,'incomplete paired panel')
   weight=provenance.get('weights',{}).get(method,{});cfg=weight.get('config',{});anchor=cfg.get('anchor_context',False);control=weight.get('control')
   info={'stage':folder.name,'method':method,'input_group':'observed_reference' if anchor else 'history_only' if weight else 'parameter_reference','training_run':cfg.get('name','none'),'training_updates':0 if control else weight.get('step',0),'training_seed':cfg.get('seed','none') if weight and not control else 'none','patients':len(spec['patients']),'scenario_seeds':len(spec['seeds']),'trajectories':len(rr),'beta_actor':cfg.get('beta_actor',''),'fql_alpha':cfg.get('alpha','') if cfg.get('algorithm')=='fql' else '', 'residual_limit_u_h':cfg.get('residual_limit_u_h','unbounded' if anchor and not control else 'none'),'model_guidance':str(cfg.get('model_guidance','')),'failures':sum(r['failed'] for r in rr),'export_sha256':weight.get('export_sha256','none')}
   by_patient={p:[r for r in rr if r['patient']==p] for p in spec['patients']}
   for k in METRICS:
    values=[statistics.mean(r[k] for r in pr) for pr in by_patient.values()];info[k+'_mean']=statistics.mean(values);info[k+'_patient_sd']=statistics.stdev(values) if len(values)>1 else 0.
   rows.append(info)
 return rows

def write_tables():
 rows=collect()
 if not rows:return ''
 csvpath=ROOT/'论文对比表_全部开发实验.csv'
 with csvpath.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 headers=['方法 / 开发阶段','输入组','残差范围U/h','累计训练更新','TIR% ↑','TBR70% ↓','TBR54% ↓','risk ↓','终止/轨迹']
 body=[];tex=['% All completed development experiments. Repeated tuning on dev; not sealed test.','% Mean +/- patient SD; training budgets and input groups are separate.','\\begin{tabular}{lllrrrrrr}','\\hline','Method & Input & Residual bound & Updates & TIR (\\%) & TBR70 (\\%) & TBR54 (\\%) & Risk & Failures \\\\','\\hline']
 for r in rows:
  cells=[html.escape(r['method'])+'<br><small>'+html.escape(r['stage'])+'</small>',html.escape(r['input_group']),html.escape(str(r['residual_limit_u_h'])),str(r['training_updates'])]+['%.2f ± %.2f'%(r[k+'_mean'],r[k+'_patient_sd']) for k in METRICS]+['%d/%d'%(r['failures'],r['trajectories'])]
  body.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
  tex.append(r['method'].replace('_',r'\_')+' & '+r['input_group'].replace('_',r'\_')+' & '+str(r['residual_limit_u_h'])+' & '+str(r['training_updates'])+' & '+' & '.join('%.2f $\\pm$ %.2f'%(r[k+'_mean'],r[k+'_patient_sd']) for k in METRICS)+' & %d/%d \\\\'%(r['failures'],r['trajectories']))
 tex+=['\\hline','\\end{tabular}'];(ROOT/'论文对比表_全部开发实验.tex').write_text('\n'.join(tex)+'\n')
 return '<h2>全部已完成开发实验：统一指标总表</h2><p>均值 ± 患者间SD；不是训练seed的SD。history_only：原72×22观测历史；observed_reference：另加最初可见基础输注参考；parameter_reference：使用模拟器公开患者参数。不同输入组、累计训练更新和调参预算不能混作公平最终排名。模型辅助实验还包含额外图模型训练成本。开发集已被反复用于选择，不能据此给出最终显著性或封存测试结论。CSV逐行保留配置、来源权重SHA和情景数量。</p><div class="table"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div><p><a href="论文对比表_全部开发实验.csv">全部实验CSV</a> · <a href="论文对比表_全部开发实验.tex">全部实验LaTeX</a></p>'
if __name__=='__main__':print(write_tables())
