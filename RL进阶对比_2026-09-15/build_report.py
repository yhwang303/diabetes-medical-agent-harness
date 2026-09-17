"""Evidence-driven review HTML and paper-style tables. No unrun result is populated."""
from pathlib import Path
import json,csv,html,datetime,statistics,math
from comparison_table import write_tables
from assess_development import assess
from final_report_sections import render as final_sections, status_text, conclusion
ROOT=Path(__file__).resolve().parent;RESULTS=ROOT/'results'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def escape(v):return html.escape(str(v))
def link(p,label=None):return '<a href="'+escape(p.relative_to(ROOT))+'">'+escape(label or p.name)+'</a>'
def table(headers,rows):return '<div class="table"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def number(v):return '—' if v is None else ('%.3f'%v if isinstance(v,float) else escape(v))
def curve(records,key,color='#176493'):
 data=[(x['step'],x[key]) for x in records if key in x and math.isfinite(x[key])]
 if not data:return ''
 low=min(v for _,v in data);high=max(v for _,v in data);xmax=max(x for x,_ in data);span=max(high-low,1e-8)
 points=' '.join('%.1f,%.1f'%(45+x/max(xmax,1)*760,160-(v-low)/span*135) for x,v in data)
 return '<svg viewBox="0 0 850 190" role="img" aria-label="'+escape(key)+'"><path d="M45 20V160H810" stroke="#888" fill="none"/><polyline points="'+points+'" fill="none" stroke="'+color+'" stroke-width="2"/><text x="0" y="25">%.3g</text><text x="0" y="160">%.3g</text><text x="45" y="185">0</text><text x="760" y="185">%d步</text></svg>'%(high,low,xmax)
def main():
 out=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RL进阶训练审核</title><style>body{font:15px/1.6 system-ui,sans-serif;color:#192c38;max-width:1300px;margin:30px auto;padding:0 24px}h1{font-size:28px}h2{margin-top:36px;border-bottom:2px solid #aaa}h3{margin-top:24px}.notice{padding:18px;background:#fff2d9;border-left:5px solid #af7800}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ced6df;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f7;padding:12px;font-size:12px}a{color:#075a93}svg{width:100%;max-height:190px}svg text{font-size:12px}details{margin:10px 0}summary{cursor:pointer}section{border-bottom:1px solid #ddd;padding-bottom:16px}</style><body><h1>RL进阶训练与独立仿真审核</h1><p>本地证据快照生成：'+datetime.datetime.now().isoformat(timespec='seconds')+'</p>']
 out+=['<div class="notice"><b>'+status_text()+'</b><p>主比较和封存判定置顶；历史开发试验、训练配置、负结果与旧seed探索保留在下方折叠明细中。</p></div>', '<p><a href="../RL训练_2026-09-15/训练审核报告.html">前一轮预测模型训练记录</a> · '+link(ROOT/'研究协议.md')+' · '+link(ROOT/'baseline_sources.md','ReBRAC/FQL依据')+' · '+link(ROOT/'model_eval_sources.md','图模型与独立评价依据')+'</p>']
 out+=['<div class="notice"><b>读表先看：</b>外部RL基线只有ReBRAC和FQL；BC是行为克隆。最终我方模型为<b>O07：固定参考有界残差ReBRAC</b>，原内部代号Ours_bound040。A组是学习策略对比、B组是非学习参考、C组是结构消融。<p>研究门槛通过≠SOTA；当前属于incremental任务适配，非incremental算法创新尚未完成。<a href="模型结构与读表说明.html">查看完整结构、损失/reward、指标和预测模型接入/回退说明</a>。</p></div>']
 out+=[final_sections()]
 out+=['<p><b>最新预算：所有模型只训练一个seed（260915）；额外seed进程已停止，历史结果保留但不冒充新的主比较。单seed结果不证明随机初始化稳定性。</b></p><h2>1. 数据、任务与读表方法</h2><p>全部225名Loop训练患者、1,653,421个起点保留；validation55人/424,866起点。72×22输入包含food/exercise的值、mask和因果记录年龄；不按模态存在性筛选。全部起点参与行为学习；两RL基线TD使用相同的合法后续状态/动作交集1,219,674行。Loop封存测试与demo未打开。模拟器为独立官方simglucose固定commit，开发adult001–004，最终adult005–010尚未用于选择。</p><p>这里一训练step是一次训练迭代，可能涉及不同网络的更新；ReBRAC的50k步对应50k次critic与25k次actor更新。50k步实际访问12,799,419次，7个完整数据遍历，约7.74轮。初始1000步短测只用于实现/吞吐。主表TIR/TBR来自实际独立轨迹，动作MAE/TD损失仅为离线诊断。前一轮的MSE属于患者预测模型，不能和策略疗效混排。</p>']
 summary=read(RESULTS/'D01_pilot50k_dev/summary.json',[]);paper=[];display=[]
 for method in ['nominal','BC','ReBRAC','FQL']:
  rows=[r for r in summary if r['method']==method]
  if not rows:continue
  grouped={p:[r for r in rows if r['patient']==p] for p in sorted(set(r['patient'] for r in rows))};values={k:[statistics.mean(r[k] for r in rs) for rs in grouped.values()] for k in ['tir_pct','tbr70_pct','tbr54_pct','tar180_pct','mean_risk','mean_bg_mg_dl','basal_u_per_day']};stats={k:(statistics.mean(v),statistics.stdev(v) if len(v)>1 else 0) for k,v in values.items()}
  failures=sum(r['failed'] for r in rows);entry={'method':method,'training_updates':0 if method=='nominal' else 50000,'training_seeds':0 if method=='nominal' else 1,'patients':len(grouped),'scenarios':len(rows),'failures':failures}
  for k,(mean,sd) in stats.items():entry[k+'_mean']=mean;entry[k+'_patient_sd']=sd
  paper.append(entry);display.append([escape(method),entry['training_updates'],str(entry['training_seeds']),*[f'{stats[k][0]:.2f} ± {stats[k][1]:.2f}' for k in values],f'{failures}/{len(rows)}'])
 out+=['<details><summary><b>2. 全部开发实验明细（含失败，不作为封存主表）</b></summary><h2>初始开发pilot</h2><p>每方法4患者×1餐食/噪声seed×3餐时bolus系数=12条3天轨迹，前6小时共同预热。下表为患者等权均值 ± <b>患者间SD</b>，目前只有1个训练seed，不能冒充多训练seed稳定性。提前终止后的剩余时间按冻结的吸收失败状态计入，不删失败情景。所有方法使用同一餐时模块；nominal参考额外知道公开的患者基础率参数，学习策略只看到历史观测。</p>',table(['方法','训练更新','训练seed','TIR% ↑','TBR70% ↓','TBR54% ↓','TAR180% ↓','平均risk ↓','平均BG mg/dL','基础U/天','提前终止'],display)]
 if paper:
  with (ROOT/'论文对比表_开发pilot.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paper[0]));w.writeheader();w.writerows(paper)
  tex=['% Development pilot only; mean +/- patient SD; one training seed.','\\begin{tabular}{lrrrrr}','\\hline','Method & TIR (\\%) & TBR70 (\\%) & TBR54 (\\%) & Risk & Failures \\\\','\\hline']
  for p in paper:tex.append(p['method']+' & '+' & '.join('%.2f $\\pm$ %.2f'%(p[k+'_mean'],p[k+'_patient_sd']) for k in ['tir_pct','tbr70_pct','tbr54_pct','mean_risk'])+' & %d/%d \\\\'%(p['failures'],p['scenarios']))
  tex+=['\\hline','\\end{tabular}'];(ROOT/'论文对比表_开发pilot.tex').write_text('\n'.join(tex)+'\n')
  out+=['<p>'+link(ROOT/'论文对比表_开发pilot.csv','CSV原数值表')+' · '+link(ROOT/'论文对比表_开发pilot.tex','LaTeX表')+' · '+link(RESULTS/'D01_pilot50k_dev/summary.json','48条情景指标')+' · '+link(RESULTS/'D01_pilot50k_dev/provenance.json','策略权重/合同/源码SHA')+'</p>']
 if (ROOT/'figures/paired_pilot_trajectories.png').exists():out+=['<h3>同一情景的连续血糖与实际基础输注</h3><p>左侧为独立模拟器真实BG，右侧为泵实际交付基础率；前6小时为共同预热，叉号标记实际提前终止。曲线没有平滑。图只展示bolus系数1.0；所有系数均计入上表。</p><img src="figures/paired_pilot_trajectories.png" style="width:100%;height:auto" alt="四名模拟成人的真实血糖与基础输注轨迹"><p>'+link(ROOT/'figures/paired_pilot_trajectories.svg','可导出SVG图')+'</p>']
 for stage in sorted(RESULTS.glob('D*')):
  if stage.name=='D01_pilot50k_dev':continue
  sr=read(stage/'summary.json',[])
  if not sr:continue
  if not (stage/'completion.json').exists():
   progress=read(stage/'provenance.json',{});out+=['<h3>评价进行中：'+escape(stage.name)+'</h3><p>已生成 '+str(len(sr))+'/'+str(progress.get('jobs','?'))+' 条轨迹；完整方法面板尚未完成，暂不展示聚合排名。'+link(stage/'summary.json','原始进度')+'</p>'];continue
  stage_rows=[]
  for method in sorted(set(v['method'] for v in sr)):
   rr=[v for v in sr if v['method']==method];stage_rows.append([escape(method),len(rr),*[number(statistics.mean(v[k] for v in rr)) for k in ['tir_pct','tbr70_pct','tbr54_pct','mean_risk']],sum(v['failed'] for v in rr)])
  out+=['<h3>追加评价：'+escape(stage.name)+'</h3><p>本阶段实际结果，训练预算/权重见来源；未与不同更新预算混作公平最终排名。'+link(stage/'provenance.json','来源')+' · '+link(stage/'summary.json','逐情景指标')+'</p>',table(['方法','情景数','TIR%','TBR70%','TBR54%','risk','终止数'],stage_rows)]
 out+=[write_tables()]
 assessment=assess();assessment_rows=[]
 for item in assessment['candidates']:
  values={c['check']:c for c in item['checks']};assessment_rows.append([escape(item['candidate']),number(values['delta_TIR_vs_nominal_min_pp']['observed']),number(values['delta_TIR_vs_BC_min_pp']['observed']),number(values['terminal_failures_allowed']['observed']),'满足本批数值容差，仍待后续验证' if item['preliminary_numeric_screen_passed'] else '本批至少一项不满足'])
 out+=['<h3>历史候选开发筛查</h3><p>按已冻结数值容差逐项计算；各行对应各自开发批次；D23为完整三情景seed，其余多数为3101探索，不混作独立封存证据。ΔTIR单位为百分点；相对nominal要求≥−2，相对原BC要求≥2；另外检查低血糖增加和提前终止。完整开发与封存结果以顶部主表为准。所有模型按用户最新要求只使用一个训练seed。</p>',table(['候选','ΔTIR vs nominal','ΔTIR vs BC','终止数','本批判定'],assessment_rows),'<p>'+link(ROOT/'results/development_assessment.json','全部数值条件与未完成项')+'</p>']

 out+=['</details><h2>3. 全部训练与改动记录</h2><p>初始策略、延长训练、图模型辅助、固定参考、有界残差及去RL消融均完整保留。旧R31的额外seed是用户改单seed之前已完成的历史探索，不进入最终主比较。</p>']
 trainrows=[];sections=[]
 for folder in sorted(RESULTS.iterdir()):
  cfg=read(folder/'config.json') if folder.is_dir() else None
  if not cfg:continue
  comp=read(folder/'completion.json',{});failure=read(folder/'failure.json');interruption=read(folder/'interruption.json');records=[]
  if (folder/'history.jsonl').exists():
   for line in (folder/'history.jsonl').read_text().splitlines():
    try:records.append(json.loads(line))
    except json.JSONDecodeError:pass
  step=comp.get('step',comp.get('steps',records[-1].get('step') if records else None));algo=cfg.get('algorithm','Graph dynamics');status='按用户指令中止，记录保留' if interruption else '失败，保留' if failure else '完成本预算' if comp else '尚无完成证据';diagnostics=sorted(folder.glob('diagnostic_*.json'));diag=read(diagnostics[-1],{}) if diagnostics else {};ev=read(folder/'best_evaluation.json',{})
  trainrows.append(['<a href="#'+escape(folder.name)+'">'+escape(folder.name)+'</a>',escape(algo),status,number(step),number(comp.get('full_passes',comp.get('full_epochs'))),number(diag.get('action_mae_u_h')),number(diag.get('td_loss')),escape('alpha='+str(cfg.get('alpha')) if algo=='Graph dynamics' else 'actor_lr='+str(cfg.get('actor_lr')))])
  section='<details id="'+escape(folder.name)+'"><summary><b>'+escape(folder.name)+'</b> · '+escape(status)+'</summary><p>'+escape(status)+'；'+link(folder/'config.json','完整配置')+' · '+link(folder/'provenance.json','数据/源码来源')+'</p>'
  if records:section+=curve(records,'mse_normalized' if algo=='Graph dynamics' else 'actor_loss')+'<p>曲线：'+escape('标准化预测MSE' if algo=='Graph dynamics' else 'actor loss，跨算法数值不可直接比较')+'；'+link(folder/'history.jsonl','全部更新日志')+'</p>'
  for label,obj in [('训练配置',cfg),('完成记录',comp),('离线动作诊断',diag),('图模型预测验证',ev),('失败现场',failure),('用户指令中止记录',interruption)]:
   if obj:section+='<details><summary>'+label+'</summary><pre>'+escape(json.dumps(obj,ensure_ascii=False,indent=2))+'</pre></details>'
  sections.append(section+'</details>')
 out+=[table(['实验','算法/模型','状态','更新步','完整遍历','动作MAE U/h','TD loss','主要设置'],trainrows),'<p>S01：作者风格初值短测；ReBRAC全零饱和记录保留。S02：用train-only动作均值初始化，ReBRAC降低学习率/加强行为正则。R00/R01/R02：统一50k pilot。R10/R11/R12：原权重、优化器和随机状态续到统一200k。G00：图模型工程短测；G01/G02：相同数据的无排序/有排序对照，分别3epoch。未完成项目不填最终数值。</p>']+sections
 graph_rows=[]
 for name in ['G01_graph_factual','G02_graph_rank','G03_graph_hinge']:
  folder=RESULTS/name;ev=read(folder/'best_evaluation.json',{});comp=read(folder/'completion.json',{});sim=read(folder/'sim_factual_forecast.json',{});hs=ev.get('horizons',{});sim_rows=sim.get('rows',[])
  if hs:graph_rows.append([escape(name),number(comp.get('full_epochs')),number(hs.get('30',{}).get('rmse_mmol_l')),number(hs.get('60',{}).get('rmse_mmol_l')),number(statistics.mean(x['horizons']['60']['rmse_mmol_l'] for x in sim_rows) if sim_rows else None),number(ev.get('plus_point1_basal_delta60_median')),number(ev.get('fraction_negative60'))])
 out+=['<h2>患者模型单独评价面板</h2><p>Loop固定8192验证历史中，30/60分钟有效3196/1699；模拟器为12条nominal参考轨迹的事后条件预测，使用实际未来动作/记录事件。图模型和旧A03输入、样本不同，不能直接排名。模拟器恒值基线60分钟RMSE为1.4288 mmol/L；表中为逐轨迹RMSE均值。方向指标受训练先验影响，不能当因果或策略验收。</p>',table(['图模型','完整epoch','Loop30 RMSE','Loop60 RMSE','Sim60 RMSE','+0.1U/h变化中位数','预测下降比例'],graph_rows)]
 probe_rows=[]
 for name in ['G01_graph_factual','G02_graph_rank','G03_graph_hinge']:
  data=read(RESULTS/name/'independent_dose_probe.json',{})
  for delta,hs in data.get('summary',{}).items():
   x=hs['60'];probe_rows.append([escape(name),delta,number(x['model_median']),number(x['simulator_median']),number(x['effect_rmse_mmol_l']),number(x['model_negative_fraction'])])
 if probe_rows:out+=['<h3>独立模拟器：改变剂量的真实效应与模型估计</h3><p>4开发成人×7历史；独立复制相同模拟器状态后改变基础率，共同事件保持一致，两个零干预副本逐输出完全相同。只向模型提供观测历史和共同事件，不提供隐状态。这里不是临床药效估计，也不是最终封存评价。</p>',table(['图模型','增加U/h','模型60min变化中位数','模拟器60min变化中位数','效应RMSE mmol/L','模型下降比例'],probe_rows),'<p>CE图模型方向一致但幅度过大；hinge首版幅度小但方向仍错误。不能将任何一项方向比例冒充动作模型已通过。</p>']
 if (ROOT/'figures/candidate_seed_stability.png').exists():out+=['<h3>历史R31跨训练seed：实际失败保留（非当前O07）</h3><p>同50k预算、同Loop数据、同开发情景。蓝色seed260915、红色seed260916、黑色纯参考；叉号为实际提前终止，曲线未平滑。首seed较好而第二seed失败，不能只报告最好的一次。</p><img src="figures/candidate_seed_stability.png" alt="我方参考残差策略跨训练seed独立轨迹" style="width:100%;height:auto"><p>'+link(ROOT/'figures/candidate_seed_stability.svg','SVG矢量图')+'</p>']
 ablation_rows=[]
 for radius,stage,method,control in [(.25,'D14_ours_bounded025_dev','Ours_bound025','Bound025_noRL'),(.5,'D15_ours_bounded050_dev','Ours_bound050','Bound050_noRL')]:
  if not (RESULTS/'D16_ours_noRL_ablations_dev/completion.json').exists() or not (RESULTS/stage/'completion.json').exists():continue
  rl=[r for r in read(RESULTS/stage/'summary.json',[]) if r['method']==method];bc=[r for r in read(RESULTS/'D16_ours_noRL_ablations_dev/summary.json',[]) if r['method']==control]
  assert {(r['patient'],r['seed'],r['factor']) for r in rl}=={(r['patient'],r['seed'],r['factor']) for r in bc}
  rt=statistics.mean(r['tir_pct'] for r in rl);bt=statistics.mean(r['tir_pct'] for r in bc);ablation_rows.append([radius,number(bt),number(rt),number(rt-bt),number(statistics.mean(r['tbr70_pct'] for r in rl)),sum(r['failed'] for r in rl)])
 if ablation_rows:out+=['<h3>我方同结构去RL消融</h3><p>同参考信息、同输出范围、同seed、同全量Loop数据和50k更新，比较是否使用RL价值目标。这里是首训练seed的开发均值差，尚不能声称跨seed稳定或统计显著；TIR改善仍须同时检查低血糖。</p>',table(['残差范围±U/h','去RL TIR%','RL TIR%','RL增加百分点','RL TBR70%','RL终止数'],ablation_rows)]
 out+=['<h2>我方结构优化：个人参考与有界残差</h2><p>R31使用最初可见6小时基础输注均值作为固定参考，再学习调整量；50k首seed开发TIR92.08%，仍低于纯参考95.30%。早期10k–40k都有提前终止，需验证训练稳定性。O01/O02进一步采用reference + L·tanh(2f/L)，L=0.25/0.5 U/h；O07取中间0.4，O08为其同结构去RL；O03/O04是两端范围去RL消融，所有组均保留完整原始训练动作目标。对输出设限不等于已证明安全或因果支持。</p><p>依据：<a href="https://arxiv.org/abs/1812.03201">Residual RL（ICRA2019）</a>的参考与学习残差叠加；<a href="https://proceedings.neurips.cc/paper/2019/hash/c2073ffa77b5357a498057413bb09d3a-Abstract.html">BEAR（NeurIPS2019）</a>关于离线动作外推误差的分析。此处是任务特定结构适配，不是BEAR复现；不能把这两篇论文当本任务的效果证据。相对初始参考±0.25/±0.5仅覆盖25.82%/42.76%的训练动作，因此对较大需求变化的能力受限。必须通过独立轨迹和去RL对照证明实际贡献，不能仅凭近似固定参考就验收。历史首组O01的12条开发轨迹结果：TIR92.22%、TBR54=0、无终止，仍低于nominal约3.07个百分点，尚未通过。</p>']
 out+=['<h2>4. 图模型改进及其边界</h2><p>依据ICML2024 H²NCM独立实现：LSTM从22维观测历史初始化7节点，胰岛素经过中间通路后影响血糖；餐食和运动使用各自通路，替代任意动作直接进入血糖残差。5分钟同步更新使候选胰岛素最早20分钟传到血糖节点，这是明确的结构先验。混合归一化预测MSE与干预排序CE；排序标签来自先验，没有真实反事实标签。图模型的较好MSE或方向一致性都不能替代独立策略收益；图模型辅助策略已经实际训练并开始独立评价，已完成的组表现失败；所有结果均在上表，这些图模型辅助候选未胜过强行为约束基线；当前O07独立结果见顶部主表。</p><h2>5. 工程核验、失败与尚未完成</h2>']
 checks=[]
 for p,label in [(RESULTS/'tests.json','离线RL公式/梯度/数据检查'),(RESULTS/'sim_checks/checks.json','独立模拟器重复性和剂量核验'),(RESULTS/'graph_tests.json','图模型延迟/梯度/目标对齐'),(RESULTS/'guidance_tests.json','模型辅助actor梯度与零系数基线一致性'),(RESULTS/'anchor_tests.json','固定历史参考与残差策略检查'),(RESULTS/'anchor_export_tests.json','已训练参考策略导出与跨患者隔离'),(RESULTS/'bounded_residual_tests.json','有界残差与全量目标保持检查'),(RESULTS/'monotonic_penalty_tests.json','定性方向损失手算检查'),(RESULTS/'research_inference_checks.json','真实评价动作与独立JSON进程一致性'),(RESULTS/'research_inference_local_checks.json','Mac本地权重/动作一致性'),(RESULTS/'final_result_checks.json','全部封存原始轨迹和成对条件重算')]:
  if p.exists():checks.append('<details><summary>'+label+'</summary><pre>'+escape(p.read_text())+'</pre></details>')
 out+=checks+['<ul><li>环境失败：远端GitHub clone超时exit128，使用固定源码归档及SHA传输；旧Gym缺pkg_resources，固定setuptools80.9.0后import及实际模拟通过。</li><li>ReBRAC/FQL/BC全部50k pilot及48条评价结果已回收；原失败权重保留，不删情景。</li><li>已完成原组200k延长训练、我方结构/损失探索及消融；最终主表固定50k、单seed。研究结果不自动完成真实双模型产品接入。</li><li>仿真没有运动机制；food/exercise保留在训练输入，但本仿真不证明运动疗效。真实患者部署、双模型产品准入均未完成。</li></ul><h2>6. 当前审核结论</h2><p>'+conclusion()+'</p></body></html>']
 (ROOT/'训练审核报告.html').write_text(''.join(out));print(ROOT/'训练审核报告.html')
if __name__=='__main__':main()
