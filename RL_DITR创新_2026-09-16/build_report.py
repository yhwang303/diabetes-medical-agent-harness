# -*- coding: utf-8 -*-
"""Plain review HTML: current evidence, explicit gaps, and complete attempt history."""
import csv,html,json,time,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def esc(x):return html.escape(str(x))
def read(p):return json.loads(p.read_text())
def link(p,label=None):return '<a href="'+esc(p.relative_to(ROOT))+'">'+esc(label or p.name)+'</a>'
def table(headers,rows):return '<div style="overflow-x:auto"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def fmt(x):return '—' if x is None else ('%.3f'%x if isinstance(x,float) else str(x))
def method_label(folder):
 p=folder/'manifest.json'
 if not p.exists():return folder.name
 m=read(p)
 if m.get('mode')=='nominal':return '固定基础率（仿真参数参考）'
 checkpoint=Path(m.get('checkpoint') or '');name=checkpoint.parent.name;code=name.split('_')[0]
 labels={'R01b':'R01b 无界标量参考','R02':'R02 有界标量参考','R03':'R03 RL-DITR分类头参考','H01':'H01 动作prefix结构实验','R04':'R04 递归+仿真事实（旧目标）','R05':'R05 递归+仿真事实（目标修正）','H02':'H02 prefix+仿真事实','H03':'H03 prefix+事实+配对目标','C01':'C01 响应结构+继承策略','C02':'C02 H02仅改奖励计算','H04':'H04 参考轨迹+因果响应','R07':'R07 H02额外策略轮次对照','H05':'H05 因果响应+延迟回报策略','R08':'R08 H02+延迟回报策略'}
 step=re.search(r'policy_(\d+)',checkpoint.name)
 completion=ROOT/'results'/name/'policy_completion.json'
 step=step.group(1).lstrip('0') if step else str(read(completion)['steps']) if completion.exists() else '待核实'
 if code in ['C01','C02']:step='继承6580（未新增策略训练）'
 if code in ['H04','R07','H05','R08']:step='继承6580 + 新增'+step
 mode={'beam':'策略+1小时逐步规划','actor':'仅策略','chunk':'策略+4小时动作块规划'}.get(m.get('mode'),str(m.get('mode')))
 return labels.get(code,name)+' / '+mode+' / '+step+'步'
def data_label(folder):
 manifest=read(folder/'manifest.json')
 if manifest.get('mode')=='nominal':return '仿真器参数参考'
 p=ROOT/'results'/Path(manifest.get('checkpoint') or '').parent.name/'config.json'
 if not p.exists():return '继承完整H02 + 原恒定干预响应训练' if Path(manifest.get('checkpoint') or '').parent.name.startswith('C01_') else '继承完整H02，0次新训练' if Path(manifest.get('checkpoint') or '').parent.name.startswith('C02_') else '待核实'
 c=read(p)
 if c.get('patient_training_origin'):c=read(ROOT/'results'/c['patient_training_origin']/'config.json')
 return 'Loop + 训练仿真' if c.get('simulation') else 'Loop'
def main():
 out=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RL-DITR基底与创新：训练审核报告</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:30px auto;padding:0 24px;color:#18222b}h1,h2,h3{line-height:1.3}table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}td,th{border:1px solid #cbd3da;padding:8px;text-align:left}th{background:#eef2f5}pre{overflow:auto;background:#f1f4f6;padding:16px}details{border:1px solid #ccd5dc;padding:12px;margin:14px 0}.status{padding:16px;border-left:5px solid #9b6118;background:#fff5df}.small{font-size:13px;color:#52606b}</style><body><h1>RL-DITR基底与创新：训练审核报告</h1>']
 out+=['<p class="small">生成于 '+time.strftime('%Y-%m-%d %H:%M:%S')+'。依据本地同步证据；远端中断后的完成状态以恢复核验为准。</p><div class="status"><b>已暂停，尚未完成新目标。</b> R05/H02/H03已完成全量单seed训练和一小时开发闭环；H02/H03均未超过固定基础率参考，等总量时序识别失败。响应结构P01/P02改善了探针方向，但C01尚未超过固定基础率参考；H04/R07额外旧目标训练没有改善控制。H05已完成新目标训练，R08及后续控制最终状态待恢复核对。尚无已验证的非incremental创新或SOTA结论。旧O07属于上一轮ReBRAC适配。</div>']
 out+=['<h2>1. 当前模型到底是什么</h2><p><b>读表顺序：</b>先看当前同数据组R05/H02/H03，再看Loop-only的R03/H01机制实验。R开头是参考或适配修正，H开头是研究候选；E开头只是评价作业编号。还没有定稿的“我方最终模型”。</p>',table(['名字','身份','与主线关系'],[['H05 / R08','当前目标消融：响应结构 / H02结构，统一延迟回报+KL','同初始化与新增一轮预算；尚无最终我方模型'],['H04','参考轨迹+血糖/latent因果响应，额外旧目标策略轮次','机制改善；额外策略训练使开发控制变差'],['C01','H04的患者结构，直接继承原H02策略','0次新增策略更新，区分结构和额外训练影响'],['R07','H02保持结构，多训练相同一轮策略','控制H04额外策略训练预算，非响应参数预算'],['P01/P02/P03/P04','血糖/latent响应机制原型、全局核及同容量MLP消融','不是独立RL控制器'],['R05','当前递归RL-DITR参考：Loop+仿真事实，修正连续策略目标','继承R04患者模型；与H02/H03统一数据和策略目标'],['H02','动作prefix结构候选：相同Loop+仿真事实','与R05比较转移结构'],['H03','H02 + 配对动作效应/风险路径监督','与H02比较配对目标；不预先声称非增量原创'],['R03 / H01','已完成的Loop-only参考 / prefix结构实验','同容量、数据、目标；H01尚未超过固定基础率参考'],['R01/R01b/R02/R04','数值后端、输出支持和策略目标探索','保留失败及技术修正，不能作为创新数量'],['旧ReBRAC / FQL','上一轮外部强化学习baseline','不再训练或仿真，只补算已有轨迹指标'],['旧BC','行为克隆：监督拟合历史动作','不是另一个强化学习算法'],['旧O07','固定参考、有界残差ReBRAC','历史结果，不再充当本轮我方主线']])]
 p=ROOT/'checks/remote_disconnect_observation.json'
 if p.exists():
  d=read(p)
  out+=['<div class="status"><b>用户要求暂停：AutoDL余额不足。</b> 2026-09-16约17:40后SSH拒绝连接。仅在用户明确说“任务开始”后恢复，不再提交训练或评价。当前远端进程状态未核实。'+link(ROOT/'阶段总结与恢复交接_2026-09-16.md','完整阶段总结与恢复顺序')+'</div>']
  h=d['H05_integrity_observed'];out+=['<p><b>晚于本地完整快照的已确认事实：</b>H05已完成'+str(h['steps'])+'更新、'+str(h['samples_seen'])+'起点访问，225人完整一轮，患者权重冻结检查通过；R08最后看到2550步。H05/R08及C02新控制分数尚未取得。'+link(p,'MCP返回值摘录（不是完整远端文件下载）')+'</p>']
  out+=[table(['已完成原型','训练数据','参数','更新数','动态240min效应MAE','时序方向%','时序终点MAE'],[[x['name'],'分时七臂504组',x['parameters'],x['steps'],fmt(x['dynamic_dev240mae']),fmt(x['timing_direction']*100),fmt(x['timing_mae'])] for x in d['temporal_response_results_observed']]),'<p>两者均100%说明普通MLP在更丰富干预覆盖下也能识别时序；卷积先验的独有优势尚未建立。单位mg/dL，开发诊断，非控制分数。</p>']
 out+=['<h3>结构与梯度</h3><pre>72×22历史（数值、缺失标记、记录年龄、时间）\n  → fR：3层Transformer / hidden256 / 8heads\n  → 初始患者latent z0 + 历史memory\n候选基础率a（U/h）+ 时间 + 当前latent\n  → fT：3层Transformer decoder，cross-attend历史\n  → 下一个latent → fP：血糖 / WTR概率\n  → reward头，value头\nπ：3层MLP → 连续有界高斯的均值/尺度\n  → 真实历史回报项 + 模型生成回报项 + 行为监督\n  → B10、K12 beam：累计折扣reward + 末端V\n  → 每5分钟执行首个基础率动作，再观察重规划</pre><p>患者训练时更新fR/fT/fP/reward/V；策略训练时冻结患者模型，只有π接收梯度。R02的status范围为[-1,1]，每5分钟reward=status/12；γ=0.9^(1/12)，价值范围按几何级数界限制。奖励形状未为追求成绩修改。</p>']
 out+=['<h3>损失与不能省略的适配</h3><pre>L_patient = MSE_glucose + CE_WTR + 0.1·MSE_consistency\n            + L_reward + L_value\nR01/R02: L_reward/L_value是标量MSE\nR03: 作者式two-hot分布交叉熵，按bins取mean（3/41 bins）\nL_policy = −mean(R_logged·logπ(a_logged|z))\n           −mean(R_imagined·logπ(a_sampled|z))\n           + MSE(policy_mean, a_logged)</pre><p>原论文是住院T2D离散注射，这里是成人T1D连续基础输注；时间、概率策略、SL和value目标均有显式适配。原文“V代替R”的稳定化细节未由公开policy trainer完整核实，本轮采用其明确列出的回报公式及尾端V，不冒称作者代码逐位复现。</p><p>'+link(ROOT/'任务与复现合同.md','完整映射、原文/源码差异与适配清单')+' · '+link(ROOT/'近期方法依据.md','近期创新依据与近邻工作')+'</p>']
 out+=['<p>'+link(ROOT/'模型结构与实验代号.md','完整结构详解：输入、计算图、参数量、loss、step/epoch与后续预测器接入')+'</p>']
 out+=['<h3>当前候选H04的结构变化</h3><pre>历史 → 冻结H02 fR → z0\n最后记录基础率 → H02参考转移 → G_ref、z_ref\n候选动作差Δa → 学习的因果卷积 → ΔG、Δz\nG = G_ref + ΔG；z = z_ref + Δz\n原status(G)/12 → 回报；干预后的z → 策略π / 价值V</pre><p>响应只用原504组恒定干预训练，未用开发标签或新增分时数据；全Loop预训练保留。P01条件核和P02全局8参数核均在84对时序探针达到100%方向一致，因此方向收益不能归功于个体化。幅度仍有误差，控制效果独立评价。</p><p>H04策略额外训练一轮；R07保持H02结构，执行同一额外轮次。该对照不消除新增响应参数和响应训练预算差异。局部线性/时间不变及点预测风险均为明确假设，尚非已验证的非增量创新。</p>']
 out+=['<h2>2. 训练数据与预算</h2><p>所有新训练只用seed 260915；全量225位训练患者、1,653,421个起点，验证55位患者、424,866个起点。food/exercise缺失是输入特征，不按模态存在性筛样本。每阶段先以1次全量遍历为预算，不冒称论文100 epoch。一步是一次优化器更新；一个epoch是每个原有起点访问一次。恢复作业不是新的seed。</p>']
 for split in ['train','validation']:
  p=ROOT/'long_horizons'/split/'manifest.json'
  if p.exists():
   d=read(p);out+=['<p>'+esc(split)+'有效未来监督数量（其他起点仍保留）：</p>',table(['分钟','有效起点数'],[[h,sum(r['horizon_counts'][h] for r in d['patients'])] for h in ['5','15','30','60','120','240']])]
 out+=['<h3>仿真扩展的同数据消融</h3><p>当前匹配组为R05/H02/H03。R05继承R04完成训练并经验证选择的递归患者模型，只重新训练策略；H02=动作prefix+相同仿真事实；H03=H02+逐时点ΔCGM/Δstatus配对差目标。三者统一采用下述修正策略目标，H02/H03在开始训练前完成配置修订。使用相同训练专用情景40101/40102；每8次患者更新混入4组×5臂、最多4小时监督。原Loop全量保留，开发探针及最终场景不用于训练。新增数据意味着不能拿旧FQL/ReBRAC结果声称公平算法胜出。此组仍是机制假说，不预先宣称足够新颖。</p><h3>R04失败与连续策略目标修正</h3><p>R04旧目标在1625步出现loss −14630.88、动作MSE 39.37。原作业在首次策略保存前停止；从完整患者阶段RNG精确重放至2000步以归档失败权重，65个共同日志点（至1625步）loss差均为0。这是恢复同一次轨迹，不是新增seed。</p><pre>B = (1/12)/(1−γ) ≈ 9.533\nL_policy_repaired = −E[(1 + clip(R_logged/B, −1, 1)) logπ(a_logged|z)]\n                    −E[stopgrad((G_imagined−V(z))/B) logπ(a_sampled|z)]</pre><p>行为监督由动作MSE改为正确的连续分布负对数似然，历史回报有界归一化后联合系数非负；想象回报加入状态价值baseline。动作MSE仅保留诊断。原status/reward形状和γ不变。这是连续目标的技术修正，不算结构创新；已验证解析反例、225位患者抽样和梯度，开发闭环仍有明显失败，技术修正不等于有效控制。</p>']
 p=ROOT/'paired_sim_train/manifest.json'
 if p.exists():
  d=read(p);out+=['<p>已生成'+str(len(d['scenarios']))+'个训练情景、'+str(d['groups'])+'组同状态、'+str(d['arms'])+'条分支、'+str(d['valid_intervals'])+'个5分钟监督标签。它们存在组内/时间关联，不能当作独立患者数量。'+link(p,'完整生成清单与哈希')+'</p>']
 p=ROOT/'results/C01_response_patient_composition/manifest.json'
 if p.exists():out+=['<p>C01组合来源已经逐权重哈希核验：H02完整患者/策略、P01血糖响应、P03潜在响应。组合操作是0次优化更新，不能把它标成重新完成一轮训练。'+link(p,'组合来源与参数量')+'</p>']
 out+=['<h3>C02：奖励一致性单因素对照</h3><p>H02全部权重、转移与策略保持精确相同；只将独立奖励头改成status(原预测血糖)/12，0次训练更新。用于隔离C01的响应结构与奖励映射改变，点预测风险仍未经不确定性校准。</p><h3>H05/R08：延迟策略目标</h3><pre>每个原Loop历史 → 4条独立4小时轨迹（3个80分钟动作块）\nJ = Σ γ^t · 原status(G_t)/12\nZ = J − 1·Σ_blocks(logπ − logπ_ref)\nA_i = Z_i − mean(Z_j, j≠i)\nL = −mean(stopgrad(A_i) · Σ_blocks logπ_i)</pre><p>患者模型及初始化策略π_ref冻结。该额外阶段用软KL替代历史动作NLL，仍遍历全部Loop历史；有限目标不加V尾端，原规划与V头保留。最终epoch预先指定。LOO/KL为已知方法适配，不称原创；块策略与实际5分钟重规划不同，软惩罚没有硬约束保证。</p>']
 p=ROOT/'checks/policy_gradient_balance.json'
 if p.exists():
  d=read(p);out+=['<p>旧目标诊断：'+str(d['patients'])+'人、'+str(d['origins'])+'固定训练起点；想象项梯度范数约为行为联合项的0.04%。这只是抽样诊断，不证明新目标必然有效。'+link(p,'原始梯度分解')+'</p>']
 out+=['<h2>3. 全部新训练尝试</h2>']
 attempt_start=len(out);attempt_rows=[]
 for p in sorted((ROOT/'results').glob('*/config.json')):
  c=read(p);folder=p.parent;logs=[]
  if 'encoder_checkpoint' in c and 'basis_degree' in c:
   evaluation=folder/('state_response_evaluation.json' if 'latent_rank' in c else 'response_evaluation.json');result=read(evaluation) if evaluation.exists() else {}
   attempt_rows.append([c['name'],'冻结H02编码器 + '+c['training_data'],result.get('status','响应算子训练中'),'响应模块 '+str(result.get('steps',''))+'步','共享已完成H02编码器；非新患者/策略训练'])
   out+=['<details><summary>'+esc(c['name'])+'：响应结构原型（尚非RL控制器）</summary><p>原Loop全量训练编码器冻结。数据为 '+esc(c['training_data'])+'；constant五臂与temporal七臂分别成组比较，不是等标签数的跨数据比较。开发标签不参与训练。结构假设须实测，不能凭机械检查声称生理真实。</p><pre>'+esc(json.dumps(c,indent=2,ensure_ascii=False))+'</pre>']
   if result and 'development' in result:
    timing=result['development'].get('equal_dose_timing',{});out+=['<p>等总量时序方向一致率：'+fmt(timing.get('direction_agreement'))+'；终点效应MAE：'+fmt(timing.get('final_effect_mae_mg_dl'))+' mg/dL。'+link(folder/'response_evaluation.json','完整机制结果')+'</p>']
   elif result:out+=['<p>潜在状态响应训练MSE：'+fmt(result.get('training_latent_effect_mse'))+'；零响应MSE：'+fmt(result.get('zero_response_mse'))+'。'+link(evaluation,'潜在状态响应训练结果（不是控制效果）')+'</p>']
   out+=['</details>'];continue
  if (folder/'history.jsonl').exists():
   for line in (folder/'history.jsonl').read_text().splitlines():
    try:logs.append(json.loads(line))
    except json.JSONDecodeError:pass
  state='快照中未见完成记录'
  if (folder/'failure.json').exists():state='中断：'+read(folder/'failure.json')['error']
  if (folder/'completion.json').exists():state=read(folder/'completion.json')['status']
  if (folder/'interruption.json').exists():state=read(folder/'interruption.json')['status']
  selected=[x for x in logs if x.get('event')=='patient_selected_for_policy'];last=logs[-1] if logs else {};budget='Loop+仿真事实'+('+配对' if c.get('simulation',{}).get('paired_weight') else '') if c.get('simulation') else '继承'+c['patient_training_origin']+'患者模型；策略用Loop' if c.get('patient_training_origin') else 'Loop'
  attempt_rows.append([c['name'],budget,state,str(last.get('stage',last.get('event','—')))+' '+str(last.get('step',last.get('steps',''))),str(selected[-1]['step'])+'步 / '+str(selected[-1]['samples'])+'访问' if selected else '待选'])
  out+=['<details><summary><b>'+esc(c['name'])+'</b> — '+esc(state)+'</summary><p>seed '+str(c['seed'])+'，batch '+str(c['batch_size'])+'，患者/策略学习率 '+str(c['patient']['lr'])+' / '+str(c['policy']['lr'])+'</p>']
  if logs:out+=['<p>最新日志：'+esc(json.dumps(logs[-1],ensure_ascii=False))+'</p>']
  rows=[[r.get('stage'),r.get('step'),fmt(r.get('rmse30_plus60')),fmt(r.get('reward_status_mse')),fmt(r.get('policy_action_mse'))] for r in logs if r.get('event')=='validation']
  if rows:out+=[table(['阶段','step','RMSE30+60 (mmol/L)','reward误差','动作模仿MSE'],rows)]
  out+=['<p>'+link(p,'完整配置')+'</p><pre>'+esc(json.dumps(c,indent=2,ensure_ascii=False))+'</pre></details>']
 out.insert(attempt_start,table(['训练代号','数据/目标','状态','最近阶段与步数','策略采用的患者检查点'],attempt_rows)+'<p>“访问”是该患者阶段在选中检查点之前的样本访问数；完整训练预算与选中检查点实际经历的更新分别记录。选择较早检查点不等于按food/exercise缺失筛掉资料。</p>')
 out+=['<h2>4. 指标和论文比较设计</h2><p>主效果量为相对完整RL-DITR的配对ΔTIR（70–180 mg/dL）。TBR&lt;54、TBR&lt;70、提前终止必须并列；同时报告TAR、CV、LBGI/HBGI、低糖事件时长、剂量、动作变化与计算代价。预测MSE不是控制成绩，奖励值也不能替代独立仿真。</p><p>真实拟合按15/30/60/120/240分钟报告患者宏平均MAE/RMSE、持续值参考、WTR Brier和缺失模态分层；共同状态/扰动动作探针报告效应误差与有限候选池遗憾。超过训练时长的预测明确标为外推。</p><p>新场景只比较RL-DITR、我方和必要消融/固定参考，FQL/ReBRAC记N/A；已有10名虚拟成人均已曝光，不能称未见患者。单训练seed的情景/患者置信区间不含训练随机性。</p><p>'+link(ROOT/'评价文献与协议建议.md','论文来源、定义、统计及失败处理')+'</p>']
 latest_dynamics={}
 for p in sorted((ROOT/'results').glob('*/dynamics.json')):
  d=read(p);key=d['checkpoint'];old=latest_dynamics.get(key)
  if old is None or d.get('action_probe_version')=='v2_actual_delivered':latest_dynamics[key]=(p,d)
 out+=['<h3>预测总览：患者宏平均RMSE，mg/dL</h3>',table(['患者模型/权重','15min','30min','60min','120min','240min','240min有效患者'],[[Path(d['checkpoint']).parent.name+' / '+Path(d['checkpoint']).name]+[fmt(d['macro_summary'][h]['macro_rmse_mg_dl']) for h in ['15','30','60','120','240']]+[d['macro_summary']['240']['patients_with_valid_targets']] for p,d in latest_dynamics.values()]),'<p>当前为固定开发抽样诊断，不是最终封存集成绩。先前动作探针使用请求剂量，v2改为实际泵送剂量；总览优先v2，旧版完整保留在下方展开记录。</p>']
 for p in sorted((ROOT/'results').glob('*/dynamics.json')):
  d=read(p);out+=['<details><summary>'+esc(p.parent.name)+'</summary>',table(['分钟','患者数','宏RMSE mg/dL','持续值RMSE','WTR Brier'],[[h,v['patients_with_valid_targets'],fmt(v['macro_rmse_mg_dl']),fmt(v['persistence_macro_rmse_mg_dl']),fmt(v['macro_wtr_brier'])] for h,v in d['macro_summary'].items()]),'<p>'+link(p,'逐患者/分层/动作效应结果')+'</p></details>']
 probe_rows=[]
 for p,d in latest_dynamics.values():
  probes=d.get('action_probes',[]);v=[]
  for h in ['30','60','120','240']:
   values=[x['horizons'][h]['effect_mae_cgm_mg_dl'] for x in probes if x['horizons'][h]['effect_mae_cgm_mg_dl'] is not None];v.append(fmt(sum(values)/len(values)) if values else '—')
  complete=[x for x in probes if 'predicted_best' in x]
  probe_rows.append([p.parent.name]+v+[fmt(100*sum(x['predicted_best']==x['simulator_best'] for x in complete)/len(complete)) if complete else '—',fmt(sum(x['candidate_pool_regret'] for x in complete)/len(complete)) if complete else '—'])
 if probe_rows:out+=['<h3>独立共同动作探针（开发诊断）</h3>',table(['实验','30min效应MAE','60min','120min','240min','候选选择一致%','候选池平均regret'],probe_rows),'<p>效应误差单位mg/dL；28个同状态探针，每个5个基础率计划，共同餐食/bolus与随机流。regret只在此候选池、此固定status效用下成立，不是全局最优差距，也不替代低糖指标。</p>']
 identification_rows=[]
 for p in sorted((ROOT/'checks').glob('*_action_identification.json')):
  d=read(p)
  for split in ['training_diagnostic','development_diagnostic']:
   for h in ['30','60','120','240']:
    v=d[split][h];identification_rows.append([d['run'],'训练干预' if split.startswith('training') else '开发干预',h,v['pairs'],fmt(v['true_effect_rms_mg_dl']),fmt(v['predicted_effect_rms_mg_dl']),fmt(v['effect_mae_mg_dl']),fmt(v['direction_agreement'])])
 if identification_rows:out+=['<details><summary>动作效应的时间响应：逐时长训练/开发诊断</summary>',table(['患者模型','数据','分钟','动作对','真实效应RMS','预测效应RMS','效应MAE','方向一致率'],identification_rows),'<p>单位mg/dL。训练诊断固定抽64组，开发为28组；所有有效对进入误差统计。方向只在真实差值超过1 mg/dL时计算，近零效应不强行判正负。R04在训练组已出现短时效应过大、长时效应消失，不能仅归因为测试泛化。阈值只用于解释方向，不参与模型训练或通过标准。</p></details>']
 response_rows=[]
 for name in ['P01_conditioned_response_operator','P02_global_response_ablation','P04_direct_prefix_response','P05_temporal_conditioned_response','P06_temporal_direct_prefix']:
  p=ROOT/'results'/name/'response_evaluation.json'
  if p.exists():
   d=read(p);t=d['development']['equal_dose_timing'];response_rows.append([name,read(p.parent/'config.json')['training_data'],d['response_parameters'],d['steps'],fmt(d['training']['240']['effect_mae_mg_dl']),fmt(d['development']['action_probe']['240']['effect_mae_mg_dl']),fmt(100*t['direction_agreement']),fmt(t['final_effect_mae_mg_dl'])])
 if response_rows:out+=['<h3>同训练预算的响应结构消融</h3>',table(['响应模块','训练干预数据','参数量（不含共享编码器）','更新数','恒定训练效应MAE','恒定开发效应MAE','重排方向一致%','重排终点效应MAE'],response_rows),'<p>效应误差为mg/dL；同一冻结H02编码器、504组、100轮/3200更新、seed260915；P01/P02/P04用原恒定五臂，P05/P06用分时七臂。同数据内部参数/更新预算可比，跨数据标签量不相同。P04为直接因果prefix MLP，与P01参数差11个；没有卷积/时间不变约束。P02只有8个共享参数，同样达到正确方向，因此个体化条件不是方向收益的必要解释。该消融只支持当前时间结构假说，不自动建立原创性或临床有效性。</p>']
 dynamic_rows=[]
 for p in sorted((ROOT/'checks').glob('*_dynamic_actions.json')):
  d=read(p);v=d['summary'];dynamic_rows.append([d['run'],v['equal_dose_pairs'],fmt(v['path_effect_mae_mg_dl']),fmt(v['final_direction_agreement']),fmt(v['reward_choice_agreement']),fmt(v['reward_regret']),v['shortened_arms']])
 if dynamic_rows:out+=['<h3>同总剂量、不同时间：独立开发探针</h3>',table(['模型','等总量计划对','路径效应MAE mg/dL','终点方向一致率','reward选择一致率','平均regret','提前终止分支'],dynamic_rows),'<p>28个共同历史，每个7条计划；84对实际输送总量相同、时间顺序不同的干预。它们不用于训练。H02在恒定动作上方向正确，但时序重排失败，不能只凭前一项宣布动作效应已识别。'+link(ROOT/'时间干预识别诊断.md','数据设计、有限线性反例与近期近邻边界')+'</p>']
 control_rows=[];historical_control_rows=[]
 for p in sorted((ROOT/'results').glob('*/summary.json')):
  if p.parent.name.startswith('E00'):continue
  d=read(p)
  if d.get('status')!='completed' or 'episodes' not in d:continue
  episodes=d['episodes'];groups={}
  for e in episodes:groups.setdefault(e['patient'],[]).append(e)
  values=[]
  for key in ['tir_lower_bound_pct','tir_upper_bound_pct','coverage_pct','tir_observed_pct','tbr70_pct','tbr54_pct','cv_pct']:
   means=[]
   for group in groups.values():
    v=[r['metrics']['bg'].get(key) for r in group];v=[x for x in v if x is not None]
    if v:means.append(sum(v)/len(v))
   values.append(fmt(sum(means)/len(means)) if means else '—')
  label=method_label(p.parent);row=[label,data_label(p.parent),len(groups),len(episodes)]+values+[sum(r['metrics']['failed'] for r in episodes)]
  (control_rows if label.startswith(('R05 ','H02 ','H03 ','C01 ','C02 ','H04 ','R07 ','H05 ','R08 ','固定基础率')) else historical_control_rows).append(row)
 if control_rows:
  out+=['<h3>当前同数据组：已完成的开发闭环（非封存）</h3><p>R05/H02/H03、响应结构及目标消融与固定参数参考。H04/R07/H05/R08在继承6580步策略上各额外训练一轮，不能按相同总预算与前三者解释。H05/R08从额外训练前初始化，不继承H04/R07。未完成的方法不填分数。</p>',table(['方法 / 推理方式 / 策略检查点','训练数据','患者','轨迹','TIR下界%','TIR上界%','覆盖%','观测BG TIR%','TBR70%','TBR54%','CV%','失败数'],control_rows),'<p>步数读取实际策略记录；完整epoch为6580步，失败重放只有2000步。患者模型另按固定验证集选择。不同训练数据的行不能用于公平算法胜出声明。BC只模仿历史动作；表中“仅策略”仍经过RL训练，不能与BC混称。固定基础率来自仿真器患者参数，是已知参数的简单控制参考。</p><p>这是已曝光开发场景；主表为模拟器BG。失败后未观测血糖不填造数据，因此须一起看覆盖率、TIR上下界和失败数；低糖/风险只针对实际观测。R03最终beam的3条提前终止由持续高糖触发，低TBR不能证明安全。R01即使TIR较高也有已知价值头缺陷。</p>']
 if historical_control_rows:out+=['<details><summary>前序结构实验、推理消融与失败版本的全部开发闭环</summary>',table(['方法 / 推理方式 / 策略检查点','训练数据','患者','轨迹','TIR下界%','TIR上界%','覆盖%','观测BG TIR%','TBR70%','TBR54%','CV%','失败数'],historical_control_rows),'</details>']
 if (ROOT/'figures/response_mechanism_and_control.svg').exists():out+=['<figure><img style="width:100%" src="figures/response_mechanism_and_control.svg" alt="响应结构的时序机制改善与尚未改善的一小时控制分别展示"><figcaption>机制改善与独立控制分开呈现；C01继承H02策略，未执行额外策略训练。4位已曝光虚拟患者、同训练seed，不是未见患者或临床验证。</figcaption></figure>']
 if (ROOT/'figures/development_patient_comparison.svg').exists():out+=['<figure><img style="width:100%" src="figures/development_patient_comparison.svg" alt="R03、H01和固定基础率的逐患者TIR及低血糖对比"><figcaption>每人3个相同开发情景。左图线段为提前终止造成的TIR未知结局范围，不是置信区间；右图低糖仅计算已观测时间。PNG/SVG图已单独检查，完整HTML浏览器渲染仍未验证。</figcaption></figure>']
 for p in sorted((ROOT/'paper_tables').glob('*/main.csv')):
  out+=['<details><summary>'+esc(p.parent.name)+'：可导出多维比较表</summary><p>'+link(p,'完整CSV（含全部指标）')+' · '+link(p.parent/'paired_comparisons.json','配对差与95%区间')+'</p>']
  with p.open() as f:records=list(csv.DictReader(f))
  keys=['tir_lower_bound_pct','tbr70_pct','tbr54_pct','tar250_pct','cv_pct','lbgi','hbgi','basal_u_per_observed_day']
  def cell(row,key):
   a=row.get(key+'_mean');b=row.get(key+'_patient_sd')
   return ('%.2f'%float(a))+(' ± %.2f'%float(b) if b else '') if a else '—'
  rows=[[method_label(ROOT/'results'/r['run']),r['group']]+[cell(r,k) for k in keys]+[r['failures']+'/'+r['episodes']] for r in records]
  out+=[table(['方法','情景','TIR下界%','TBR70%','TBR54%','TAR250%','CV%','LBGI','HBGI','基础量U/日','失败'],rows),'<p>均值±患者标准差；开发情景每组4位已曝光虚拟患者。TIR下界包含提前终止影响，其余风险、剂量仅按观测期计算；覆盖率、TIR上界、CGM指标、动作变化及配对区间见完整CSV/JSON。单seed不提供训练随机性区间。</p></details>']

 timing_rows=[]
 for p in sorted((ROOT/'results').glob('*/inference_timing.json')):
  d=read(p);groups={}
  for b in d.get('batches',[]):groups.setdefault(b['batch_size'],[]).append(b['seconds'])
  for n,values in groups.items():
   v=sorted(values);timing_rows.append([method_label(p.parent),n,len(v),fmt(v[len(v)//2]*1000),fmt(v[min(len(v)-1,int(.95*len(v)))]*1000)])
 if timing_rows:out+=['<details><summary>计算代价：实测推理批次时间</summary>',table(['方法','同时推理患者数','批次数','中位毫秒/批','P95毫秒/批'],timing_rows),'<p>GPU worker内计算，含结果取回，不含JSON传输和仿真器；训练/其他评测可能共享GPU，批量也不同，因此是实际资源记录，不是受控吞吐排名。最终效率比较仍需相同硬件与空闲条件。</p></details>']
 out+=['<h2>5. 旧T01：只离线补算，不是新封存实验</h2><p>下表均来自既有54条/方法的相同历史情景。失败后的未知时间没有填伪造血糖；CV/风险等只计算实际观测，因此必须结合覆盖率和失败数阅读，不能单独据此排名。保留旧固定分母惩罚TIR。</p>']
 p=ROOT/'legacy_panel/summary.json'
 if p.exists():
  d=read(p);out+=[table(['历史方法','旧惩罚TIR%','覆盖率%','观测TIR%','TBR70%','TBR54%','CV%','LBGI','HBGI','失败/54'],[[r['method']]+[fmt(r[k]) for k in ['old_failure_penalized_tir_pct','coverage_pct','tir_observed_pct','tbr70_pct','tbr54_pct','cv_pct','lbgi','hbgi']]+[str(r['failures'])+'/54'] for r in d['rows']]),'<p>'+link(ROOT/'legacy_panel/observed_metrics.csv','完整新增指标CSV')+' · '+link(p,'来源说明')+'</p>']
 out+=['<h2>6. 已知失败、验证和未完成项</h2><ul><li>R01融合FP32 attention出现NaN，有限数据/权重下已对照数学后端与BF16定位；R01b恢复原状态。</li><li>首版无界reward/value回归偏离原输出支持范围；R02修正，不列为算法创新。</li><li>批量beam初版缩进错误已修复并保留日志；数值检查不代表策略有效。</li><li>延迟目标首次BF16烟测因autocast复用no-grad采样权重而断梯度；改为有梯度采样后detach，batch256三次更新通过。烟测未形成正式权重。</li><li>新四小时索引首用仿真环境缺pyarrow，改用已装pyarrow的原生数据环境，无新增依赖。</li><li>R02策略因固定验证动作MSE恶化至26.48而提前停止；负历史回报×连续log密度存在鼓励降低历史动作密度的病态方向。原检查点和独立失败评价保留。</li><li>完整按作者目标对齐的基底、创新训练、机制消融、新冻结主结果及最终Harness研究结论仍未完成。</li><li>没有声称SOTA、顶会原创性或临床安全；尚未接入Core和用户ICML预测器。</li></ul>']
 out+=['<details><summary>检查证据</summary><ul>'+''.join('<li>'+link(p)+'</li>' for p in sorted((ROOT/'checks').glob('*.json')))+'</ul></details><p class="small">HTML已生成；完整浏览器视觉渲染未验证。所有临床、工程与研究结论分别记录。</p></body></html>']
 (ROOT/'训练审核报告.html').write_text('\n'.join(out));print('report written')

if __name__=='__main__':main()
