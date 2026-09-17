"""Matched main comparisons, honest scope and frozen acceptance evidence."""
import json,csv,statistics,html
from pathlib import Path
import numpy as np
from paper_comparison import write_paper_tex
ROOT=Path(__file__).resolve().parent
ORDER=['BC','ReBRAC_BC100','FQL_alpha100','Ours_bound040','nominal','Anchor_FQL','Bound040_noRL']
LABEL={'nominal':'固定基础率参考（非学习）','BC':'BC（行为克隆，监督模仿）','ReBRAC_BC100':'ReBRAC（2023，β=100）','FQL_alpha100':'FQL（2025，α=100）','Anchor_FQL':'固定参考 + FQL','Bound040_noRL':'O07去RL（同结构行为克隆）','Ours_bound040':'O07：固定参考有界残差 ReBRAC（我方）'}
def read(p):return json.loads(p.read_text()) if p.exists() else None
def status_text():
 a=read(ROOT/'results/T01_frozen_sealed_comparison/assessment.json')
 if a:return '封存研究评价通过；可作为Harness研究接入候选，尚未完成产品准入。' if a['all_passed'] else '封存研究评价未通过；当前模型不能作为已验收RL组件。'
 return '完整开发评价通过，冻结后的独立封存评价进行中；尚未最终验收。'
def conclusion():
 a=read(ROOT/'results/T01_frozen_sealed_comparison/assessment.json')
 scope='本轮只能判断公开成人T1D模拟器中的3天基础输注任务。共同餐时bolus按公开CR计算，初始6小时使用nominal基础率；因此固定历史参考有较强先验优势。单训练seed、6个封存模拟患者、没有运动生理机制，均限制外推。真实Loop封存数据未用于本轮策略疗效证明。封存adult009的TBR70/TBR54分别9.20%/3.13%，个体低血糖问题仍明显，不能由总体通过推导患者安全。'
 if not a:return status_text()+scope
 if a['all_passed']:
  return '结论：该权重通过预先规定的单seed独立仿真研究门槛，可进入Harness真实RL研究接入阶段。该版本属于ReBRAC的任务适配和增量结构改进，不满足已证明非incremental算法创新或SOTA的结论。封存TIR94.97%，较ReBRAC高25.00pp、FQL高39.88pp、同结构去RL高21.78pp；较固定参考高1.13pp，但TBR70为2.05%，高于参考1.87%及ReBRAC0.96%，不宣称全面领先或算法普遍优越。'+scope+' 独立研究JSON推理已验证，不等于Core已注册；产品仍需受控执行器、输入来源和适用性合同、双模型有效证据及发布闸门验收，不能直接给真实患者使用。'
 failed=[k for k,v in a['checks'].items() if not v['passed']]
 return '结论：当前权重没有通过封存研究门槛，失败项目为 '+', '.join(failed)+'。开发集较好不能替代未见患者表现；保持封存结果，不据此继续调参后冒充一次独立测试。'+scope

def render():
 out=['<h2 id="main-comparison">主比较：单训练seed、相同训练迭代数与评价情景</h2><p>所有学习模型均seed260915、50,000次训练迭代（ReBRAC每2次迭代更新actor一次，即25,000次actor/50,000次critic更新；BC每次更新actor）；使用全部225名Loop训练患者，未增加模拟器训练。每一项 ± 是<b>患者间标准差</b>，不是训练seed标准差。TIR越高越好，TBR与risk越低越好。原始学习基线输入72×22；参考组额外保留最初可见6小时的固定基础率参考；有界组输出参考±0.4 U/h。用同结构去RL对照分离RL价值目标的贡献，不能把输入与动作范围不同的整组优势全归于算法。<b>“同预算”指最终比较权重各自的50k训练迭代，非等量优化器调用或等算力，不是累计搜索预算相同：</b>按用户要求，我方模型进行了更多结构/损失/范围探索，外部baseline只保留单seed可比结果；因此这是本任务系统配置对比，不是等额超参数搜索下的算法排名。</p>']
 for label,stage,stem in [('完整开发集（4患者；用于选择）','D23_full_development_comparison','完整开发'),('封存集（6患者；冻结后一次性评价）','T01_frozen_sealed_comparison','封存主比较')]:
  d=ROOT/'results'/stage
  if not (d/'completion.json').exists():
   rs=read(d/'summary.json') or [];out.append('<h3>'+label+'</h3><p>尚未完成，已回收'+str(len(rs))+'条；不展示不完整排名。</p>');continue
  rs=read(d/'summary.json');completion=read(d/'completion.json');assert len(rs)==completion['jobs'];pids=sorted({r['patient'] for r in rs});counts={m:len([r for r in rs if r['method']==m]) for m in ORDER};assert len(set(counts.values()))==1
  fields=['tir_pct','tbr70_pct','tbr54_pct','tar180_pct','tar250_pct','mean_risk','mean_bg_mg_dl','basal_u_per_day'];paper=[];display=[]
  for m in ORDER:
   rows=[r for r in rs if r['method']==m];entry={'method':m,'training_seed':0 if m=='nominal' else 260915,'training_updates':0 if m=='nominal' else 50000,'input_group':'published nominal' if m=='nominal' else 'history + fixed reference' if m in ['Anchor_FQL','Bound040_noRL','Ours_bound040'] else 'history only','output_range':'reference +/-0.4 then [0,20]' if m in ['Bound040_noRL','Ours_bound040'] else '[0,20]','patients':len(pids),'trajectories':len(rows),'failures':sum(r['failed'] for r in rows)}
   for k in fields:
    v=[statistics.mean(r[k] for r in rows if r['patient']==p) for p in pids];entry[k+'_mean']=statistics.mean(v);entry[k+'_patient_sd']=statistics.stdev(v)
   group={'BC':'A. 学习策略对比','nominal':'B. 非学习参考','Anchor_FQL':'C. 结构对照与消融'}.get(m)
   if group:display.append('<tr><td colspan="8"><b>'+group+'</b></td></tr>')
   paper.append(entry);display.append('<tr>'+''.join('<td>'+str(x)+'</td>' for x in [LABEL[m],entry['input_group'],entry['output_range'],*[('%.2f ± %.2f'%(entry[k+'_mean'],entry[k+'_patient_sd'])) for k in ['tir_pct','tbr70_pct','tbr54_pct','mean_risk']],str(entry['failures'])+'/'+str(len(rows))])+'</tr>')
  path=ROOT/('论文对比表_'+stem+'.csv')
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paper[0]));w.writeheader();w.writerows(paper)
  write_paper_tex(paper,ROOT/('论文对比表_'+stem+'.tex'),stage.startswith('T'))
  out+=['<h3>'+label+'</h3><div class="table"><table><tr>'+''.join('<th>'+x+'</th>' for x in ['方法','输入组','动作范围 U/h','TIR% ↑','TBR70% ↓','TBR54% ↓','risk ↓','终止'])+'</tr>'+''.join(display)+'</table></div><p><a href="'+path.name+'">完整指标CSV</a> · <a href="论文对比表_'+stem+'.tex">论文LaTeX表</a> · <a href="results/'+stage+'/provenance.json">权重与冻结来源</a> · <a href="results/'+stage+'/summary.json">全部情景指标</a></p>']
  differences=[];allpairs={}
  for ref in ['nominal','BC','ReBRAC_BC100','FQL_alpha100','Anchor_FQL','Bound040_noRL']:
   values=np.array([statistics.mean(r['tir_pct'] for r in rs if r['patient']==p and r['method']=='Ours_bound040')-statistics.mean(r['tir_pct'] for r in rs if r['patient']==p and r['method']==ref) for p in pids]);ci=np.quantile(np.random.default_rng(9981).choice(values,size=(10000,len(pids))).mean(1),[.025,.975]);allpairs[ref]={'mean_pp':float(values.mean()),'ci95_pp':ci.tolist(),'patient_differences_pp':values.tolist()};differences.append('<tr><td>'+LABEL[ref]+'</td><td>%+.2f</td><td>[%+.2f, %+.2f]</td></tr>'%(values.mean(),*ci))
  (d/'paired_comparisons.json').write_text(json.dumps({'method':'paired patient cluster bootstrap10000 seed9981; exploratory intervals, no multiplicity correction','comparisons':allpairs},indent=2))
  out+=['<h4>我方相对各对照的TIR差值（百分点）</h4><table><tr><th>对照</th><th>平均差值</th><th>配对患者bootstrap95%区间</th></tr>'+''.join(differences)+'</table><p>先在患者内平均9个共同情景，再按患者重采样；患者数少，区间仅描述本模拟面板，不代表真实人群或随机初始化稳健性。多对照区间未校正多重比较；只有相对BC区间属于预定验收项。</p>']
 if (ROOT/'figures/sealed_patient_comparison.png').exists():out+=['<h3>封存患者逐人结果</h3><img src="figures/sealed_patient_comparison.png" style="width:100%;height:auto" alt="每位封存患者在全部方法上的TIR"><details><summary>全部6名封存患者的连续轨迹（固定首情景seed，bolus系数1.0）</summary><p>未按效果挑选患者或情景；所有9情景均计入主表。阴影为共同预热，叉号为实际终止；不平滑曲线。</p><img src="figures/sealed_trajectories.png" style="width:100%;height:auto" alt="6名封存患者BG和实际基础输注"></details>']
 a=read(ROOT/'results/T01_frozen_sealed_comparison/assessment.json')
 if a:
  raw=read(ROOT/'results/T01_frozen_sealed_comparison/summary.json');patient_rows=[]
  for pid in sorted({r['patient'] for r in raw}):
   rr=[r for r in raw if r['patient']==pid and r['method']=='Ours_bound040'];patient_rows.append('<tr><td>adult%03d</td>'%pid+''.join('<td>%.2f</td>'%statistics.mean(r[k] for r in rr) for k in ['tir_pct','tbr70_pct','tbr54_pct'])+'</tr>')
  out+=['<h3>我方封存患者逐人风险（每人9情景均值）</h3><table><tr><th>模拟患者</th><th>TIR%</th><th>TBR70%</th><th>TBR54%</th></tr>'+''.join(patient_rows)+'</table><p>验收是本协议的总体研究门槛，不是逐患者临床安全标准。平均TIR较高仍可能伴随某位患者较多低血糖，不能据总均值放行真实患者。</p>']
  out+=['<h3>封存验收逐项结果</h3><table><tr><th>条件</th><th>实测</th><th>门槛</th><th>结果</th></tr>'+''.join('<tr><td>'+html.escape(k)+'</td><td>%.4f</td><td>%s</td><td>%s</td></tr>'%(v['observed'],v['threshold'],'通过' if v['passed'] else '未通过') for k,v in a['checks'].items())+'</table><p>性能容差在首次闭环结果前冻结；用户后续只修改训练seed数为1。<a href="simulation_contract_v3_single_seed.json">完整合同</a> · <a href="configs/T01_frozen_sealed_comparison.json">封存前权重/源码冻结计划</a> · <a href="results/T01_frozen_sealed_comparison/assessment.json">原始判定</a></p>']
 bundle=ROOT/'artifacts/O07_research_candidate'
 if bundle.exists():out+=['<h3>权重与研究推理交付</h3><p><a href="artifacts/O07_research_candidate/STATUS.md">最新研究状态</a> · <a href="artifacts/O07_research_candidate/README.md">使用说明与边界</a> · <a href="artifacts/O07_research_candidate.zip">下载完整推理包ZIP</a> · <a href="artifacts/O07_research_candidate/manifest.json">文件SHA清单</a> · <a href="artifacts/O07_research_candidate/policy.npz">CPU推理权重</a> · <a href="results/research_inference_checks.json">实际评价轨迹的独立进程一致性检查</a></p><p>固定权重、训练归一化和参考来源显式绑定。不同患者、归一化错误、未来/过旧参考、无效历史、权重篡改会拒绝。SHA清单不是外部签名，研究CLI不是Core已注册执行器；产品的来源信任和唯一发布仍须单独接入与验收。</p>']
 out+=['<h3>本轮结论与适用边界</h3><p>'+conclusion()+'</p>']
 return ''.join(out)
