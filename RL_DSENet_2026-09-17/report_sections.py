"""Evidence-driven additions to the offline report; no hand-written scores."""
import csv
import html
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(path):
    file=ROOT/path
    return json.loads(file.read_text()) if file.exists() else None

def f(v):return '—' if v is None else '%.2f'%v

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(('<th>' if i==0 else '<td>')+html.escape(str(v))+('</th>' if i==0 else '</td>') for i,v in enumerate(row))+'</tr>' for row in rows)+'</tbody></table></div>'

def sections():
    selection=read('checks/patch_selection.json');freeze=read('configs/final_freeze.json');final=read('paper_tables/final_control_analysis.json');heldout=read('results/F00_forecast_heldout/evaluation.json')
    patch='';evaluation='';lead=''
    if selection:
        rows=[]
        for r in selection['results']:
            p=r['patch'];rows.append([r['name']+(' ★' if r['name']==selection['selected'] else ''),str(p['m_patch_len'])+'/'+str(p['m_stride']),str(p['patch_len'])+'/'+str(p['stride']),r['parameters'],r['selected_epoch'],f(r['selection_score'])]+[f(r['full_validation'][str(h)]['rmse_mg_dl']) for h in (30,60,120,240)])
        patch='<h2>四个 patch 超参数 · 完整初筛</h2><p>每格单位为5分钟采样点。五个新增候选与 D01 均使用 seed260915、相同完整起点顺序、3 epoch、相同优化器与目标。★仅由预先规定的固定验证子集分数选择，未用留出患者或最终控制场景选参数。</p>'+table(['配置','Global P/S','Local P/S','参数量','选中轮','选择分数↓','30min','60min','120min','240min'],rows)+'<p>选择分数为每位验证患者固定至多128个起点、四时长患者等权RMSE的均值；右侧四列为选中轮的全量验证RMSE，因此二者均值不必相等。P03是预定子集分数最优；D01的全量四时长均值仍略低约0.02mg/dL，本轮没有事后改选模规则，不能把patch筛选宣传为确定的性能提升。短预算初筛不证明全局最优或充分收敛。与 P01 相比，P02/P03/P04/P05 分别只变一个值；stride 同时改变 token 数、Fusion 容量和局部物理感受野，不能称等容量消融。P05最后局部patch完全由复制padding构成，按原结构保留。</p><p><a href="checks/patch_selection.json">选择记录、各时长低糖误差与训练时间</a> · <a href="checks/patch_grid_cuda.json">五组真实CUDA检查</a></p>'
        with (ROOT/'paper_tables/patch_search.csv').open('w') as handle:
            w=csv.writer(handle);w.writerow(['Configuration','Global P/S','Local P/S','Parameters','Selected epoch','Selection score','Full RMSE30','Full RMSE60','Full RMSE120','Full RMSE240']);w.writerows(rows)
    if heldout:
        rows=[]
        for name in ('persistence','trend','dsenet'):
            rows.append([{'persistence':'Persistence','trend':'Linear trend','dsenet':'DSENet-Loop (ours)'}[name]]+[f(heldout['summary_patient_equal'][name][str(h)]['rmse_mg_dl']) for h in (30,60,120,240)])
        evaluation+='<h2>留出患者预测 · 一次性评价</h2><p>模型及完整依赖冻结后才打开 Loop sealed_test：'+str(len(heldout['patients']))+'位患者、'+format(sum(p['evaluated_origins'] for p in heldout['patients']),',')+'个原始起点，与训练/验证患者互斥。没有重拟合归一化、没有选择新checkpoint。患者等权 RMSE，mg/dL。</p>'+table(['方法','30min↓','60min↓','120min↓','240min↓'],rows)+'<p><a href="results/F00_forecast_heldout/evaluation.json">逐患者留出预测指标及低糖分层</a> · <a href="paper_tables/forecast_heldout.tex">论文 LaTeX 表</a></p>'
        intervals=read('paper_tables/forecast_heldout_intervals.json')
        if intervals:
            evaluation+='<p>各时长有有效目标的患者数及与Persistence的配对误差差值如下；没有该时长目标的患者不补造误差。</p>'+table(['时长','有效患者','Δ RMSE','95%患者区间'],[[str(a['horizon_min'])+' min',a['patients'],f(a['delta_rmse_mg_dl']),'['+f(a['ci95'][0])+', '+f(a['ci95'][1])+']'] for a in intervals['results']])
        lowrows=[[name]+[f(heldout['summary_patient_equal'][name][str(h)]['low_rmse_mg_dl']) for h in (30,60,120,240)] for name in ('persistence','dsenet')]
        evaluation+='<details open><summary>留出患者的真实未来低糖子集</summary>'+table(['方法','30min','60min','120min','240min'],lowrows)+'<p>未来CGM&lt;70 mg/dL的患者等权RMSE。120分钟DSENet略高约0.01mg/dL，不能说所有低糖分层均改善；尚无概率校准或低糖事件识别结论。</p></details>'
        with (ROOT/'paper_tables/forecast_heldout.csv').open('w') as handle:
            w=csv.writer(handle);w.writerow(['Method','RMSE30','RMSE60','RMSE120','RMSE240']);w.writerows(rows)
        latex=['\\begin{tabular}{lrrrr}','\\toprule','Method & 30 min & 60 min & 120 min & 240 min \\\\','\\midrule']+[' & '.join(row)+' \\\\' for row in rows]+['\\bottomrule','\\end{tabular}','% Patient-equal RMSE (mg/dL), 32 held-out Loop patients; one training seed.']
        body='\n'.join(latex)+'\n'
        (ROOT/'paper_tables/forecast_heldout.tex').write_text('\\begin{table}[t]\n\\centering\n\\small\n\\caption{Patient-equal glucose prediction RMSE (mg/dL) on held-out Loop patients. Lower is better.}\n\\label{tab:loop-forecast}\n'+body+'\\par\n\\footnotesize Single training seed 260915. The effective patient counts at 30/60/120/240 minutes are 32/32/31/31; unavailable targets are not imputed.\n\\end{table}\n')
    if final:
        rows=[]
        for name,item in final['methods'].items():
            a=item['aggregate'];g=a['bg'];tir=f(g['tir_lower_bound_pct'])
            if g['tir_upper_bound_pct']-g['tir_lower_bound_pct']>1e-6:tir+='–'+f(g['tir_upper_bound_pct'])
            rows.append([item['label'],tir,f(g['tbr70_pct']),f(g['tbr54_pct']),str(a['failures'])+'/'+str(a['episodes']),f(g['coverage_pct'])])
        evaluation+='<h2>论文主表 · 冻结的新场景控制比较</h2><p>10位已曝光虚拟成人 × 2个新场景seed × 3种bolus条件，每方法60条3天轨迹（前6小时预热）。只控制基础率；共享餐时bolus辅助。seed260915是唯一训练seed，场景seed不是训练重复。所有原始指标重新计算，并核对方法间场景完全相同。</p>'+table(['方法','BG TIR / bounds %↑','TBR70 %↓','TBR54 %↓','终止/总数','覆盖 %'],rows)+'<p><strong>这是一张系统比较表。</strong>已有BC/ReBRAC/FQL等未重训，训练信息、动作范围与推理预算并非全部一致；新方法使用额外配对仿真监督和固定观测参考，不能把整张表的差值全部归因于DSENet或RL算法。RL-DITR两行是本项目方法适配，不是原论文临床成绩。只有最后三行共享同一DSENet世界模型、候选池及奖励，适合检验策略作用。</p><p class="note">TIR范围是提前终止后的未知结局上下界，不是置信区间。TBR仅覆盖观测阶段；低TBR不能替提前终止的模型证明安全。主要比较为预先选定的仅策略模型，最终结果不再用于调参。</p><p><a href="paper_tables/control_final.tex">论文主表 LaTeX</a> · <a href="paper_tables/control_final.csv">完整指标 CSV</a> · <a href="paper_tables/final_control_analysis.json">CGM、逐患者指标、风险与配对区间</a> · <a href="configs/final_freeze.json">运行前冻结的权重、代码和场景</a></p>'
        texfile=ROOT/'paper_tables/control_final.tex'
        if texfile.exists() and not texfile.read_text().startswith('\\begin{table*}'):
            body=texfile.read_text()
            texfile.write_text('\\begin{table*}[t]\n\\centering\n\\small\n\\caption{Frozen new-scenario basal-control evaluation on 10 previously exposed virtual adults. Each method has 60 trajectories: two scenario seeds and three shared meal-bolus conditions.}\n\\label{tab:control-main}\n'+body+'\\par\n\\vspace{3pt}\n\\parbox{\\textwidth}{\\footnotesize One training seed (260915); all prior baseline weights are reused without retraining. TIR ranges bound unknown outcomes after early termination, not confidence intervals; TBR uses observed intervals only. Training information, action support and planning budgets differ across systems. The final three rows share the same DSENet world model, seven candidate plans and reward. Common nominal-basal warmup lasts six hours; evaluation lasts the remaining 66 hours. These are new scenarios on known virtual patients, not unseen-patient or clinical results.}\n\\end{table*}\n')
        comparisons=[]
        for name in ('F01_hold','F06_Ours_bound040','F09_H02','F10_planner'):
            r=final['paired_comparisons'][name]
            def ci(key):
                a=r[key]
                return '—' if a is None else f(a['difference'])+' ['+f(a['ci95'][0])+', '+f(a['ci95'][1])+']'
            comparisons.append([final['methods'][name]['label'],ci('tir_lower_bound_pct'),ci('tbr70_pct'),ci('tbr54_pct')])
        evaluation+='<h2>优势有多确定，代价是什么</h2><p>下表是“仅策略 − 对照”的百分点差及95%患者配对bootstrap区间（10,000次）。先在每位患者内平均六场景，再重采样10位患者；不能把60轨迹当60个独立患者。单训练seed的区间不覆盖训练随机性，也未作多重比较校正，应视为配对描述性证据。TBR差值为正意味着低糖更多。</p>'+table(['对照','Δ TIR及95%区间','Δ TBR70及95%区间','Δ TBR54及95%区间'],comparisons)
        a=final['methods']['F11_actor']['aggregate'];g=a['bg'];hold=final['methods']['F01_hold']['aggregate']['bg'];planner=final['methods']['F10_planner']['aggregate']['bg']
        lead='<p class="verdict"><strong>最终仅策略：</strong>BG TIR '+f(g['tir_lower_bound_pct'])+'%，TBR70 '+f(g['tbr70_pct'])+'%，TBR54 '+f(g['tbr54_pct'])+'%，终止 '+str(a['failures'])+'/'+str(a['episodes'])+'。相对固定基础率，TIR '+('%+.2f'%(g['tir_lower_bound_pct']-hold['tir_lower_bound_pct']))+' 个百分点，TBR70 '+('%+.2f'%(g['tbr70_pct']-hold['tbr70_pct']))+' 个百分点；相对同后端无RL规划，TIR '+('%+.2f'%(g['tir_lower_bound_pct']-planner['tir_lower_bound_pct']))+' 个百分点。是否有统计支持须同时看患者配对区间。</p>'
        latency=read('checks/inference_latency.json')
        if latency:
            evaluation+='<h2>模型规模与推理开销</h2>'+table(['模式','batch','中位数ms','p95 ms'],[[r['mode'],r['batch_size'],f(r['median_ms']),f(r['p95_ms'])] for r in latency['rows']])+'<p>RTX4090，Torch2.0.0+cu118，4个CPU线程；其余CUDA任务退出后测量，每组20次预热、100次计时。包含历史编码、预测、参考/候选选择及输出拷回CPU，不含JSON传输和仿真；这是一次顺序基准，不是跨硬件速度保证。</p><p>DSENet '+format(latency['parameters']['dsenet'],',')+'参数；冻结H02历史模块（存储包含未使用旧头）'+format(latency['parameters']['history_encoder_including_unused_legacy_heads'],',')+'；参考适配器 '+format(latency['parameters']['reference_adapter'],',')+'；响应核 '+format(latency['parameters']['response'],',')+'；策略 '+format(latency['parameters']['policy'],',')+'。不能把316,782写成整套模型总参数量。</p>'
        risks=read('paper_tables/control_risk_audit.json')
        if risks:
            risk=risks['methods']['F11_actor'];worst=max(risk['patients'],key=lambda p:p['tbr70_pct'] if p['tbr70_pct'] is not None else -1)
            lead+='<p class="verdict"><strong>平均值掩盖的风险：</strong>adult'+format(worst['patient'],'03d')+' 的TBR70为 '+f(worst['tbr70_pct'])+'%，TBR54为 '+f(worst['tbr54_pct'])+'%；该患者有 '+str(worst['prolonged_under54_120min_events'])+' 次连续至少120分钟低于54 mg/dL的事件。模型不能据平均TIR被称为安全可靠的控制器。</p>'
            evaluation+='<h2>逐患者风险 · 不能被平均值覆盖</h2>'+table(['虚拟成人','TIR %','TBR70 %','TBR54 %','低糖事件最长累计min','持续<54至少120min事件'],[[format(p['patient'],'03d'),f(p['tir_lower_bound_pct']),f(p['tbr70_pct']),f(p['tbr54_pct']),p['max_hypo_event_low_minutes'],p['prolonged_under54_120min_events']] for p in risk['patients']])+'<p>低于70持续15分钟开始一个事件，恢复15分钟结束；事件内累计低糖分钟不等于完全连续低糖。最后一列单独计连续低于54至少120分钟的事件。各方法与终点未恢复事件见<a href="paper_tables/control_risk_audit.json">完整风险审计</a>。</p><p>后续改进应先增加有监督的减量/暂停候选和低糖恢复轨迹，验证动作效应支持范围，再重训策略并使用新的冻结场景。不能直接在这个最终面板上反复改模型，再宣称一次性验证。</p>'
        evaluation+='<p><strong>本轮可以支持的结论：</strong>相对固定观测基础率，RL的TIR改善伴随低糖增加，是收益与风险的交换；不能称全指标胜出。旧有界RL的TIR为'+f(final['methods']['F06_Ours_bound040']['aggregate']['bg']['tir_lower_bound_pct'])+'%，新模型点估计更高，但配对差值区间包含零，不能声称显著优于这一最强旧参考。相对同后端无RL规划，RL降低了TBR70，患者配对区间支持这一方向；TIR差值区间包含零，不能宣称TIR显著优于规划器。这比只报告最好的TIR更准确地说明了RL的作用。</p>'
        evaluation+='<p>当前证据适用于公开仿真的既定基础输注任务。固定参考来自预热阶段的实际输注，但预热使用了模拟器标称基础率；这是重要的控制先验。没有检验持续基础需求漂移、餐时bolus自主决策、真实患者治疗效果或临床安全。预测误差改善不能直接证明RL优势，已曝光患者上的新场景也不代表患者外推。</p>'
    return {'patch':patch,'evaluation':evaluation,'lead':lead,'completed':final is not None and heldout is not None}
