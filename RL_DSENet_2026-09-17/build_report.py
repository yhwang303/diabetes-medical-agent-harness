"""Generate an offline HTML evidence report and traceable paper tables."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from report_sections import sections

ROOT=Path(__file__).resolve().parent

def read(path):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else None

def fmt(v):
    return '未取得' if v is None else f'{v:.2f}'

def main():
    selection=read('checks/patch_selection.json')
    selected=selection['selected'] if selection else 'D01_forecast'
    forecast=read('results/'+selected+'/validation_full.json')
    world=read('results/D05_selected_world/development_diagnostics.json') or read('results/D02_reference_response/development_diagnostics.json')
    checks=read('checks/forecast_cuda.json')
    coverage=read('forecast_masks/manifest.json')
    tables=ROOT/'paper_tables';tables.mkdir(exist_ok=True)
    rows=[];comparison=[]
    if forecast:
        summary=forecast['summary_patient_equal']
        for method in ('persistence','trend','dsenet'):
            rows.append([{'persistence':'Persistence','trend':'Linear trend (30-min history)','dsenet':'DSENet–Loop (ours, seed 260915)'}[method]]+[summary[method][str(h)]['rmse_mg_dl'] for h in (30,60,120,240)])
        for horizon in (30,60,120,240):
            delta=np.array([r['methods']['dsenet'][str(horizon)]['rmse_mg_dl']-r['methods']['persistence'][str(horizon)]['rmse_mg_dl'] for r in forecast['patients'] if r['methods']['dsenet'][str(horizon)]['rmse_mg_dl'] is not None])
            rng=np.random.default_rng(260917);bootstrap=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1);lo,hi=np.percentile(bootstrap,[2.5,97.5])
            comparison.append({'horizon_min':horizon,'patient_count':len(delta),'delta_rmse_mg_dl':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
        with (tables/'forecast_rmse.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['Method','30min RMSE (mg/dL)','60min RMSE (mg/dL)','120min RMSE (mg/dL)','240min RMSE (mg/dL)']);writer.writerows(rows)
        latex='\\begin{tabular}{lrrrr}\n\\toprule\nMethod & 30 min & 60 min & 120 min & 240 min \\\\\n\\midrule\n'
        for row in rows:latex+=row[0].replace('–','--')+' & '+' & '.join(fmt(v) for v in row[1:])+' \\\\\n'
        latex+='\\bottomrule\n\\end{tabular}\n% RMSE in mg/dL, patient-equal on 55 Loop validation patients.\n% Single training seed; development validation, not sealed test or RL control.\n'
        (tables/'forecast_rmse.tex').write_text(latex)
        (tables/'paired_patient_intervals.json').write_text(json.dumps(comparison,indent=2))
    table=''.join('<tr><th>'+html.escape(r[0])+'</th>'+''.join('<td>'+fmt(v)+'</td>' for v in r[1:])+'</tr>' for r in rows)
    intervals=''.join(f'<tr><th>{r["horizon_min"]} min</th><td>{r["delta_rmse_mg_dl"]:.2f}</td><td>[{r["ci95_low"]:.2f}, {r["ci95_high"]:.2f}]</td></tr>' for r in comparison)
    low=''
    if forecast:
        for method in ('persistence','dsenet'):
            low+='<tr><th>'+method+'</th>'+''.join('<td>'+fmt(forecast['summary_patient_equal'][method][str(h)]['low_rmse_mg_dl'])+'</td>' for h in (30,60,120,240))+'</tr>'
    control_rows=[]
    for name,label in [('E01_hold_dev','Fixed last observed basal'),('E02_planner_dev','D03 drifting no-RL planner'),('E03_actor_dev','D03 drifting RL policy'),('E04_beam_dev','D03 drifting RL + planning'),('E05_bounded_planner_dev','D04 fixed-reference planner'),('E06_bounded_actor_dev','D04 fixed-reference RL'),('E07_bounded_beam_dev','D04 fixed-reference RL + planning'),('E08_selected_planner_dev','Selected patch: no-RL planner'),('E09_selected_actor_dev','Selected patch: RL policy'),('E10_selected_beam_dev','Selected patch: RL + planning')]:
        result=read('results/'+name+'/summary.json')
        if not result:continue
        episodes=result['episodes']
        def average(key):
            patients=sorted(set(e['patient'] for e in episodes))
            values=[e['metrics']['bg'].get(key) for e in episodes]
            return float(np.mean([np.mean([e['metrics']['bg'][key] for e in episodes if e['patient']==pid]) for pid in patients])) if all(v is not None for v in values) else None
        control_rows.append([label,average('tir_lower_bound_pct'),average('tir_upper_bound_pct'),average('tbr70_pct'),average('tbr54_pct'),sum(e['metrics']['failed'] for e in episodes),len(episodes),average('coverage_pct')])
    control_body=''.join('<tr><th>'+html.escape(r[0])+'</th><td>'+fmt(r[1])+('–'+fmt(r[2]) if abs(r[1]-r[2])>.001 else '')+'</td><td>'+fmt(r[3])+'</td><td>'+fmt(r[4])+'</td><td>'+str(r[5])+'/'+str(r[6])+'</td><td>'+fmt(r[7])+'</td></tr>' for r in control_rows)
    with (tables/'control_development.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['Method','BG TIR lower %','BG TIR upper %','Observed TBR70 %','Observed TBR54 %','Failures','Episodes','Coverage %']);writer.writerows(control_rows)
    control_html='<div class="scroll"><table><thead><tr><th>Method</th><th>BG TIR / bounds %↑</th><th>TBR70 %↓</th><th>TBR54 %↓</th><th>提前终止</th><th>覆盖率 %</th></tr></thead><tbody>'+ (control_body or '<tr><th colspan="6">闭环评价尚未完成</th></tr>')+'</tbody></table></div>'
    world_html='<p>参考模型与动作响应尚未完成验证；没有可发布的策略控制成绩。</p>'
    if world:
        world_rows=''
        for key,r in world['results'].items():
            world_rows+='<tr><th>'+html.escape(key)+'</th>'+''.join('<td>'+fmt(r[k])+'</td>' for k in ('effect_path_mae_mg_dl','zero_response_path_mae_mg_dl','candidate_regret','hold_reference_regret','reference_rmse_mg_dl'))+'</tr>'
        world_html=f'<p class="verdict">决策开发门槛：{"通过" if world["decision_gate_passed"] else "未通过"}。通过也不等于闭环控制优势。</p><div class="scroll"><table><thead><tr><th>已曝光探针</th><th>效应 MAE↓</th><th>零响应 MAE↓</th><th>候选 regret↓</th><th>保持参考 regret↓</th><th>参考轨迹 RMSE↓</th></tr></thead><tbody>{world_rows}</tbody></table></div><p>误差单位 mg/dL；regret 为同一候选池在原 status 折扣回报下的损失。截断候选池不补造缺失回报，不参与完整池排序。此表没有患者外推或医学安全含义。</p>'
    status='预测训练已完成；强化学习控制优势尚未建立' if forecast else 'DSENet 预测训练进行中；尚无最终控制结论'
    label_counts=''
    if coverage:
        label_counts=''.join('<tr><th>'+name+'</th><td>'+format(v['origins'],',')+'</td>'+''.join('<td>'+format(v['labels_per_horizon'][h-1],',')+'</td>' for h in (6,12,24,48))+'</tr>' for name,v in coverage.items())
    extra=sections()
    if extra['completed']:status='单seed训练与冻结评测完成；收益和风险按实测报告'
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DSENet × RL-DITR · 训练审核报告</title>
<style>:root{{--paper:#f5f8fb;--ink:#16324f;--body:#26313c;--teal:#157f86;--amber:#b46c00;--line:#cbd9e5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--body);font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1160px;margin:auto;padding:52px 40px 80px}}header{{border-top:8px solid var(--teal);padding-top:24px}}.eyebrow{{color:var(--teal);font:13px/1.5 ui-monospace,monospace;letter-spacing:.08em}}h1{{font:700 clamp(28px,4vw,48px)/1.25 "Songti SC",Georgia,serif;color:var(--ink);margin:16px 0}}h2{{font-size:25px;color:var(--ink);margin:44px 0 14px}}p{{max-width:88ch}}a{{color:var(--teal);text-underline-offset:4px}}a:focus-visible{{outline:3px solid var(--amber)}}.verdict{{padding:14px 20px;border-left:4px solid var(--amber);background:#fff6e8}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin:30px 0;background:var(--line)}}.flow div{{background:#fff;padding:18px}}.flow strong{{display:block;color:var(--ink)}}.flow small{{display:block;color:#556b7b}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:white;font-variant-numeric:tabular-nums;font-size:14px;margin:16px 0}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}thead th{{background:var(--ink);color:white}}tbody th:first-child{{white-space:normal;min-width:180px;max-width:280px}}tbody tr:last-child th,tbody tr:last-child td{{border-bottom:2px solid var(--ink)}}.formula{{padding:24px;background:#e6f1f3;color:var(--ink);font:18px/1.8 ui-monospace,monospace;overflow:auto}}.note{{font-size:14px;color:#556b7b}}details{{border-block:1px solid var(--line);padding:16px 0;margin:16px 0}}summary{{cursor:pointer;font-weight:600}}code{{font-size:.9em}}footer{{margin-top:48px;border-top:1px solid var(--line);padding-top:20px;font-size:14px}}@media(max-width:700px){{main{{padding:28px 18px}}.flow{{grid-template-columns:1fr 1fr}}th,td{{padding:9px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}.scroll{{overflow:visible}}details{{break-inside:avoid}}a{{color:inherit}}}}</style>
<main><header><div class="eyebrow">RESEARCH EVIDENCE · 2026-09-17 · SINGLE SEED 260915</div><h1>DSENet × RL-DITR<br>从血糖预测到动作决策</h1><p class="verdict">{status}。报告只呈现已取得证据；未完成实验不会填成优势。</p><p>保留 DSENet 主框架，兼容完整 Loop 训练池，并用有明确参考动作语义的患者模型连接 RL。当前是研究实验，不是患者给药产品。</p></header>
{extra["lead"]}<div class="flow"><div><strong>事实预测</strong><small>DSENet 双流 · 缺失与尺度核验</small></div><div><strong>干预后果</strong><small>真实参考分支 + 因果响应</small></div><div><strong>动作决策</strong><small>统一轨迹 reward · 候选 regret</small></div><div><strong>闭环控制</strong><small>无 RL / 策略 / 规划同条件比较</small></div></div>
<h2>这次保留与修改了什么</h2><p>长短窗口双流、全局双向门控 Mamba、局部 LoRE、Router 和 Fusion 均保留。初版单 CGM 通道，6小时历史、4小时预测；输注、餐食、运动和缺失记录通过外部历史编码器参与后续动作模型。新 DSENet 的各配置参数量见超参数表，选中配置为 {selected}。</p><p>修复原局部注意力掩码方向、可选 Fusion 分支广播/宽度；用观测掩码计算可逆归一化，避免缺失占位进入统计。先将原 Loop z-score 还原 mmol/L，指标统一乘18转 mg/dL，没有叠加第二套 StandardScaler。完整 CUDA 前后向与两向 Mamba 梯度已验证。</p><p><a href="dsenet/upstream.json">上游提交与原文件哈希</a> · <a href="checks/forecast_cuda.json">真实 CUDA 检查</a> · <a href="实验协议.md">实验协议</a> · <a href="审计落实清单.md">逐项审计落实</a></p>
<h2>全部起点保留，标签用途分开</h2><p>患者划分先于切窗。事实 CGM 标签使用已有 CGM 连续片段，动作响应仍要求完整治疗前缀。这样既不因 food/exercise 缺失删起点，也不把未知动作当作已知。下表是有效标签数量，重叠标签不能当作独立患者或独立实验。</p><div class="scroll"><table><thead><tr><th>划分</th><th>原起点</th><th>30 min</th><th>60 min</th><th>120 min</th><th>240 min</th></tr></thead><tbody>{label_counts}</tbody></table></div>
{extra["patch"]}<h2>预测对比表 · 完整验证集</h2><p>55位验证患者，患者等权 RMSE（mg/dL，越低越好）。三种方法使用相同历史和标签；DSENet仅seed260915。该表可导出论文格式，但必须保留“开发验证集、单seed、血糖预测”的限定，不能改写成RL控制实验。</p><div class="scroll"><table><thead><tr><th>Method</th><th>30 min↓</th><th>60 min↓</th><th>120 min↓</th><th>240 min↓</th></tr></thead><tbody>{table or '<tr><th colspan="5">全量验证尚未完成</th></tr>'}</tbody></table></div><p><a href="paper_tables/forecast_rmse.csv">下载 CSV</a> · <a href="paper_tables/forecast_rmse.tex">下载 LaTeX</a> · <a href="results/{selected}/validation_full.json">逐患者原始指标</a></p>
<details open><summary>患者配对差值与低血糖子集</summary><p>差值为 DSENet减持续值，负数代表误差更小。95%区间对患者进行10,000次配对bootstrap；不包含训练随机性，也不是封存泛化区间。</p><table><thead><tr><th>时长</th><th>Δ RMSE</th><th>95%区间</th></tr></thead><tbody>{intervals}</tbody></table><p>未来真实 CGM&lt;70 mg/dL 的子集 RMSE：</p><div class="scroll"><table><thead><tr><th>方法</th><th>30 min</th><th>60 min</th><th>120 min</th><th>240 min</th></tr></thead><tbody>{low}</tbody></table></div><p class="note">总体误差更小不保证低糖识别或控制更安全；尚未评价概率校准。子集只在有低糖标签的患者中平均，完整计数见JSON。</p></details>
<h2>三段训练，各自在学什么</h2><div class="scroll"><table><thead><tr><th>阶段</th><th>更新参数与监督</th><th>预算</th><th>优化器</th></tr></thead><tbody><tr><th>DSENet事实预测</th><td>全部DSENet参数；观测CGM的masked MSE</td><td>每配置3完整epoch；225人全部起点</td><td>AdamW 3e-4，batch256</td></tr><tr><th>参考与动作响应</th><td>仅适配器和响应核；参考分支MSE + 配对效应MSE</td><td>504组×7分支；100epoch / 3200更新</td><td>AdamW 1e-3，16组/batch</td></tr><tr><th>RL策略</th><td>仅306→128→7策略网络；模型回报期望 − 0.05 KL</td><td>1完整Loop epoch；6580更新；seed260915</td><td>AdamW 3e-4，batch256</td></tr></tbody></table></div><p>三者职责不同：MSE训练预测与动作后果；RL策略在冻结世界模型上优化候选计划的回报。精确枚举7个候选得到期望策略梯度，没有用旧V作尾部回报，也没有继承旧策略。患者模型冻结前后逐张量hash一致。</p><div class="formula">J(π) = Σₐ π(a|h) Q<sub>world</sub>(h,a) − 0.05 KL(π || prior)<br>Q<sub>world</sub> = Σₜ 0.9<sup>t/12</sup> status(Gₜ) / 12</div><p>策略输入为256维历史上下文、48点参考预测、当前实际基础率与持久观测参考。候选为保持参考，或在三个80分钟块之一改变±0.25 U/h；每5分钟只执行首步。prior为保持0.4、其余各0.1。策略加规划版本取策略top3及保持参考后按world回报重排；当前实现先评分7条再筛选，并未实现树搜索或宣称搜索加速。精确有限候选策略优化是本项目对RL-DITR模型式思路的适配，不能描述为原论文训练器的完整复现。点预测status也不等于校准的期望风险。</p><p>开发时发现不断围绕“上一次动作”重定中心会累积漂移并造成严重低糖，因此保存D03负结果，改为固定预热结束观测基础率。这个动作空间修订不是医学安全规则；其收益不能全部归因于RL。</p>
<h2>动作参考必须有真实含义</h2><div class="formula">G(h, a) = DSENet(h) + ReferenceAdapter(h, a_ref)<br>　　　　 + Response(h, a − a_ref)</div><p>ReferenceAdapter由真正执行参考基础率的仿真分支监督。Response由同状态其他动作分支减参考分支的CGM差监督；相同动作响应严格为零，动作后缀不改变早期预测。旧H02只提供冻结历史编码，旧血糖、reward和V头不参与新轨迹效用。</p>{world_html}
{extra["evaluation"]}<h2>同后端闭环控制 · 开发面板与失败记录</h2><p>4位已知虚拟成人×3种bolus条件，每条3天（前6小时预热）。共享餐时bolus辅助，只控制基础率。每个版本内部共享候选池、DSENet世界模型与reward；D03连续重定中心的失败结果、D04持久参考修订、选中patch后的版本分开保留。固定参考只读取预热结束最后实际基础率。</p>{control_html}<p class="note">提前终止时TIR显示未知结局上下界，非置信区间；TBR仅基于已观察阶段，低TBR不证明失败后安全。开发成绩不作为封存证据。</p><p><a href="paper_tables/control_development.csv">下载开发控制表 CSV</a></p>
<h2>可复现记录与结论边界</h2><p>已取回644个证据文件并逐个核对SHA256。H05完成6,580更新；R08日志到4,950步、最后权重4,000步，无完成标记；C02的E43只有启动清单。没有重训已有baseline，也没有为缺失控制结果补分。</p><p>正式论文控制主表按运行前冻结的统一新场景评价，完成结果见上表。已有FQL/ReBRAC使用不同训练监督时，只能注明差异作系统比较。10个虚拟成人已曝光，新场景不能称未见患者。提前终止、低糖、覆盖率和计算时间必须与TIR一起报告。</p><p><a href="recovery/local_verification.json">旧证据核验</a> · <a href="审计正文_参考对话.md">审计正文存档</a></p>
<h2>模型与复现交付</h2><p><a href="delivery/DSENet_RL_single_seed_model.tar.gz">完整研究推理模型包</a> · <a href="delivery/model_bundle_manifest.json">包内依赖与哈希</a> · <a href="checks/model_bundle.json">隔离目录真实CUDA加载验证</a> · <a href="README.md">运行说明</a> · <a href="阶段总结与恢复交接_2026-09-17.md">阶段总结与恢复交接</a> · <a href="paper_tables/results_paragraph_en.md">英文结果段落</a></p><p>模型包包括预测器、world、策略、旧H02历史编码依赖、归一化和源码。已在独立目录加载，输出与原运行目录逐值一致；没有仅凭权重文件存在就称可用。此检查不等于Harness受控执行器已注册。</p><footer>来源：用户提供的昨日记录、指定审计对话、DSENet上游代码及本轮真实运行输出。原baseline、旧实验和Core闸门均保留。没有临床安全或SOTA声明。</footer></main></html>'''
    (ROOT/'训练审核报告.html').write_text(document)
    print(ROOT/'训练审核报告.html')

if __name__=='__main__':main()
