"""Render the checked JSON into HTML and matching manuscript tables."""
import csv,html,json
from pathlib import Path
R=Path(__file__).resolve().parent
G=[('tir_lower_bound_pct','TIR70–180 % ↑'),('tbr70_pct','TBR70 % ↓'),('tbr54_pct','TBR54 % ↓'),('tar180_pct','TAR180 % ↓'),('tar250_pct','TAR250 % ↓'),('sd_mg_dl','SD mg/dL ↓'),('cv_pct','CV % ↓'),('lbgi','LBGI ↓'),('hbgi','HBGI ↓')]
esc=html.escape

def main():
 d=json.loads((R/'analysis/final.json').read_text());ms=d['manifest']['methods'];dev=d['new_vs_old_development'];out=R/'delivery';out.mkdir(exist_ok=True)
 def fmt(a,k):return '%.2f ± %.2f'%(a[k]['mean'],a[k]['sd'])
 def table(panel):
  rows=d['panels'][panel];headers=['Method / publication','Training records']+[n for k,n in G]+['Failures ↓','Coverage % ↑','BG&lt;54 ≥120min events ↓']
  body=[];tex=[]
  complete=[v for v in rows.values() if v['failures']==0 and v['coverage_pct']['mean']==100]
  best={k:(max if k=='tir_lower_bound_pct' else min)(v[k]['mean'] for v in complete) for k,_ in G}
  for m in ms:
   a=rows[m['key']];complete_run=a['failures']==0 and a['coverage_pct']['mean']==100
   cells=[esc(m['label']+' · '+str(m['venue'])+' '+str(m['year'])),esc('warm-up' if m['key']=='hold' else ('L+S1' if m['key']=='ours' and d['manifest']['selected_candidate']=='old' else 'L+S1+S2'))];plain=[]
   for k,_ in G:
    value=fmt(a,k)
    if k=='tir_lower_bound_pct' and not complete_run:value='[%.2f, %.2f]'%(a[k]['mean'],a['tir_upper_bound_pct']['mean'])
    elif not complete_run:value+=' †'
    isbest=complete_run and abs(a[k]['mean']-best[k])<1e-8
    cells.append('<b>'+value+'</b>' if isbest else value);plain.append((value,isbest))
   cells += [str(a['failures'])+'/60',fmt(a,'coverage_pct'),str(a['prolonged54'])+(' †' if not complete_run else '')]
   body.append('<tr class="'+('ours' if m['key']=='ours' else '')+'">'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
   texvals=[]
   for value,b in plain:
    v=value.replace('±',r'$\pm$').replace(' †',r'$^{\dagger}$');texvals.append(r'\textbf{'+v+'}' if b else v)
   tex.append(m['label'].replace('*',r'$^{*}$')+' & '+cells[1]+' & '+' & '.join(texvals)+' & '+str(a['failures'])+'/60 & '+('%.2f'%a['coverage_pct']['mean'])+' & '+str(a['prolonged54'])+(r'$^{\dagger}$' if not complete_run else '')+r' \\')
  caption=('Native requested actions; only common simulator pump constraints.' if panel=='native' else 'Identical CGM-only Patek-style attenuation for every controller; report as control-layer sensitivity, not learned-model improvement.')
  textext='''% Requires booktabs, graphicx. One training seed 260915. Patient-level mean +- SD.
% BG is scoring ground truth; CGM only is observable. 10 known adults, 6 cases each.
% TIR intervals bound missing outcomes, NOT confidence intervals. Dagger: observed-prefix only.
% Training objectives/budgets differ; this is a matched-evaluation system comparison.
\\begin{table*}[t]
\\centering\\scriptsize
\\caption{'''+caption+''' New scenario seeds 92111/92112; 3 days with 6-hour warm-up. Bold denotes numerical best among fully observed methods only; no significance claim.}
\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrrrrrrrr}
\\toprule
Method & Data & TIR$\\uparrow$ & TBR70$\\downarrow$ & TBR54$\\downarrow$ & TAR180$\\downarrow$ & TAR250$\\downarrow$ & SD$\\downarrow$ & CV$\\downarrow$ & LBGI$\\downarrow$ & HBGI$\\downarrow$ & Fail & Coverage & Long54 \\\\
\\midrule
'''+ '\n'.join(tex)+'''
\\bottomrule
\\end{tabular}}
\\par\\smallskip\\raggedright\\footnotesize
TIR is time in 70--180 mg/dL; TIR/TBR/TAR/CV/Coverage are percentages, glucose SD is in mg/dL. Values are patient-level mean $\\pm$ SD (10 patients, six cases each), not multiple-training-seed statistics. TIR intervals bound missing outcomes, not confidence intervals; $^{\\dagger}$ denotes observed-prefix metrics after early termination, which must not be ranked against complete follow-up. Long54 counts continuous BG below 54 mg/dL for at least 120 minutes. L is Loop; S1 is paired simulation; S2 is the fixed 160 PPO training trajectories, never development or confirmation trajectories. RL-DITR$^*$ is a public-data continuous-basal task adaptation. The retained ours uses L+S1; other learned methods also receive S2. Reward learning, architecture and budgets are not identical.
\\end{table*}
'''
  (out/('comparison_'+panel+'.tex')).write_text(textext)
  return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
 sections=[]
 native=d['panels']['native'];o=native['ours'];selected=d['manifest']['selected_candidate'];passed=[k for k,v in dev['joint'].items() if v['passed']]
 native_tir=(fmt(o,'tir_lower_bound_pct') if o['failures']==0 and o['coverage_pct']['mean']==100 else '[%.2f, %.2f]'%(o['tir_lower_bound_pct']['mean'],o['tir_upper_bound_pct']['mean']))
 status=('本轮没有候选达到预先约定的联合标准；原版继续保留为主模型。' if not passed else '开发集通过联合标准的候选：'+', '.join(passed)+'。最终选择和确认结果分别记录，不能用确认集挑选。')
 sections.append('<section class="finding"><h2>结论与版本</h2><p><strong>'+esc(status)+'</strong></p><p>最终确认面板采用 '+esc(selected)+'，TIR '+native_tir+'%，TBR70 '+fmt(o,'tbr70_pct')+'%，TBR54 '+fmt(o,'tbr54_pct')+'%，失败 '+str(o['failures'])+'/60。这是新场景成绩，与历史 97.08 ± 4.50 / 1.55 ± 3.89 / 0.57 ± 1.79 不应混成同一张表。</p><p>历史版本及P03/D05/D06、原代码和轨迹未覆盖。此次未发布替代版到 GitHub。每个数字均由真实轨迹重新计分，原始BG评分器未改。</p></section>')
 failed_external=[m for m in ms if m['key'] not in ('ours','hold') and native[m['key']]['failures']]
 if failed_external:
  failures='；'.join(m['label']+' '+str(native[m['key']]['failures'])+'/60' for m in failed_external)
  sections.append('<section class="finding"><h2>不能省略的基线失稳结果</h2><p>原生输出提前终止：'+esc(failures)+'。共享S补训并不保证每种算法变强；本轮按预先固定的末checkpoint评价，没有拿新旧版本按确认成绩择优。</p><p><b>这是一张透明的同协议系统重评表，尚不能单独作为充分调优后的算法SOTA证据。</b>多项通用离线RL适配失稳、任务动作表示/先验及既往调参预算不同，均限制优势归因。不能以崩溃基线数量多来宣称全面优越。按照用户最新指令，本批跑完后停止，不继续补训练或调参。</p></section>')
 full_external=[m for m in ms if m['key'] not in ('ours','hold') and native[m['key']]['failures']==0 and native[m['key']]['coverage_pct']['mean']==100]
 if full_external:
  strongest=max(full_external,key=lambda m:native[m['key']]['tir_lower_bound_pct']['mean']);a=native[strongest['key']];delta=o['tir_lower_bound_pct']['mean']-a['tir_lower_bound_pct']['mean']
  sections.append('<section><h2>确认面板的主要比较</h2><p>完整轨迹外部方法中，TIR最高的是 '+esc(strongest['label'])+'：'+fmt(a,'tir_lower_bound_pct')+'%；ours为'+fmt(o,'tir_lower_bound_pct')+'%，差值'+('%+.2f'%delta)+'个百分点。该方法TBR70/TBR54为'+fmt(a,'tbr70_pct')+' / '+fmt(a,'tbr54_pct')+'%，ours为'+fmt(o,'tbr70_pct')+' / '+fmt(o,'tbr54_pct')+'%。应同时看达标时间和风险，不能只按TIR宣布全面优势。</p><p>有提前终止的方法未参与这项完整轨迹排名，其结果仍全部保留在主表。数值差异不等于单seed条件下已证明统计或临床优势。</p></section>')
 sections.append('<section><h2>修了什么、没有修好什么</h2><p>DSENet漏报会影响我方世界模型的动作评价与训练信号，但闭环TIR/TBR来自模拟器实际BG，漏报不会直接把这些指标算漂亮。其他外部方法不调用DSENet，因此不是同一个预测器缺陷；本轮另查到的实现顺序差异才是相应重训原因。</p><ul><li>FQL 修正 target Polyak 的参数来源；ReBRAC 恢复旧 actor / 新 critic 的 target 来源及作者更新相位；LOM 修正 delayed target 相位。同 seed、原预算重训，没有拿新旧结果择优。</li><li>七个 model-free 基线补训相同的公开S记录，RL-DITR使用原R03结构做相同来源适配。共享S共392,832 transitions；Loop全1,653,421起点保留。训练数据与本轮确认场景隔离。</li><li>低糖控制尝试：连续趋势减量17/30、真实回报PPO的2/4/8轮固定候选、冻结DSENet主干的W01尾部校准。内部版本只列消融，不冒充外部基线。</li><li>预测漏报仍是实质限制：富集严重低糖的168条原候选轨迹上，W01的&lt;54漏报168→167，不能称已解决。这个条件样本比例不是全体患者漏报率；平均预测误差改善也不是控制收益。</li></ul></section>')
 sections.append('<section><h2>公平性和评测合同</h2><p>两张表所有行使用相同10名已知虚拟成人、3种bolus因子（0.8/1/1.2）、场景种子92111/92112、3天时长、6小时预热、5分钟决策。餐时bolus由公共场景给定；RL只控制基础率，单位U/h。策略输入只有观测CGM、实际输注和已有历史特征，不使用未来餐食、隐藏BG或患者生理参数。</p><p><b>主表取消旧的共同anchor±0.25投影。</b>各方法原生输出仅接受相同模拟泵约束；我们模型本身的候选支持仍为观测anchor附近，属于模型设计，明确保留。ours还显式保留从首个控制前观测得到的persistent anchor；这些前馈外部策略使用近期72×22历史，没有额外拼接这项长期记忆。因此这里没有消除表示/先验差异，不能把结果单独归因于DSENet。第二张表给所有方法同一个brake17，单列安全层贡献。不能将该层收益算成RL或DSENet预测改进。另有一项明确的接口适配：ours被减量至原候选支持外时，先请求原下档anchor−0.25再接受同一brake，恢复支持后再调用actor；不伪造历史、不在支持外使用Q。此re-entry本身也是控制系统组成，不能声称两张表只差一个无副作用函数。</p><p>L指Loop；S1指旧配对仿真训练臂；S2指本次正式PPO的160条训练轨迹，不含开发或确认轨迹。共享训练记录补齐了原L/L+S不对称的一部分；仍不能称“所有训练条件完全相同”：原预训练、模型表示、调参史、优化预算和回报不同。最终保留的D06仍使用DSENet预测CGM上的RL-DITR status有限时域回报；传统RL基线使用观测CGM status/12、γ=.991258，BC仅做模仿。内部PPO尝试使用BG整形回报/γ=.997，但没有晋升，不能误写成正文ours的训练方式。原D06只使用S1，而基线已获得S1+S2。因此正文可称同协议系统比较，不能单凭此表宣称纯算法普遍SOTA。</p><p>一个训练seed260915。±是10名患者的患者均值之间的SD；每名患者先平均6个场景。不是多seed稳定性或95%CI。新场景属于已知患者，不能写成未见患者泛化。</p></section>')
 for panel,title in [('native','正文对比：原生动作'),('brake17','控制层敏感性：所有方法 + 同一brake17')]:
  sections.append('<section id="'+panel+'"><h2>'+title+'</h2>'+table(panel)+'<p><b>↑越大越好；↓越小越好。</b>Mean BG与胰岛素总量不设单向优劣箭头。TIR [下界,上界] 来自提前终止后的未知结局，不是置信区间。†为观测前缀；有失败的低TBR或低风险不能与完整轨迹直接排名。粗体仅表示完整轨迹方法中的数值最优，不代表统计显著。</p><p><a href="comparison_'+panel+'.csv">CSV</a> · <a href="comparison_'+panel+'.tex">LaTeX大表</a></p></section>')
 body=[]
 for key,a in dev['methods'].items():
  gate='历史参考' if key=='old' else ('通过' if dev['joint'][key]['passed'] else '未通过：'+', '.join(k for k,v in dev['joint'][key]['gates'].items() if not v))
  body.append('<tr><td>'+esc(key)+'</td>'+''.join('<td>'+fmt(a,k)+'</td>' for k in ['tir_lower_bound_pct','tbr70_pct','tbr54_pct','tar180_pct','sd_mg_dl','cv_pct'])+'<td>'+esc(gate)+'</td></tr>')
 sections.append('<section><h2>开发消融与失败记录</h2><p>本表为已曝光91701/91702；与上面的确认场景分开。预先约定TIR不降、低糖/高糖/SD/CV不升、至少一项低糖下降、无新增失败。没有为得到“成功”放宽门槛。</p><div class="scroll"><table><tr><th>内部候选</th><th>TIR↑</th><th>TBR70↓</th><th>TBR54↓</th><th>TAR180↓</th><th>SD↓</th><th>CV↓</th><th>联合标准</th></tr>'+''.join(body)+'</table></div><p>W01只训练reference adapter与action-response：双流DSENet/GateMamba/LoRE/Router/Fusion及历史编码器逐值冻结。旧actor的推理不直接用response选动作，换world只改变它接收的reference；不能宣称actor已学会利用新预测。PPO则保持世界模型冻结，用真实仿真回报更新现有actor，没有强迫选择未来减量。</p></section>')
 sections.append('<section><h2>PPO为何变化很小：只读行为诊断</h2><p>从160条正式训练轨迹各固定抽62个状态，共9920个，不使用确认集。相同状态下，PPO02/04/08相比原D06的实际首动作变化分别为315（3.18%）、188（1.90%）、149（1.50%）。未来60min CGM&lt;54关联子集143个状态中，仅1/1/3个状态改变；原D06已经在125/143（87.41%）选择当前候选集合的减量档。</p><p>这支持本轮PPO执行行为改变很小，不能由概率或训练loss变化代替控制改进。该子集由后续已发生结局定义，仅是关联诊断，不证明反事实收益、不证明低糖不可避免。已按用户指令停止扩展实验。<a href="../world_tail/PPO首动作只读诊断.md">完整诊断与抽样合同</a></p></section>')
 source_rows=[]
 for method,values in d['source_order_sensitivity'].items():
  for version,a in values.items():
   source_rows.append('<tr><td>'+esc(method+' / '+version)+'</td>'+''.join('<td>'+fmt(a,k)+'</td>' for k in ['tir_lower_bound_pct','tbr70_pct','tbr54_pct','sd_mg_dl'])+'<td>'+str(a['failures'])+'/60</td></tr>')
 sections.append('<section><h2>实现顺序修复的单独复评</h2><p>仍为旧917开发场景及原共同±0.25 U/h设置；仅检查原预算重训后的影响，不与新确认表混排，也不把这些版本列为额外外部方法。</p><div class="scroll"><table><tr><th>方法/版本</th><th>TIR↑</th><th>TBR70↓</th><th>TBR54↓</th><th>SD↓</th><th>失败</th></tr>'+''.join(source_rows)+'</table></div></section>')
 extra=[]
 for m in ms:
  a=native[m['key']];extra.append('<tr><td>'+esc(m['label'])+'</td>'+''.join('<td>'+fmt(a,k)+'</td>' for k in ['mean_mg_dl','basal_u_per_observed_day','bolus_u_per_observed_day','action_tv_per_observed_day'])+'</tr>')
 sections.append('<section><h2>输注与额外指标</h2><div class="scroll"><table><tr><th>Method</th><th>Mean BG mg/dL</th><th>Basal U/observed day</th><th>Bolus U/observed day</th><th>Action TV (U/h)/observed day</th></tr>'+''.join(extra)+'</table></div><p>用量少不自动更好：必须结合低糖、高糖、实际输注和失败结局。提前终止的用量与波动均为观测时段统计。完整CGM次级指标、逐患者均值、患者级bootstrap区间及原始文件SHA见<a href="../analysis/final.json">完整分析JSON</a>。区间仅为描述性患者bootstrap，未校正多重比较，不据此宣称所有指标显著优越。</p></section>')
 sources=[('Patek 2012 · 连续风险减量','https://pmc.ncbi.nlm.nih.gov/articles/PMC4607512/'),('2015 PLGS · 预测暂停与及时恢复','https://diabetesjournals.org/care/article/38/7/1197/30982/Predictive-Low-Glucose-Insulin-Suspension-Reduces'),('PPO · 原始方法','https://arxiv.org/abs/1707.06347'),('FQL · ICML2025','https://proceedings.mlr.press/v267/park25f.html'),('LOM · ICLR2025','https://proceedings.iclr.cc/paper_files/paper/2025/hash/be62c4a943675195ff5a2a98d5b9724f-Abstract-Conference.html'),('GFP · ICLR2026','https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html'),('RL-DITR · Nature Medicine2023','https://www.nature.com/articles/s41591-023-02552-9')]
 sections.append('<section><h2>文献与复现边界</h2><p><a href="本轮代码与依赖.tar.gz">本轮代码与依赖</a> · <a href="归档清单.json">逐文件SHA归档清单</a> · <a href="../复现说明.md">复现说明</a></p><ul>'+''.join('<li><a href="'+u+'">'+esc(t)+'</a></li>' for t,u in sources)+'</ul><p>RL-DITR*是公开Loop上的连续基础率任务适配，不是原私有医院队列复现。LOM为按论文与作者代码核验的任务实现；近年通用离线RL算法并非专为本场景设计。Patek的17分钟是CGM滞后补偿，当前CGM-only适配不含需要临床CF的IOB模块，不能冒充完整SSM或临床系统。</p><p><a href="../实验合同.md">实验合同</a> · <a href="../共享数据预注册.md">共享数据合同</a> · <a href="../基线实现核验.md">基线核验</a> · <a href="../literature/方法依据.md">详细文献</a> · <a href="../configs/final_manifest.json">冻结最终模型清单</a></p></section>')
 css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#163248;font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}header,main,footer{max-width:1640px;margin:auto}header{background:#17384f;color:#fff;padding:44px}h1{font-size:34px;line-height:1.3}h2{font-size:24px;margin-top:0}main{padding:26px}section{background:#fff;padding:28px;margin-bottom:24px;border:1px solid #d9e2e8;border-radius:8px}.finding{border-left:5px solid #bc8742}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:10px 9px;border-bottom:1px solid #dce4e9;white-space:nowrap;text-align:right}th{background:#17384f;color:white}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f4f7f9}.ours{background:#e5f2ee!important}a{color:#1d6b98}header a{color:#d1edf8}footer{padding:24px 44px}li{margin:8px 0}@media print{@page{size:A3 landscape;margin:12mm}header{padding:12px}main{padding:0}section{border:0;padding:12px}.scroll{overflow:visible}table{font-size:7pt}td,th{padding:4px}nav{display:none}}'''
 text='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>同协议复评与低血糖修复报告</title><style>'+css+'</style></head><body><header><p>DSENET–RL · RESEARCH AUDIT · 2026-09-21</p><h1>同协议复评与低血糖修复报告</h1><p>保留原版，修正受影响基线，重新验证控制结局。</p><nav><a href="#native">原生动作大表</a>　<a href="#brake17">统一减量层大表</a></nav></header><main>'+''.join(sections)+'</main><footer>公开仿真研究 · 单训练seed · 无临床安全或真实患者有效性声明</footer></body></html>'
 (out/'最终公平对比与低血糖修复报告.html').write_text(text)
 print('REPORT_BUILT')
if __name__=='__main__':main()
