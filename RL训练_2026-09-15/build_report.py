# -*- coding: utf-8 -*-
from pathlib import Path
import json,html,datetime
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
def read(p,default=None):
 return json.loads(p.read_text()) if p.exists() else default
def esc(x):return html.escape(str(x))
def num(x):return f'{x:.4f}' if isinstance(x,float) else esc(x)
def table(headers,rows):
 return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def chart(points,title):
 if not points:return ''
 points=[p for p in points if p[1]<1e6];lo=min(y for x,y in points);hi=max(y for x,y in points);dx=max(x for x,y in points) or 1
 xy=' '.join(f'{45+x/dx*820:.1f},{175-(y-lo)/(hi-lo or 1)*145:.1f}' for x,y in points)
 return f'<figure><figcaption>{esc(title)}（横轴优化步，纵轴数值）</figcaption><svg viewBox="0 0 900 205" role="img"><path d="M45 20V175H865" stroke="#777" fill="none"/><polyline points="{xy}" stroke="#1465a3" stroke-width="2" fill="none"/><text x="3" y="30">{hi:.2f}</text><text x="3" y="175">{lo:.2f}</text><text x="45" y="198">0</text><text x="825" y="198">{dx}</text></svg></figure>'

def main():
 audit=read(R/'local_multimodal_audit.json') or read(R/'multimodal_audit.json',{});data=[]
 for split in ['train','validation']:
  man=read(R/f'packed_{split}_manifest.json',{});ps=man.get('patients',[])
  data.append([split,man.get('patient_count','待核验'),f"{man.get('sample_count',0):,}",f"{sum(p['horizon30_samples'] for p in ps):,}",f"{sum(p['horizon60_samples'] for p in ps):,}"])
 experiments=[]
 for p in sorted(R.iterdir(),key=lambda p:(not p.name.startswith('S00'),p.name)):
  if p.is_dir() and (p/'config.json').exists():
   cfg=read(p/'config.json');done=read(p/'completion.json') or read(p/'failure.json',{});ev=read(p/'full_validation_evaluation.json') or read(p/'final_evaluation.json') or read(p/'best_evaluation.json',{});hist=[]
   if (p/'history.jsonl').exists():
    for line in (p/'history.jsonl').read_text().splitlines():
     try:hist.append(json.loads(line))
     except json.JSONDecodeError:pass
   experiments.append((p,cfg,done,ev,hist))
 completed=[x for x in experiments if x[2].get('status')=='completed_budgeted_experiment' and not x[1]['name'].startswith('S00')]
 accepted=[x for x in completed if x[2].get('gate_prediction') and x[2].get('gate_direction')]
 if completed:
  conclusion='当前不能作为 Harness 中可发布个体化给药结论的 RL 模型。'
  detail='本轮完成了下列真实数据患者模型实验；预测/动作响应、反事实辨识、策略及独立闭环的证据必须分别成立。'
  if accepted:detail+='部分患者模型通过了基本研究screen；这仍不等于策略已通过独立评价或来源合同验收。'
  else:detail+='本轮尝试尚未同时通过预测与剂量响应的基本研究screen，因此策略优化阶段暂不启动，保留患者模型权重与负结果。'
 else:conclusion='实验进行中，尚未形成最终模型适用性结论。';detail='以下只列已经产生的真实文件和结果；未运行项目不算完成。'
 cards='<h2>1. 当前结论</h2><div class="notice"><b>'+conclusion+'</b><p>'+detail+'</p></div>'
 cards+='<p>用途：成人 T1D 历史泵基础输注研究。独立预测器 F 未替换；Harness、真实泵和患者用药均未接入。本页中的患者模型 fR/fT/fP 是 RL 的前置组件，不将它单独命名为已完成的 RL 策略。</p>'
 cards+='<h2>2. 数据与统一预处理</h2>'+table(['划分','患者','5分钟合格起点（全部保留）','可连续30分钟起点','可连续60分钟起点'],data)
 cards+='<p>30/60分钟列表示连续有效的5分钟动作序列，允许各步真实基础率变化；与交接中的“全区间恒定基础率样本”不是同一统计。所有5分钟起点继续参与训练，后续不足用 loss mask；没有为 food/exercise 删患者或缩减起点。模型可见数据集规模、实际优化访问次数和完整epoch数分开报告。</p>'
 cards+='<p>72×22 输入：CGM、此前5分钟基础量、记录Bolus、记录碳水、运动记录数；每通道观测值、mask、距最近记录时间、时间是否有依据，加2维相对时间。无插值、无前向填充；张量中的缺失0总是带mask。运动0条记录表示没有记录，不能推出没有运动。</p>'
 coverage=read(R/'modality_coverage.json',{})
 if coverage:cards+=table(['全部合格历史窗口','food有记录','exercise有记录','两者都有记录'],[[esc(split)]+[f'{v[k]:,} / {v[k]/v["samples"]:.2%}' for k in ['food_history_samples','exercise_history_samples','both_history_samples']] for split,v in coverage.items()])
 rows=[]
 for k,v in audit.get('normalization',{}).items():rows.append([esc(k),num(v['mean']),num(v['scale']),f"{v['observed_training_rows']:,}"])
 cards+=table(['train-only数值通道','均值','标准差/缩放','真实观测统计行'],rows)
 cards+='<p>归一化按train合格起点历史行并集统计，每行只计一次；280患者hash与未来扰动检查通过，本地/远端独立环境读取结果一致。封存32名测试患者和17名demo患者不参与调参；本轮未上传其患者表。</p>'
 cards+='<h2>3. 实验总表</h2>'
 rows=[]
 for p,c,d,e,hist in experiments:
  hs=e.get('horizons_minutes',{});rows.append([f'<a href="#{esc(c["name"])}">{esc(c["name"])}</a>',('失败，记录保留' if d.get('status','').startswith('failed') else '完成有限预算实验') if d else '运行/结果未完整',esc(c['model']),f"lr={c['lr']}<br>unroll={c['unroll']}<br>batch={c['batch_size']}",d.get('steps',hist[-1].get('step','—') if hist else '—'),f"{d.get('samples_seen', next((x.get('samples_seen',0) for x in reversed(hist) if x.get('event')=='train'),0)):,}<br>完整epoch={d.get('full_epochs','—')}<br>患者={d.get('patients_seen', next((x.get('patients_seen','—') for x in reversed(hist) if x.get('event')=='train'),'—'))}",num(hs.get('30',{}).get('model',{}).get('rmse_mmol_l','—')),num(hs.get('60',{}).get('model',{}).get('rmse_mmol_l','—'))])
 cards+=table(['实验','状态','网络设置','训练参数','优化步','实际访问/完整轮次','30min RMSE','60min RMSE'],rows)
 cards+='<p><b>评价口径：</b>过程与最终复评都来自validation，不是独立封存测试。一般轨迹输入日志中的后续基础率，属于回顾性动作条件拟合，动作可能携带后续CGM反馈；恒定动作子集另列。失败实验若只有较早checkpoint指标，应以其best_evaluation和failure记录中的步数为准。</p>'
 cards+='<p>绝对血糖预测、残差预测、展开长度、学习率等修改以各实验配置为准。S00是运行烟测，不能作为正式效果证据。所有RMSE/MAE单位为 mmol/L。</p>'
 cards+='<h2>4. 每次尝试：设置、曲线、结果与限制</h2>'
 for p,c,d,e,hist in experiments:
  cards+=f'<section id="{esc(c["name"])}"><h3>{esc(c["name"])}</h3>'
  evaluated_step=d.get('best_checkpoint_step',d.get('last_saved_checkpoint_step','待完整结果'))
  scope='全量validation：所有424,866个起点' if (p/'full_validation_evaluation.json').exists() else '按患者抽样validation（样本数见下表）'
  cards+='<p><b>以下主要指标对应：权重第 '+esc(evaluated_step)+' 步；'+scope+'。</b></p>'

  if c.get('reason'):cards+='<p><b>原计划/尝试原因（完成程度看状态）：</b>'+esc(c['reason'])+'</p>'
  cards+='<details><summary>完整配置与完成记录</summary><pre>'+esc(json.dumps({'config':c,'completion':d},indent=2,ensure_ascii=False))+'</pre></details>'
  cards+=chart([(x['step'],x['glucose_mse']) for x in hist if x.get('event')=='train'],'训练GLU MSE')
  cards+=chart([(x['step'],x['score_rmse30_plus60']) for x in hist if x.get('event')=='eval'],'固定验证集：30/60分钟RMSE之和（模型选择指标）')
  checkpoints=[]
  for ep in sorted(p.glob('eval_*.json')):
   q=read(ep);hs=q['horizons_minutes'];dose=q['action_sensitivity']
   checkpoints.append([f'<a href="results/{esc(p.name)}/{ep.name}">{int(ep.stem.split("_")[1])}</a>',num(hs.get('30',{}).get('model',{}).get('rmse_mmol_l','—')),num(hs.get('60',{}).get('model',{}).get('rmse_mmol_l','—')),num(dose['fraction_negative_60']),num(dose['median_delta60_per_plus1u_h'])])
  if checkpoints:cards+='<details><summary>全部已保存检查点的过程评价（固定抽样validation）</summary>'+table(['优化步/原始JSON','30min RMSE','60min RMSE','+1 U/h下降比例','60min变化中位数'],checkpoints)+'</details>'
  rows=[]
  for horizon,v in e.get('horizons_minutes',{}).items():
   rows.append([horizon,v['model']['n'],num(v['model']['rmse_mmol_l']),num(v['last']['rmse_mmol_l']),num(v['linear']['rmse_mmol_l']),num(v['model']['mae_mmol_l']),num(v['wtr_accuracy']),esc(v.get('patient_bootstrap95_model_minus_last_mae'))])
  cards+=table(['分钟','有效样本','模型RMSE','Last-value RMSE','线性趋势RMSE','模型MAE','WTR准确率','患者bootstrap95%：模型−Last MAE'],rows)
  sens=e.get('action_sensitivity',{});held=e.get('held_action_subset',{})
  if held:
   cards+='<h4>恒定基础率片段复核（回顾性筛选，仍非独立干预）</h4>'+table(['分钟','样本','模型RMSE','Last RMSE','线性RMSE'],[[h,v['model']['n'],num(v['model']['rmse_mmol_l']),num(v['last']['rmse_mmol_l']),num(v['linear']['rmse_mmol_l'])] for h,v in held.items()])
  cards+='<p><b>固定历史，恒定基础率增加1 U/h：</b>'+esc(json.dumps(sens,ensure_ascii=False))+'</p>'
  local=read(p/'local_dose_response.json',{})
  if local.get('weights_loaded_and_exactly_verified'):cards+='<p><b>更小扰动（+0.1U/h；限60分钟连续资格历史）：</b>'+esc(json.dumps({k:v for k,v in local.items() if k!='records'},ensure_ascii=False))+'</p>'
  diagnostic=read(p/'input_diagnostics.json',{})
  if diagnostic:
   cards+='<h4>完成后输入诊断（相同起点、未重新训练）</h4><p>固定未来基础率可能与日志治疗不符；隐藏food/exercise可能超出输入分布。只说明输入依赖现象，不估计干预后果或特征因果收益。</p>'
   rows=[]
   for group,hs in diagnostic['summary'].items():
    for horizon,v in hs.items():rows.append([esc(group),horizon,v['logged_actions']['n'],num(v['logged_actions']['rmse_mmol_l']),num(v['hold_initial_rate']['rmse_mmol_l']),num(v['mask_food_exercise_history']['rmse_mmol_l'])])
   cards+=table(['历史记录分组','分钟','相同样本数','原输入RMSE','固定初始基础率RMSE','隐藏food/exercise RMSE'],rows)
  if e:cards+='<p>预测screen：<b>'+('通过' if e['gate_prediction_beats_both_baselines_30_60'] else '未通过')+'</b>；动作方向screen：<b>'+('通过' if e['gate_direction_sanity_only'] else '未通过')+'</b>。这些只是研究排错，不能证明反事实剂量后果真实。</p>'
  rows=[]
  for name,v in e.get('strata',{}).items():rows.append([esc(name),v['model']['n'],num(v['model']['rmse_mmol_l']),num(v['last']['rmse_mmol_l'])])
  cards+=table(['60分钟分层','样本','模型RMSE','Last RMSE'],rows)
  rows=[]
  for pt in e.get('patients',[]):
   v=pt['horizons'].get('60')
   if v:rows.append([esc(pt['patient']),v['model']['n'],num(v['model']['mae_mmol_l']),num(v['last']['mae_mmol_l'])])
  cards+='<details><summary>各验证患者60分钟MAE</summary>'+table(['患者','样本','模型MAE','Last MAE'],rows)+'</details>'
  links=[]
  for fname,label in [('config.json','config'),('history.jsonl','逐批日志'),('full_validation_evaluation.json','全量验证JSON'),('final_evaluation.json','扩大抽样验证JSON'),('best_evaluation.json','选中检查点评价'),('failure.json','失败记录'),('provenance.json','源码/数据指纹'),('input_diagnostics.json','输入诊断'),('local_dose_response.json','小剂量扰动')]:
   if (p/fname).exists():links.append(f'<a href="results/{esc(p.name)}/{fname}">{label}</a>')
  cards+='<p>原始文件：'+' · '.join(links)+'</p></section>'
 cards+='<h2>5. 论文、源码与此次适配</h2>'
 rows=[['fR / fT','3层Transformer、256宽、8头；因果历史编码+初始历史交叉注意','连续基础率和时间投影替代原注射/时段embedding'],['fP','GLU MSE + WTR CE','内部患者模型头，独立F保留'],['一致性','修复作者padding反用；不循环回卷标签','双方有梯度；每条轨迹随机一个有效后续位置，代码/协议可查'],['策略三项目标','论文历史RL、模拟RL、动作监督；公开代码缺历史项等问题','仅在前置模型通过后实现/运行，不能用未运行代码冒充结果'],['未来事件','初始历史包含food/exercise','没有真实未来state/food/Bolus进入新动作rollout；未知未来情景仍需独立定义'],['动作语义','浮点U/h；积分量为区间U','不映射为整数注射量；产品时长未冻结']]
 cards+=table(['组成','依据/处理','边界'],rows)
 cards+='<p><a href="method_audit.md">完整逐源码行方法审计</a> · <a href="experiment_protocol.md">实验协议与版本化追加</a> · <a href="method_implementation_review.md">实现独立审阅</a> · <a href="runtime_notes.md">运行证据与边界</a> · <a href="https://github.com/rlditr23/RL-DITR/tree/5080fdbe43979f9b8a9616fbc120aae4ac25842a">作者固定commit</a> · <a href="https://www.nature.com/articles/s41591-023-02552-9">RL-DITR原论文</a></p>'
 cards+='<h2>6. 环境、故障与所有调整</h2>'+table(['记录','实际结果','处理'],[['SSH中文路径','macOS Unix socket超长，首次连接失败','在本项目固定cwd使用相对.adlc；重测成功'],['E00：Python3.11+新Torch','环境已创建；多GB Torch下载过慢，主动取消','保留取消日志；未算模型训练'],['E01：专用.venv-native','Python3.8.10 / Torch2.0.0+cu118 / numpy1.24.4 / pandas2.0.3 / pyarrow17.0.0','复用镜像CUDA；新依赖只装入项目venv；280文件与统计跨环境复核'],['train上传等待','MCP调用120秒超时，但传输继续并完成225文件','以远端逐文件hash为准，不能仅凭调用返回判断缺文件'],['GPU模型工程验证','初版4项通过；修复后增加普通/无梯度推理一致性，共5项通过','证明实现链路，不证明疗效'],['A01评价中断','2,000步出现非有限评价值；未保存该步现场，根因未定','保留失败；后续评价前checkpoint与现场定位、避开旧快速路径，未将异常样本删除'],['独立小扰动脚本错误','首次脚本漏载checkpoint，结果已标INVALID','修复后显式加载并逐张量一致性检查，再重跑；主训练/评价不受影响，错误数值不作结果证据']])
 support=read(R/'action_support.json',{})
 if support:
  cards+='<h3>动作支持描述统计</h3><p>按当前CGM、30分钟趋势、前一基础率、记录碳水历史分箱。仅描述已测历史的条件变异；不能据此证明没有未测混杂，也不能将跨患者变异当作随机干预。</p>'+table(['分箱宽度','≥50样本组数','覆盖样本','加权条件动作SD U/h','低SD<0.1组样本占比'],[[esc(v['bin_widths_glucose_trend_prevrate_recordedcarbs']),v['groups_n_atleast50'],v['samples_in_these_groups'],num(v['sample_weighted_mean_conditional_action_sd']),num(v['fraction_of_supported_samples_in_sd_under_0_1_groups'])] for v in support.get('coarse_groups',[])])
 cards+='<h2>7. 能否用于大项目：结论依据与下一步</h2><p><b>'+conclusion+'</b></p><ul><li>数据层：Loop真实观察性CGM/输注可以训练患者模型与历史动作学习；现有记录的food/exercise已保留。稀疏记录、确定性闭环行为、缺少可靠在线到达时间仍是辨识边界。</li><li>模型层：以第4节各次实测为准。预测误差好不等于动作反事实可信；不得把策略在自身患者模型中取得高奖励当作独立效果。</li><li>策略层：若前置模型screen未通过，策略训练停止在依赖处，报告负结果。没有伪造策略奖励或独立仿真收益。</li><li>产品层：真实来源合同D02、独立F训练重叠核查、未来事件与公开生理仿真、受控worker及双证据闸门验收尚未完成。现有Harness工程不因此改成真实模型已验收。</li></ul>'
 if (ROOT/'conclusion.md').exists():cards+='<pre class="conclusion">'+esc((ROOT/'conclusion.md').read_text())+'</pre>'
 cards+='<h2>8. 可复跑入口与证据</h2><pre>远端目录：/root/autodl-tmp/diabetes-agent\n.venv-native/bin/python RL训练_2026-09-15/test_model.py\n.venv-native/bin/python RL训练_2026-09-15/train.py --config RL训练_2026-09-15/configs/实验配置.json\n# 每个真实运行的完整配置在其results目录，重跑需使用新name保留原结果。</pre>'
 cards+='<p><a href="results/local_multimodal_audit.json">本地22维审计</a> · <a href="results/remote_multimodal_audit.json">远端22维审计</a> · <a href="results/packed_train_manifest.json">训练清单</a> · <a href="results/packed_validation_manifest.json">验证清单</a></p>'
 artifacts=read(ROOT/'artifacts/manifest.json',[])
 if artifacts:
  cards+='<h3>所有尝试的模型权重</h3><p>仅供复核与研究，不具备Harness发布资格。下列权重已从训练检查点导出；完整优化器检查点继续保存在远端对应results目录。best按固定验证集30+60分钟RMSE之和选择。</p>'
  cards+=table(['实验','版本','权重所在步','文件大小','SHA-256'],[[esc(a['run']),f'<a href="artifacts/{esc(a["file"])}">{esc(a["kind"])}</a>',a['step'],f'{a["bytes"]/1e6:.1f} MB',f'<code>{a["sha256"]}</code>'] for a in artifacts])
 for path,label in [('results/delivery_verification.json','本地交付核验'),('results/packed_sha256.json','全部训练数组SHA'),('results/modality_coverage.json','稀疏记录覆盖率'),('results/environment_native_versions.txt','远端环境版本'),('results/A02_lower_learning_rate/local_dose_response_unloaded_INVALID.json','漏载权重的无效诊断记录')]:
  if (ROOT/path).exists():cards+=f'<p><a href="{path}">{label}</a></p>'
 style='body{max-width:1200px;margin:30px auto;padding:0 22px;font:15px/1.65 system-ui,sans-serif;color:#182431;background:#fff}h1{font-size:28px}h2{margin-top:40px;border-bottom:2px solid #ced8e1;padding-bottom:8px}h3{font-size:21px}.notice{background:#fff2d9;padding:18px;border-left:5px solid #cc8817}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px}td,th{border:1px solid #d0d8df;padding:9px;text-align:left;vertical-align:top}th{background:#edf2f6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:15px;font-size:12px}a{color:#145d9b}figure{margin:18px 0}svg{width:100%;max-height:240px}svg text{font-size:12px}section{margin:30px 0;padding-bottom:24px;border-bottom:1px solid #ddd}summary{cursor:pointer;color:#145d9b}nav a{margin-right:18px}.conclusion{font:inherit}'
 document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RL训练与审核记录</title><style>'+style+'</style><body><h1>RL训练与审核记录 · Loop / RL-DITR</h1><p>更新：'+datetime.datetime.now().isoformat(timespec='seconds')+' · 真实数据 / RTX 4090 · 全部尝试与负结果保留</p>'+cards+'</body></html>'
 (ROOT/'训练审核报告.html').write_text(document);print(ROOT/'训练审核报告.html')
if __name__=='__main__':main()
