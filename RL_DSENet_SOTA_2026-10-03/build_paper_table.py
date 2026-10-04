"""Render the existing confirmation evidence as paper tables; never run models."""
import hashlib
import html
import json
from pathlib import Path

R = Path(__file__).resolve().parent
OUT = R / 'delivery_paper_r1'
SOURCE = R / 'checks/panels_confirmation_r1.json'
SOURCE_SHA = 'b2a0896586d185fe620a461e510c3e64cb3e800ad7b5c1471f2223f69786bde4'
TOL = 1e-8
OURS = 'ppo_world_quantile'
METRICS = [
    ('tir_observed_pct', 'TIR', '70–180 % ↑', 1, 3),
    ('tbr70_pct', 'TBR70', '<70 % ↓', -1, 3),
    ('tbr54_pct', 'TBR54', '<54 % ↓', -1, 3),
    ('tar180_pct', 'TAR180', '>180 % ↓', -1, 3),
    ('tar250_pct', 'TAR250', '>250 % ↓', -1, 3),
    ('sd_mg_dl', 'SD', 'mg/dL ↓', -1, 3),
    ('cv_pct', 'CV', '% ↓', -1, 3),
    ('lbgi', 'LBGI', '↓', -1, 3),
    ('hbgi', 'HBGI', '↓', -1, 3),
    ('risk', 'Risk', 'LBGI + HBGI ↓', -1, 6),
]
LABELS = {
    'ppo_world_quantile': ('DSENet–World PPO40', '本文主方案 · 冻结候选'),
    'ppo_wide': ('Wide PPO40', '本文内部对照 · 旧模型特征'),
    'd06_frozen': ('D06', '本文旧方案 · 冻结'),
    'physiology': ('IOB / Zone', '生理规则对照'),
    'hold': ('Hold', '观测基础率对照'),
    'mpc_event_risk': ('Quantile + event-risk MPC', '本文世界模型 · 固定规划器'),
    'mpc_point_world': ('Point-world MPC', '本文世界模型 · 固定规划器'),
    'mpc_quantile_median': ('Quantile-median MPC', '本文世界模型 · 固定规划器'),
    'mpc_expected': ('Quantile-expected MPC', '本文世界模型 · 固定规划器'),
    'iql_wide': ('IQL · new-task', '外部算法适配 · 固定20k'),
    'retained_bc': ('BC', '通用基线 · 冻结 + 投影'),
    'retained_iql': ('IQL', 'ICLR 2022 · 冻结 + 投影'),
    'retained_td3bc': ('TD3+BC', 'NeurIPS 2021 · 冻结 + 投影'),
    'retained_rebrac': ('ReBRAC', 'NeurIPS 2023 · 冻结 + 投影'),
    'retained_fql': ('FQL', 'ICML 2025 · 冻结 + 投影'),
    'retained_lom': ('LOM', 'ICLR 2025 · 冻结 + 投影'),
    'retained_gfp': ('GFP', 'ICLR 2026 · 冻结 + 投影'),
    'retained_ditr': ('RL-DITR*', 'Nature Medicine 2023 · 任务适配'),
}
EXTERNAL = ['retained_bc', 'retained_td3bc', 'retained_iql', 'retained_rebrac',
            'retained_fql', 'retained_lom', 'retained_gfp', 'retained_ditr', 'iql_wide', OURS]
ALL = ['hold', 'physiology', 'd06_frozen', 'mpc_point_world', 'mpc_quantile_median',
       'mpc_expected', 'mpc_event_risk', 'ppo_wide'] + EXTERNAL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks(values, direction):
    """Two distinct unrounded levels; ties compare to the level's anchor."""
    ordered = sorted(values, key=lambda n: -direction * values[n])
    result = {n: 0 for n in values}
    for level in (1, 2):
        if not ordered:
            break
        anchor = values[ordered[0]]
        tied = [n for n in ordered if abs(values[n] - anchor) <= TOL]
        for n in tied:
            result[n] = level
        ordered = [n for n in ordered if n not in tied]
    return result


def main():
    assert sha(SOURCE) == SOURCE_SHA, 'Audited input changed'
    assert not OUT.exists(), 'Preserve prior deliveries'
    bundle = json.loads(SOURCE.read_text())
    panels = {n.removeprefix('confirmation_'): p for n, p in bundle['panels'].items()}
    assert set(panels) == set(ALL) and len(panels) == 18
    assert sum(p['episodes'] for p in panels.values()) == 1080
    assert all(p['scoring_verified_exact'] and not p['smoke'] for p in panels.values())
    assert sum(p['complete'] is not None for p in panels.values()) == 12
    paper_source = R / 'paper_comparison_sources.json'
    papers = json.loads(paper_source.read_text())
    audit = {'source_sha256': SOURCE_SHA, 'paper_source_sha256': sha(paper_source),
             'tie_tolerance_absolute': TOL, 'ranking_uses_unrounded_means': True,
             'failed_rows_ranked': False, 'tables': {}, 'cells': []}

    def paper_table(names, table_id, caption):
        eligible = [n for n in names if panels[n]['complete'] is not None]
        rank_maps = {key: ranks({n: panels[n]['complete']['bg'][key]['mean'] for n in eligible}, direction)
                     for key, _, _, direction, _ in METRICS}
        rank_maps['long54'] = ranks({n: panels[n]['observed_event_totals']['bg']['prolonged_under54_120min_events']
                                    for n in eligible}, -1)
        audit['tables'][table_id] = {'rows': names, 'eligible': eligible, 'ranks': rank_maps}
        headings = '<th scope="col">Method / 方法</th>' + ''.join(
            '<th scope="col">%s<small>%s</small></th>' % (html.escape(title), html.escape(unit))
            for _, title, unit, _, _ in METRICS)
        headings += '<th scope="col">长严重低糖<small>事件数 ↓</small></th><th scope="col">失败<small>病例 / 60</small></th><th scope="col">覆盖<small>%</small></th>'
        rows = []
        for n in names:
            p = panels[n]; complete = p['complete'] is not None
            label, sub = LABELS[n]
            note = sub + (' · Observed†' if not complete else '')
            cells = ['<th scope="row">' + html.escape(label) + '<small>' + html.escape(note) + '</small></th>']
            for key, _, _, _, places in METRICS:
                stat = p['observed']['bg'][key]
                assert stat['mean'] is not None and stat['sd'] is not None
                rank = rank_maps[key].get(n, 0)
                body = '<span class="mean">%.*f</span><span class="sd">± %.*f</span>' % (places, stat['mean'], places, stat['sd'])
                if rank == 1:
                    body = '<strong class="best">' + body + '</strong>'
                elif rank == 2:
                    body = '<span class="second">' + body + '</span>'
                cells.append('<td data-metric="%s" data-rank="%d">%s</td>' % (key, rank, body))
                audit['cells'].append({'table': table_id, 'method': n, 'metric': key,
                                       'mean': stat['mean'], 'sd': stat['sd'], 'places': places, 'rank': rank})
            count = p['observed_event_totals']['bg']['prolonged_under54_120min_events']
            rank = rank_maps['long54'].get(n, 0)
            val = str(count) + ('' if complete else '†')
            val = '<strong class="best">' + val + '</strong>' if rank == 1 else '<span class="second">' + val + '</span>' if rank == 2 else val
            cells += ['<td data-metric="long54" data-rank="%d">%s</td>' % (rank, val),
                      '<td>%d/60</td>' % p['failed_episodes'],
                      '<td>%.3f</td>' % p['observed']['bg']['coverage_pct']['mean']]
            cls = 'ours' if n == OURS else ('incomplete' if not complete else '')
            rows.append('<tr class="%s" data-method="%s">%s</tr>' % (cls, n, ''.join(cells)))
        return '<figure id="%s"><figcaption>%s</figcaption><p class="scope">排名范围：本表 %d 个完整部署。失败行只报告已观测前缀。</p><div class="scroll" tabindex="0" aria-label="横向滚动查看全部指标"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div></figure>' % (
            table_id, caption, len(eligible), headings, ''.join(rows))

    external_table = paper_table(EXTERNAL, 'external', 'Table 1. 本文主模型与外部算法的任务适配对照')
    all_table = paper_table(ALL, 'all', 'Table 2. 全部 18 个冻结部署：加入项目内世界模型、MPC 与规则对照')
    ours = panels[OURS]['complete']['bg']
    comparison = {}
    for n in EXTERNAL:
        if n == OURS or panels[n]['complete'] is None:
            continue
        comparison[n] = {key: ours[key]['mean'] - panels[n]['complete']['bg'][key]['mean']
                         for key, *_ in METRICS}
    external_ranks = audit['tables']['external']['ranks']
    ours_unique_best = [key for key, *_ in METRICS if external_ranks[key][OURS] == 1 and
                        sum(x == 1 for x in external_ranks[key].values()) == 1]
    assert set(ours_unique_best) == {'tir_observed_pct', 'tbr54_pct', 'tar180_pct', 'sd_mg_dl', 'hbgi', 'risk'}
    assert external_ranks['tar250_pct'][OURS] == 1
    failure_bounds = {n: panels[n]['coverage_and_tir_bounds']['bg']['tir_upper_bound_pct']['mean']
                      for n in EXTERNAL if panels[n]['complete'] is None}
    assert all(x < ours['tir_observed_pct']['mean'] for x in failure_bounds.values())
    audit['interpretation'] = {'relative_d06_gate_is_not_sota_definition': True,
        'sota_does_not_require_every_metric_best': True,
        'external_comparison_complete_baselines': list(comparison),
        'external_unique_best_among_ten_metrics': ours_unique_best,
        'ours_minus_external_complete': comparison,
        'incomplete_external_tir_optimistic_upper_bounds': failure_bounds,
        'field_sota_established': False,
        'field_sota_reason': 'Limited task-adapted comparators, unequal training data/budgets, single training seed and known virtual patients; no matched external-baseline uncertainty analysis.',
        'new_training_or_simulator_execution': False}

    failure_rows = []
    for n in ALL:
        p = panels[n]
        if p['complete'] is not None:
            continue
        cov = p['coverage_and_tir_bounds']['bg']
        failure_rows.append('<tr><th scope="row">%s</th><td>%d/60</td><td>%.3f</td><td>[%.3f, %.3f]</td><td>%d</td></tr>' % (
            LABELS[n][0], p['failed_episodes'], cov['coverage_pct']['mean'],
            cov['tir_lower_bound_pct']['mean'], cov['tir_upper_bound_pct']['mean'], p['unknown_tail_intervals'] * 5))
    sources = []
    for m in papers['methods']:
        urls = ' · '.join('<a href="%s">%s</a>' % (html.escape(m[k], quote=True), title)
                          for k, title in [('paper_url', '原论文'), ('code_url', '作者代码')] if m.get(k))
        sources.append('<li><b>%s</b>（%s）：%s %s</li>' % (
            html.escape(LABELS[m['id']][0]), html.escape(m['venue_year']), html.escape(m['adaptation_note']), urls))
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSENet + RL · 论文对比表与 SOTA 判断</title><style>
:root{--ink:#151515;--muted:#595959;--line:#b3b3b3;--paper:#fff;--link:#214666}*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.75 "Songti SC","STSong","Noto Serif CJK SC",serif}
main{max-width:1860px;padding:32px 32px 72px;margin:auto}h1{font-size:30px;font-weight:600;margin:8px 0 12px;line-height:1.4}h2{font-size:21px;margin:40px 0 12px}p{margin:12px 0}
.meta,nav,.scope,.notes,.sd{font-family:-apple-system,"PingFang SC",sans-serif}.meta{font-size:12px;letter-spacing:.09em;color:var(--muted)}
nav{font-size:13px;display:flex;gap:22px;flex-wrap:wrap;padding:14px 0;border-bottom:2px solid var(--ink)}a{color:var(--link);text-underline-offset:3px}
.verdict{max-width:1150px;font-size:17px;margin:22px 0 24px}.verdict b{font-weight:700}.notes{font-size:13px;color:var(--muted);max-width:1300px;line-height:1.8}.notes p{margin:6px 0}
figure{margin:30px 0 14px}figcaption{font-size:17px;margin-bottom:5px;font-weight:600}.scope{font-size:12px;color:var(--muted);margin:0 0 12px}
.scroll{overflow:auto;max-width:100%;scrollbar-gutter:stable}.scroll:focus{outline:2px solid var(--link);outline-offset:4px}
table{border-collapse:separate;border-spacing:0;border-top:2px solid var(--ink);border-bottom:2px solid var(--ink);width:100%;min-width:1510px;font:14px/1.4 "Times New Roman","Songti SC",serif;font-variant-numeric:tabular-nums}
th,td{padding:11px 8px;text-align:center;white-space:nowrap;vertical-align:middle;font-weight:400}thead th{border-bottom:1px solid var(--ink);font-weight:600;padding:12px 8px;background:#fff}
th:first-child{text-align:left;min-width:222px;width:222px;position:sticky;left:0;background:#fff;z-index:1;border-right:1px solid #eee}
thead th:first-child{z-index:2}small{display:block;font:11px/1.6 -apple-system,"PingFang SC",sans-serif;color:var(--muted);font-weight:400;white-space:normal}
tbody th{font-weight:400}.mean{display:block}.sd{display:block;font-size:10px;line-height:1.65;color:#595959;white-space:nowrap;font-weight:400}
strong.best{font-weight:800}strong.best .sd{font-weight:700;color:#151515}.second .mean,.second .sd{ text-decoration:underline;text-underline-offset:3px;text-decoration-thickness:1px}.second:not(:has(.mean)){text-decoration:underline;text-underline-offset:3px}
.ours th,.ours td{border-top:1px solid var(--ink)}.ours th{font-weight:700}.incomplete{color:#626262}.incomplete th{color:#626262}
.text{max-width:1180px}.text ul{padding-left:24px}.quote{border-left:3px solid var(--ink);padding:10px 20px;margin:22px 0;font-size:17px}.audit table{min-width:780px}.audit th:first-child{min-width:180px}
.key{font-family:"Times New Roman",serif}.key b{font-weight:800}.key u{text-underline-offset:3px}details{margin:24px 0}summary{cursor:pointer}footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
@media(max-width:650px){main{padding:20px 14px 44px}h1{font-size:25px}.verdict{font-size:16px}th:first-child{min-width:185px;width:185px}nav{gap:14px}}
@page{size:A3 landscape;margin:12mm}@media print{main{max-width:none;padding:0}nav,.screen-only{display:none}.scroll{overflow:visible}table{min-width:0;font-size:10px}th,td{padding:6px 4px}th:first-child{position:static;min-width:130px;width:130px}small,.sd{font-size:8px}thead{display:table-header-group}tr{break-inside:avoid}h2,figcaption{break-after:avoid}a{color:#151515}.notes{font-size:10px}figure{margin-top:20px}}
</style></head><body><main><div class="meta">CONFIRMATION STUDY · 2026-10-04 · SINGLE TRAINING SEED</div>
<h1>DSENet + RL：论文对比表与 SOTA 判断</h1>
<nav><a href="#external">外部模型对比</a><a href="#all">全部18方法大表</a><a href="#judgment">SOTA 判断</a><a href="#failures">失败与未知尾</a><a href="#sources">论文来源</a><a href="../delivery_closeout_r1/完整研究报告/阶段研究报告.html">完整研究档案</a></nav>
<div class="verdict"><b>当前判断：World PPO40 在本次外部任务适配对照中，TIR、总风险等主要指标领先；现有证据尚不足以确立整个领域的 SOTA。</b><p>“相对 D06 降低低糖且其他指标不退步”是创新过程约束，不是 SOTA 的定义。SOTA 应针对明确任务、比较对象和评价指标判断，也不要求每一项指标全部第一。下文据外部模型比较重新解释结果。</p></div>
<div class="notes"><p class="key"><b>粗体：本表最优</b>；<u>下划线：本表第二优</u>。按未舍入均值排名，容差1e−8；并列同标，第二优为下一不同数值层级。不是显著性标记。</p><p>真实 BG 为主指标。上行为均值，下行为 ± 患者间样本标准差；10名虚拟患者，每人6个病例。长严重低糖为BG&lt;54 mg/dL持续≥120分钟的累计事件数。失败数和覆盖率不作成绩排名。</p><p>† 提前终止的方法仅有Observed前缀，排除最优／次优排名。百分比显示三位小数，Risk显示六位小数以避免掩盖极小差距。表1与表2的比较范围不同，标记分别计算；所有18行均可在表2查到。</p></div>
''' + external_table + '''
<p class="notes">表1比较本文主方案与8个旧基线适配、1个新任务IQL适配。它们不是9个独立算法；两个IQL为不同任务适配实例。完整比较行只有本文、BC、旧IQL和新IQL，六个提前终止方法仍保留，不把失败前缀当成完整随访。</p>
''' + all_table + '''
<p class="notes">表2增加项目内消融／对照及基于本文训练世界模型的MPC。它们不是新增外部论文基线，MPC也不是PPO。World PPO与event-risk MPC的Risk为1.658866和1.658930：数值第一／第二的差仅0.000064，不能据加粗或下划线声称可靠优势。</p>
<section id="judgment" class="text"><h2>怎样判断当前结果</h2>
<p><b>1. 与本次外部适配基线相比：主要指标领先成立。</b>World PPO40的TIR为97.115%，高于BC的89.545%、旧IQL的91.774%及新IQL的95.107%；综合风险为1.658866，分别低于3.245909、3.033590、2.212558。在表1的10项连续指标中，它有6项单独最优，TAR250并列最优。这里的计数只是描述，相关指标并非独立证据，也不是事后制定的SOTA通过规则。</p>
<p>保留真实取舍：BC的TBR70较低（1.132% vs 1.225%）、LBGI较低（0.481 vs 0.496）；新IQL的CV略低（18.058% vs 18.113%）。与旧IQL相比，World PPO在上述10项均值上均改善；与新IQL相比8项改善、1平、CV略差。这些都是样本均值比较，不是逐患者或统计显著性结论。</p>
<p>六个失败基线不能按Observed低糖值参加排名；但其TIR乐观上界均低于World PPO的完整TIR，因此当前记录支持本文主方案的TIR领先，并非只因把这些行排除才出现第一。该结论仅适用于固定部署和本批场景，不代表对应算法的最佳可达能力。</p>
<p><b>2. 与项目内所有完整控制器相比：各有优点，没有统一的全指标冠军。</b>本文event-risk MPC的TIR更高（97.681%）、TBR54更低（0.004%）、血糖SD更低（21.720 mg/dL），World PPO的TBR70与LBGI更低。规则对照低糖最少，同时高糖更多。若讨论整个世界模型系列，MPC成绩也是本文结果；若讨论“世界模型＋RL给药”，应使用World PPO，不能把MPC的最好数值拼接到PPO名下。</p>
<p><b>3. 是否已经证明领域SOTA：尚未证明，而不是已经证明不可能是SOTA。</b>限制来自比较证据：当前外部方法是具体任务适配，训练数据、奖励、交互预算和调参充分程度并不等价；六个基线提前终止；只有单训练seed和10名已见虚拟成人；现有配对区间以D06为参考，不能移用来宣称对外部基线的显著优势。当前也没有覆盖所有同任务强参考实现。未通过D06无退步约束不是这项结论的否决条件。</p>
<p>原论文任务也不能直接混排：FQL报告的是OGBench/D4RL任务，RL-DITR研究的是住院T2D胰岛素方案，均不等于本项目成人T1D基础输注仿真；其论文分数不能直接与本表TIR比较。<a href="https://proceedings.mlr.press/v267/park25f.html">FQL原文</a>；<a href="https://www.nature.com/articles/s41591-023-02552-9">RL-DITR原文</a>。</p>
<div class="quote">可用于报告的表述：在本研究固定的成人T1D基础输注仿真比较中，DSENet–World PPO40相对所评估的外部算法适配获得最高平均TIR；在完整随访对照中，其严重低糖暴露、血糖SD及综合风险等指标也领先，同时保留TBR70、LBGI及CV方面的取舍。该结果支持本比较集内的性能优势，尚不能外推为领域SOTA。</div>
<p class="notes">本页修正此前把开发约束与SOTA结论并置的解读，不修改已冻结的实验、成绩或模型选择。没有新训练、新场景、重新调参或事后改变评分公式。</p></section>
<section id="failures" class="text audit"><h2>提前终止与未知尾部</h2><div class="scroll" tabindex="0"><table><thead><tr><th>方法</th><th>失败 / 60</th><th>BG覆盖 %</th><th>TIR下界—上界 %</th><th>未知尾分钟</th></tr></thead><tbody>''' + ''.join(failure_rows) + '''</tbody></table></div><p class="notes">共50个原生提前终止病例。TIR上下界是未知结局的范围，不是置信区间；Observed低糖为0不等于全程无低糖。完整随访方法TIR上下界与其均值相同。全部病例引用见完整报告的病例审计CSV。</p></section>
<section id="sources" class="text"><h2>论文身份与实际适配</h2><ul>''' + ''.join(sources) + '''</ul></section>
<footer>数据：18部署 × 60病例 = 1,080条确认记录。数值来源SHA256：''' + SOURCE_SHA + '''。<br><a href="排名与判定核查.json">查看排名、精确值与判定范围</a> · <a href="../delivery_closeout_r1/完整研究报告/指标长表.csv">下载全部精确指标CSV</a> · <a href="../delivery_closeout_r1/确认对比大表.csv">原18方法横向CSV</a><br>公开仿真研究结果；单训练seed260915、10名已见虚拟成人。打印样式为A3横向。</footer></main></body></html>'''
    OUT.mkdir()
    path = OUT / '论文对比表.html'
    path.write_text(page)
    audit.update({'status': 'built', 'html_sha256': sha(path), 'builder_sha256': sha(__file__),
                  'raw_scoring_rerun': False, 'all_original_scores_preserved': True})
    (OUT / '排名与判定核查.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'html': str(path), 'external_rows': len(EXTERNAL), 'all_rows': len(ALL),
                      'numeric_cells': len(audit['cells']), 'source_unchanged': sha(SOURCE) == SOURCE_SHA}, ensure_ascii=False))


if __name__ == '__main__':
    main()
