"""Make a short landing page by copying the audited confirmation table verbatim.

Presentation only: no scoring, ranking, model selection or experiment execution.
The frozen full-report builder and all research evidence remain unchanged.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

R = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--papers', type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    report.relative_to(R)
    output = report.parent / '结果总览.html'
    proof_path = report.parent / '总览核查.json'
    paper_copy = report.parent / '论文来源.json'
    wide_csv = report.parent / '确认对比大表.csv'
    assert not any(p.exists() for p in (output, proof_path, paper_copy, wide_csv)), 'Never overwrite a delivery'
    manifest = json.loads((report / '构建清单.json').read_text())
    checks = json.loads((report / '核查结果.json').read_text())
    assert manifest['status'] == 'final' and checks['status'] == 'passed'
    assert manifest['builder_sha256'] == sha(R / 'build_report.py')
    assert manifest['protocol_sha256'] == sha(R / 'protocol.json')
    assert manifest['scorer_sha256'] == sha(R.parent / 'RL_DITR创新_2026-09-16/control_metrics.py')
    for name, digest in manifest['outputs_sha256'].items():
        assert sha(report / name) == digest, 'Full report changed: ' + name
    confirmed = [p for p in manifest['panels'].values() if p['split'] == 'confirmation']
    assert len(confirmed) == 18
    source = (report / '阶段研究报告.html').read_text()
    tables = re.findall(r'(<section id="panel-\d+-comparison"><h2>独立确认场景.*?</section>)', source, re.S)
    assert len(tables) == 1, 'Need exactly one full confirmation comparison section'
    section = tables[0]
    first_table = re.search(r'<table>.*?</table>', section, re.S).group(0)
    display_rows = []
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', first_table, re.S):
        display_rows.append([' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', cell)).split())
            for cell in re.findall(r'<(?:th|td)[^>]*>(.*?)</(?:th|td)>', row, re.S)])
    assert len(display_rows) == 19 and all(len(row) == 18 for row in display_rows)
    style = re.search(r'<style>(.*?)</style>', source, re.S).group(1)
    style += '''
    #papers table{min-width:1100px;table-layout:fixed}
    #papers th,#papers td{text-align:left;white-space:normal;vertical-align:top;overflow-wrap:anywhere}
    #papers thead th:first-child{left:auto;width:155px;min-width:155px;max-width:none}
    #papers th:nth-child(2){width:260px}#papers th:nth-child(3){width:120px}
    #papers th:nth-child(4){width:440px}#papers th:last-child{width:125px}
    '''
    summary = json.loads((report / '报告解读.json').read_text())
    paragraphs = summary['closeout_overview_paragraphs']
    assert 1 <= len(paragraphs) <= 5 and all(isinstance(p, str) and p.strip() for p in paragraphs)
    papers = json.loads(args.papers.read_text())
    methods = papers['methods']
    expected = {'retained_' + x for x in ('bc', 'td3bc', 'iql', 'rebrac', 'fql', 'lom', 'gfp', 'ditr')}
    assert {m['id'] for m in methods} == expected and len(methods) == 8
    esc = html.escape
    rows = []
    for method in methods:
        links = []
        for field, label in (('paper_url', '原论文'), ('code_url', '作者代码')):
            url = method.get(field)
            if url:
                assert url.startswith('https://')
                links.append('<a href="%s">%s</a>' % (esc(url, quote=True), label))
        rows.append('<tr>' + ''.join('<td>' + esc(str(method.get(k) or '无单一指定来源')) + '</td>'
                    for k in ('display_name', 'paper_title', 'venue_year', 'adaptation_note')) +
                    '<td>' + ' · '.join(links) + '</td></tr>')
    prefix = esc(report.name, quote=True) + '/'
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    text += '<title>DSENet + RL · 结果总览与论文模型对比</title><style>' + style + '</style><body>'
    text += '<header><h1>DSENet + RL：当前结果总览</h1><p>2026-10-04 · 本轮收尾 · 18 个冻结方法 × 60 个确认场景</p>'
    text += '<nav><a href="#reading">结论</a><a href="#' + re.search(r'id="([^"]+)"', section).group(1) + '">大对比表</a>'
    text += '<a href="#papers">论文来源</a><a href="' + prefix + '阶段研究报告.html">完整结果与全部开发尝试</a></nav></header><main>'
    text += '<section id="reading"><h2>如何读这份结果</h2>' + ''.join('<p class="lead">' + esc(p) + '</p>' for p in paragraphs)
    text += '<p>按用户要求暂停新方案与新训练；本表来自已冻结的同场景实测。论文方法列为本项目任务适配结果，不是复制原论文其他任务的分数。</p></section>'
    text += section
    text += '<section id="papers"><h2>论文模型身份与适配边界</h2><div class="table-scroll"><table><thead><tr>'
    text += ''.join('<th>' + x + '</th>' for x in ('表中方法', '原论文', '会议／年份', '本项目适配边界', '一手来源'))
    text += '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>'
    text += '<section><h2>完整结果与可追溯证据</h2><p><a href="确认对比大表.csv">下载横向大表 CSV（与页面相同的两位小数展示值）</a></p><p>'
    text += ' · '.join('<a href="' + prefix + f + '">' + label + '</a>' for f, label in (
        ('阶段研究报告.html', '全部 25 行开发 + 18 行确认'), ('指标长表.csv', '全部指标精确 CSV'),
        ('患者配对差.csv', '患者配对差与区间'), ('病例审计.csv', '全部病例及失败记录'), ('构建清单.json', '来源与 SHA')))
    text += '</p><p>本页大表从已核验的完整报告逐字节复制；数值、粗体和失败标记未另行计算。横向滚动可查看全部指标。完整原始轨迹另外归档，病例 CSV 保留引用与 SHA。</p></section></main></body></html>'
    with output.open('x') as stream:
        stream.write(text)
    shutil.copyfile(args.papers, paper_copy)
    with wide_csv.open('x', encoding='utf-8-sig', newline='') as stream:
        csv.writer(stream).writerows(display_rows)
    with wide_csv.open(encoding='utf-8-sig', newline='') as stream:
        assert list(csv.reader(stream)) == display_rows
    assert section in output.read_text()
    proof = dict(status='passed', full_report_sha256=sha(report/'阶段研究报告.html'),
        full_report_manifest_sha256=sha(report/'构建清单.json'), overview_sha256=sha(output),
        exact_confirmation_section_sha256=hashlib.sha256(section.encode()).hexdigest(),
        confirmation_section_copied_verbatim=True, confirmation_methods=18, paper_methods=8,
        wide_csv_sha256=sha(wide_csv), wide_csv_rows=18, wide_csv_columns=18,
        wide_csv_display_values_exact=True,
        paper_sources_sha256=sha(paper_copy), script_sha256=sha(__file__),
        numeric_scores_recomputed=False, models_executed=False, visual_review='pending')
    with proof_path.open('x') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == '__main__':
    main()
