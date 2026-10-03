"""Read-only full raw/JSON/CSV/HTML/LaTeX consistency audit, original runtime."""
import csv
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from check_final_analysis import check
import final_analysis as fa
import build_report as br


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.panel = None
        self.tables = {}
        self.in_table = False
        self.cell = None
        self.row = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'section':
            self.panel = attrs.get('id')
        if tag == 'table' and self.panel in ['native', 'brake17']:
            self.in_table = True
            self.tables[self.panel] = []
        if not self.in_table:
            return
        if tag == 'tr':
            self.row = []
        if tag in ['td', 'th']:
            self.cell = {'text':'', 'bold':False}
        if tag == 'b' and self.cell is not None:
            self.cell['bold'] = True
    def handle_data(self, data):
        if self.in_table and self.cell is not None:
            self.cell['text'] += data
    def handle_endtag(self, tag):
        if self.in_table:
            if tag in ['td', 'th'] and self.cell is not None:
                self.row.append(self.cell)
                self.cell = None
            if tag == 'tr':
                self.tables[self.panel].append(self.row)
            if tag == 'table':
                self.in_table = False
        if tag == 'section':
            self.panel = None


def main():
    data = json.loads((R/'analysis/final.json').read_text())
    manifest = json.loads((R/'configs/final_manifest.json').read_text())
    assert data['manifest'] == manifest and not manifest['selection_pending']
    parser = Tables()
    parser.feed((R/'delivery/最终公平对比与低血糖修复报告.html').read_text())
    out = {'status':'passed', 'raw_episodes':0, 'rows':[], 'warnings':[]}
    first_jobs = None
    for panel in manifest['panels']:
        rows = data['panels'][panel]
        complete = [a for a in rows.values() if a['failures']==0 and a['coverage_pct']['mean']==100]
        assert complete
        best = {k:(max if k=='tir_lower_bound_pct' else min)(a[k]['mean'] for a in complete) for k,_ in br.G}
        csv_rows = list(csv.DictReader((R/f'delivery/comparison_{panel}.csv').open()))
        assert len(csv_rows) == len(manifest['methods'])
        html_rows = parser.tables[panel]
        assert len(html_rows) == len(manifest['methods'])+1
        assert all(len(r)==len(html_rows[0]) for r in html_rows)
        tex = (R/f'delivery/comparison_{panel}.tex').read_text()
        spec = re.search(r'\\begin\{tabular\}\{([lcr]+)\}', tex).group(1)
        assert len(spec) == 14
        tex_rows = [x for x in tex.splitlines() if ' & ' in x]
        assert len(tex_rows) == len(manifest['methods'])+1
        assert all(x.count(' & ')+1==len(spec) for x in tex_rows)
        for i, method in enumerate(manifest['methods']):
            key = method['key']
            folder = R/'results'/f'C_{panel}_{key}'
            independent = check(folder)
            out['raw_episodes'] += independent['episodes']
            a = rows[key]
            rerun = fa.read(folder)
            for k in fa.G+fa.D:
                assert rerun[k] == a[k]
                for stat in ['mean','sd']:
                    assert float(csv_rows[i][k+'_'+stat]) == a[k][stat]
            for k in ['raw_sha256','patients','failure_reasons','jobs','failures','episodes','prolonged54']:
                # JSON serializes failure-reason tuples as lists; normalize only container types.
                assert json.loads(json.dumps(rerun[k])) == a[k], (panel,key,k)
            if first_jobs is None:
                first_jobs = a['jobs']
            assert first_jobs == a['jobs']
            em = json.loads((folder/'manifest.json').read_text())
            assert em['common_limit_u_h'] is None
            assert em['brake_config'] == (json.loads((R/'configs/brake17.json').read_text()) if panel=='brake17' else None)
            if method['checkpoint']:
                assert em['checkpoint_sha256'] == method['checkpoint_sha256']
            assert csv_rows[i]['method'] == method['label']
            assert csv_rows[i]['training_information'] == method['training_information']
            for k in ['failures','episodes','prolonged54']:
                assert int(csv_rows[i][k]) == a[k]
            hr = html_rows[i+1]
            assert hr[0]['text'].startswith(method['label']+' · ')
            complete_run = a['failures']==0 and a['coverage_pct']['mean']==100
            tr = tex_rows[i+1].split(' & ')
            for j,(k,_) in enumerate(br.G):
                value = '%.2f ± %.2f'%(a[k]['mean'],a[k]['sd'])
                if k=='tir_lower_bound_pct' and not complete_run:
                    value = '[%.2f, %.2f]'%(a[k]['mean'],a['tir_upper_bound_pct']['mean'])
                elif not complete_run:
                    value += ' †'
                bold = complete_run and abs(a[k]['mean']-best[k])<1e-8
                assert hr[j+2] == {'text':value,'bold':bold}
                tv = value.replace('±',r'$\pm$').replace(' †',r'$^{\dagger}$')
                if bold:
                    tv = r'\textbf{'+tv+'}'
                assert tr[j+2] == tv
            assert hr[11]['text'] == str(a['failures'])+'/60'
            assert hr[12]['text'] == '%.2f ± %.2f'%(a['coverage_pct']['mean'],a['coverage_pct']['sd'])
            assert hr[13]['text'] == str(a['prolonged54'])+(' †' if not complete_run else '')
            assert tr[11] == str(a['failures'])+'/60'
            assert tr[12] == '%.2f'%a['coverage_pct']['mean']
            assert tr[13].rstrip().removesuffix(r'\\').rstrip() == str(a['prolonged54'])+(r'$^{\dagger}$' if not complete_run else '')
            out['rows'].append(dict(panel=panel,method=key,full_raw_and_units='passed',table_numbers='passed',incomplete_and_bold_rules='passed'))
        # All printed statistics need visible explanations rather than source comments.
        visible_tex = '\n'.join(s for s in tex.splitlines() if not s.lstrip().startswith('%'))
        if 'dagger' not in visible_tex or ('patient' not in visible_tex.lower() and 'patient-level' not in visible_tex.lower()):
            out['warnings'].append(panel+': inspect visible LaTeX explanatory notes')
    paths = [R/'analysis/final.json', R/'configs/final_manifest.json',R/'build_report.py',R/'final_analysis.py']+list((R/'delivery').glob('comparison_*'))+[R/'delivery/最终公平对比与低血糖修复报告.html']
    out['sha256'] = {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (R/'checks/final_delivery_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'status':out['status'],'raw_episodes':out['raw_episodes'],'rows':len(out['rows']),'warnings':out['warnings']}))


if __name__ == '__main__':
    main()
