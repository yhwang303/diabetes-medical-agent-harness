"""Matched-panel aggregation with patient clusters and explicit missing outcomes."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RL_DITR创新_2026-09-16'))
from control_metrics import summarize

METRICS=('tir_lower_bound_pct','tir_upper_bound_pct','tbr70_pct','tbr54_pct','tar180_pct','tar250_pct','coverage_pct','cv_pct','lbgi','hbgi','risk','hypo_events_per_observed_day')

def mean(values):
    values=[float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None

def paired_interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(260917)
    boot=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return {'difference':float(a.mean()),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'patients':len(a)}

def main():
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text());all_methods={};jobs=None
    for method in freeze['methods']:
        folder=ROOT/'results'/method['name'];summary=json.loads((folder/'summary.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
        assert summary['status']=='completed' and len(summary['episodes'])==freeze['episode_count']
        if jobs is None:jobs=manifest['jobs']
        assert jobs==manifest['jobs'],'Methods do not share the same scenarios'
        assert manifest['checkpoint_sha256']==method.get('sha256')
        patient_rows={};failure_reasons=[]
        for episode in summary['episodes']:
            raw=json.loads((folder/(episode['key']+'.json')).read_text())
            recalculated=summarize(raw['records'],(raw['job']['total_minutes']-360)//5,raw['failure_reason'] is not None)
            assert recalculated==episode['metrics'],'Metric recomputation mismatch'
            if raw['failure_reason']:failure_reasons.append({'key':episode['key'],'reason':raw['failure_reason']})
        for patient in freeze['patients']:
            episodes=[e for e in summary['episodes'] if e['patient']==patient];assert len(episodes)==freeze['episode_count']//len(freeze['patients'])
            row={source:{k:mean([e['metrics'][source].get(k) for e in episodes]) for k in METRICS} for source in ('bg','cgm')}
            for k in ('basal_u_per_observed_day','bolus_u_per_observed_day','action_tv_per_observed_day','pump_changed_fraction'):
                row[k]=mean([e['metrics'].get(k) for e in episodes])
            row['failures']=sum(e['metrics']['failed'] for e in episodes);patient_rows[str(patient)]=row
        aggregate={source:{k:mean([p[source][k] for p in patient_rows.values()]) for k in METRICS} for source in ('bg','cgm')}
        for k in ('basal_u_per_observed_day','bolus_u_per_observed_day','action_tv_per_observed_day','pump_changed_fraction'):aggregate[k]=mean([p[k] for p in patient_rows.values()])
        aggregate['failures']=len(failure_reasons);aggregate['episodes']=len(summary['episodes'])
        all_methods[method['name']]={'label':method['label'],'training_information':method['training_information'],'aggregate':aggregate,'patients':patient_rows,'failure_reasons':failure_reasons,'wall_seconds':summary['wall_seconds']}
    primary=all_methods[freeze['primary_method']];comparisons={}
    for name,result in all_methods.items():
        if name==freeze['primary_method']:continue
        values={}
        for metric in ('tir_lower_bound_pct','tir_upper_bound_pct','tbr70_pct','tbr54_pct','risk'):
            differences=[]
            for patient in freeze['patients']:
                a=primary['patients'][str(patient)]['bg'][metric];b=result['patients'][str(patient)]['bg'][metric]
                if a is not None and b is not None:differences.append(a-b)
            values[metric]=paired_interval(differences) if differences else None
        values['interpretation']='Primary minus comparator; TIR endpoints account for missing outcomes. TBR/risk are observed-only. Patient-cluster CI excludes training-seed uncertainty.'
        comparisons[name]=values
    output={'matched_jobs_verified':True,'all_raw_metrics_recomputed':True,'patients':len(freeze['patients']),'scenarios_per_patient':freeze['episode_count']//len(freeze['patients']),'single_training_seed':260915,'methods':all_methods,'paired_comparisons':comparisons,'freeze_sha256':hashlib.sha256((ROOT/'configs/final_freeze.json').read_bytes()).hexdigest()}
    tables=ROOT/'paper_tables';tables.mkdir(exist_ok=True)
    (tables/'final_control_analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    with (tables/'control_final.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['Method','Training information','BG TIR lower %','BG TIR upper %','TBR70 observed %','TBR54 observed %','TAR180 observed %','CV observed %','Risk observed','Failures','Episodes','Coverage %','Basal U/day','Action total variation/day'])
        for item in all_methods.values():
            a=item['aggregate'];g=a['bg'];writer.writerow([item['label'],item['training_information'],g['tir_lower_bound_pct'],g['tir_upper_bound_pct'],g['tbr70_pct'],g['tbr54_pct'],g['tar180_pct'],g['cv_pct'],g['risk'],a['failures'],a['episodes'],g['coverage_pct'],a['basal_u_per_observed_day'],a['action_tv_per_observed_day']])
    latex=['\\begin{tabular}{lrrrrr}','\\toprule','Method & TIR (\\%) $\\uparrow$ & TBR$_{70}$ (\\%) $\\downarrow$ & TBR$_{54}$ (\\%) $\\downarrow$ & Failures & Coverage (\\%) \\\\','\\midrule']
    def f(v):return '--' if v is None else '%.2f'%v
    for item in all_methods.values():
        a=item['aggregate'];g=a['bg'];tir=f(g['tir_lower_bound_pct'])
        if g['tir_upper_bound_pct']-g['tir_lower_bound_pct']>1e-6:tir+='--'+f(g['tir_upper_bound_pct'])
        label=item['label'].replace('_','\\_').replace('±','$\\pm$')
        latex.append(' & '.join([label,tir,f(g['tbr70_pct']),f(g['tbr54_pct']),str(a['failures'])+'/'+str(a['episodes']),f(g['coverage_pct'])])+' \\\\')
    latex+=['\\bottomrule','\\end{tabular}','% Known virtual adults; frozen new scenarios, shared meal bolus and six-hour warmup.','% Single training seed 260915; systems differ in training information and action support.','% TIR ranges are missing-outcome bounds, not confidence intervals. TBR is observed-only.']
    (tables/'control_final.tex').write_text('\n'.join(latex)+'\n')
    print(json.dumps({n:r['aggregate'] for n,r in all_methods.items()},indent=2))

if __name__=='__main__':main()
