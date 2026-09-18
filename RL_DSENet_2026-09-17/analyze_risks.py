"""Descriptive heterogeneity and event-duration audit; never used for model selection."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent

def main():
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text());results={}
    for method in freeze['methods']:
        episodes=json.loads((ROOT/'results'/method['name']/'summary.json').read_text())['episodes'];patients=[]
        for pid in freeze['patients']:
            group=[e for e in episodes if e['patient']==pid]
            row={'patient':pid,'failures':sum(e['metrics']['failed'] for e in group)}
            for metric in ('tir_lower_bound_pct','tir_upper_bound_pct','tbr70_pct','tbr54_pct','cv_pct'):
                values=[e['metrics']['bg'].get(metric) for e in group];values=[v for v in values if v is not None]
                row[metric]=statistics.mean(values) if values else None
            row['max_hypo_event_low_minutes']=max((e['metrics']['bg'].get('max_hypo_low_minutes',0) for e in group),default=0)
            row['prolonged_under54_120min_events']=sum(e['metrics']['bg'].get('prolonged_under54_120min_events',0) for e in group)
            row['right_censored_events']=sum(e['metrics']['bg'].get('right_censored_hypo_events',0) for e in group)
            patients.append(row)
        results[method['name']]={'label':method['label'],'patients':patients,'max_event_low_minutes':max(r['max_hypo_event_low_minutes'] for r in patients),'prolonged_under54_120min_events':sum(r['prolonged_under54_120min_events'] for r in patients),'right_censored_events':sum(r['right_censored_events'] for r in patients)}
    (ROOT/'paper_tables/control_risk_audit.json').write_text(json.dumps({'status':'post_evaluation_descriptive_audit_no_model_changes','event_rule':'BG below70 for15min starts event;15min recovery closes it; max duration counts low intervals within event, not necessarily a continuous run','severe_rule':'continuous BG below54 for at least120min','methods':results},indent=2))

if __name__=='__main__':main()
