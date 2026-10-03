"""Check event miss rates and realized open-loop utility on matched original plans."""
import csv
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent


def utility(g):
    mmol=np.asarray(g,dtype=float)/18
    mg=mmol*18
    risk=10*(1.509*(np.log(np.maximum(mg,1))**1.084-5.381))**2
    reward=np.where(mg<70,-1,1-np.clip(risk,0,15.5)/7.75)
    return float(np.sum(reward*(.9**(np.arange(len(g))/12))/12))


def main():
    d=np.load(R/'analysis/point_predictions.npz');rows=[];ranking=[]
    for i,key in enumerate(d['point_ids']):
        actual=[]
        for k in range(7):
            p=R/'results/P1_branches'/(str(key)+'__full%d.json'%k)
            if not p.exists():continue
            branch=json.loads(p.read_text());t=branch['point']['minute']
            future=[r for r in branch['records'] if t<r['minute']<=t+240]
            if len(future)!=48:continue
            cgm=np.array([r['cgm_mg_dl'] for r in future]);bg=np.array([r['bg_mg_dl'] for r in future])
            pred=d['candidate_cgm_mg_dl'][i,k]
            actual.append((k,utility(cgm)))
            rows.append(dict(point=str(key),label=branch['point']['label'],patient=branch['job']['patient'],plan=k,
                predicted_min_cgm=float(pred.min()),actual_min_cgm=float(cgm.min()),actual_min_bg=float(bg.min()),
                cgm_mae=float(np.abs(pred-cgm).mean()),min_cgm_bias=float(pred.min()-cgm.min()),
                false_safe70_cgm=bool(pred.min()>=70 and cgm.min()<70),
                false_safe54_cgm=bool(pred.min()>=54 and cgm.min()<54),
                false_safe70_bg=bool(pred.min()>=70 and bg.min()<70),
                false_safe54_bg=bool(pred.min()>=54 and bg.min()<54),
                predicted_q=float(d['q'][i,k]),realized_q=utility(cgm)))
        if len(actual)==7:
            true=np.array([v for _,v in actual]);a=int(d['logits'][i].argmax());q=int(d['q'][i].argmax());kl=int((d['q'][i]+.05*np.log([.4]+[.1]*6)).argmax())
            ranking.append(dict(point=str(key),true_best=int(true.argmax()),actor=a,q=q,kl=kl,
                                actor_regret=float(true.max()-true[a]),q_regret=float(true.max()-true[q]),kl_regret=float(true.max()-true[kl])))
    with (R/'analysis/forecast_events.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for label in ('severe','mild','negative'):
        subset=[r for r in rows if r['label']==label];s={'arms':len(subset)}
        for threshold in (54,70):
            for sensor in ('bg','cgm'):
                positive=[r for r in subset if r['actual_min_'+sensor]<threshold]
                safe=[r for r in subset if r['predicted_min_cgm']>=threshold]
                misses=sum(r['false_safe'+str(threshold)+'_'+sensor] for r in subset)
                s[str(threshold)+'_'+sensor]=dict(true_low=len(positive),misses=misses,predicted_safe=len(safe),
                    missed_fraction=misses/len(positive) if positive else None,
                    false_safe_fraction=misses/len(safe) if safe else None)
        summary[label]=s
    out=dict(summary=summary,rankings=ranking,complete_pools=len(ranking),
             event_enriched_development=True,probabilities_not_calibrated=True,
             limitation='Arms/windows are correlated. Predictions target CGM; BG columns are a separate safety-label comparison.')
    (R/'analysis/forecast_events.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
