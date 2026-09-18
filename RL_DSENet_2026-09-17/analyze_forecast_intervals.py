"""Paired patient bootstrap; exclude only horizons without observed targets."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    data=json.loads((ROOT/'results/F00_forecast_heldout/evaluation.json').read_text());results=[]
    for horizon in (30,60,120,240):
        pairs=[(p['methods']['dsenet'][str(horizon)]['rmse_mg_dl'],p['methods']['persistence'][str(horizon)]['rmse_mg_dl']) for p in data['patients']]
        assert all((a is None)==(b is None) for a,b in pairs)
        delta=np.array([a-b for a,b in pairs if a is not None]);rng=np.random.default_rng(260917)
        bootstrap=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
        results.append({'horizon_min':horizon,'delta_rmse_mg_dl':float(delta.mean()),'ci95':np.percentile(bootstrap,[2.5,97.5]).tolist(),'patients':len(delta),'patients_without_target':len(pairs)-len(delta)})
    (ROOT/'paper_tables/forecast_heldout_intervals.json').write_text(json.dumps({'bootstrap_seed':260917,'resamples':10000,'cluster':'patient','training_seed_uncertainty_included':False,'results':results},indent=2))

if __name__=='__main__':main()
