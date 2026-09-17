"""Checks physical bookkeeping, reproducibility and explicit observable feature construction."""
import json,hashlib
import numpy as np
from sim_eval import History,simulate,ROOT,NORMALIZER

def main():
 h=History();h.append(0,180,1/12,2,30);h.append(5,198,.1,None,None)
 x=h.rows[-1];norm=json.loads(NORMALIZER.read_text());assert abs(x[0]-(11-norm['cgm_mmol_l']['mean'])/norm['cgm_mmol_l']['scale'])<1e-6
 assert np.array_equal(x[5:10],[1,1,0,0,0]);assert np.array_equal(x[15:20],[1,1,1,1,0]);assert np.all(x[[2,3,4]]==0);assert abs(x[12]-np.log1p(5/60))<1e-6
 for k in range(2,73):h.append(k*5,180,.1,None,None)
 s=h.state();assert s.shape==(72,22) and np.isfinite(s).all();assert np.allclose(s[:,20],1/12) and np.allclose(s[:,21],np.arange(-71,1)/12)
 first=json.loads((ROOT/'results/sim_checks/nominal_first.json').read_text());second=simulate(1,3101,days=3)
 assert first['records']==second['records'],'Same seeded official environment did not reproduce';assert first['metrics']==second['metrics'];assert len(second['records'])==864
 assert abs(sum(x['meal_g'] for x in second['records'])-sum(x['meal_g'] for x in first['records']))<1e-8
 out={'passed':True,'checks':['observed-value normalization','missing is masked not absent event','causal log-age and 72x22 time alignment','exact deterministic 3-day repeated trace','5-minute increments and per-step delivered-dose conservation (asserted inside simulate)'],'simulator_has_no_exercise_physiology':True,'record_sha256':hashlib.sha256(json.dumps(second['records'],sort_keys=True).encode()).hexdigest(),'contract_sha256':hashlib.sha256((ROOT/'simulation_contract_v1.json').read_bytes()).hexdigest(),'clinical_ready':False}
 (ROOT/'results/sim_checks/checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
