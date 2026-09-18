"""Reuse frozen external baseline exports; no optimization or new weights."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'RL进阶对比_2026-09-15'))
from numpy_policy import NumpyPolicy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--mode');args=ap.parse_args();policies={};rngs={}
    print(json.dumps({'ready':True,'frozen_baseline':True}),flush=True)
    for line in sys.stdin:
        try:
            req=json.loads(line);start=time.perf_counter();actions=[]
            for history,key,seed in zip(req['history'],req['case_keys'],req['scenario_seeds']):
                if key not in policies:policies[key]=NumpyPolicy(args.checkpoint);rngs[key]=np.random.default_rng(seed+20000)
                action=float(10*(policies[key](history,rngs[key].normal(size=(1,1)))[0,0]+1))
                if not np.isfinite(action) or not -1e-5<=action<=20.00001:raise ValueError('Invalid baseline output')
                actions.append(float(np.clip(action,0,20)))
            print(json.dumps({'actions_u_h':actions,'batch_seconds':time.perf_counter()-start}),flush=True)
        except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)

if __name__=='__main__':main()
