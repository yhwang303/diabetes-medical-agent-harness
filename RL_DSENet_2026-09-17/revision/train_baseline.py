"""Single-seed, fixed-budget external baseline training; no simulator selection."""
import argparse,importlib,json,hashlib,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'RL进阶对比_2026-09-15'))
from rl_data import Replay

def make_agent(cfg):return importlib.import_module(cfg['algorithm']+'_algorithm').Agent(cfg)
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=ROOT/'results'/cfg['name'];out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
    write(out/'config.json',cfg);source=out/'sources';source.mkdir()
    for p in ROOT.glob('*algorithm.py'):shutil.copy2(p,source/p.name)
    shutil.copy2(__file__,source/Path(__file__).name)
    for name in ('rl_algorithms.py','rl_data.py'):
        shutil.copy2(ROOT.parents[1]/'RL进阶对比_2026-09-15'/name,source/name)
    train=Replay('train');val=Replay('validation');agent=make_agent(cfg).cuda();agent.optimizers()
    write(out/'provenance.json',{'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
        'train_manifest':train.manifest,'validation_manifest':val.manifest,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'selection':'fixed final budget; no simulator or sealed-test checkpoint selection','started_at':time.strftime('%Y-%m-%dT%H:%M:%S')})
    permutation=torch.randperm(train.size,device='cuda');cursor=0;seen=0;start=time.time();window=[];step=0
    try:
        with (out/'history.jsonl').open('w',buffering=1) as log:
            for step in range(1,cfg['updates']+1):
                if cursor==train.size:permutation=torch.randperm(train.size,device='cuda');cursor=0
                ids=permutation[cursor:min(cursor+cfg['batch_size'],train.size)];cursor+=len(ids);seen+=len(ids)
                metrics=agent.update(train.batch(ids),step)
                if not all(np.isfinite(v) for v in metrics.values()):raise RuntimeError('Nonfinite training metric')
                window.append(metrics)
                if step%500==0:
                    keys=set().union(*(x.keys() for x in window));event={k:float(np.mean([x[k] for x in window if k in x])) for k in keys}
                    event.update(step=step,seen=seen,seconds=time.time()-start,updates_per_second=step/(time.time()-start));log.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True);window=[]
                if step%10000==0 or step==cfg['updates']:
                    checkpoint={'agent':agent.state_dict(),'config':cfg,'step':step,'seen':seen,
                        'optimizers':{k:v.state_dict() for k,v in vars(agent).items() if isinstance(v,torch.optim.Optimizer)},
                        'schedulers':{k:v.state_dict() for k,v in vars(agent).items() if k.endswith('_scheduler')},
                        'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'permutation':permutation,'cursor':cursor}
                    torch.save(checkpoint,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        agent.eval()
        with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(91526);idx=torch.randperm(val.size,device='cuda')[:8192];actions=[];errs=[]
            for i in range(0,len(idx),256):
                b=val.batch(idx[i:i+256]);p=agent.act(b['state']);actions.extend((10*(p+1)).flatten().tolist());errs.extend((10*(p-b['action'])).flatten().tolist())
        write(out/'diagnostic.json',{'scope':'validation logged-action diagnostic only','mae_u_h':float(np.abs(errs).mean()),'action_quantiles_u_h':np.quantile(actions,[0,.01,.5,.99,1]).tolist()})
        write(out/'completion.json',{'status':'fixed_budget_training_complete','step':step,'seen':seen,'seconds':time.time()-start,'training_seed':cfg['seed'],'final_checkpoint_sha256':hashlib.sha256((out/'last.pt').read_bytes()).hexdigest()})
        print(json.dumps({'complete':cfg['name'],'seconds':time.time()-start}),flush=True)
    except Exception:
        write(out/'failure.json',{'step':step,'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
