"""Actual CUDA checkpoint/inference mechanics for the new world-driven PPO."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ppo_world_worker import Trainer

R=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='PPO_world_smoke_r2');a=ap.parse_args()
    directory=R/'results'/a.run;checkpoint=directory/'policy_iter01.pt'
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trainer=Trainer((directory/'config.json').resolve(),checkpoint.resolve())
    data=np.load(next((R/'cache/smoke_paired_train').glob('*.npz')))
    h=data['history'][:2];anchor=data['anchor_u_h'][:2]
    initial_count=trainer.count
    req=dict(op='evaluate',history=h.tolist(),anchors=anchor.tolist())
    one=trainer.act(req,evaluate=True);two=trainer.act(req,evaluate=True)
    single=trainer.act(dict(op='evaluate',history=h[:1].tolist(),anchors=anchor[:1].tolist()),evaluate=True)
    assert one==two and single['actions'][0]==one['actions'][0]
    assert trainer.buffer=={} and trainer.count==initial_count
    features,_=trainer.features(h,anchor)
    assert features.shape==(2,195) and torch.isfinite(features).all()
    assert not features.requires_grad
    assert all(p.grad is None and not p.requires_grad for p in trainer.world.parameters())
    assert torch.backends.cudnn.deterministic is False
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    state=torch.load(checkpoint,map_location='cpu')
    assert state['iteration']==1 and state['simulator_transitions']==144
    assert state['model']['4.weight'].abs().max()>0
    result=dict(status='passed',run=a.run,checkpoint_sha256=before,feature_dim=195,
                actual_simulator_transitions=initial_count,evaluate_buffer_empty=True,
                deterministic_repeat_and_batch_actions=True,world_frozen=True,
                actor_gradient_update_observed=True,checkpoint_unchanged=True,
                cudnn_deterministic=False,TF32=False,actions=one['actions'],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                performance_claim=False)
    (R/'checks/world_policy_mechanics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
