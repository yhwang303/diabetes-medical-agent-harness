"""Frozen actor/Q/KL controls. No patient identity or future data in requests."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

R = Path(__file__).resolve().parent
B = R.parent / 'RL_DSENet_2026-09-17'
sys.path.insert(0, str(B))
from world_model import Patient
from policy_bounded import Policy, candidate_actions, history_anchor


class Frozen:
    def __init__(self):
        torch.set_num_threads(2)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        ck = torch.load(B/'results/D06_selected_policy/policy.pt', map_location='cpu')
        wp = B/ck['config']['world_checkpoint']
        assert hashlib.sha256(wp.read_bytes()).hexdigest() == ck['config']['world_sha256']
        wc = torch.load(wp, map_location='cpu'); cfg = wc['config']
        self.world = Patient(B/cfg['forecast_checkpoint'], B.parent/cfg['context_checkpoint']).cuda().eval()
        self.world.load_state_dict(wc['model']); self.world.requires_grad_(False)
        self.actor = Policy().cuda().eval(); self.actor.load_state_dict(ck['policy']); self.actor.requires_grad_(False)
        self.beta = ck['config']['kl_beta']
        assert self.beta == .05

    @torch.no_grad()
    def infer(self, req, mode='actor', diagnostics=True):
        x = torch.tensor(np.asarray(req['history'], dtype='float32'), device='cuda')
        assert x.ndim == 3 and x.shape[1:] == (72, 22) and torch.isfinite(x).all()
        anchor = torch.tensor(req['anchor_u_h'], device='cuda', dtype=x.dtype)
        z, f = self.world.encode(x); rate = self.world.reference_rate(x)
        assert anchor.shape == rate.shape and torch.isfinite(anchor).all()
        assert ((rate-anchor).abs() <= .26).all(), 'out-of-support current rate; explicit recovery required'
        plans = candidate_actions(rate, anchor); ref = self.world.reference(z, f, rate)
        logits = self.actor(z, ref, rate, anchor); actor_choice = logits.argmax(-1)
        if diagnostics or mode != 'actor':
            pred = self.world.trajectories(z, f, rate, plans)
            q = self.world.utility(pred)
            assert torch.isfinite(q).all()
            kl = q + self.beta*self.actor.prior.log()
        chosen = actor_choice if mode == 'actor' else (q if mode == 'q' else kl).argmax(-1)
        actions = plans[torch.arange(len(x), device='cuda'), chosen, 0].cpu().tolist()
        reviews = []
        if diagnostics:
            approx_anchor = history_anchor(x)
            for i in range(len(x)):
                reviews.append(dict(chosen=int(chosen[i]), actor_choice=int(actor_choice[i]),
                                    q_choice=int(q[i].argmax()), kl_choice=int(kl[i].argmax()),
                                    q=q[i].cpu().tolist(), logits=logits[i].cpu().tolist(),
                                    predicted_min_mg_dl=(pred[i].amin(-1)*18).cpu().tolist(),
                                    anchor=float(anchor[i]), rate=float(rate[i]), training_anchor=float(approx_anchor[i]),
                                    plan=plans[i, chosen[i]].cpu().tolist(), action_u_h=actions[i]))
        return {'actions_u_h': actions, 'reviews': reviews}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--review', choices=['actor','q','kl'], default='actor')
    ap.add_argument('--checkpoint'); ap.add_argument('--fast', action='store_true'); args=ap.parse_args()
    f=Frozen(); print(json.dumps({'ready':True}), flush=True)
    for line in sys.stdin:
        try:
            start=time.perf_counter(); out=f.infer(json.loads(line), args.review, not args.fast)
            out['batch_seconds']=time.perf_counter()-start
            print(json.dumps(out, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({'error':repr(error)}), flush=True)


if __name__ == '__main__': main()
