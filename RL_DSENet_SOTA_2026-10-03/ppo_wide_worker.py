"""Diagnostic A: real-reward PPO with wide basal actions and frozen DSENet features.

This worker does not use the old world to score candidate actions. BG is accepted
only with completed training outcomes, never by the inference feature function.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

R = Path(__file__).resolve().parent
P = R.parent
B = P / 'RL_DSENet_2026-09-17'
sys.path.insert(0, str(B))
from world_model import Patient
from physiologic_features import FEATURE_DIM, FEATURE_NAMES, features as physical_features

MULTIPLIERS = [0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]
INITIAL_PRIOR = [.005, .01, .035, .15, .6, .15, .035, .01, .005]
DEFAULTS = dict(seed=260915, iterations=40, actor_lr=3e-4, value_lr=1e-3,
                clip=.2, gamma=.997, gae_lambda=.99, entropy=.01, epochs=4,
                batch_size=512, gradient_clip=.5, reward_profile='risk',
                low_risk_weight=2., reward_scale=10., terminal_penalty=100.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.Tanh(),
                         nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, outputs))


def risk_components(bg):
    """Kovatchev transform; return the unweighted high/low components separately."""
    bg = np.asarray(bg, dtype=np.float64)
    if bg.ndim != 1 or not len(bg) or not np.isfinite(bg).all() or (bg <= 0).any():
        raise ValueError('Training BG must be a finite positive one-dimensional series')
    f = 1.509 * (np.log(np.maximum(bg, 1.)).clip(min=0)**1.084 - 5.381)
    risk = 10. * f**2
    return np.where(f >= 0, risk, 0.), np.where(f < 0, risk, 0.)


class Trainer:
    def __init__(self, config_path, checkpoint=None):
        if not config_path.is_absolute():
            raise ValueError('--config must be an absolute path')
        self.c = dict(DEFAULTS, **json.loads(config_path.read_text()))
        c = self.c
        if c['seed'] != 260915 or c['reward_profile'] != 'risk':
            raise ValueError('Only seed 260915 and the explicit risk profile are implemented')
        if c.get('action_multipliers', MULTIPLIERS) != MULTIPLIERS:
            raise ValueError('The nine physical action multipliers are frozen')
        if c.get('teacher_kl', 0.) != 0.:
            raise ValueError('This diagnostic has no teacher KL regularization')
        if not 0 < c['gamma'] < 1 or not 0 <= c['gae_lambda'] <= 1:
            raise ValueError('Invalid discount or GAE coefficient')
        if c['low_risk_weight'] < 1 or c['terminal_penalty'] < 0:
            raise ValueError('Low risk weight must be >=1 and terminal penalty nonnegative')
        if any(c[k] <= 0 for k in ('actor_lr', 'value_lr', 'clip', 'epochs', 'batch_size',
                                  'iterations', 'reward_scale', 'gradient_clip')) or c['entropy'] < 0:
            raise ValueError('Invalid PPO optimizer or budget configuration')
        if not torch.cuda.is_available():
            raise RuntimeError('This diagnostic requires the restored CUDA/Mamba runtime')
        torch.set_num_threads(2)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        world_path = B / 'results/D05_selected_world/world.pt'
        binding = json.loads((B / 'results/D06_selected_policy/config.json').read_text())
        if sha(world_path) != binding['world_sha256']:
            raise ValueError('D05 hash differs from the retained D06 dependency binding')
        if FEATURE_DIM != len(FEATURE_NAMES):
            raise ValueError('Physiologic feature names and dimension disagree')
        wc = torch.load(world_path, map_location='cpu')
        cfg = wc['config']
        forecast_path = B / cfg['forecast_checkpoint']
        context_path = P / cfg['context_checkpoint']
        if sha(forecast_path) != cfg['forecast_sha256'] or sha(context_path) != cfg['context_sha256']:
            raise ValueError('Frozen P03/H02 dependency hash mismatch')
        self.world = Patient(forecast_path, context_path).cuda().eval()
        self.world.load_state_dict(wc['model'])
        self.world.requires_grad_(False)
        random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
        self.policy = network(306 + FEATURE_DIM, len(MULTIPLIERS)).cuda()
        self.value = network(306 + FEATURE_DIM, 1).cuda()
        nn.init.zeros_(self.policy[-1].weight)
        with torch.no_grad():
            self.policy[-1].bias.copy_(torch.tensor(INITIAL_PRIOR, device='cuda').log())
        nn.init.zeros_(self.value[-1].weight); nn.init.zeros_(self.value[-1].bias)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=c['actor_lr'])
        self.vo = torch.optim.Adam(self.value.parameters(), lr=c['value_lr'])
        self.multipliers = torch.tensor(MULTIPLIERS, device='cuda')
        self.buffer = {}; self.iteration = 0; self.count = 0
        self.output = Path(c.get('output_dir', R / 'results' / c['name'])).resolve()
        if R != self.output and R not in self.output.parents:
            raise ValueError('All PPO artifacts must remain in this research directory')
        self.provenance = dict(
            world_sha256=sha(world_path), forecast_sha256=sha(forecast_path),
            context_sha256=sha(context_path), feature_names=list(FEATURE_NAMES),
            normalization_sha256=sha(P / 'Loop数据集/训练管线_v2/prepared/normalization.json'),
            input_dim=306 + FEATURE_DIM, action_multipliers=MULTIPLIERS,
            initial_prior=INITIAL_PRIOR, teacher_kl=False,
            world_candidate_scoring=False, hidden_inputs=False,
            source_sha256={p.name: sha(p) for p in (
                Path(__file__), R / 'physiologic_features.py', B / 'world_model.py',
                B / 'forecast_model.py', P / 'RL_DITR创新_2026-09-16/ditr_model.py')})
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location='cpu')
            if state['config'] != c or state['provenance'] != self.provenance:
                raise ValueError('Checkpoint configuration or feature/weight/source provenance changed')
            self.policy.load_state_dict(state['model']); self.value.load_state_dict(state['value'])
            self.opt.load_state_dict(state['opt']['actor']); self.vo.load_state_dict(state['opt']['value'])
            self.iteration = state['iteration']; self.count = state['simulator_transitions']
            random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
            torch.set_rng_state(state['rng_cpu']); torch.cuda.set_rng_state_all(state['rng_cuda'])

    @torch.no_grad()
    def features(self, histories, anchors):
        history = np.asarray(histories, dtype=np.float32)
        anchor = np.asarray(anchors, dtype=np.float32)
        if history.ndim != 3 or history.shape[1:] != (72, 22) or not np.isfinite(history).all():
            raise ValueError('Expected a finite batch of 72x22 observable histories')
        if anchor.shape != (len(history),) or not np.isfinite(anchor).all() or (anchor <= 0).any() or (anchor > 20).any():
            raise ValueError('Expected positive observed warmup anchors, in U/h')
        extra = np.asarray(physical_features(history, anchor), dtype=np.float32)
        if extra.shape != (len(history), FEATURE_DIM) or not np.isfinite(extra).all():
            raise ValueError('Physiologic feature contract mismatch')
        x = torch.as_tensor(history, device='cuda')
        a = torch.as_tensor(anchor, device='cuda')
        z, forecast = self.world.encode(x); rate = self.world.reference_rate(x)
        # Wide executed actions are permitted here; the old +/-0.26 check is inapplicable.
        reference = self.world.reference(z, forecast, rate)
        result = torch.cat([z, reference/10, rate[:, None]/5, a[:, None]/5,
                            torch.as_tensor(extra, device='cuda')], -1)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite frozen-world features')
        return result, a

    @torch.no_grad()
    def act(self, req, evaluate=False):
        x, anchor = self.features(req['history'], req['anchors'])
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        choices = logits.argmax(-1) if evaluate else dist.sample()
        if not evaluate:
            ids = [str(i) for i in req['indices']]
            if len(ids) != len(x) or len(set(ids)) != len(ids):
                raise ValueError('One distinct trajectory index is required per training state')
            values = self.value(x).flatten(); logp = dist.log_prob(choices)
            for i, key in enumerate(ids):
                self.buffer.setdefault(key, []).append((x[i].cpu(), choices[i].cpu(),
                                                        logp[i].cpu(), values[i].cpu()))
        actions = (anchor * self.multipliers[choices]).clamp(0, 20)
        return dict(actions=actions.cpu().tolist(), action_indices=choices.cpu().tolist(),
                    entropy=dist.entropy().cpu().tolist())

    def update(self, req):
        c = self.c
        if not self.buffer or set(req['outcomes']) != set(self.buffer):
            raise ValueError('Update must provide exactly the buffered training trajectories')
        if self.iteration >= c['iterations']:
            raise ValueError('Frozen training-iteration budget exhausted')
        destination = self.output / ('policy_iter%02d.pt' % (self.iteration + 1))
        if destination.exists():
            raise FileExistsError(str(destination))
        xs=[]; acts=[]; logs=[]; advs=[]; returns=[]; rewards_all=[]
        low_all=[]; high_all=[]; bg_all=[]; terminal_count=0
        for key, entries in self.buffer.items():
            record = req['outcomes'][key]
            if type(record['terminal']) is not bool:
                raise ValueError('terminal must explicitly distinguish real termination from time limit')
            if record.get('failure_reason') not in (None, 'native_environment_done'):
                raise ValueError('Technical failure cannot become a PPO terminal reward')
            bg = np.asarray(record['bg'], dtype=np.float64)
            if len(bg) != len(entries):
                raise ValueError('One post-action BG outcome is required per buffered action')
            high, low = risk_components(bg)
            rewards = -(high + c['low_risk_weight'] * low) / (12. * c['reward_scale'])
            vals = np.array([float(e[3]) for e in entries])
            terminal = record['terminal']; terminal_count += int(terminal)
            if terminal:
                bootstrap = 0.; rewards[-1] -= c['terminal_penalty']
            else:
                # Administrative 3-day truncation still has a valid continuation state.
                with torch.no_grad():
                    f, _ = self.features([record['history']], [record['anchor']])
                    bootstrap = float(self.value(f).item())
            gae = 0.; advantage = np.empty(len(vals), dtype=np.float64)
            for t in range(len(vals)-1, -1, -1):
                next_value = bootstrap if t == len(vals)-1 else vals[t+1]
                delta = rewards[t] + c['gamma'] * next_value - vals[t]
                gae = delta + c['gamma'] * c['gae_lambda'] * gae
                advantage[t] = gae
            xs.extend(e[0] for e in entries); acts.extend(e[1] for e in entries)
            logs.extend(e[2] for e in entries); advs.extend(advantage); returns.extend(advantage+vals)
            rewards_all.extend(rewards); low_all.extend(low); high_all.extend(high); bg_all.extend(bg)
        x=torch.stack(xs).cuda(); a=torch.stack(acts).cuda(); oldlog=torch.stack(logs).cuda()
        advantages=torch.as_tensor(np.asarray(advs), dtype=torch.float32, device='cuda')
        advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
        targets=torch.as_tensor(np.asarray(returns), dtype=torch.float32, device='cuda')
        diagnostics=[]
        for _ in range(c['epochs']):
            order=torch.randperm(len(x), device='cuda')
            for start in range(0, len(x), c['batch_size']):
                ix=order[start:start+c['batch_size']]
                dist=torch.distributions.Categorical(logits=self.policy(x[ix]))
                newlog=dist.log_prob(a[ix]); logratio=newlog-oldlog[ix]; ratio=logratio.exp()
                pg=-torch.minimum(ratio*advantages[ix],
                                  ratio.clamp(1-c['clip'], 1+c['clip'])*advantages[ix]).mean()
                entropy=dist.entropy().mean(); loss=pg-c['entropy']*entropy
                vloss=(self.value(x[ix]).flatten()-targets[ix]).square().mean()
                if not torch.isfinite(loss) or not torch.isfinite(vloss):
                    raise ValueError('Nonfinite PPO loss; no checkpoint written')
                self.opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                self.opt.step()
                self.vo.zero_grad(set_to_none=True); vloss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                self.vo.step()
                diagnostics.append([float(pg), float(vloss), float(entropy),
                                    float(((ratio-1)-logratio).mean()),
                                    float(((ratio-1).abs()>c['clip']).float().mean())])
        self.iteration += 1; self.count += len(x)
        self.output.mkdir(parents=True, exist_ok=True)
        state=dict(config=c, provenance=self.provenance, model=self.policy.state_dict(),
                   value=self.value.state_dict(), opt=dict(actor=self.opt.state_dict(), value=self.vo.state_dict()),
                   iteration=self.iteration, simulator_transitions=self.count,
                   rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                   rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all())
        # Exclusive creation preserves an existing result even after a duplicate update.
        with destination.open('xb') as handle:
            torch.save(state, handle)
        bg=np.asarray(bg_all); stats=np.mean(diagnostics, axis=0)
        result=dict(iteration=self.iteration, transitions=self.count, batch_transitions=len(x),
                    mean_reward=float(np.mean(rewards_all)), terminal_episodes=terminal_count,
                    truncated_episodes=len(self.buffer)-terminal_count,
                    raw_costs=dict(mean_high_risk=float(np.mean(high_all)), mean_low_risk=float(np.mean(low_all)),
                                   bg_below70_fraction=float(np.mean(bg<70)), bg_below54_fraction=float(np.mean(bg<54))),
                    policy_loss=float(stats[0]), value_loss=float(stats[1]), entropy=float(stats[2]),
                    approx_update_kl=float(stats[3]), clip_fraction=float(stats[4]),
                    sampled_action_counts=torch.bincount(a, minlength=len(MULTIPLIERS)).cpu().tolist(),
                    checkpoint=str(destination), checkpoint_sha256=sha(destination))
        self.buffer={}
        return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    args=ap.parse_args(); trainer=Trainer(args.config, args.checkpoint)
    print(json.dumps(dict(ready=True, iteration=trainer.iteration,
                          feature_dim=306+FEATURE_DIM, provenance=trainer.provenance)), flush=True)
    for line in sys.stdin:
        req=json.loads(line)
        if req['op']=='act': out=trainer.act(req)
        elif req['op']=='evaluate': out=trainer.act(req, evaluate=True)
        elif req['op']=='update': out=trainer.update(req)
        elif req['op']=='close': break
        else: raise ValueError('Unknown worker operation: '+str(req['op']))
        print(json.dumps(out, allow_nan=False), flush=True)


if __name__=='__main__':
    main()
