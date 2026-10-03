"""Real-reward PPO over frozen WorldModelV2 action-response features.

Only act/update are inherited from the tested wide-action PPO. Its initializer
is deliberately not called: no D05/H02 weights or old actor are instantiated.
The new actor evaluates all nine fixed 60-minute plans at every decision.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

from ppo_wide_worker import Trainer as WideTrainer, network, MULTIPLIERS, INITIAL_PRIOR
from physiologic_features import FEATURE_DIM, FEATURE_NAMES, features as physical_features
from world_control_worker import load_world, quantile_weights, sha
from train_world_policy import validate_config

R = Path(__file__).resolve().parent
P = R.parent
INPUT_DIM = 195
PLAN_FEATURE_NAMES = ('median_min_over100', 'median_6h_endpoint_over100',
                      'q05_min_over100_or_point_median', 'p_bg_below70',
                      'p_bg_below54', 'expected_weighted_mean_risk_over10')


def candidate_plans(anchors, multipliers):
    """N x 9 x 72 U/h; first 12 steps at multiplier, remaining 60 at anchor."""
    plans = anchors[:, None, None].expand(-1, len(MULTIPLIERS), 72).clone()
    plans[:, :, :12] = (anchors[:, None] * multipliers[None, :]).clamp(0, 20)[:, :, None]
    return plans


def response_features(output, weights):
    """Nine sets of six summaries; marginal quadrature, never joint-tail CVaR."""
    q = output['cgm_quantiles_mgdl']
    median = output['cgm_median_mgdl']
    p70, p54 = output['p_bg_below70'], output['p_bg_below54']
    if (q.shape[1:] != (9, 72, len(weights)) or median.shape != q.shape[:3]
            or p70.shape != q.shape[:2] or p54.shape != p70.shape):
        raise ValueError('Expected nine complete six-hour world predictions')
    if (not torch.isfinite(q).all() or (q.diff(dim=-1) < 0).any()
            or not torch.isfinite(p70).all() or not torch.isfinite(p54).all()
            or ((p54 < 0) | (p54 > p70) | (p70 > 1)).any()):
        raise ValueError('Invalid quantiles or BG event probabilities')
    f = 1.509 * (q.clamp(20, 600).log().pow(1.084) - 5.381)
    risk = 10 * f.square() * torch.where(f < 0, 2., 1.)
    expected_risk = (risk * weights).sum(-1).mean(-1)
    return torch.stack((median.min(-1).values / 100, median[..., -1] / 100,
                        q[..., 0].min(-1).values / 100, p70, p54,
                        expected_risk / 10), -1).flatten(1)


class Trainer(WideTrainer):
    def __init__(self, config_path, checkpoint=None):
        if not config_path.is_absolute():
            raise ValueError('--config must be an absolute resolved run config')
        self.c = json.loads(config_path.read_text())
        c = self.c
        protocol = json.loads((R / 'protocol.json').read_text())
        validate_config(c, protocol, bound=True)
        self.world, world_provenance = load_world(c['world_checkpoint'])
        for config_key, provenance_key in (
                ('world_checkpoint_sha256', 'world_checkpoint_sha256'),
                ('world_training_provenance_sha256', 'training_provenance_sha256'),
                ('world_completion_sha256', 'completion_sha256')):
            if c[config_key] != world_provenance[provenance_key]:
                raise ValueError('Frozen world dependency changed: ' + config_key)
        if world_provenance['training_variant'] != c['world_variant']:
            raise ValueError('Requested point/quantile world variant mismatch')
        if world_provenance['budget_override'] and not c['allow_world_smoke_budget']:
            raise ValueError('A smoke-budget world cannot supply formal policy features')
        if FEATURE_DIM != 28 or len(FEATURE_NAMES) != FEATURE_DIM:
            raise ValueError('Frozen physiology feature contract changed')
        if self.world.context[-2].out_features != 64:
            raise ValueError('This policy requires a 64-dimensional world context')
        levels = world_provenance['quantile_levels']
        self.quantile_weights = torch.tensor(quantile_weights(levels), dtype=torch.float32, device='cuda')
        self.multipliers = torch.tensor(MULTIPLIERS, dtype=torch.float32, device='cuda')
        random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
        self.policy = network(INPUT_DIM, len(MULTIPLIERS)).cuda()
        self.value = network(INPUT_DIM, 1).cuda()
        nn.init.zeros_(self.policy[-1].weight)
        with torch.no_grad():
            self.policy[-1].bias.copy_(torch.tensor(INITIAL_PRIOR, device='cuda').log())
        nn.init.zeros_(self.value[-1].weight); nn.init.zeros_(self.value[-1].bias)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=c['actor_lr'])
        self.vo = torch.optim.Adam(self.value.parameters(), lr=c['value_lr'])
        self.buffer = {}; self.iteration = 0; self.count = 0
        self.output = Path(c['output_dir']).resolve()
        self.output.relative_to((R / 'results').resolve())
        sources = [Path(__file__).resolve(), R / 'train_world_policy.py', R / 'ppo_wide_worker.py',
                   R / 'physiologic_features.py', R / 'world_control_worker.py', R / 'protocol.json']
        self.provenance = dict(
            controller_family='frozen_world_features_real_PPO', input_dim=INPUT_DIM,
            world=world_provenance, feature_layout=dict(context=64, dsenet_4h=48,
                physiology=28, anchor=1, candidate_responses=54),
            physiology_feature_names=list(FEATURE_NAMES), plan_feature_names=list(PLAN_FEATURE_NAMES),
            candidate_plan=dict(multipliers=MULTIPLIERS, intervention_steps=12, horizon_steps=72,
                                remaining_action='observed_warmup_anchor', execute_steps=1),
            action_multipliers=MULTIPLIERS, initial_prior=INITIAL_PRIOR, teacher_kl=False,
            world_frozen=True, p03_frozen=True, model_imagined_rollouts=False,
            BG_reward_only=True, hidden_inference_inputs=False,
            point_q05_channel='median' if c['world_variant'] == 'point' else 'quantile_0.05',
            risk_feature=dict(high_weight=1., low_weight=2., glucose_clip_mg_dl=[20., 600.],
                              integration='midpoint quantile bins with constant endpoint tails'),
            source_sha256={str(p.relative_to(P)): sha(p) for p in sources},
            runtime=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), precision=c['precision'], threads=2,
                         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                         cudnn_deterministic=torch.backends.cudnn.deterministic))
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location='cpu')
            if state['config'] != c or state['provenance'] != self.provenance:
                raise ValueError('Policy checkpoint config or frozen feature provenance changed')
            self.policy.load_state_dict(state['model'], strict=True)
            self.value.load_state_dict(state['value'], strict=True)
            self.opt.load_state_dict(state['opt']['actor']); self.vo.load_state_dict(state['opt']['value'])
            self.iteration = state['iteration']; self.count = state['simulator_transitions']
            random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
            torch.set_rng_state(state['rng_cpu']); torch.cuda.set_rng_state_all(state['rng_cuda'])

    @torch.no_grad()
    def features(self, histories, anchors):
        history = np.asarray(histories, dtype=np.float32)
        anchor = np.asarray(anchors, dtype=np.float32)
        if (history.ndim != 3 or history.shape[1:] != (72, 22) or len(history) == 0
                or not np.isfinite(history).all() or anchor.shape != (len(history),)
                or not np.isfinite(anchor).all() or ((anchor <= 0) | (anchor > 20)).any()):
            raise ValueError('Expected finite observed histories and positive warmup anchors <=20 U/h')
        extra = physical_features(history, anchor)
        h = torch.as_tensor(history, device='cuda')
        a = torch.as_tensor(anchor, device='cuda')
        cache = self.world.encode(h, a)
        output = self.world.predict(cache, candidate_plans(a, self.multipliers))
        response = response_features(output, self.quantile_weights)
        features = torch.cat((cache['context'], cache['dsenet_cgm_4h_mgdl'] / 100,
                              torch.as_tensor(extra, device='cuda'), a[:, None] / 2, response), -1)
        if features.shape != (len(history), INPUT_DIM) or not torch.isfinite(features).all():
            raise ValueError('World policy feature contract mismatch')
        return features, a

    def act(self, req, evaluate=False):
        required = {'op', 'history', 'anchors'} | (set() if evaluate else {'indices'})
        if set(req) != required:
            raise ValueError('Inference accepts only history and anchors; labels and hidden fields are forbidden')
        return super().act(req, evaluate=evaluate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args(); trainer = Trainer(args.config, args.checkpoint)
    print(json.dumps(dict(ready=True, iteration=trainer.iteration,
                          feature_dim=INPUT_DIM, provenance=trainer.provenance), allow_nan=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request['op'] == 'close':
            break
        try:
            if request['op'] == 'act': result = trainer.act(request)
            elif request['op'] == 'evaluate': result = trainer.act(request, evaluate=True)
            elif request['op'] == 'update': result = trainer.update(request)
            else: raise ValueError('Unknown worker operation')
        except Exception as error:
            print(json.dumps(dict(error=repr(error))), flush=True)
            raise  # A partial failed update may not be retried on this worker.
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
