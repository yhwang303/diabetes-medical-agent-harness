"""Independent Loop implementation of LOM (ICLR 2025), paper Eqs. 15--18.

No upstream source is incorporated. Adaptations are recorded in
recent_baselines_research.md. A fixed GMM pretraining stage precedes SARSA,
mode-value regression and advantage-weighted imitation within one mode.
"""
import copy
import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'RL进阶对比_2026-09-15'))
from rl_algorithms import MLP, masked_mean


DEFAULTS = {
    'gmm_pretrain_steps': 10000,
    'num_mixtures': 10,
    'lom_temperature': 0.2,
    'lom_weight_clip': 50.0,
    'lom_smooth_noise': 0.2,
    'lom_noise_clip': 0.5,
    'gmm_lr': 1e-3,
    'mode_lr': 1e-3,
    'policy_delay': 2,
}


class MixtureDensity(nn.Module):
    def __init__(self, state_dim, components):
        super().__init__()
        self.components = components
        self.net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, components * 3))

    def forward(self, state):
        raw_mean, raw_log_sd, logits = self.net(state).chunk(3, dim=-1)
        means = raw_mean.tanh().clamp(-0.9999, 0.9999)
        sd = raw_log_sd.clamp(-15, 0).exp()
        return means, sd, logits.log_softmax(dim=-1)

    def nll(self, state, action):
        means, sd, log_weights = self(state)
        log_p = torch.distributions.Normal(means, sd).log_prob(action)
        return -(log_p + log_weights).logsumexp(dim=-1).mean()


class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**DEFAULTS, **cfg}
        self.algo = 'lom'
        self.state_dim = cfg.get('state_dim', 1584)
        self.components = self.cfg['num_mixtures']
        width, layers = cfg['width'], cfg['layers']
        self.gmm = MixtureDensity(self.state_dim, self.components)
        self.actor = MLP(self.state_dim, width, layers, tanh=True)
        if 'initial_action_center' in cfg:
            nn.init.constant_(self.actor[-2].bias,
                              math.atanh(cfg['initial_action_center']))
        # One behavior-Q network, as in the paper's Algorithm 1 (K=1).
        self.q = MLP(self.state_dim + 1, width, layers)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.mode = MLP(self.state_dim + self.components, width, layers)

    def optimizers(self):
        c = self.cfg
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c['actor_lr'])
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=c['critic_lr'])
        self.gmm_opt = torch.optim.Adam(self.gmm.parameters(), lr=c['gmm_lr'])
        self.mode_opt = torch.optim.Adam(self.mode.parameters(), lr=c['mode_lr'])

    def act(self, state, noise=None):
        return self.actor(state)

    def behavior_q(self, state, action, target=False):
        return (self.q_target if target else self.q)(torch.cat((state, action), -1))

    def mode_inputs(self, state):
        n, k = len(state), self.components
        expanded = state[:, None, :].expand(n, k, self.state_dim).reshape(n * k, -1)
        indicators = torch.eye(k, device=state.device, dtype=state.dtype)
        indicators = indicators.expand(n, k, k).reshape(n * k, k)
        return expanded, torch.cat((expanded, indicators), -1)

    @torch.no_grad()
    def choose_mode(self, state):
        _, mode_input = self.mode_inputs(state)
        selected = self.mode(mode_input).reshape(len(state), self.components).argmax(-1)
        means, sd, _ = self.gmm(state)
        index = selected[:, None]
        mean = means.gather(1, index)
        sigma = sd.gather(1, index)
        return (mean + sigma * torch.randn_like(mean)).clamp(-1, 1), mean, selected

    def critic_loss(self, batch):
        c = self.cfg
        with torch.no_grad():
            noise = (torch.randn_like(batch['next_action']) * c['lom_smooth_noise'])
            next_action = (batch['next_action'] + noise.clamp(
                -c['lom_noise_clip'], c['lom_noise_clip'])).clamp(-1, 1)
            target = batch['reward'] + c['gamma'] * self.behavior_q(
                batch['next_state'], next_action, target=True)
        pred = self.behavior_q(batch['state'], batch['action'])
        # Censored/gapped rows are omitted from TD, never made terminal examples.
        return masked_mean((pred - target).square(), batch['q_valid']), pred, target

    def update(self, batch, step):
        c = self.cfg
        state, mask = batch['state'], batch['q_valid']
        if step <= c['gmm_pretrain_steps']:
            self.gmm_opt.zero_grad(set_to_none=True)
            loss = self.gmm.nll(state, batch['action'])
            loss.backward()
            self.gmm_opt.step()
            return {'gmm_loss': float(loss), 'stage_gmm': 1.0, 'td_rows': 0}

        self.gmm.requires_grad_(False)
        q_loss, q, target = self.critic_loss(batch)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        expanded_state, mode_input = self.mode_inputs(state)
        with torch.no_grad():
            means, sd, _ = self.gmm(state)
            # One Monte Carlo draw per mode approximates Eq. 17's expectation.
            samples = means + sd * torch.randn_like(means)
            mode_target = self.behavior_q(expanded_state, samples.reshape(-1, 1))
        mode_mask = mask.expand(-1, self.components).reshape(-1, 1)
        mode_loss = masked_mean((self.mode(mode_input) - mode_target).square(), mode_mask)
        self.mode_opt.zero_grad(set_to_none=True)
        mode_loss.backward()
        self.mode_opt.step()

        chosen, mode_mean, _ = self.choose_mode(state)
        with torch.no_grad():
            advantage = self.behavior_q(state, chosen) - self.behavior_q(state, mode_mean)
            # Clipping in log space equals min(exp(A/beta), C) without overflow.
            weights = (advantage / c['lom_temperature']).clamp_max(
                math.log(c['lom_weight_clip'])).exp()
        prediction = self.act(state)
        valid_mode_error = weights * (prediction - chosen).square()
        missing_target_bc = (prediction - batch['action']).square()
        # Preserve every origin: without valid successor evidence, retain logged
        # behavior cloning instead of giving an unreliable Q-weighted label.
        actor_loss = torch.where(mask.bool(), valid_mode_error, missing_target_bc).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        rl_step = step - c['gmm_pretrain_steps']
        if rl_step % c['policy_delay'] == 0:
            with torch.no_grad():
                for target_p, source_p in zip(self.q_target.parameters(), self.q.parameters()):
                    target_p.lerp_(source_p, c['tau'])
        return {
            'critic_loss': float(q_loss), 'mode_loss': float(mode_loss),
            'actor_loss': float(actor_loss), 'weights': float(masked_mean(weights, mask)),
            'q_mean': float(masked_mean(q, mask)),
            'target_mean': float(masked_mean(target, mask)),
            'td_rows': int(mask.sum()), 'stage_gmm': 0.0,
        }
