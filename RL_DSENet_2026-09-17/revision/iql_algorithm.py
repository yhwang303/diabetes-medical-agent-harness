"""IQL task adaptation; equations/order follow the author's MIT implementation.

Provenance and deliberate Loop/common-capacity changes: iql_implementation.md.
Missing futures are omitted from TD/value losses, never made terminal. They
remain ordinary Gaussian behavior-cloning samples, preserving all origins.
"""
import copy
import math

import torch
from torch import nn

from rl_algorithms import MLP


class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.get('state_dim', 1584)
        width, layers = cfg.get('width', 256), cfg.get('layers', 3)
        self.actor = MLP(dim, width, layers, tanh=True)
        if 'initial_action_center' in cfg:
            last = next(m for m in reversed(self.actor) if isinstance(m, nn.Linear))
            nn.init.constant_(last.bias, math.atanh(cfg['initial_action_center']))
        self.log_std = nn.Parameter(torch.zeros(1))
        # IQL does not introduce the LayerNorm used by the common ReBRAC critic.
        self.q1 = MLP(dim + 1, width, layers)
        self.q2 = MLP(dim + 1, width, layers)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
        self.value = MLP(dim, width, layers)

    def optimizers(self):
        c = self.cfg
        self.actor_opt = torch.optim.Adam(
            list(self.actor.parameters()) + [self.log_std], lr=c.get('actor_lr', 3e-4))
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=c.get('critic_lr', 3e-4))
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=c.get('value_lr', 3e-4))
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt, T_max=c.get('updates', 50000))

    def act(self, s, noise=None):
        # The frozen evaluation protocol uses the Gaussian mean, not samples.
        return self.actor(s)

    @torch.no_grad()
    def target_update(self):
        tau = self.cfg.get('tau', .005)
        for live, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for src, dst in zip(live.parameters(), target.parameters()):
                dst.lerp_(src, tau)

    def update(self, b, step):
        c = self.cfg
        s, a = b['state'], b['action']
        valid = b['q_valid'].reshape(-1).bool()
        n_valid = int(valid.sum())
        value_loss = s.new_zeros(())
        q_loss = s.new_zeros(())
        target_mean = s.new_zeros(())
        weights = torch.ones_like(a)
        expectile = c.get('expectile', .7)
        beta = c.get('beta', 3.)

        if n_valid:
            sv, av = s[valid], a[valid]
            sa = torch.cat([sv, av], -1)
            with torch.no_grad():
                q_target = torch.minimum(self.q1_target(sa), self.q2_target(sa))
            diff = q_target - self.value(sv)
            value_loss = (torch.where(diff > 0, expectile, 1 - expectile)
                          * diff.square()).mean()
            self.value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            self.value_opt.step()
            # Official actor update uses the newly updated value estimate.
            with torch.no_grad():
                advantage = q_target - self.value(sv)
                # Clamp before exp is mathematically equivalent to exp().min(100)
                # but avoids overflowing intermediate tensors.
                weights[valid] = (beta * advantage).clamp(max=math.log(100.)).exp()

        mean = self.actor(s)
        dist = torch.distributions.Normal(mean, self.log_std.clamp(-5., 2.).exp())
        actor_loss = -(weights * dist.log_prob(a)).sum(-1).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.actor_scheduler.step()

        if n_valid:
            with torch.no_grad():
                # No terminal factor: q_valid denotes an observed contiguous
                # transition. An unavailable next observation is not a death.
                target = b['reward'][valid] + c['gamma'] * self.value(b['next_state'][valid])
                target_mean = target.mean()
            q_loss = ((self.q1(sa) - target).square()
                      + (self.q2(sa) - target).square()).mean()
            self.q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_opt.step()
            self.target_update()

        assert torch.isfinite(actor_loss) and torch.isfinite(q_loss) and torch.isfinite(value_loss)
        return {
            'actor_loss': float(actor_loss.detach()),
            'critic_loss': float(q_loss.detach()),
            'value_loss': float(value_loss.detach()),
            'bc_mse': float((mean.detach() - a).square().mean()),
            'adv_weight_mean': float(weights.mean()),
            'target_mean': float(target_mean),
            'td_rows': n_valid,
            'actor_rows': len(s),
            'actor_lr': self.actor_opt.param_groups[0]['lr'],
        }
