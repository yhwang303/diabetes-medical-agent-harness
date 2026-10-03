"""IQL equations adapted from ikostrikov/implicit_q_learning, commit 09d7002.

The local upstream MIT license/source are recorded by replay provenance. This
is a new public-simulator task implementation, not an official medical model.

MIT License — Copyright (c) 2021 Ilya Kostrikov, Ashvin Nair, Sergey Levine
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import copy
import math

import torch
from torch import nn


def mlp(inputs, hidden, outputs=1, tanh=False):
    layers = []
    for width in hidden:
        layers += [nn.Linear(inputs, width), nn.ReLU()]
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class IQL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, hidden = config['state_dim'], config['hidden_dims']
        self.actor = mlp(dim, hidden, tanh=True)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.q1 = mlp(dim+1, hidden)
        self.q2 = mlp(dim+1, hidden)
        self.value = mlp(dim, hidden)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

    def optimizers(self):
        c = self.config
        self.actor_opt = torch.optim.Adam(list(self.actor.parameters())+[self.log_std], lr=c['actor_lr'])
        self.q_opt = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=c['critic_lr'])
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=c['value_lr'])
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=c['updates'])

    def act(self, state):
        """Deterministic Gaussian mean in [-1,1]; no sampling at evaluation."""
        return self.actor(state)

    def update(self, batch):
        c = self.config
        state, action = batch['state'], batch['action']
        state_action = torch.cat([state, action], dim=-1)
        with torch.no_grad():
            target_q = torch.minimum(self.q1_target(state_action), self.q2_target(state_action))
        difference = target_q-self.value(state)
        value_loss = (torch.where(difference > 0, c['expectile'], 1-c['expectile'])*difference.square()).mean()
        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_opt.step()
        # Author order: new V, pre-update target Q, then actor.
        with torch.no_grad():
            advantage = target_q-self.value(state)
            weights = (c['advantage_temperature']*advantage).clamp_max(math.log(c['advantage_weight_clip'])).exp()
        mean = self.actor(state)
        distribution = torch.distributions.Normal(mean, self.log_std.clamp(c['log_std_min'], c['log_std_max']).exp())
        actor_loss = -(weights*distribution.log_prob(action)).sum(-1).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.actor_scheduler.step()
        with torch.no_grad():
            bootstrap = self.value(batch['next_state'])
            # True physiological terminals only. Time limits keep final-state V.
            target = batch['reward']+c['gamma']*(~batch['terminal']).to(bootstrap.dtype)*bootstrap
        q1, q2 = self.q1(state_action), self.q2(state_action)
        q_loss = ((q1-target).square()+(q2-target).square()).mean()
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        with torch.no_grad():
            for live, frozen in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for source, destination in zip(live.parameters(), frozen.parameters()):
                    destination.lerp_(source, c['target_tau'])
        result = dict(value_loss=float(value_loss.detach()), actor_loss=float(actor_loss.detach()),
                      critic_loss=float(q_loss.detach()), advantage_mean=float(advantage.mean()),
                      advantage_weight_mean=float(weights.mean()), target_mean=float(target.mean()),
                      action_bc_mse=float((mean.detach()-action).square().mean()),
                      actor_lr=self.actor_opt.param_groups[0]['lr'])
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError('Nonfinite IQL optimization result')
        return result

    def optimizer_state(self):
        return dict(actor=self.actor_opt.state_dict(), q=self.q_opt.state_dict(),
                    value=self.value_opt.state_dict(), actor_scheduler=self.actor_scheduler.state_dict())
