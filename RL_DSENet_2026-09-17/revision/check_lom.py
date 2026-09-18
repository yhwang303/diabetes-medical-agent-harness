"""Synthetic mechanism checks; not a Loop or closed-loop performance test."""
import json
import torch
from lom_algorithm import Agent


def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(not torch.equal(x, p) for x, p in zip(before, module.parameters()))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(260915)
    cfg = dict(algorithm='lom', state_dim=12, width=16, layers=2,
               gamma=0.9912583890453033, tau=0.005, actor_lr=1e-3,
               critic_lr=3e-4, gmm_pretrain_steps=2, num_mixtures=3,
               lom_smooth_noise=0.0, initial_action_center=-0.92)
    agent = Agent(cfg)
    agent.optimizers()
    batch = dict(state=torch.randn(8, 12), next_state=torch.randn(8, 12),
                 action=torch.rand(8, 1) * .1 - .95,
                 next_action=torch.rand(8, 1) * .1 - .95,
                 reward=torch.randn(8, 1), q_valid=torch.ones(8, 1, dtype=torch.bool))
    initial = {k: snapshot(getattr(agent, k)) for k in ('gmm', 'actor', 'q', 'mode')}
    agent.update(batch, 1)
    agent.update(batch, 2)
    assert changed(initial['gmm'], agent.gmm)
    assert all(not changed(initial[k], getattr(agent, k)) for k in ('actor', 'q', 'mode'))
    frozen = snapshot(agent.gmm)
    for step in range(3, 7):
        info = agent.update(batch, step)
        assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
    assert not changed(frozen, agent.gmm)
    assert all(changed(initial[k], getattr(agent, k)) for k in ('actor', 'q', 'mode'))
    assert agent.act(batch['state']).shape == (8, 1)
    assert (agent.act(batch['state']).abs() <= 1).all()

    # A missing successor is excluded, not imputed as terminal reward.
    masked = {**batch, 'q_valid': batch['q_valid'].clone()}
    masked['q_valid'][-1] = False
    loss_before = agent.critic_loss(masked)[0]
    altered = {**masked, 'reward': masked['reward'].clone()}
    altered['reward'][-1] = 1e6
    assert torch.equal(loss_before, agent.critic_loss(altered)[0])
    # LOM's target uses logged successor actions rather than current actor actions.
    target_before = agent.critic_loss(batch)[2]
    altered = {**batch, 'next_action': torch.ones_like(batch['next_action'])}
    assert not torch.equal(target_before, agent.critic_loss(altered)[2])
    # Missing-successor origins still train the actor by logged-action BC.
    actor_before = snapshot(agent.actor)
    no_successors = {**batch, 'q_valid': torch.zeros_like(batch['q_valid']),
                     'action': torch.full_like(batch['action'], 0.9)}
    info = agent.update(no_successors, 7)
    assert changed(actor_before, agent.actor)
    assert info['td_rows'] == 0 and info['actor_loss'] > 0

    # Force distinct mode values; all states must select mode index 2.
    class FixedMode(torch.nn.Module):
        def forward(self, x):
            return (x[:, -3:] * x.new_tensor([0., 1., 2.])).sum(-1, keepdim=True)
    agent.mode = FixedMode()
    sampled, mean, selected = agent.choose_mode(batch['state'])
    assert (selected == 2).all()
    assert torch.equal(mean, agent.gmm(batch['state'])[0][:, 2:3])
    assert (sampled.abs() <= 1).all()
    print(json.dumps(dict(status='passed', checks=[
        'GMM-only pretraining', 'GMM frozen during RL', 'actor/Q/mode updates',
        'finite bounded actions', 'q_valid excludes missing targets',
        'SARSA logged next action', 'missing-successor actor BC retention',
        'best-mode selection and selected mean'],
        scope='synthetic mechanism checks, not efficacy'), indent=2))


if __name__ == '__main__':
    main()
