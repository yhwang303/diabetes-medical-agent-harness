"""Public numerical/data-contract checks; does not score control performance."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_ditr_shared import GAMMA, historical_target, status_score, corrected_policy_loss
from ditr_shared_data import MixedData
from ditr_model import DITRAgent
from ditr_losses import policy_loss
from ditr_data import tensor


class ConstantPatient:
    def encode(self, state):
        return state

    def value(self, z):
        return torch.ones((len(z), 1), device=z.device) * 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-root')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    device = args.device
    batch = {'target': torch.tensor([[5., 6., 0.], [5., 6., 0.]], device=device),
             'mask': torch.tensor([[True, True, False]] * 2, device=device),
             'terminal': torch.tensor([True, False], device=device),
             'final_state': torch.zeros(2, 72, 22, device=device)}
    result = historical_target(ConstantPatient(), batch)
    reward = status_score(batch['target']) / 12
    base = reward[0, 0] + GAMMA * reward[0, 1]
    torch.testing.assert_close(result[0], base)
    torch.testing.assert_close(result[1], base + GAMMA ** 2 * 3)
    info = {'terminal_zero_bootstrap': True, 'truncation_retains_bootstrap': True,
            'masked_tail_does_not_discount': True, 'device': device}
    if args.shared_root:
        data = MixedData(Path(args.shared_root), seed=260915)
        raw = data.batch(128)
        assert int(raw['source_is_simulation'].sum()) == 64
        assert np.all(raw['k'] < raw['length'])
        info.update(mixed_shape=list(raw['state'].shape), simulation_samples=64,
                    loop_samples=64, loop_origins=data.loop.total,
                    simulation_transitions=data.shared.size,
                    genuine_terminal_samples=int(raw['terminal'].sum()))
        if args.checkpoint:
            ck = torch.load(args.checkpoint, map_location='cpu')
            agent = DITRAgent(**ck['config']['model']).to(device).eval()
            agent.load_state_dict(ck['agent'], strict=True)
            small = tensor({k: v[:2] for k, v in raw.items()}, device=device)
            small['terminal'].zero_()
            with torch.no_grad():
                torch.manual_seed(260915)
                previous, _ = policy_loss(agent, small, 12)
                torch.manual_seed(260915)
                corrected, _ = corrected_policy_loss(agent, small, 12)
            torch.testing.assert_close(corrected, previous, atol=1e-6, rtol=1e-6)
            info['nonterminal_original_legacy_loss_equivalent'] = True
    print(json.dumps(info, allow_nan=False))


if __name__ == '__main__':
    main()
