"""Optional synthetic capacity probe on the user's CUDA machine, not efficacy training.

No installation/network operations. Requires an existing CUDA PyTorch environment.
"""
import argparse
import json
import time
import torch
from torch import nn


class CapacityProbe(nn.Module):
    def __init__(self, width=256, heads=8, layers=3):
        super().__init__()
        self.project = nn.Linear(14, width)
        self.history = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width*4, batch_first=True), layers)
        self.action_time = nn.Linear(2, width)
        self.dynamics = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width*4, batch_first=True), layers)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out))
                                   for out in [2, 1, 1]])

    def forward(self, history, actions):
        state = self.history(self.project(history))[:, -1:]
        tokens = state + self.action_time(actions)
        n = tokens.shape[1]
        causal_mask = torch.triu(torch.ones(n, n, device=tokens.device, dtype=torch.bool), diagonal=1)
        latent = self.dynamics(tokens, mask=causal_mask)
        return sum(head(latent).float().square().mean() for head in self.heads)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--unroll', type=int, default=12)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--output', default='cuda_capacity_result.json')
    a = p.parse_args()
    if min(a.batch_size, a.unroll, a.steps) < 1:
        raise ValueError('Sizes and steps must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; CPU/MPS cannot verify RTX 4090 memory')
    torch.manual_seed(20260914)
    model = CapacityProbe().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = torch.randn(a.batch_size, 72, 14, device='cuda')
    actions = torch.randn(a.batch_size, a.unroll, 2, device='cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            loss = model(history, actions)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            loss = model(history, actions)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'parameters': sum(p.numel() for p in model.parameters()),
              'batch': a.batch_size, 'history': 72, 'unroll': a.unroll,
              'dtype': str(dtype), 'max_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
              'max_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
              'seconds_per_step': (time.perf_counter()-started)/a.steps,
              'scope': 'synthetic capacity only; no clinical learning; excludes frozen external predictor, ensemble and beam search'}
    with open(a.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
