"""Finite-horizon model-based policy over seven explicitly timed plans.

This is an RL-DITR task adaptation, not an exact reproduction of its policy:
we integrate rewards on the DSENet world trajectory and omit an uncalibrated V.
"""
import torch
from torch import nn

def candidate_actions(rate):
    patterns=torch.zeros(7,48,device=rate.device)
    for block in range(3):
        patterns[1+2*block,block*16:(block+1)*16]=-.25
        patterns[2+2*block,block*16:(block+1)*16]=.25
    return (rate[:,None,None]+patterns[None]).clamp(0,20)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(305,128),nn.Tanh(),nn.Linear(128,7))
        self.register_buffer('prior',torch.tensor([.4,.1,.1,.1,.1,.1,.1]))
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():self.network[-1].bias.copy_(self.prior.log())

    def forward(self,context,reference,rate):
        return self.network(torch.cat([context,reference/10,rate[:,None]/5],-1))
