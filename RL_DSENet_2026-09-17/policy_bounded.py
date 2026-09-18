"""Finite-horizon model-based policy over seven explicitly timed plans.

This is an RL-DITR task adaptation, not an exact reproduction of its policy:
we integrate rewards on the DSENet world trajectory and omit an uncalibrated V.
"""
import torch
from torch import nn

def candidate_actions(rate, anchor):
    patterns=torch.zeros(7,48,device=rate.device)
    for block in range(3):
        patterns[1+2*block,block*16:(block+1)*16]=-.25
        patterns[2+2*block,block*16:(block+1)*16]=.25
    return (anchor[:,None,None]+patterns[None]).clamp(0,20)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(306,128),nn.Tanh(),nn.Linear(128,7))
        self.register_buffer('prior',torch.tensor([.4,.1,.1,.1,.1,.1,.1]))
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():self.network[-1].bias.copy_(self.prior.log())

    def forward(self,context,reference,rate,anchor):
        return self.network(torch.cat([context,reference/10,rate[:,None]/5,anchor[:,None]/5],-1))


def history_anchor(history):
    rates=(history[:,:,1]*0.14462788945609448+0.09945811581924525)*12
    current=rates[:,-1]
    eligible=(history[:,:,6]>.5)&((rates-current[:,None]).abs()<=.25)
    index=eligible.long().argmax(1)
    assert eligible.any(1).all()
    return rates.gather(1,index[:,None]).squeeze(1)
