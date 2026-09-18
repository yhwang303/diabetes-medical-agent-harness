"""TD3+BC (NeurIPS 2021), continuous Loop task adaptation of the paper updates."""
import copy
import math
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'RL进阶对比_2026-09-15'))
from rl_algorithms import MLP, masked_mean

class Agent(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;self.algo='td3bc'
        self.actor=MLP(1584,cfg['width'],cfg['layers'],tanh=True)
        nn.init.constant_(self.actor[-2].bias,math.atanh(cfg['initial_action_center']))
        self.q1=MLP(1585,cfg['width'],cfg['layers']);self.q2=MLP(1585,cfg['width'],cfg['layers'])
        self.actor_target=copy.deepcopy(self.actor).requires_grad_(False)
        self.q1_target=copy.deepcopy(self.q1).requires_grad_(False);self.q2_target=copy.deepcopy(self.q2).requires_grad_(False)
    def optimizers(self):
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=self.cfg['actor_lr'])
        self.q_opt=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=self.cfg['critic_lr'])
    def act(self,s,noise=None):return self.actor(s)
    def update(self,b,step):
        s,a,m=b['state'],b['action'],b['q_valid'];c=self.cfg
        with torch.no_grad():
            noise=(torch.randn_like(a)*c['target_noise']).clamp(-c['noise_clip'],c['noise_clip'])
            na=(self.actor_target(b['next_state'])+noise).clamp(-1,1);sa=torch.cat([b['next_state'],na],-1)
            target=b['reward']+c['gamma']*torch.minimum(self.q1_target(sa),self.q2_target(sa))
        sa=torch.cat([s,a],-1);q1,q2=self.q1(sa),self.q2(sa)
        loss=masked_mean((q1-target).square()+(q2-target).square(),m)
        self.q_opt.zero_grad(set_to_none=True);loss.backward();self.q_opt.step()
        info={'critic_loss':float(loss),'q_mean':float(q1.mean()),'td_rows':int(m.sum())}
        if step%c['policy_delay']==0:
            self.q1.requires_grad_(False);pa=self.actor(s);q=self.q1(torch.cat([s,pa],-1))
            scale=c['alpha']/masked_mean(q.abs(),m).detach().clamp_min(1e-6)
            aloss=(pa-a).square().mean()-scale*masked_mean(q,m)
            self.actor_opt.zero_grad(set_to_none=True);aloss.backward();self.actor_opt.step();self.q1.requires_grad_(True)
            with torch.no_grad():
                for online,target_module in ((self.actor,self.actor_target),(self.q1,self.q1_target),(self.q2,self.q2_target)):
                    for src,dst in zip(online.parameters(),target_module.parameters()):dst.lerp_(src,c['tau'])
            info['actor_loss']=float(aloss)
        return info
