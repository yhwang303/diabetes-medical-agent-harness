"""GFP (ICLR 2026) PyTorch task adaptation of the MIT-licensed author implementation.

Official: Simple-Robotics/guided-flow-policy, agents/gfp.py. The selected
configuration uses actor targets, actor-reference softmax guidance, and
normalized Q loss. No online phase or simulator samples are used here.
"""
import copy
import math
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'RL进阶对比_2026-09-15'))
from rl_algorithms import MLP, TwinQ, masked_mean

class Flow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(torch.arange(32)*(-math.log(10000)/31)))
        self.network = MLP(1584+1+64,cfg['width'],cfg['layers'],style='fql')
    def forward(self, s, x, t):
        angle=t*self.frequencies
        return self.network(torch.cat([s,x,angle.sin(),angle.cos()],-1))

class Agent(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.algo='gfp'
        self.actor=MLP(1585,cfg['width'],cfg['layers'],style='fql')
        self.flow=Flow(cfg)
        self.q=TwinQ(cfg['width'],cfg['layers'],'fql')
        self.q_target=copy.deepcopy(self.q).requires_grad_(False)
        # Same train-only mean initialization used by the frozen FQL comparator.
        nn.init.constant_(self.actor[-1].bias,cfg['initial_action_center'])
        nn.init.constant_(self.flow.network[-1].bias,cfg['initial_action_center'])
    def optimizers(self):
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=self.cfg['actor_lr'])
        self.flow_opt=torch.optim.Adam(self.flow.parameters(),lr=self.cfg['actor_lr'])
        self.q_opt=torch.optim.Adam(self.q.parameters(),lr=self.cfg['critic_lr'])
    def act(self,s,noise=None):
        if noise is None:noise=torch.randn_like(s[:,:1])
        return self.actor(torch.cat([s,noise],-1)).clamp(-1,1)
    @torch.no_grad()
    def flow_action(self,s,z):
        x=z.clone();n=self.cfg['flow_steps']
        for k in range(n):x=x+self.flow(s,x,torch.full_like(x,k/n))/n
        return x.clamp(-1,1)
    @torch.no_grad()
    def guidance(self,s,a,actor_a,lam):
        delta=self.q_target(s,a).mean(-1,keepdim=True)-self.q_target(s,actor_a).mean(-1,keepdim=True)
        return torch.sigmoid(lam*delta/self.cfg['eta_temperature'])
    def critic_loss(self,b):
        with torch.no_grad():
            target=b['reward']+self.cfg['gamma']*self.q_target(b['next_state'],self.act(b['next_state'])).mean(-1,keepdim=True)
        q=self.q(b['state'],b['action'])
        return masked_mean((q-target).square().mean(-1,keepdim=True),b['q_valid']),q,target
    def update(self,b,step):
        s,a=b['state'],b['action'];m=b['q_valid'];c=self.cfg
        for opt in (self.actor_opt,self.flow_opt,self.q_opt):opt.zero_grad(set_to_none=True)
        q_loss,q,target=self.critic_loss(b)
        noise=torch.randn_like(a);teacher=self.flow_action(s,noise)
        raw=self.actor(torch.cat([s,noise],-1));distill=(raw-teacher).square().mean()
        self.q.requires_grad_(False)
        actor_q=self.q(s,raw.clamp(-1,1)).mean(-1,keepdim=True)
        lam=masked_mean(actor_q.abs(),m).detach().clamp_min(1e-6).reciprocal()
        actor_loss=c['alpha']*distill-lam*masked_mean(actor_q,m)
        weight=self.guidance(s,a,raw.detach().clamp(-1,1),lam)
        z=torch.randn_like(a);t=torch.rand_like(a);x=(1-t)*z+t*a
        flow_error=(self.flow(s,x,t)-(a-z)).square()
        flow_loss=(weight*flow_error).mean()
        actor_loss.backward();self.q.requires_grad_(True)
        flow_loss.backward();q_loss.backward()
        # Author target_update reads the pre-optimizer critic, a one-step lag.
        with torch.no_grad():
            for src,dst in zip(self.q.parameters(),self.q_target.parameters()):dst.lerp_(src,c['tau'])
        for opt in (self.actor_opt,self.flow_opt,self.q_opt):opt.step()
        return {'actor_loss':float(actor_loss),'critic_loss':float(q_loss),'flow_loss':float(flow_loss),
                'distill_loss':float(distill),'guidance_mean':float(weight.mean()),'q_mean':float(q.mean()),'td_rows':int(m.sum())}
