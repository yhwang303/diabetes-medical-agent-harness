"""Conservative online actor refinement from real simulator rewards.
Frozen DSENet/world features; no simulator BG or patient ID in inference inputs.
"""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parent;P=R.parent;B=P/'RL_DSENet_2026-09-17'
sys.path.insert(0,str(P/'RL_DSENet_局部风险_2026-09-21'))
from worker import Frozen

def physical(logits):return torch.stack([logits[:,[0,3,4,5,6]].max(-1).values,logits[:,1],logits[:,2]],-1)

class Trainer:
 def __init__(self):
  self.c=json.loads((R/'configs'/(sys.argv[sys.argv.index('--config')+1]+'.json')).read_text());torch.manual_seed(self.c['seed']);np.random.seed(self.c['seed'])
  f=Frozen();self.world=f.world;self.policy=f.actor;self.policy.requires_grad_(True)
  self.teacher=copy.deepcopy(self.policy).requires_grad_(False)
  self.value=nn.Sequential(nn.Linear(306,128),nn.Tanh(),nn.Linear(128,1)).cuda()
  nn.init.zeros_(self.value[-1].weight);nn.init.zeros_(self.value[-1].bias)
  self.opt=torch.optim.Adam(self.policy.parameters(),lr=self.c['actor_lr']);self.vo=torch.optim.Adam(self.value.parameters(),lr=self.c['value_lr'])
  self.buffer={};self.count=0;self.iteration=0
 @torch.no_grad()
 def features(self,histories,anchors):
  x=torch.tensor(np.asarray(histories,dtype=np.float32),device='cuda');anchor=torch.tensor(anchors,device='cuda',dtype=x.dtype)
  z,f=self.world.encode(x);rate=self.world.reference_rate(x)
  assert ((rate-anchor).abs()<=.26).all(), 'persistent anchor support'
  ref=self.world.reference(z,f,rate)
  return torch.cat([z,ref/10,rate[:,None]/5,anchor[:,None]/5],-1),anchor
 @torch.no_grad()
 def act(self,req):
  x,anchor=self.features(req['history'],req['anchors']);logits=physical(self.policy.network(x));dist=torch.distributions.Categorical(logits=logits)
  actions=dist.sample();v=self.value(x).flatten();lp=dist.log_prob(actions)
  for i,key in enumerate(req['indices']):self.buffer.setdefault(key,[]).append((x[i].cpu(),actions[i].cpu(),lp[i].cpu(),v[i].cpu()))
  basal=(anchor+torch.tensor([0.,-.25,.25],device='cuda')[actions]).clamp(0,20)
  return {'actions':basal.cpu().tolist()}
 def update(self,req):
  c=self.c;xs=[];acts=[];logs=[];advs=[];returns=[];all_rewards=[];all_done=[]
  for key,entries in self.buffer.items():
   records=req['outcomes'][str(key)];bg=np.array(records['bg']);assert len(bg)==len(entries)
   rewards=(-((bg-120)/60)**2-4*(np.maximum(70-bg,0)/16)**2).clip(-20,0)/12
   vals=np.array([float(e[3]) for e in entries]);term=bool(records['terminal'])
   if term:bootstrap=0.;rewards[-1]-=20./12
   else:
    with torch.no_grad():f,_=self.features([records['history']],[records['anchor']]);bootstrap=float(self.value(f).item())
   gae=0.;av=np.zeros_like(vals)
   for t in range(len(vals)-1,-1,-1):
    nv=bootstrap if t==len(vals)-1 else vals[t+1]
    delta=rewards[t]+c['gamma']*nv-vals[t];gae=delta+c['gamma']*c['gae_lambda']*gae;av[t]=gae
   xs.extend(e[0] for e in entries);acts.extend(e[1] for e in entries);logs.extend(e[2] for e in entries)
   advs.extend(av);returns.extend(av+vals);all_rewards.extend(rewards);all_done.append(term)
  x=torch.stack(xs).cuda();a=torch.stack(acts).cuda();oldlog=torch.stack(logs).cuda()
  advantages=torch.tensor(np.array(advs),dtype=torch.float32,device='cuda');advantages=(advantages-advantages.mean())/advantages.std().clamp_min(1e-6)
  targets=torch.tensor(np.array(returns),dtype=torch.float32,device='cuda')
  with torch.no_grad():teacher=torch.distributions.Categorical(logits=physical(self.teacher.network(x)))
  diagnostics=[]
  for epoch in range(c['epochs']):
   order=torch.randperm(len(x),device='cuda')
   for start in range(0,len(x),c['batch_size']):
    ix=order[start:start+c['batch_size']];dist=torch.distributions.Categorical(logits=physical(self.policy.network(x[ix])))
    ratio=(dist.log_prob(a[ix])-oldlog[ix]).exp();pg=-torch.minimum(ratio*advantages[ix],ratio.clamp(1-c['clip'],1+c['clip'])*advantages[ix]).mean()
    ref=torch.distributions.Categorical(logits=teacher.logits[ix]);kl=torch.distributions.kl_divergence(ref,dist).mean()
    loss=pg+c['teacher_kl']*kl-c['entropy']*dist.entropy().mean()
    self.opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.policy.parameters(),.5);self.opt.step()
    vloss=(self.value(x[ix]).flatten()-targets[ix]).square().mean();self.vo.zero_grad(set_to_none=True);vloss.backward();torch.nn.utils.clip_grad_norm_(self.value.parameters(),1.);self.vo.step()
    assert torch.isfinite(loss) and torch.isfinite(vloss)
    diagnostics.append([float(loss),float(vloss),float(kl)])
  self.count+=len(x);self.iteration+=1
  base=torch.load(B/'results/D06_selected_policy/policy.pt',map_location='cpu');base['policy']=self.policy.state_dict();base['ppo_config']=c;base['iteration']=self.iteration;base['simulator_transitions']=self.count
  base['value']=self.value.state_dict();base['optimizers']={'actor':self.opt.state_dict(),'value':self.vo.state_dict()};base['rng_cpu']=torch.get_rng_state();base['rng_cuda']=torch.cuda.get_rng_state_all()
  path=R/'results'/c['name']/('policy_iter%02d.pt'%self.iteration);torch.save(base,path)
  result=dict(iteration=self.iteration,transitions=self.count,mean_reward=float(np.mean(all_rewards)),losses=np.mean(diagnostics,axis=0).tolist(),terminal_episodes=sum(all_done),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  self.buffer={};return result

def main():
 t=Trainer();print(json.dumps({'ready':True}),flush=True)
 for line in sys.stdin:
  req=json.loads(line)
  if req['op']=='act':out=t.act(req)
  elif req['op']=='update':out=t.update(req)
  elif req['op']=='close':break
  else:raise ValueError(req['op'])
  print(json.dumps(out,allow_nan=False),flush=True)
if __name__=='__main__':main()
