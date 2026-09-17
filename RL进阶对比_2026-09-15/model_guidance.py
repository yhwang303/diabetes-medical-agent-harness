"""Differentiable short-horizon physiology regularizer for real offline actor-critic updates."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from graph_model import GraphPatient
from rl_data import ROOT

def context_features(state):
 s=state.reshape(-1,72,22)
 # Observable history only. Fixed scales are documented summaries, not learned test statistics.
 return torch.stack([s[:,-1,0],s[:,-1,0]-s[:,-7,0],s[:,-12:,1].mean(1),s[:,-12:,7].sum(1)/3,s[:,-12:,8].sum(1)/3,s[:,-12:,9].sum(1)/3],1)

def status_score(normalized,mean,scale):
 bg=((normalized*scale+mean)*18).clamp_min(1);risk=10*(1.509*(torch.log(bg)**1.084-5.381)).square()
 return torch.where(bg<70,-torch.ones_like(bg),1-risk.clamp(0,15.5)/7.75)

class ModelGuidance:
 def __init__(self,cfg,device='cuda'):
  self.cfg=cfg;self.models=[];self.provenance={};self.device=device
  for name in cfg['models']:
   p=ROOT/'results'/name/'best.pt';ck=torch.load(p,map_location=device);model=GraphPatient(ck['config']).to(device);model.load_state_dict(ck['model']);model.eval().requires_grad_(False);self.models.append(model);self.provenance[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'step':ck['step'],'config':ck['config']}
  bank=ROOT/'event_bank_v1.npz';manifest=json.loads(bank.with_suffix('.json').read_text());assert hashlib.sha256(bank.read_bytes()).hexdigest()==manifest['sha256'];self.provenance['event_bank']=manifest
  with np.load(bank) as f:self.context=torch.from_numpy(f['context']).to(device);self.events=torch.from_numpy(f['events']).to(device)
  norm=json.loads((ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text());self.mean=norm['cgm_mmol_l']['mean'];self.scale=norm['cgm_mmol_l']['scale']
 def __call__(self,state,action):
  with torch.no_grad():
   query=context_features(state);dist=torch.cdist(query,self.context);near=dist.topk(self.cfg['neighbors'],largest=False).indices;choice=torch.randint(near.shape[1],(len(state),self.cfg['scenarios']),device=state.device);chosen=near.gather(1,choice);events=self.events[chosen].reshape(-1,12,6)
  count=self.cfg['scenarios'];actions=(10*(action+1))[:,None,:].expand(-1,count,12).reshape(-1,12);preds=[]
  for model in self.models:
   with torch.no_grad():z=model.encode(state);z=z[:,None,:].expand(-1,count,-1).reshape(-1,7)
   preds.append(model.rollout(z,actions,events).reshape(len(state),count,12))
  pred=torch.stack(preds);discount=torch.pow(action.new_tensor(self.cfg['gamma']),torch.arange(12,device=action.device));ret=(status_score(pred,self.mean,self.scale)*discount).sum(-1)/12
  mean_return=ret.mean((0,2));disagreement=(pred.std(0,unbiased=False)*self.scale).mean((1,2));loss=(-mean_return+self.cfg['disagreement_weight']*disagreement).mean()
  return loss,{'model_return':float(mean_return.detach().mean()),'model_disagreement_mmol_l':float(disagreement.detach().mean()),'model_penalty':float(loss.detach())}
