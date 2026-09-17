"""Independent H2NCM-inspired seven-node discrete dynamics; no author code is copied."""
import torch
from torch import nn

class GraphPatient(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.cfg=cfg;hidden=cfg.get('history_width',128);width=cfg.get('node_width',64)
  self.encoder=nn.LSTM(22,hidden,num_layers=2,batch_first=True);self.initial=nn.Linear(hidden,6)
  # z[0:7], basal_Uh/2, normalized_bolus, bolus_mask, normalized_exercise, exercise_mask, normalized_carb, carb_mask
  parents=[[0,3,4,5,6],[1,7,8,9],[1,2],[2,3],[4,10,11],[4,5],[6,12,13]]
  mask=torch.zeros(7,14)
  for i,columns in enumerate(parents):mask[i,columns]=1
  if cfg.get('dense_ablation'):mask[:]=1
  self.register_buffer('parent_mask',mask)
  self.w1=nn.Parameter(torch.empty(7,14,width));self.b1=nn.Parameter(torch.zeros(7,1,width));self.w2=nn.Parameter(torch.empty(7,width,width));self.b2=nn.Parameter(torch.zeros(7,1,width));self.w3=nn.Parameter(torch.empty(7,width,1));self.b3=nn.Parameter(torch.zeros(7,1,1))
  for k in range(7):nn.init.xavier_uniform_(self.w1[k]);nn.init.xavier_uniform_(self.w2[k]);nn.init.uniform_(self.w3[k],-.001,.001)
 def encode(self,history):
  history=history.reshape(-1,72,22)
  if self.cfg.get('hide_modalities'):
   history=history.clone();history[:,:,[3,4,8,9,13,14,18,19]]=0
  _,(h,_)=self.encoder(history);latent=torch.tanh(self.initial(h[-1]));return torch.cat([history[:,-1,0:1],latent],-1)
 def transition(self,z,basal,events):
  # Events are normalized value + explicit observed mask pairs, never inferred absence.
  ext=torch.cat([basal[:,None]/2,events[:,0:2],events[:,4:6],events[:,2:4]],-1)
  x=torch.cat([z,ext],-1)[None,:,:]*self.parent_mask[:,None,:]
  h=torch.tanh(torch.bmm(x,self.w1)+self.b1);h=torch.tanh(torch.bmm(h,self.w2)+self.b2);delta=(torch.bmm(h,self.w3)+self.b3).squeeze(-1).transpose(0,1)
  return z+delta
 def rollout(self,z,actions,events):
  if self.cfg.get('future_events') is False:events=torch.zeros_like(events)
  if self.cfg.get('hide_modalities'):
   events=events.clone();events[:,:,2:]=0
  pred=[]
  for t in range(actions.shape[1]):z=self.transition(z,actions[:,t],events[:,t]);pred.append(z[:,0])
  return torch.stack(pred,1)
 def forward(self,history,actions,events):return self.rollout(self.encode(history),actions,events)

def finite_difference_monotonic_penalty(means,candidates):
 spacing=candidates[:,1:]-candidates[:,:-1];valid=spacing>1e-6
 slopes=(means[:,1:]-means[:,:-1])/spacing.clamp_min(1e-6)
 penalties=torch.relu(slopes).square()
 return (penalties*valid).sum()/valid.sum().clamp_min(1)

def patient_losses(model,b,alpha,phi=1.):
 z=model.encode(b['state']);pred=model.rollout(z,b['actions'],b['events']);mask=b['mask'].float()
 mse=((pred-b['target']).square()*mask).sum()/mask.sum().clamp_min(1)
 rank=pred.new_zeros(())
 if model.cfg.get('rank_weight',alpha)>0:
  # Same history, same exogenous-event scenario; labels are priors, not observed counterfactuals.
  base=b['actions'][:,0];candidates=torch.stack([base*.5,base,(base*1.5+.05).clamp(max=20)],1)
  acts=candidates[:,:,None].expand(-1,-1,pred.shape[1]);ez=z[:,None,:].expand(-1,3,-1).reshape(-1,7);events=b['events'][:,None,:,:].expand(-1,3,-1,-1).reshape(-1,pred.shape[1],6)
  counter=model.rollout(ez,acts.reshape(-1,pred.shape[1]),events).reshape(-1,3,pred.shape[1]);later=mask.clone();later[:,:3]=0;valid=later.sum(1)>0
  if valid.any():
   means=(counter*later[:,None,:]).sum(-1)/later.sum(-1,keepdim=True).clamp_min(1)
   if model.cfg.get('rank_loss')=='finite_difference_hinge':rank=finite_difference_monotonic_penalty(means[valid],candidates[valid])
   else:rank=nn.functional.cross_entropy(phi*means[valid],torch.zeros(int(valid.sum()),dtype=torch.long,device=pred.device))
 return (1-alpha)*mse+model.cfg.get('rank_weight',alpha)*rank,{'mse_normalized':float(mse.detach()),'ranking_penalty':float(rank.detach())}
