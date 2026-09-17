"""RL-DITR architecture adaptation: causal fR, cross-attentive fT, glucose and WTR heads.
Continuous basal U/h and elapsed 5-minute steps replace injection/meal-slot embeddings.
No future observed features enter the rollout; future encodings are loss targets only.
"""
import math
import torch
from torch import nn

class MLP(nn.Sequential):
 def __init__(self,i,h,o): super().__init__(nn.Linear(i,h),nn.ReLU(),nn.Linear(h,o))

class PatientModel(nn.Module):
 def __init__(self,width=256,layers=3,ff=2048,dropout=.4,residual=False):
  super().__init__(); self.residual=residual; self.width=width
  t=torch.arange(128).float()[:,None]; div=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000)/width))
  pos=torch.zeros(128,width);pos[:,0::2]=torch.sin(t*div);pos[:,1::2]=torch.cos(t*div);self.register_buffer('pos',pos)
  self.input=nn.Sequential(nn.Linear(22+width,width),nn.ReLU(),nn.LayerNorm(width))
  # Sequence-first avoids the PyTorch 2.0 eval fast path; weights and math are unchanged.
  self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,8,ff,dropout,batch_first=False),layers,enable_nested_tensor=False)
  self.action=nn.Linear(1,width);self.time=nn.Linear(1,width)
  self.transition_input=nn.Sequential(nn.Linear(width*4,width),nn.ReLU(),nn.LayerNorm(width))
  self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,8,ff,dropout,batch_first=False),layers)
  self.glucose=MLP(width,width,1);self.wtr=MLP(width,width,2)
 def encode(self,x):
  b,l,_=x.shape;p=self.pos[:l].expand(b,-1,-1);z=self.input(torch.cat([x,p],-1)); mask=torch.ones(l,l,dtype=torch.bool,device=x.device).triu(1)
  return self.encoder(z.transpose(0,1),mask=mask).transpose(0,1)
 def rollout(self,state,actions,encoded=None):
  memory=self.encode(state) if encoded is None else encoded
  z=memory[:,-1];features=[];latent=[];glus=[];wtrs=[]
  initial_glu=state[:,-1,0]*2.9487731123159437+7.602434716830251
  for h in range(actions.shape[1]):
   a=self.action(actions[:,h:h+1]/5);dt=self.time(torch.full_like(actions[:,h:h+1],(h+1)/12))
   features.append(torch.cat([z,a,dt,self.pos[72+h].expand(len(z),-1)],-1))
   q=self.transition_input(torch.stack(features,1)); l=q.shape[1]
   z=self.decoder(q.transpose(0,1),memory.transpose(0,1),tgt_mask=torch.ones(l,l,dtype=torch.bool,device=q.device).triu(1))[-1]
   latent.append(z);g=self.glucose(z).squeeze(-1)
   glus.append(g+initial_glu if self.residual else g);wtrs.append(self.wtr(z))
  return torch.stack(glus,1),torch.stack(wtrs,1),torch.stack(latent,1),memory
