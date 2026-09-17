import json
import numpy as np
import torch
from graph_model import GraphPatient,patient_losses
from dynamics_data import Dynamics,ROOT

def main():
 torch.set_num_threads(2);torch.manual_seed(11);cfg={'history_width':32,'node_width':16};m=GraphPatient(cfg);state=torch.randn(2,72,22);actions=torch.ones(2,12,requires_grad=True);events=torch.zeros(2,12,6);pred=m(state,actions,events)
 early=torch.autograd.grad(pred[:,:3].sum(),actions,retain_graph=True)[0];late=torch.autograd.grad(pred[:,3].sum(),actions,retain_graph=True)[0];assert torch.equal(early,torch.zeros_like(early));assert late[:,0].abs().sum()>0
 b={'state':state,'actions':actions.detach(),'events':events,'target':torch.randn(2,12),'mask':torch.tensor([[True]*12,[True]+[False]*11])};loss,info=patient_losses(m,b,.05);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());old=float(patient_losses(m,b,0)[0]);b['target'][1,1:]=1e5;assert abs(float(patient_losses(m,b,0)[0])-old)<1e-6
 data=Dynamics('train',device='cpu');assert data.size==1653421;ids=torch.arange(8);batch=data.batch(ids);assert torch.isfinite(batch['target']).all() and torch.isfinite(batch['events']).all();row=data.arrays['start'][ids];f=data.arrays['features'];assert torch.equal(batch['events'][:,0],f[row+1][:,[2,7,3,8,4,9]]);assert torch.equal(batch['actions'][:,0],data.extra['action_grid'][row]);assert torch.allclose(batch['target'][:,0],(data.extra['glucose_grid'][row+1]-data.mean)/data.scale)
 out={'passed':True,'checks':['no direct insulin effect before 20 minutes in graph','nonzero insulin path at 20 minutes','finite factual and prior-ranking gradients for all learnable parameters','invalid future targets excluded','all 1653421 train origins retained','actual five-minute action and next-row event/target alignment'],'clinical_ready':False};(ROOT/'results/graph_tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
