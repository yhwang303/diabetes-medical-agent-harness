import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from model import PatientModel
from train import losses
class ModelTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(4);torch.manual_seed(42)
  self.device='cuda' if torch.cuda.is_available() else 'cpu'
  self.m=PatientModel().to(self.device)
  x=torch.randn(2,72,22,device=self.device)
  self.b={'state':x,'next_state':torch.randn_like(x),'action':torch.ones(2,3,device=self.device),'target':torch.ones(2,3,device=self.device)*7,'mask':torch.tensor([[1,1,1],[1,0,0]],device=self.device,dtype=torch.bool),'k':torch.tensor([2,0],device=self.device)}
 def test_causal_encoder(self):
  self.m.eval();x=self.b['state'];z=self.m.encode(x);xx=x.clone();xx[:,40:]+=100
  torch.testing.assert_close(z[:,:40],self.m.encode(xx)[:,:40],atol=2e-5,rtol=2e-5)
 def test_grad_and_no_grad_inference_agree(self):
  self.m.eval();x=self.b['state'];a=self.b['action']
  expected=self.m.rollout(x,a)[0]
  with torch.no_grad():actual=self.m.rollout(x,a)[0]
  torch.testing.assert_close(expected,actual,atol=2e-5,rtol=2e-5)
 def test_no_future_targets_in_rollout(self):
  self.m.eval();b=self.b;p=self.m.rollout(b['state'],b['action'])[0];b['next_state']+=100;b['target']+=100
  torch.testing.assert_close(p,self.m.rollout(b['state'],b['action'])[0],rtol=0,atol=0)
 def test_masked_targets_and_finite_grad(self):
  self.m.eval();loss,parts=losses(self.m,self.b);b=dict(self.b);b['target']=b['target'].clone();b['target'][~b['mask']]=1000
  ll,_=losses(self.m,b);torch.testing.assert_close(loss,ll);loss.backward()
  for name in ['encoder','decoder','glucose','wtr','action']:
   parameters=list(getattr(self.m,name).parameters());self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
 def test_checkpoint_reload(self):
  self.m.eval();x=self.b['state'];a=self.b['action'];before=self.m.rollout(x,a)[0]
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'results') as td:
   p=Path(td)/'check.pt';torch.save(self.m.state_dict(),p);second=PatientModel().to(self.device).eval();second.load_state_dict(torch.load(p,map_location=self.device));torch.testing.assert_close(before,second.rollout(x,a)[0],atol=0,rtol=0)
if __name__=='__main__':unittest.main(verbosity=2)
