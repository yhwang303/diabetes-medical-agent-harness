import torch,json
from graph_model import finite_difference_monotonic_penalty
from rl_data import ROOT

a=torch.tensor([[0.,1.,2.]])
correct=torch.tensor([[3.,2.,1.]],requires_grad=True);loss=finite_difference_monotonic_penalty(correct,a);assert float(loss)==0.;loss.backward();assert torch.equal(correct.grad,torch.zeros_like(correct))
wrong=torch.tensor([[1.,2.,4.]],requires_grad=True);loss=finite_difference_monotonic_penalty(wrong,a);assert float(loss)==2.5;loss.backward();assert torch.isfinite(wrong.grad).all()
assert float(finite_difference_monotonic_penalty(torch.tensor([[100.,2.,1.]]),torch.tensor([[0.,0.,1.]])))==0.
result={'passed':True,'checks':['zero penalty and gradient once direction is correct','hand-computed positive slope penalty 2.5','equal-action edges excluded'],'clinical_ready':False};(ROOT/'results/monotonic_penalty_tests.json').write_text(json.dumps(result,indent=2));print(result)
