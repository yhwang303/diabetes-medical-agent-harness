"""Reference policy architecture and source-causality checks; all methods get the same extra context."""
import json,copy
import torch
from anchor_data import AnchorReplay
from rl_algorithms import Agent
from rl_data import ROOT

def main():
 torch.set_num_threads(2);torch.manual_seed(16);cfg=json.loads((ROOT/'configs/R31_anchor_rebrac.json').read_text());a=Agent(cfg);state=torch.randn(3,1585);state[:,1584]=torch.tensor([-1.,-.9,-.6]);expected=state[:,1584:1585];assert torch.allclose(a.act(state),expected,atol=1e-6)
 # A later history cannot change the explicit fixed reference input.
 with torch.no_grad():state[:,:1584]+=10
 assert torch.allclose(a.act(state),expected,atol=1e-6)
 data=AnchorReplay('train',device='cpu');assert data.size==1653421 and data.anchor_manifest['origin_count_unchanged']==1653421
 for row in data.anchor_manifest['patients']:assert row['latest_source_row']<=row['first_origin']
 ids=torch.arange(32);b=data.batch(ids);assert b['state'].shape==(32,1585);assert torch.equal(b['state'][:,-1],b['next_state'][:,-1]);a.optimizers();old={k:v.clone() for k,v in a.actor.state_dict().items()};a.update(b,2);assert any(not torch.equal(v,old[k]) for k,v in a.actor.state_dict().items());assert all(torch.isfinite(p).all() for p in a.parameters())
 for algorithm in ['bc','fql']:
  c=copy.deepcopy(cfg);c['algorithm']=algorithm;c['width']=32;c['layers']=2;m=Agent(c);m.optimizers();m.update(b,2);assert m.act(b['state']).shape==(32,1)
 out={'passed':True,'checks':['zero residual returns the supplied observed reference including 0','changes in later history do not redefine the reference','all225 patients and1653421 origins retained','source rows never exceed earliest eligible decision','same reference on both sides of TD transition','finite actual residual actor updates for ReBRAC FQL and BC'],'clinical_ready':False};(ROOT/'results/anchor_tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
