"""Validate each patch/stride setting with real Mamba forward and gradients."""
import json
import torch
from forecast_model import Forecast, ROOT, DEFAULT

GRID = {
    'P01_g12s6_l2s1': dict(m_patch_len=12,m_stride=6,patch_len=2,stride=1),
    'P02_g24s6_l2s1': dict(m_patch_len=24,m_stride=6,patch_len=2,stride=1),
    'P03_g12s3_l2s1': dict(m_patch_len=12,m_stride=3,patch_len=2,stride=1),
    'P04_g12s6_l4s1': dict(m_patch_len=12,m_stride=6,patch_len=4,stride=1),
    'P05_g12s6_l2s2': dict(m_patch_len=12,m_stride=6,patch_len=2,stride=2),
}

def main():
    torch.set_num_threads(4);torch.manual_seed(260915);results={}
    (ROOT/'configs').mkdir(exist_ok=True)
    for name,overrides in GRID.items():
        config=dict(DEFAULT,**overrides);model=Forecast(config).cuda().eval()
        x=torch.randn(3,72,22,device='cuda');x[:,:,5]=1;x[:,::5,5]=0
        y=model(x);assert y.shape==(3,48) and torch.isfinite(y).all()
        y.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        results[name]={'status':'passed','parameters':sum(p.numel() for p in model.parameters()),'config':overrides}
        (ROOT/'configs'/(name+'.json')).write_text(json.dumps(overrides,indent=2))
        del model
    (ROOT/'checks/patch_grid_cuda.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
