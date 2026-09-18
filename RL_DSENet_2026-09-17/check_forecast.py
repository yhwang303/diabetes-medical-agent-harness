"""Real CUDA checks; never replace Mamba with a stand-in."""
import json
from pathlib import Path
import torch
from types import SimpleNamespace
from forecast_model import Forecast, ROOT
from models.DSENet import Fusion_Head, Model
from layers.Local_Stream import _ScaledDotProductAttention

def main():
    torch.manual_seed(260915)
    model = Forecast().cuda().eval()
    x = torch.randn(3, 72, 22, device='cuda')
    x[:, :, 5] = 1
    x[:, ::4, 5] = 0
    pred = model(x)
    assert pred.shape == (3, 48) and torch.isfinite(pred).all()
    altered = x.clone()
    altered[:, ::4, 0] = 12345
    torch.testing.assert_close(pred, model(altered))
    loss = pred.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert model.backbone.long_encoder.layers[0].fwd.mamba.in_proj.weight.grad.abs().sum() > 0
    assert model.backbone.long_encoder.layers[0].bwd.mamba.in_proj.weight.grad.abs().sum() > 0
    assert model.backbone.router.W_w.weight.grad.abs().sum() > 0
    torch.testing.assert_close(pred[:1], model(x[:1]), atol=1e-4, rtol=1e-4)
    missing = x.clone(); missing[:, :, 5] = 0
    try:
        model(missing)
    except ValueError:
        pass
    else:
        raise AssertionError('All-missing history accepted')
    constant = x.clone(); constant[:, :, 0] = 0
    assert torch.isfinite(model(constant)).all()
    dense = x.clone(); dense[:, :, 5] = 1
    legacy_config = dict(model.config, revin=True)
    original_revin = Model(SimpleNamespace(**legacy_config)).cuda().eval()
    original_revin.load_state_dict(model.backbone.state_dict())
    physical = dense[:, :, :1]*model.scale+model.mean
    torch.testing.assert_close(model(dense), original_revin(physical, None, None).squeeze(-1), atol=1e-4, rtol=1e-4)
    # Unbound helper has no object dependencies; check intended local mask.
    mask = _ScaledDotProductAttention.get_local_mask(None, 5, 3)
    assert mask.shape == (5, 5) and not mask.diag().any()
    assert not mask[2, 1] and mask[2, 0]
    assert not _ScaledDotProductAttention.get_local_mask(None, 1, 3).any()
    for concat in (True, False):
        for individual in (True, False):
            head = Fusion_Head(concat, individual, 2, 2, 28, 16, 12, 7).cuda()
            y = head(torch.randn(3, 2, 4, 4, device='cuda'), torch.randn(3, 2, 4, 3, device='cuda'),
                     torch.full((3,), .4, device='cuda'), torch.full((3,), .6, device='cuda'))
            assert y.shape == (3, 2, 7) and torch.isfinite(y).all()
            y.sum().backward()
    result = {'status': 'passed', 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'parameters': sum(p.numel() for p in model.parameters()),
              'checks': ['real_Mamba_CUDA_forward_backward', 'both_Mamba_directions_receive_gradient',
                         'router_gradient', 'missing_placeholder_invariance', 'all_missing_rejected',
                         'constant_finite', 'dense_mask_matches_original_RevIN', 'batch_independence_eval', 'local_mask_direction',
                         'four_fusion_branches_forward_backward'], 'medical_evidence': False}
    (ROOT / 'checks').mkdir(exist_ok=True)
    (ROOT / 'checks/forecast_cuda.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
