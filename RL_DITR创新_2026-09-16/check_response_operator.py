"""Causality, identity and convolution checks for the response mechanism."""
import json
import torch
from ditr_data import ROOT
from response_operator import ResponseOperator

torch.manual_seed(260915);m=ResponseOperator(conditioned=False);z=torch.zeros(2,256)
with torch.no_grad():m.coefficients.copy_(torch.linspace(-.02,.03,8))
actions=torch.randn(2,48);pred=m(z,actions);kernel=m.basis@m.coefficients;manual=torch.stack([torch.stack([(kernel[:t+1]*a[:t+1].flip(0)).sum() for t in range(48)]) for a in actions])
torch.testing.assert_close(pred,manual);torch.testing.assert_close(m(z,torch.zeros_like(actions)),torch.zeros_like(actions))
changed=actions.clone();changed[:,24:]+=5;torch.testing.assert_close(m(z,changed)[:,:24],pred[:,:24]);loss=pred.square().mean();loss.backward();assert torch.isfinite(m.coefficients.grad).all() and m.coefficients.grad.abs().sum()>0
result={'status':'passed','zero_intervention_exact':True,'no_future_action_influence':True,'manual_causal_convolution_parity':True,'finite_nonzero_gradient':True,'scope':'mechanical checks only, no physiological or efficacy guarantee'}
(ROOT/'checks/response_operator_mechanics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
