"""DSENet reference-calibrated patient model, with a causal action response.

The frozen RL-DITR history encoder is auxiliary context. Its old glucose,
reward and value heads do not participate in new trajectory scores.
"""
import sys
from pathlib import Path
import torch
from torch import nn
from forecast_model import Forecast, ROOT

OLD = ROOT.parent / 'RL_DITR创新_2026-09-16'
sys.path.insert(0, str(OLD))
from ditr_model import DITRAgent, status_score
from response_operator import ResponseOperator

class Patient(nn.Module):
    def __init__(self, forecast_checkpoint, context_checkpoint):
        super().__init__()
        ck = torch.load(forecast_checkpoint, map_location='cpu')
        self.forecast = Forecast(ck['config']['model'])
        self.forecast.load_state_dict(ck['model'])
        ck = torch.load(context_checkpoint, map_location='cpu')
        agent = DITRAgent(**ck['config']['model']); agent.load_state_dict(ck['agent'])
        self.history_encoder = agent.patient
        self.forecast.requires_grad_(False); self.history_encoder.requires_grad_(False)
        self.reference_adapter = nn.Sequential(nn.Linear(256+48+1, 128), nn.Tanh(), nn.Linear(128, 48))
        nn.init.zeros_(self.reference_adapter[-1].weight); nn.init.zeros_(self.reference_adapter[-1].bias)
        self.response = ResponseOperator(width=256, hidden=64, horizon=48)

    def encode(self, history):
        self.forecast.eval(); self.history_encoder.eval()
        with torch.no_grad():
            z = self.history_encoder.encode(history)[:, -1]
            forecast = self.forecast(history)
        return z, forecast

    @staticmethod
    def reference_rate(history):
        # Reconstruct actual basal from the previous 5min interval, not nominal.
        if not (history[:, -1, 6] > .5).all():
            raise ValueError('Last delivered basal is unavailable')
        return (history[:, -1, 1]*0.14462788945609448+0.09945811581924525)*12

    def reference(self, context, forecast, rate):
        return forecast + self.reference_adapter(torch.cat([context, forecast/10, rate[:, None]/5], -1))

    def trajectories(self, context, forecast, reference_rate, actions):
        reference = self.reference(context, forecast, reference_rate)
        delta = actions-reference_rate[:, None, None]
        return reference[:, None, :actions.shape[-1]] + self.response(context, delta)

    @staticmethod
    def utility(glucose):
        discount = .9**(torch.arange(glucose.shape[-1], device=glucose.device)/12)
        return (status_score(glucose)*discount/12).sum(-1)
