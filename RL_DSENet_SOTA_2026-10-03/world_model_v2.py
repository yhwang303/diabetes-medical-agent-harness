"""Research-only causal basal world model; no clinical or calibration guarantee.

The genuine frozen DSENet supplies its original 48-step forecast. A new history
encoder and nonlinear causal action network learn corrections and a 72-step
extension. Missing event records remain masked; logged exposure summaries are
not estimates of true insulin/carbohydrate on board. No ensemble or RNG reset.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


FEATURE_NAMES = (
    'cgm_mmol_l', 'basal_u_prev_5min', 'bolus_recorded_u_prev_5min',
    'carbs_recorded_g_prev_5min', 'exercise_event_count_prev_5min',
)
DEFAULT_QUANTILES = (.05, .1, .25, .5, .75, .9, .95)


def load_frozen_dsenet(checkpoint_path):
    """Load the existing Forecast implementation and an explicit trusted checkpoint.

    The caller records/checks the checkpoint SHA; there is no default checkpoint
    or fallback. This reads the existing source without modifying it.
    """
    source = Path(__file__).resolve().parent.parent / 'RL_DSENet_2026-09-17/forecast_model.py'
    spec = importlib.util.spec_from_file_location('_world_v2_legacy_forecast', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    forecast = module.Forecast(checkpoint['config']['model'])
    forecast.load_state_dict(checkpoint['model'], strict=True)
    return forecast.requires_grad_(False).eval()


class CausalGatedBlock(nn.Module):
    """Channel-only normalization and left padding preserve future-action causality."""
    def __init__(self, width, dilation):
        super().__init__()
        self.left_padding = 2 * dilation
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, 2 * width, 3, dilation=dilation)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        value, gate = self.conv(F.pad(y, (self.left_padding, 0))).chunk(2, dim=1)
        return x + self.output((value.tanh() * gate.sigmoid()).transpose(1, 2))


class WorldModelV2(nn.Module):
    """encode(history, anchor) -> cache; predict(cache, actions) -> tensor dict.

    history: float32 (N,72,22), existing observed-only Loop normalization.
    anchor: float32 (N,), U/h. actions: float32 (N,A,H), 1 <= H <= 72.
    Each action controls the NEXT five-minute interval; output k is its endpoint.
    The caller enforces the research action contract; this model never clips it.
    """
    def __init__(self, forecast, normalization, width=64, quantiles=DEFAULT_QUANTILES):
        super().__init__()
        quantiles = tuple(quantiles)
        if (not quantiles or tuple(sorted(set(quantiles))) != quantiles
                or quantiles[0] <= 0 or quantiles[-1] >= 1 or .5 not in quantiles):
            raise ValueError('Quantiles must be unique, increasing, in (0,1), and contain .5')
        self.forecast = forecast.requires_grad_(False).eval()
        self.max_horizon = 72
        self.median_index = quantiles.index(.5)
        self.register_buffer('quantile_levels', torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer('history_mean', torch.tensor(
            [normalization[name]['mean'] for name in FEATURE_NAMES], dtype=torch.float32))
        self.register_buffer('history_scale', torch.tensor(
            [normalization[name]['scale'] for name in FEATURE_NAMES], dtype=torch.float32))
        if not bool(torch.all(self.history_scale > 0)):
            raise ValueError('Normalization scales must be positive')
        # Midpoints of the logged previous-five-minute intervals; generic bases,
        # not a pharmacokinetic model or a prescribed insulin action duration.
        ages = torch.arange(71, -1, -1, dtype=torch.float32) * 5 + 2.5
        taus = torch.tensor([30., 90., 180., 360.])
        self.register_buffer('exposure_weights', torch.exp(-ages[:, None] / taus[None]))
        self.history_encoder = nn.GRU(22, width, batch_first=True)
        self.forecast_encoder = nn.Sequential(nn.Linear(48, 32), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(width + 32 + 24 + 2, width), nn.Tanh())
        self.future_input = nn.Linear(5, width)
        self.action_blocks = nn.ModuleList([
            CausalGatedBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 32, 64)
        ])
        self.quantile_head = nn.Linear(width, len(quantiles))
        # Mutually exclusive horizon events: no BG<70, BG<70 but no BG<54,
        # and any BG<54. Their probabilities imply P54 <= P70.
        self.event_head = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 3))

    def train(self, mode=True):
        super().train(mode)
        self.forecast.eval()
        return self

    def encode(self, history, anchor_u_h):
        if history.ndim != 3 or history.shape[1:] != (72, 22):
            raise ValueError('history must have shape (N,72,22)')
        if history.dtype != torch.float32 or anchor_u_h.dtype != torch.float32:
            raise ValueError('history and anchor must be float32')
        if anchor_u_h.shape != history.shape[:1] or anchor_u_h.device != history.device:
            raise ValueError('anchor must have shape (N,) on the history device')
        if not bool(torch.isfinite(history).all() and torch.isfinite(anchor_u_h).all()):
            raise ValueError('Nonfinite history or anchor')
        if bool((anchor_u_h < 0).any()):
            raise ValueError('Negative basal anchor')
        observed = history[:, :, 5:10] > .5
        if not bool(observed[:, :, 0].any(dim=1).all()):
            raise ValueError('Every history needs an observed CGM')
        # Enforce placeholder invariance before either encoder receives the data.
        cleaned = torch.cat([torch.where(observed, history[:, :, :5], 0.), history[:, :, 5:]], -1)
        physical = cleaned[:, :, :5] * self.history_scale + self.history_mean
        physical = torch.where(observed, physical, 0.)
        scales = physical.new_tensor([18. / 100., 12. / 5., 1. / 5., 1. / 50., 1. / 5.])
        encoder_input = torch.cat([physical * scales, cleaned[:, :, 5:20], cleaned[:, :, 20:] / 6.], -1)
        _, hidden = self.history_encoder(encoder_input)
        # Keep basal, recorded bolus, and recorded carbohydrate separate.
        doses = physical[:, :, 1:4] / physical.new_tensor([5., 5., 50.])
        exposure = torch.einsum('ntc,tk->nck', doses, self.exposure_weights)
        coverage = torch.einsum('ntc,tk->nck', observed[:, :, 1:4].float(), self.exposure_weights)
        coverage = coverage / self.exposure_weights.sum(0)[None, None]
        logged_summary = torch.cat([exposure.flatten(1), coverage.flatten(1)], -1)
        last_index = torch.where(observed[:, :, 0], torch.arange(72, device=history.device), -1).max(1).values
        last_cgm = physical[:, :, 0].gather(1, last_index[:, None]).squeeze(1) * 18.
        with torch.no_grad():
            self.forecast.eval()
            dsenet = self.forecast(cleaned).float() * 18.
        if dsenet.shape != (len(history), 48) or not bool(torch.isfinite(dsenet).all()):
            raise ValueError('Frozen DSENet must return finite (N,48) mmol/L predictions')
        forecast_feature = self.forecast_encoder((dsenet - last_cgm[:, None]) / 100.)
        context = self.context(torch.cat([
            hidden[-1], forecast_feature, logged_summary,
            anchor_u_h[:, None] / 5., last_cgm[:, None] / 100.,
        ], -1))
        return {
            'context': context, 'anchor_u_h': anchor_u_h, 'last_cgm_mgdl': last_cgm,
            'dsenet_cgm_4h_mgdl': dsenet, 'logged_exposure_summary': logged_summary,
        }

    def predict(self, cache, actions_u_h):
        if actions_u_h.ndim != 3 or actions_u_h.shape[0] != len(cache['context']):
            raise ValueError('actions must have shape (N,A,H) matching cache')
        n, arms, horizon = actions_u_h.shape
        if arms < 1 or not 1 <= horizon <= self.max_horizon:
            raise ValueError('Need at least one arm and 1 <= H <= 72')
        if actions_u_h.dtype != torch.float32 or actions_u_h.device != cache['context'].device:
            raise ValueError('actions must be float32 on the cache device')
        if not bool(torch.isfinite(actions_u_h).all()) or bool((actions_u_h < 0).any()):
            raise ValueError('Actions must be finite nonnegative basal rates; no clipping')
        index = torch.arange(horizon, device=actions_u_h.device)
        available = index < 48
        dsenet = cache['dsenet_cgm_4h_mgdl'][:, index.clamp_max(47)]
        # Beyond four hours use an explicitly marked observed-CGM reference,
        # never present a repeated/extrapolated DSENet value as its own forecast.
        baseline = torch.where(available[None], dsenet, cache['last_cgm_mgdl'][:, None])
        anchor = cache['anchor_u_h'][:, None, None]
        features = torch.stack([
            actions_u_h / 5., (actions_u_h - anchor) / 5.,
            ((index + 1) / float(self.max_horizon))[None, None].expand(n, arms, -1),
            ((baseline - cache['last_cgm_mgdl'][:, None]) / 100.)[:, None].expand(-1, arms, -1),
            available.float()[None, None].expand(n, arms, -1),
        ], -1)
        x = self.future_input(features) + cache['context'][:, None, None]
        x = x.reshape(n * arms, horizon, -1)
        for block in self.action_blocks:
            x = block(x)
        raw = self.quantile_head(x).float().reshape(n, arms, horizon, -1)
        median = baseline[:, None] + 40. * raw[:, :, :, self.median_index]
        left_gaps = F.softplus(raw[:, :, :, :self.median_index]) * 10. + .01
        right_gaps = F.softplus(raw[:, :, :, self.median_index + 1:]) * 10. + .01
        left = median[..., None] - left_gaps.flip(-1).cumsum(-1).flip(-1)
        right = median[..., None] + right_gaps.cumsum(-1)
        quantiles = torch.cat([left, median[..., None], right], -1)
        # This event describes the entire requested horizon, so it may depend on
        # every candidate action. Per-step CGM outputs remain prefix-causal.
        event_logits = self.event_head(torch.cat([x.mean(1), x[:, -1]], -1)).float().reshape(n, arms, 3)
        event_probability = event_logits.softmax(-1)
        return {
            'cgm_quantiles_mgdl': quantiles,
            'cgm_median_mgdl': median,
            'quantile_levels': self.quantile_levels,
            'bg_event_logits': event_logits,
            'bg_event_probabilities': event_probability,
            'p_bg_below70': event_probability[:, :, 1:].sum(-1),
            'p_bg_below54': event_probability[:, :, 2],
            'dsenet_available': available,
        }

    def forward(self, history, anchor_u_h, actions_u_h):
        return self.predict(self.encode(history, anchor_u_h), actions_u_h)
