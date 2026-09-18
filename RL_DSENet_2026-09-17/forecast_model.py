"""Original DSENet dual streams with observed-only, reversible normalization."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'dsenet'))
from models.DSENet import Model

DEFAULT = dict(seq_len=72, label_len=36, pred_len=48, enc_in=1, c_out=1,
               m_layers=1, d_state=16, d_conv=2, m_patch_len=6, m_stride=3,
               e_layers=2, n_heads=4, d_model=64, d_ff=128, dropout=0.1,
               fc_dropout=0.1, head_dropout=0., individual=False,
               patch_len=6, stride=3, padding_patch='end', revin=False,
               affine=False, subtract_last=False, decomposition=False,
               kernel_size=25, concat=True, local_ws=3)

class Forecast(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(DEFAULT if config is None else config)
        self.backbone = Model(SimpleNamespace(**self.config))
        norm = json.loads((ROOT.parent / 'Loop数据集/训练管线_v2/prepared/normalization.json').read_text())['cgm_mmol_l']
        self.register_buffer('mean', torch.tensor(norm['mean']))
        self.register_buffer('scale', torch.tensor(norm['scale']))

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (72, 22):
            raise ValueError('Expected frozen Loop 72x22 history')
        mask = history[:, :, 5:6] > .5
        count = mask.sum(1, keepdim=True)
        if torch.any(count == 0):
            raise ValueError('No observed CGM in history')
        # Frozen Loop z-score -> physical mmol/L. Missing placeholders excluded.
        x = history[:, :, :1] * self.scale + self.mean
        center = (x * mask).sum(1, keepdim=True) / count
        variance = ((x - center).square() * mask).sum(1, keepdim=True) / count
        scale = (variance + 1e-5).sqrt().detach()
        center = center.detach()
        normalized = torch.where(mask, (x - center) / scale, torch.zeros_like(x))
        prediction = self.backbone(normalized, None, None)
        return (prediction * scale + center).squeeze(-1)
