"""Temporal Convolutional Network for LOB direction.

Dilated causal convolutions (Bai/Kolter/Koltun 2018). Fast on CPU,
competitive with Transformer on financial time-series. Often the most
underrated baseline in modern architecture bake-offs.

Matches `src.teacher.MultiStreamTransformer` interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TCNConfig:
    n_channels: int = 128             # channel width inside residual blocks
    n_blocks: int = 6                 # dilation doubles each block → receptive field 2^n
    kernel_size: int = 3
    dropout: float = 0.15
    d_model: int = 192                # projection for head fusion

    batch_size: int = 256
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    reg_loss_weight: float = 0.3
    label_smoothing: float = 0.05
    early_stop_patience: int = 6


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation   # causal left-padding
        self.pad = pad
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        # x: (B, C, T)
        y = F.pad(x, (self.pad, 0))
        y = self.relu(self.conv1(y))
        y = self.drop(y)
        y = F.pad(y, (self.pad, 0))
        y = self.relu(self.conv2(y))
        y = self.drop(y)
        return self.relu(y + self.res(x))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        lob_channels: int = 3 * 20,
        cfg: TCNConfig = TCNConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim
        self.lob_channels = lob_channels

        layers = []
        in_ch = lob_channels
        for b in range(cfg.n_blocks):
            dil = 2 ** b
            layers.append(_TemporalBlock(in_ch, cfg.n_channels, cfg.kernel_size,
                                          dilation=dil, dropout=cfg.dropout))
            in_ch = cfg.n_channels
        self.tcn = nn.Sequential(*layers)

        self.lob_proj = nn.Sequential(
            nn.LayerNorm(cfg.n_channels),
            nn.Linear(cfg.n_channels, cfg.d_model),
            nn.GELU(),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        fused = cfg.d_model * 2
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 3),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, lob: torch.Tensor, feat: torch.Tensor):
        B = lob.shape[0]
        # (B, 3, 20, T) → (B, 60, T) channels-first for Conv1d
        x = lob.reshape(B, self.lob_channels, self.lob_time_dim)
        h = self.tcn(x)                       # (B, C, T)
        lob_pool = h.mean(dim=-1)             # (B, C)
        lob_tok = self.lob_proj(lob_pool)     # (B, d_model)
        feat_tok = self.feat_proj(feat)
        fused = torch.cat([lob_tok, feat_tok], dim=-1)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
