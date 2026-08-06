"""Mamba (State-Space Model) classifier for LOB direction.

Uses `mambapy` (pure PyTorch) for CPU compatibility. On GPU-enabled machines
(RunPod), swap to Mamba-2 with the CUDA kernel (`mamba-ssm` package) for
~10× speedup — interface identical.

Matches `src.teacher.MultiStreamTransformer` interface:
    forward(lob: (B, 3, 20, T), feat: (B, F)) -> (logits (B, 3), reg (B,))
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MambaModelConfig:
    d_model: int = 192
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dropout: float = 0.15

    # Training
    batch_size: int = 256
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    reg_loss_weight: float = 0.3
    label_smoothing: float = 0.05
    early_stop_patience: int = 6


class MambaClassifier(nn.Module):
    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        lob_channels: int = 3 * 20,
        cfg: MambaModelConfig = MambaModelConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim
        self.lob_channels = lob_channels

        # Project LOB channels to d_model per time step
        self.input_proj = nn.Sequential(
            nn.Linear(lob_channels, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        from mambapy.mamba import Mamba, MambaConfig
        m_cfg = MambaConfig(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand_factor=cfg.expand_factor,
        )
        self.mamba = Mamba(m_cfg)

        # Feature tower
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Heads on [pooled_mamba, feat] concat.
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
        # (B, 3, 20, T) → (B, T, 60)
        x = lob.permute(0, 3, 1, 2).reshape(B, self.lob_time_dim, -1)
        x = self.input_proj(x)                  # (B, T, d_model)
        h = self.mamba(x)                       # (B, T, d_model)
        lob_pool = h.mean(dim=1)                # (B, d_model)
        feat_tok = self.feat_proj(feat)         # (B, d_model)
        fused = torch.cat([lob_pool, feat_tok], dim=-1)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
