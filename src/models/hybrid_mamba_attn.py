"""Hybrid Mamba + Attention classifier (Jamba-like architecture).

Interleaves Mamba SSM blocks with multi-head attention layers:
    Block 1: Mamba (linear, long-context)
    Block 2: Mamba
    Block 3: Attention (global mixing)
    Block 4: Mamba
    Block 5: Mamba
    Block 6: Attention
    ...

Jamba (AI21, 2024) / Zamba (Zyphra, 2024) empirically outperform both
pure Transformer and pure Mamba on long-context tasks. On LOB time-series
with 50-500 tick windows, this means:
  - Mamba layers cheaply process the sequence (O(n) recurrence)
  - Attention layers provide selective cross-token mixing where it matters

Interface matches `src.teacher.MultiStreamTransformer`:
    forward(lob: (B, 3, 20, T), feat: (B, F)) -> (logits (B, 3), reg (B,))

NOTE: Uses mambapy (pure PyTorch) — works on CPU + GPU. On GPU-enabled
machines with CUDA kernels, swap to `mamba-ssm` package for ~10× speedup.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HybridMambaAttnConfig:
    d_model: int = 192
    n_blocks: int = 6                # total SSM+attn blocks
    attn_every: int = 3              # attention layer at every Nth block (e.g. every 3rd)
    n_heads: int = 8
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
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


class _AttentionBlock(nn.Module):
    """Single self-attention layer (pre-norm, residual)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + self.drop(h)


class _MambaBlockWrap(nn.Module):
    """Wrap mambapy MambaBlock with pre-norm + residual (already has its own)."""

    def __init__(self, d_model: int, d_state: int, d_conv: int,
                 expand: int, dropout: float):
        super().__init__()
        from mambapy.mamba import MambaBlock, MambaConfig
        cfg = MambaConfig(
            d_model=d_model, n_layers=1, d_state=d_state,
            d_conv=d_conv, expand_factor=expand,
        )
        self.mamba = MambaBlock(cfg)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mambapy MambaBlock expects (B, T, D) and returns same.
        h = self.norm(x)
        h = self.mamba(h)
        return x + self.drop(h)


class HybridMambaAttnClassifier(nn.Module):
    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        lob_channels: int = 3 * 20,
        cfg: HybridMambaAttnConfig = HybridMambaAttnConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim
        self.lob_channels = lob_channels

        # Input projection: (B, 3, 20, T) -> (B, T, d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(lob_channels, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # Build interleaved block stack
        blocks = []
        for b in range(cfg.n_blocks):
            # Attention at every cfg.attn_every-th block (1-indexed sense: b+1).
            if (b + 1) % cfg.attn_every == 0:
                blocks.append(_AttentionBlock(cfg.d_model, cfg.n_heads, cfg.dropout))
            else:
                blocks.append(_MambaBlockWrap(
                    cfg.d_model, cfg.mamba_d_state, cfg.mamba_d_conv,
                    cfg.mamba_expand, cfg.dropout,
                ))
        self.blocks = nn.ModuleList(blocks)

        # Feat tower
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Heads
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
        x = self.input_proj(x)                 # (B, T, d_model)

        for blk in self.blocks:
            x = blk(x)

        lob_pool = x.mean(dim=1)               # (B, d_model)
        feat_tok = self.feat_proj(feat)        # (B, d_model)
        fused = torch.cat([lob_pool, feat_tok], dim=-1)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def block_summary(self) -> list[str]:
        """Which block type at each position — diagnostic."""
        out = []
        for i, blk in enumerate(self.blocks):
            kind = "attn" if isinstance(blk, _AttentionBlock) else "mamba"
            out.append(f"block_{i}_{kind}")
        return out
