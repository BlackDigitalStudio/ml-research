"""Mixture-of-Experts on volatility regimes.

Unlike parameter-efficient MoE (Switch Transformer etc.) where gating
activates 1-of-N experts PER TOKEN, we use **regime-level MoE**:
  - One expert per market regime (e.g., high-vol, low-vol, trending,
    mean-reverting)
  - Gating by regime classifier on current market state
  - Only one expert active per sample (hard-gate, no interpolation)

Why this works on non-stationary financial data:
  - Different market regimes have DIFFERENT signal structures
  - A single model averages across regimes → suboptimal per-regime
  - Separate experts → each specialized, combined intelligently

Regime features (default 4 regimes):
  - high_vol (vol_ratio > 1.5)
  - low_vol  (vol_ratio < 0.5)
  - trending (abs(hurst - 0.5) > 0.15)
  - mean_rev (otherwise)

Inference:
    regime_id = compute_regime(feat[:, vol_ratio, hurst])
    pred = experts[regime_id](lob, feat)

Training: can either
  (a) Route samples hard to their regime-expert at both train+test
      (each expert sees only its regime's data). Simpler, works well on
      enough samples.
  (b) Soft routing with entropy regularization on gating. Fancier, not used
      here.

This module uses approach (a).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# Feature indices in X_feat (from src/features.py:FEATURE_KEYS)
FEAT_VOL_RATIO = 21
FEAT_HURST = 23


def compute_regime_hard(feat: torch.Tensor,
                         vol_thresh_hi: float = 1.5,
                         vol_thresh_lo: float = 0.5,
                         hurst_trend_band: float = 0.15) -> torch.Tensor:
    """Return (B,) int64 regime id ∈ {0: high_vol, 1: low_vol, 2: trending, 3: mean_rev}.

    Priority order: high_vol > low_vol > trending > mean_rev.
    A sample with high_vol AND trending routes to high_vol (single expert).
    """
    vol_ratio = feat[:, FEAT_VOL_RATIO]
    hurst = feat[:, FEAT_HURST]

    high_vol = vol_ratio > vol_thresh_hi
    low_vol = (vol_ratio < vol_thresh_lo) & ~high_vol
    trending = (torch.abs(hurst - 0.5) > hurst_trend_band) & ~high_vol & ~low_vol

    regime = torch.full(feat.shape[:1], 3, dtype=torch.long, device=feat.device)
    regime[high_vol] = 0
    regime[low_vol] = 1
    regime[trending] = 2
    return regime


REGIME_NAMES = ("high_vol", "low_vol", "trending", "mean_rev")


@dataclass
class RegimeMoEConfig:
    n_experts: int = 4
    vol_thresh_hi: float = 1.5
    vol_thresh_lo: float = 0.5
    hurst_trend_band: float = 0.15
    # Gating mode: 'hard' (argmax single expert) or 'top_k' (average top K)
    gate_mode: str = "hard"
    top_k: int = 1


class RegimeMoE(nn.Module):
    """Wrap N copies of a classifier, route inputs by regime.

    On forward:
      1. Compute regime per sample from feat features
      2. Group samples by regime
      3. Run each expert on its group
      4. Scatter outputs back to original sample order

    This is memory-heavy (N × base_model params) but enables per-regime
    specialization. On our setup (64 GB VRAM), 4× Transformer V1 = ~12M total
    — fits easily. 4× Mamba-2-30M = ~120M — still fits.

    Training: each expert sees only its regime's samples each epoch. If a
    regime has very few samples, consider reducing n_experts.
    """

    def __init__(
        self,
        experts: list[nn.Module],      # N experts; each has .forward(lob, feat) -> (logits, reg)
        cfg: RegimeMoEConfig = RegimeMoEConfig(),
    ):
        super().__init__()
        assert len(experts) == cfg.n_experts, (
            f"expected {cfg.n_experts} experts, got {len(experts)}"
        )
        self.experts = nn.ModuleList(experts)
        self.cfg = cfg

    def _regime_of(self, feat: torch.Tensor) -> torch.Tensor:
        return compute_regime_hard(
            feat,
            vol_thresh_hi=self.cfg.vol_thresh_hi,
            vol_thresh_lo=self.cfg.vol_thresh_lo,
            hurst_trend_band=self.cfg.hurst_trend_band,
        )

    def forward(self, lob: torch.Tensor, feat: torch.Tensor):
        """Route (lob, feat) through regime-specific expert, return (logits, reg)."""
        B = lob.shape[0]
        regime = self._regime_of(feat)
        # Preallocate outputs (assume 3-class logits)
        device = lob.device
        logits_out = torch.zeros(B, 3, device=device, dtype=torch.float32)
        reg_out = torch.zeros(B, device=device, dtype=torch.float32)

        for r in range(self.cfg.n_experts):
            mask = regime == r
            n_r = int(mask.sum().item())
            if n_r == 0:
                continue
            # Route via index select — cheaper than masked_select/assign.
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            lob_r = lob.index_select(0, idx)
            feat_r = feat.index_select(0, idx)
            l_r, r_r = self.experts[r](lob_r, feat_r)
            logits_out.index_copy_(0, idx, l_r.to(logits_out.dtype))
            reg_out.index_copy_(0, idx, r_r.to(reg_out.dtype))
        return logits_out, reg_out

    def regime_distribution(self, feat: torch.Tensor) -> dict[str, int]:
        """Diagnostic: count samples per regime in a batch."""
        regime = self._regime_of(feat).cpu().numpy()
        return {REGIME_NAMES[r]: int((regime == r).sum())
                for r in range(self.cfg.n_experts)}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_regime_moe(
    expert_factory: Callable[[], nn.Module],
    cfg: RegimeMoEConfig = RegimeMoEConfig(),
) -> RegimeMoE:
    """Convenience builder: calls `expert_factory()` N times to make experts."""
    experts = [expert_factory() for _ in range(cfg.n_experts)]
    return RegimeMoE(experts, cfg)
