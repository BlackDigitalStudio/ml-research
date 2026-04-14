"""Chronos-Bolt fine-tuning adapter for LOB classification.

Chronos-Bolt (Amazon, 2024) is a T5-based time-series foundation model
pretrained on diverse univariate series. We use its ENCODER as a frozen
(or fine-tunable) representation of the BTC mid-price trajectory, then
add a classification head that fuses the Chronos representation with
our handcrafted feature vector.

Matches interface of `src.teacher.MultiStreamTransformer`:
    forward(lob: (B, 3, 20, T), feat: (B, F)) -> (logits (B, 3), reg (B,))

Input contract:
    lob[:, 0] is bid volume, lob[:, 1] is ask volume, lob[:, 2] is trade flow
    We derive the midprice trajectory from... actually LOB here doesn't carry
    mid directly. We use top-of-book volume-weighted mid as the proxy:
        mid[t] = (best_bid_vol[t] + best_ask_vol[t]) / 2   <-- NOT price
    This is a placeholder; the proper fix is caching best_bid/ask PRICES in
    X_lob when building samples for Chronos experiments. For now we use
    the mean across LOB levels as a stand-in signal for the encoder.

Design:
    - Sequence of length lob_time_dim (default 50) is fed to Chronos encoder
      after padding/resampling to a multiple of patch_size (16).
    - Encoder produces (B, n_patches, d_chronos). We mean-pool → (B, d_chronos).
    - Handcrafted features (B, num_feat) go through a small MLP → (B, d_proj).
    - Concat → fused head → (logits, reg).

Available models:
    - amazon/chronos-bolt-tiny   (~9M  params, fastest)
    - amazon/chronos-bolt-mini   (~21M)
    - amazon/chronos-bolt-small  (~48M, recommended baseline)
    - amazon/chronos-bolt-base   (~205M, requires more CPU)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


ChronosSize = Literal["tiny", "mini", "small", "base"]


@dataclass
class ChronosAdapterConfig:
    model_name: str = "amazon/chronos-bolt-small"
    freeze_encoder: bool = True          # fine-tune head only (fast first pass)
    freeze_for_epochs: int = 5           # then unfreeze (progressive unfreezing)
    d_proj: int = 192                    # projection dim for head
    head_hidden: int = 256
    dropout: float = 0.15
    # Multivariate path: iterate channel-independently through encoder and
    # aggregate across channels. Trades throughput for signal quality.
    # None = single-channel (top-level imbalance only, legacy behaviour).
    # "all_levels" = bid/ask level-0 ... level-4 prices + qtys (20 channels).
    # "full" = all 80 LOB channels.
    multivariate_mode: str | None = None

    # Training knobs (same interface as other bake-off models)
    batch_size: int = 128
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    reg_loss_weight: float = 0.3
    label_smoothing: float = 0.05
    early_stop_patience: int = 6


class ChronosClassifier(nn.Module):
    """Chronos-Bolt encoder + classification head for LOB direction."""

    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        cfg: ChronosAdapterConfig = ChronosAdapterConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim

        # Load pretrained backbone
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(cfg.model_name, device_map="cpu")
        backbone = pipe.model
        self.chronos_config = backbone.chronos_config
        cc = self.chronos_config

        def _cfg(name, default=None):
            if hasattr(cc, name):
                return getattr(cc, name)
            if isinstance(cc, dict):
                return cc.get(name, default)
            return default
        self.patch_size = _cfg("input_patch_size", 16)
        self.patch_stride = _cfg("input_patch_stride", 16)
        self.reg_token_id = _cfg("reg_token_id", 1)

        # Reusable pieces: patch + norm + input_patch_embedding + encoder.
        # Everything decoder-side we throw away.
        self.patch = backbone.patch
        self.instance_norm = backbone.instance_norm
        self.input_patch_embedding = backbone.input_patch_embedding
        self.encoder = backbone.encoder
        # Keep reg_token embedding for consistency with Chronos pretraining.
        self.shared = backbone.shared
        self.d_chronos = self.encoder.config.d_model

        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.input_patch_embedding.parameters():
                p.requires_grad = False

        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_proj),
            nn.GELU(),
            nn.Linear(cfg.d_proj, cfg.d_proj),
        )

        # Lob-embedding projection to fuse with feat
        self.lob_proj = nn.Sequential(
            nn.LayerNorm(self.d_chronos),
            nn.Linear(self.d_chronos, cfg.d_proj),
            nn.GELU(),
        )

        fused_dim = cfg.d_proj * 2
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 3),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def _lob_to_series(self, lob: torch.Tensor) -> torch.Tensor:
        """Reduce (B, 3, 20, T) → (B, C, T) where C depends on multivariate_mode.

        C=1 (default): top-level imbalance — smallest info, fastest.
        C=20 ("all_levels"): bid_vol[0..4] + ask_vol[0..4] + bid_imb[0..4]
              + ask_imb[0..4] — the 5 best levels of each side with
              derived imbalance channels.
        C=60 ("full"): all LOB channels 3×20 (bid_vol, ask_vol, trade_flow)
              stacked. Maximum signal.
        """
        if self.cfg.multivariate_mode is None:
            bid_l1 = lob[:, 0, 0, :]
            ask_l1 = lob[:, 1, 0, :]
            tot = bid_l1 + ask_l1 + 1e-6
            return ((bid_l1 - ask_l1) / tot).unsqueeze(1)           # (B, 1, T)
        if self.cfg.multivariate_mode == "all_levels":
            # First 5 levels bid + ask + per-level imbalance.
            bid_vol = lob[:, 0, :5, :]                              # (B, 5, T)
            ask_vol = lob[:, 1, :5, :]
            tot = bid_vol + ask_vol + 1e-6
            imb = (bid_vol - ask_vol) / tot
            return torch.cat([bid_vol, ask_vol, imb, imb], dim=1)   # (B, 20, T)
        if self.cfg.multivariate_mode == "full":
            B, _, _, T = lob.shape
            return lob.reshape(B, 60, T)                             # (B, 60, T)
        raise ValueError(f"unknown multivariate_mode={self.cfg.multivariate_mode}")

    def _encode_chronos(self, series: torch.Tensor) -> torch.Tensor:
        """Run series (B, T) through Chronos encoder → pooled (B, d_chronos)."""
        B, T = series.shape
        # Pad T up to next multiple of patch_size
        pad = (-T) % self.patch_size
        if pad > 0:
            series = F.pad(series, (pad, 0))

        # Mask: all real (no NaN in our case)
        mask = torch.ones_like(series, dtype=torch.bool)

        # Follow Chronos-Bolt's encode pipeline:
        #   1. instance_norm
        #   2. patch (non-overlapping 16-stride)
        #   3. input_patch_embedding
        #   4. prepend REG token
        #   5. T5 encoder stack
        # Adapted from ChronosBoltModelForForecasting.encode().
        normalized, _ = self.instance_norm(series)
        patched_context = self.patch(normalized)            # (B, n_patches, patch_size)
        patched_mask = torch.nan_to_num(
            self.patch(mask.to(normalized.dtype)), nan=0.0
        )
        # Concat context with mask (Chronos convention).
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        input_embeds = self.input_patch_embedding(patched_context)

        # REG token
        reg_input_ids = torch.full(
            (B, 1), self.reg_token_id, device=input_embeds.device, dtype=torch.long
        )
        reg_embeds = self.shared(reg_input_ids)
        input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)

        attention_mask = torch.cat(
            [patched_mask.sum(dim=-1) > 0, torch.ones(B, 1, device=input_embeds.device).bool()],
            dim=-1,
        )

        enc_out = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state              # (B, N+1, d_chronos)

        # Pool: mean over valid positions
        mask_f = attention_mask.float().unsqueeze(-1)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(
        self, lob: torch.Tensor, feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        series = self._lob_to_series(lob)                # (B, C, T)
        B, C, T = series.shape
        # Channel-independent encoding: flatten channel into batch dim,
        # run encoder once, then reshape + mean-pool across channels.
        flat = series.reshape(B * C, T)
        pooled = self._encode_chronos(flat)              # (B*C, d_chronos)
        pooled = pooled.view(B, C, -1).mean(dim=1)       # (B, d_chronos)
        lob_tok = self.lob_proj(pooled)
        feat_tok = self.feat_proj(feat)
        fused = torch.cat([lob_tok, feat_tok], dim=-1)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    def unfreeze_encoder(self) -> None:
        """Progressive unfreezing — call after freeze_for_epochs."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.input_patch_embedding.parameters():
            p.requires_grad = True

    def num_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
