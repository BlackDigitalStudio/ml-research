"""MOMENT-1-large fine-tuning adapter for LOB classification.

MOMENT (AutonLab / CMU, 2024) is a T5-based time-series foundation model
pretrained on The Time-Series Pile. 341M params, default context length 512.

We use its `embed()` method to get per-sample (B, 1024) representations,
freeze the encoder, train only classification + feat-fusion head.

Interface matches `src.teacher.MultiStreamTransformer`:
    forward(lob: (B, 3, 20, T), feat: (B, F)) -> (logits (B, 3), reg (B,))

Install (from github — PyPI momentfm has numpy incompat on Python 3.12):
    pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MOMENTAdapterConfig:
    model_repo: str = "AutonLab/MOMENT-1-large"
    freeze_encoder: bool = True
    d_proj: int = 192
    head_hidden: int = 256
    dropout: float = 0.15
    context_len: int = 512      # MOMENT native context
    reduction: str = "mean"     # mean | concat
    # Multivariate: None = top-level imbalance only (C=1).
    # "all_levels" = best 5 levels × bid/ask/imb (C=20).
    # "full" = full 60 LOB channels natively multivariate.
    multivariate_mode: str | None = None

    # Training
    batch_size: int = 128
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    reg_loss_weight: float = 0.3
    label_smoothing: float = 0.05
    early_stop_patience: int = 6


class MOMENTClassifier(nn.Module):
    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        cfg: MOMENTAdapterConfig = MOMENTAdapterConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim

        from momentfm import MOMENTPipeline
        self.backbone = MOMENTPipeline.from_pretrained(
            cfg.model_repo,
            model_kwargs={"task_name": "embedding"},
        )
        if cfg.freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Infer embedding dim by one dry forward
        with torch.no_grad():
            dry = torch.zeros(1, 1, cfg.context_len)
            d_moment = self.backbone.embed(x_enc=dry, reduction=cfg.reduction).embeddings.shape[-1]
        self.d_moment = int(d_moment)

        # Projection + fusion
        self.lob_proj = nn.Sequential(
            nn.LayerNorm(self.d_moment),
            nn.Linear(self.d_moment, cfg.d_proj),
            nn.GELU(),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_proj),
            nn.GELU(),
            nn.Linear(cfg.d_proj, cfg.d_proj),
        )

        fused = cfg.d_proj * 2
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 3),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def _lob_to_series(self, lob: torch.Tensor) -> torch.Tensor:
        """Project (B, 3, 20, T) → (B, C, T). MOMENT natively multivariate."""
        if self.cfg.multivariate_mode is None:
            bid_l1 = lob[:, 0, 0, :]
            ask_l1 = lob[:, 1, 0, :]
            tot = bid_l1 + ask_l1 + 1e-6
            return ((bid_l1 - ask_l1) / tot).unsqueeze(1)           # (B, 1, T)
        if self.cfg.multivariate_mode == "all_levels":
            bid_vol = lob[:, 0, :5, :]
            ask_vol = lob[:, 1, :5, :]
            tot = bid_vol + ask_vol + 1e-6
            imb = (bid_vol - ask_vol) / tot
            return torch.cat([bid_vol, ask_vol, imb, imb], dim=1)   # (B, 20, T)
        if self.cfg.multivariate_mode == "full":
            B, _, _, T = lob.shape
            return lob.reshape(B, 60, T)
        raise ValueError(f"unknown multivariate_mode={self.cfg.multivariate_mode}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → MOMENT embed → (B, d_moment) via channel-independent pool.
        MOMENT accepts (B, C, T) natively but returns (B, C, D) — we mean over C.
        """
        B, C, T = x.shape
        ctx = self.cfg.context_len
        if T < ctx:
            pad = torch.zeros(B, C, ctx - T, dtype=x.dtype, device=x.device)
            x = torch.cat([pad, x], dim=-1)
        elif T > ctx:
            x = x[..., -ctx:]
        # Run channel-independently (pools over C)
        flat = x.reshape(B * C, 1, x.shape[-1])
        out = self.backbone.embed(x_enc=flat, reduction=self.cfg.reduction)
        emb = out.embeddings.view(B, C, -1).mean(dim=1)
        return emb

    def forward(self, lob: torch.Tensor, feat: torch.Tensor):
        series = self._lob_to_series(lob)
        emb = self._encode(series)
        lob_tok = self.lob_proj(emb)
        feat_tok = self.feat_proj(feat)
        fused = torch.cat([lob_tok, feat_tok], dim=-1)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    def num_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
