"""TimesFM 2.5 200M fine-tuning adapter for LOB classification.

Google's TimesFM (2024) is a decoder-only time-series foundation model
pretrained on 100B+ time-series datapoints. On our task:
  - Freeze tokenizer + stacked_xf (encoder) — ~200M params, skip training
  - Add small classification head on pooled hidden states
  - Concatenate with handcrafted feat tower
  - Fine-tune only head (fast) or progressive unfreeze

Interface matches `src.teacher.MultiStreamTransformer`:
    forward(lob: (B, 3, 20, T), feat: (B, F)) -> (logits (B, 3), reg (B,))

Caveat: uses `timesfm` package (not `transformers` since HF checkpoint
naming differs from transformers' TimesFmModel). Install via:
    pip install git+https://github.com/google-research/timesfm.git
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class TimesFMAdapterConfig:
    model_repo: str = "google/timesfm-2.5-200m-pytorch"
    freeze_encoder: bool = True
    d_proj: int = 192
    head_hidden: int = 256
    dropout: float = 0.15
    patch_len: int = 32       # TimesFM 2.5 default
    context_len: int = 512    # must be multiple of patch_len
    # Channel-independent multivariate: run encoder over each LOB channel
    # separately, then mean-pool across channels. None = univariate proxy.
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


def _load_timesfm_backbone(repo: str) -> tuple[nn.Module, nn.ModuleList]:
    """Download weights, load inner model, return (tokenizer, stacked_xf).
    Caller uses these as fixed encoder; no dependency on forecast/output heads.
    """
    import timesfm
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(repo, "model.safetensors")
    wrapper = timesfm.TimesFM_2p5_200M_torch()
    wrapper.model.load_checkpoint(path=weights, torch_compile=False)
    # Discard forecast heads — we use encoder only.
    del wrapper.model.output_projection_point
    del wrapper.model.output_projection_quantiles
    return wrapper.model.tokenizer, wrapper.model.stacked_xf


class TimesFMClassifier(nn.Module):
    """TimesFM 2.5 encoder + LOB fusion + classification head."""

    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        cfg: TimesFMAdapterConfig = TimesFMAdapterConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim

        tokenizer, stacked_xf = _load_timesfm_backbone(cfg.model_repo)
        self.tokenizer = tokenizer
        self.stacked_xf = stacked_xf
        # Infer d_model from tokenizer output dim (1280 for 2.5-200M).
        with torch.no_grad():
            self.d_model = int(self.tokenizer(
                torch.zeros(1, 1, 2 * cfg.patch_len)
            ).shape[-1])

        if cfg.freeze_encoder:
            for p in self.tokenizer.parameters():
                p.requires_grad = False
            for p in self.stacked_xf.parameters():
                p.requires_grad = False

        # Feature tower
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_proj),
            nn.GELU(),
            nn.Linear(cfg.d_proj, cfg.d_proj),
        )

        # LOB encoder output projection
        self.lob_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, cfg.d_proj),
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
        """Project (B, 3, 20, T) → (B, C, T). C=1 default or 20/60 multivariate."""
        if self.cfg.multivariate_mode is None:
            bid_l1 = lob[:, 0, 0, :]
            ask_l1 = lob[:, 1, 0, :]
            tot = bid_l1 + ask_l1 + 1e-6
            return ((bid_l1 - ask_l1) / tot).unsqueeze(1)          # (B, 1, T)
        if self.cfg.multivariate_mode == "all_levels":
            bid_vol = lob[:, 0, :5, :]
            ask_vol = lob[:, 1, :5, :]
            tot = bid_vol + ask_vol + 1e-6
            imb = (bid_vol - ask_vol) / tot
            return torch.cat([bid_vol, ask_vol, imb, imb], dim=1)
        if self.cfg.multivariate_mode == "full":
            B, _, _, T = lob.shape
            return lob.reshape(B, 60, T)
        raise ValueError(f"unknown multivariate_mode={self.cfg.multivariate_mode}")

    def _encode(self, series: torch.Tensor) -> torch.Tensor:
        """Run (B, T) → encoder → (B, d_model) pooled."""
        B, T = series.shape
        patch_len = self.cfg.patch_len
        # Pad T up to next multiple of patch_len
        pad = (-T) % patch_len
        if pad > 0:
            series = torch.nn.functional.pad(series, (pad, 0))
        Tp = series.shape[1]
        N = Tp // patch_len
        # Shape: (B, N, patch_len)
        patches = series.view(B, N, patch_len)
        masks = torch.zeros_like(patches)  # no padding
        inp = torch.cat([patches, masks], dim=-1)  # (B, N, 2*patch_len)
        # TimesFM's `load_checkpoint()` populates internal tensors that
        # aren't all registered as nn.Module parameters, so `.to(device)`
        # at training time misses them. Force device sync per-forward —
        # no-op after the first call.
        device = inp.device
        self.tokenizer.to(device)
        self.stacked_xf.to(device)
        emb = self.tokenizer(inp)                  # (B, N, d_model)
        h = emb
        for layer in self.stacked_xf:
            h, _ = layer(h, torch.zeros(B, N, dtype=torch.long, device=h.device), None)
        return h.mean(dim=1)                       # (B, d_model)

    def forward(self, lob: torch.Tensor, feat: torch.Tensor):
        series = self._lob_to_series(lob)                # (B, C, T)
        B, C, T = series.shape
        # Channel-independent: run TimesFM encoder per channel, mean-pool.
        flat = series.reshape(B * C, T)
        pooled = self._encode(flat)                      # (B*C, d_model)
        pooled = pooled.view(B, C, -1).mean(dim=1)       # (B, d_model)
        lob_tok = self.lob_proj(pooled)
        feat_tok = self.feat_proj(feat)
        fused = torch.cat([lob_tok, feat_tok], dim=-1)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    def num_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def unfreeze_encoder(self) -> None:
        for p in self.tokenizer.parameters():
            p.requires_grad = True
        for p in self.stacked_xf.parameters():
            p.requires_grad = True
