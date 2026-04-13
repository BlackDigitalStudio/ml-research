"""PatchTST — "A Time Series is Worth 64 Words" (Nie et al., ICLR 2023).

Channel-independent patching + Transformer on patch tokens. Purpose-built for
multivariate time-series forecasting; often beats standard Transformer on
finance/energy benchmarks at fewer params.

Interface matches `src.teacher.MultiStreamTransformer`:
    forward(lob: (B, 3, 20, T), feat: (B, F)) -> (logits (B, 3), reg (B,))

Design choices (conservative defaults, tweak via PatchTSTConfig):
  - channel-independent backbone (each of C=60 LOB channels flows through
    the *same* transformer weights; concat at head)
  - non-overlapping patches of size `patch_len`
  - RevIN instance-norm on input (per-channel, per-sample)
  - classification head fuses pooled patch-tokens with handcrafted feat vector
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PatchTSTConfig:
    # Model capacity — defaults sized similar to the Transformer teacher.
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.15

    # Patching — LOB time dim is 50 ticks = 5 sec. patch_len=10 → 5 patches.
    patch_len: int = 10
    stride: int = 10         # non-overlapping

    # Training — kept separate in PatchTSTConfig so train_teacher can read it.
    batch_size: int = 256
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    reg_loss_weight: float = 0.3
    label_smoothing: float = 0.05
    early_stop_patience: int = 6


class _RevIN(nn.Module):
    """Reversible instance normalization (Kim et al., 2022)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = x.std(dim=-1, keepdim=True).detach() + self.eps
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class PatchTST(nn.Module):
    """PatchTST classifier + regressor for LOB + handcrafted features."""

    def __init__(
        self,
        num_feat: int = 34,
        lob_time_dim: int = 50,
        lob_channels: int = 3 * 20,  # 3 channels × 20 levels
        cfg: PatchTSTConfig = PatchTSTConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.lob_time_dim = lob_time_dim
        self.lob_channels = lob_channels

        assert lob_time_dim % cfg.stride == 0, (
            f"time dim {lob_time_dim} not divisible by stride {cfg.stride}"
        )
        self.num_patches = lob_time_dim // cfg.stride

        self.revin = _RevIN(lob_channels)

        # Patch embedding — linear projection of patch_len → d_model.
        self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)

        # Positional encoding over patches (shared across channels).
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Channel-independent Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # After encoder: (B*C, N, d_model). Pool over N (mean), reshape to
        # (B, C * d_model). Then project to fused token.
        self.channel_fuse = nn.Sequential(
            nn.LayerNorm(lob_channels * cfg.d_model),
            nn.Linear(lob_channels * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Feat projection + fusion.
        self.feat_proj = nn.Sequential(
            nn.Linear(num_feat, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        fused_dim = cfg.d_model * 2

        # Heads.
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 3),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self, lob: torch.Tensor, feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # lob: (B, 3, 20, T=50) → (B, C=60, T)
        B = lob.shape[0]
        x = lob.reshape(B, self.lob_channels, self.lob_time_dim)

        # RevIN per channel.
        x = self.revin(x)

        # Patching: (B, C, T) → (B, C, N, patch_len)
        x = x.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)
        # Shape now: (B, C, N, patch_len).
        B_, C_, N_, P_ = x.shape

        # Flatten channels with batch for channel-independent transformer.
        x = x.reshape(B_ * C_, N_, P_)
        tokens = self.patch_embed(x)           # (B*C, N, d_model)
        tokens = tokens + self.pos_emb          # broadcast (1, N, d_model)
        enc = self.encoder(tokens)              # (B*C, N, d_model)

        # Mean-pool across patches, restore (B, C, d_model).
        pooled = enc.mean(dim=1)                # (B*C, d_model)
        pooled = pooled.view(B_, C_, -1)        # (B, C, d_model)
        lob_tok = self.channel_fuse(pooled.flatten(1))  # (B, d_model)

        feat_tok = self.feat_proj(feat)         # (B, d_model)
        fused = torch.cat([lob_tok, feat_tok], dim=-1)  # (B, 2*d_model)

        logits = self.cls_head(fused)
        reg = self.reg_head(fused).squeeze(-1)
        return logits, reg

    # Convenience param count for logs.
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_pretrained_backbone(self, path: str) -> dict:
        """Load weights from an SSL-pretrained `PatchTSTBackbone` into this
        classifier's matching layers (revin, patch_embed, pos_emb, encoder).

        Skips layers with shape mismatch — caller should ensure pretraining
        config matches (d_model, n_layers, patch_len). Returns a report
        with loaded/skipped tensor counts.
        """
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=True)
        own = self.state_dict()
        loaded = []
        skipped = []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                skipped.append((k, tuple(v.shape),
                                tuple(own[k].shape) if k in own else None))
        self.load_state_dict(own)
        return {"loaded": loaded, "skipped": skipped, "n_loaded": len(loaded)}
