"""PatchTST backbone + reconstruction head for SSL pretraining.

Mirrors the architecture used by `src.models.patchtst.PatchTST` so the
pretrained backbone can be loaded directly into the downstream classifier
via state_dict transfer.

Key design choice: the backbone is **channel-independent** — the same
patch embedding + transformer encoder weights are applied to each LOB
channel independently, then mean-pooled. This is what makes it transfer
well across different feature sets and task heads.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BackboneConfig:
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.15
    patch_len: int = 16
    stride: int = 16


class _RevIN(nn.Module):
    """Reversible instance normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = x.std(dim=-1, keepdim=True).detach() + self.eps
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class PatchTSTBackbone(nn.Module):
    """Channel-independent PatchTST encoder. Returns (B, C, N, d_model).

    Input:  (B, C, T)
    Output: (B, C, N, d_model)  where N = T // stride

    `C` and `T` need not match between pretrain and downstream — the
    architecture is naturally agnostic to both. Only `patch_len` / `stride`
    must match (so patch embedding weights apply identically).
    """

    def __init__(self, num_channels: int, time_dim: int, cfg: BackboneConfig = BackboneConfig()):
        super().__init__()
        self.cfg = cfg
        self.num_channels = int(num_channels)
        self.time_dim = int(time_dim)
        assert time_dim % cfg.stride == 0, "time_dim must be multiple of stride"
        self.num_patches = time_dim // cfg.stride

        self.revin = _RevIN(num_channels)
        self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        x = self.revin(x)
        # Patching: (B, C, T) → (B, C, N, P)
        x = x.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)
        N, P = x.shape[-2], x.shape[-1]
        # Channel-independent: (B*C, N, P) → linear → (B*C, N, d_model)
        x = x.reshape(B * C, N, P)
        tok = self.patch_embed(x) + self.pos_emb
        enc = self.encoder(tok)            # (B*C, N, d_model)
        return enc.view(B, C, N, -1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PatchTSTReconstructor(nn.Module):
    """Pretrain head: predict masked time-step values from backbone output.

    Reconstruction is per-channel, per-time-step. We project each patch
    embedding back to patch_len raw values, and the loss compares only
    masked positions.

    Loss = MSE on masked time steps, summed over channels.
    """

    def __init__(self, num_channels: int, time_dim: int, cfg: BackboneConfig = BackboneConfig()):
        super().__init__()
        self.backbone = PatchTSTBackbone(num_channels, time_dim, cfg)
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.patch_len),
        )

    def forward(self, masked_input: torch.Tensor) -> torch.Tensor:
        """masked_input: (B, C, T) → reconstructed (B, C, T)."""
        B, C, T = masked_input.shape
        enc = self.backbone(masked_input)            # (B, C, N, d_model)
        N = enc.shape[2]
        # Project each patch back to patch_len values.
        recon_patches = self.head(enc)                # (B, C, N, patch_len)
        # Stitch patches back to time dimension: (B, C, N*patch_len) = (B, C, T)
        recon = recon_patches.view(B, C, N * self.cfg.patch_len)
        return recon

    @staticmethod
    def loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MSE on masked positions only.
        recon, target: (B, C, T) — predicted vs ground truth
        mask: (B, T) — True = masked position (we predict here)
        """
        # Broadcast mask across channel dim: (B, 1, T)
        m = mask.unsqueeze(1).float()
        # Sum-of-squares on masked positions, divided by num masked × C
        diff = (recon - target) ** 2 * m
        denom = m.sum() * recon.shape[1] + 1e-9
        return diff.sum() / denom

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
