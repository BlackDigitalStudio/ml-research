"""Self-supervised pretraining loop for PatchTSTBackbone on raw LOB.

Trains via masked-prediction over flat-schema depth parquets. Saves
backbone weights (not the reconstruction head) so downstream classifiers
can load them via `model.backbone.load_state_dict(...)`.

Run on GPU pod for serious training. ~64M unlabeled snapshots → 1M
windows × N epochs is the typical recipe. With Mamba-2 / PatchTST scale
(few M params), expect 6-12 hours on 1× RTX PRO 4500.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LOBWindowDataset, collate_masked
from .model import BackboneConfig, PatchTSTReconstructor


@dataclass
class PretrainConfig:
    # Architecture
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    num_channels: int = 80          # 4 streams (bid_p, bid_q, ask_p, ask_q) × 20 levels
    window_size: int = 256          # ticks per training window (~25.6s of LOB)

    # Masking
    mask_ratio: float = 0.20

    # Training
    batch_size: int = 256
    samples_per_epoch: int = 50_000
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05

    # Data
    num_workers: int = 4

    # I/O
    save_every_epochs: int = 1
    seed: int = 42


def pretrain_loop(
    depth_paths: Sequence[Path | str],
    output_dir: Path | str,
    cfg: PretrainConfig = PretrainConfig(),
    device: str | None = None,
) -> dict:
    """Run masked-prediction pretraining on LOB windows.

    Saves `<output_dir>/backbone_epoch{N}.pt` per epoch + `final_backbone.pt`
    at the end. These can be loaded into downstream PatchTST classifiers
    via `model.backbone.load_state_dict(torch.load(...))`.

    Returns a dict with training history.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"[ssl] device={device} window={cfg.window_size} mask_ratio={cfg.mask_ratio}")
    print(f"[ssl] depth files: {len(depth_paths)}")

    ds = LOBWindowDataset(
        depth_paths=depth_paths,
        window_size=cfg.window_size,
        mask_ratio=cfg.mask_ratio,
        samples_per_epoch=cfg.samples_per_epoch,
        seed=cfg.seed,
        normalize=True,
    )
    print(f"[ssl] indexed {len(ds.files)} flat-schema files, "
          f"samples/epoch={cfg.samples_per_epoch}")

    dl = DataLoader(
        ds, batch_size=cfg.batch_size,
        shuffle=False,            # dataset already random
        num_workers=cfg.num_workers,
        collate_fn=collate_masked,
        pin_memory=(device != "cpu"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = PatchTSTReconstructor(
        num_channels=cfg.num_channels,
        time_dim=cfg.window_size,
        cfg=cfg.backbone,
    ).to(device)
    n_params = model.num_params()
    print(f"[ssl] model params: {n_params/1e6:.2f}M (backbone+head)")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(dl)
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        t0 = time.time()
        for batch in dl:
            x = batch.input.to(device, non_blocking=True)
            y = batch.target.to(device, non_blocking=True)
            m = batch.mask.to(device, non_blocking=True)
            optim.zero_grad()
            recon = model(x)
            loss = model.loss(recon, y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            ep_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        ep_loss /= max(n_seen, 1)
        dt = time.time() - t0
        history.append({"epoch": epoch + 1, "loss": ep_loss, "seconds": dt})
        print(f"[ssl] epoch {epoch + 1:3d}/{cfg.epochs}  loss={ep_loss:.5f}  "
              f"{n_seen} samples in {dt:.0f}s ({n_seen/dt:.0f}/s)")

        # Save backbone state every N epochs.
        if (epoch + 1) % cfg.save_every_epochs == 0:
            ckpt = output_dir / f"backbone_epoch{epoch + 1}.pt"
            torch.save(model.backbone.state_dict(), ckpt)

    # Final backbone weights — what downstream classifiers load.
    torch.save(model.backbone.state_dict(), output_dir / "final_backbone.pt")
    # Also save the full reconstructor for diagnostics.
    torch.save(model.state_dict(), output_dir / "final_reconstructor.pt")
    return {"history": history, "output_dir": str(output_dir)}
