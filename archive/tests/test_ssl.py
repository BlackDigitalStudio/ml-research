"""Contract tests for SSL masked-prediction pretraining."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.ssl.dataset import LOBWindowDataset, collate_masked
from src.ssl.model import BackboneConfig, PatchTSTBackbone, PatchTSTReconstructor


def _make_synth_depth_parquet(path: Path, n_rows: int = 10_000,
                                seed: int = 7, levels: int = 20):
    """Build a minimal flat-schema depth parquet for testing."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n_rows, dtype=np.int64) * 100 + 1_700_000_000_000
    # Random walk prices + exponential qtys
    bp = 67000.0 + np.cumsum(rng.normal(0, 1, size=(n_rows, levels)), axis=0) * 0.01
    bp = bp.astype(np.float64)
    bq = rng.exponential(1.0, size=(n_rows, levels)).astype(np.float64)
    ap = bp + rng.uniform(0.5, 2.0, size=(n_rows, levels))
    aq = rng.exponential(1.0, size=(n_rows, levels)).astype(np.float64)

    fsl = pa.list_(pa.float64(), levels)
    def _fsl(arr):
        return pa.FixedSizeListArray.from_arrays(
            pa.array(arr.reshape(-1), type=pa.float64()), levels
        )
    table = pa.table({
        "timestamp": pa.array(ts, type=pa.int64()),
        "bid_prices": _fsl(bp),
        "bid_qtys": _fsl(bq),
        "ask_prices": _fsl(ap),
        "ask_qtys": _fsl(aq),
    })
    pq.write_table(table, str(path))


def test_dataset_yields_correct_shape():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "depth.parquet"
        _make_synth_depth_parquet(p, n_rows=5000)
        ds = LOBWindowDataset([p], window_size=128, mask_ratio=0.15,
                               samples_per_epoch=20, seed=1)
        b = ds[0]
        assert b.input.shape == (80, 128)   # 4 streams × 20 levels, T=128
        assert b.target.shape == (80, 128)
        assert b.mask.shape == (128,)
        # input should be zeroed where mask=True
        assert (b.input[:, b.mask] == 0).all()
        # target at those positions is ORIGINAL (not zeroed)
        assert not (b.target[:, b.mask] == 0).all()


def test_dataset_skips_non_flat_schema():
    """Legacy list<list<f64>> files should be silently skipped."""
    with tempfile.TemporaryDirectory() as td:
        # flat
        flat_p = Path(td) / "flat.parquet"
        _make_synth_depth_parquet(flat_p, n_rows=2000)
        # legacy (wrong schema) — just a trivial int64 column
        legacy_p = Path(td) / "legacy.parquet"
        pq.write_table(pa.table({"timestamp": pa.array([1, 2, 3], type=pa.int64())}),
                        str(legacy_p))

        ds = LOBWindowDataset([flat_p, legacy_p], window_size=64,
                               samples_per_epoch=5, seed=2)
        # only flat should be indexed
        assert len(ds.files) == 1


def test_mask_ratio_respected():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "depth.parquet"
        _make_synth_depth_parquet(p, n_rows=3000)
        ds = LOBWindowDataset([p], window_size=200, mask_ratio=0.25,
                               samples_per_epoch=10, seed=3)
        masks = [ds[i].mask.sum().item() for i in range(10)]
        avg = sum(masks) / len(masks)
        assert abs(avg - 50) <= 2   # 25% of 200


def test_collate_batches_correctly():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "depth.parquet"
        _make_synth_depth_parquet(p, n_rows=2000)
        ds = LOBWindowDataset([p], window_size=64, samples_per_epoch=8)
        batch = collate_masked([ds[i] for i in range(4)])
        assert batch.input.shape == (4, 80, 64)
        assert batch.target.shape == (4, 80, 64)
        assert batch.mask.shape == (4, 64)


def test_backbone_output_shape():
    cfg = BackboneConfig(d_model=96, n_layers=2, patch_len=16, stride=16)
    bb = PatchTSTBackbone(num_channels=80, time_dim=128, cfg=cfg)
    x = torch.randn(4, 80, 128)
    enc = bb(x)
    # (B, C, N, d_model) — N = 128/16 = 8
    assert enc.shape == (4, 80, 8, 96)


def test_reconstructor_forward_backward():
    cfg = BackboneConfig(d_model=64, n_layers=2, patch_len=8, stride=8)
    m = PatchTSTReconstructor(num_channels=80, time_dim=64, cfg=cfg)
    x = torch.randn(2, 80, 64)
    target = torch.randn(2, 80, 64)
    mask = torch.zeros(2, 64, dtype=torch.bool); mask[:, ::4] = True
    recon = m(x)
    assert recon.shape == (2, 80, 64)
    loss = m.loss(recon, target, mask)
    assert loss.requires_grad
    loss.backward()


def test_reconstructor_loss_only_on_mask():
    cfg = BackboneConfig(d_model=32, n_layers=1, patch_len=8, stride=8)
    m = PatchTSTReconstructor(num_channels=80, time_dim=64, cfg=cfg)
    target = torch.randn(1, 80, 64)
    recon = target.clone()           # perfect reconstruction
    mask = torch.zeros(1, 64, dtype=torch.bool); mask[:, ::4] = True
    loss = m.loss(recon, target, mask)
    assert loss.item() == pytest.approx(0.0)

    # Differ only at MASKED positions → loss non-zero
    recon2 = target.clone()
    recon2[:, :, mask[0]] = 0.0
    loss2 = m.loss(recon2, target, mask)
    assert loss2.item() > 0.0

    # Differ only at UNMASKED positions → loss still 0
    recon3 = target.clone()
    recon3[:, :, ~mask[0]] = 999.0
    loss3 = m.loss(recon3, target, mask)
    assert loss3.item() == pytest.approx(0.0)
