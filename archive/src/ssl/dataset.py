"""LOBWindowDataset — stream random LOB windows from flat depth parquets.

Memory-efficient: opens parquets via pyarrow, samples random windows on
demand. No full-dataset materialization.

For each `__getitem__`, picks a random parquet file weighted by row count,
then a random start position, returns a (channels, T) tensor along with
a binary mask of which time steps to predict (BERT-style masked modeling).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass
class MaskedLOBBatch:
    """One mini-batch of (input, target, mask) for masked LOB modeling."""
    input: torch.Tensor      # (B, C, T) — float32, masked positions zeroed
    target: torch.Tensor     # (B, C, T) — float32, ORIGINAL values
    mask: torch.Tensor       # (B, T)    — bool, True = predict here


class LOBWindowDataset(Dataset):
    """Random LOB windows from flat-schema depth parquets.

    `__len__` is virtual — returns `samples_per_epoch`. `__getitem__` ignores
    the index and returns a uniformly random window across all files. Use
    `num_workers > 0` in DataLoader for parallel I/O.

    File-weighted sampling: each parquet is picked with probability
    proportional to its row count, so longer files contribute more samples.
    """

    def __init__(
        self,
        depth_paths: Sequence[Path | str],
        window_size: int = 256,
        mask_ratio: float = 0.15,
        samples_per_epoch: int = 100_000,
        seed: int = 42,
        normalize: bool = True,
        levels: int = 20,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.mask_ratio = float(mask_ratio)
        self.samples_per_epoch = int(samples_per_epoch)
        self.normalize = bool(normalize)
        self.levels = int(levels)

        # Index files: (path, n_rows). Skip non-flat-schema files silently
        # (recorder hourly files use legacy nested schema).
        self.files: list[tuple[Path, int]] = []
        for p in depth_paths:
            p = Path(p)
            try:
                meta = pq.read_metadata(str(p))
                schema = pq.read_schema(str(p))
            except Exception:
                continue
            names = set(schema.names)
            if not {"bid_prices", "bid_qtys", "ask_prices", "ask_qtys"} <= names:
                continue
            n = meta.num_rows
            if n > self.window_size + 1:
                self.files.append((p, n))

        if not self.files:
            raise FileNotFoundError("No flat-schema depth parquets found for SSL pretraining.")

        total = sum(n for _, n in self.files)
        # Weights for file-sampling: ∝ n_rows.
        self.weights = np.array([n / total for _, n in self.files], dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        # Per-file lazy parquet handles. Keep open across samples to avoid
        # repeated open/close (pq.ParquetFile is lightweight + supports
        # row-group level random access).
        self._handles: dict[Path, pq.ParquetFile] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _handle(self, path: Path) -> pq.ParquetFile:
        h = self._handles.get(path)
        if h is None:
            h = pq.ParquetFile(str(path))
            self._handles[path] = h
        return h

    def _read_window(self, path: Path, start: int, length: int) -> np.ndarray:
        """Read [start:start+length) rows from a flat-schema depth parquet.
        Returns (C=4*levels, length) f32 array.
        """
        # ParquetFile doesn't support row-level slicing across row-groups
        # directly; cheapest correct path: read full table (cached by OS),
        # slice. For pretraining at random positions this is fine on the
        # network mfs — OS page cache amortizes after first touch.
        h = self._handle(path)
        tbl = h.read(columns=["bid_prices", "bid_qtys", "ask_prices", "ask_qtys"])
        tbl = tbl.combine_chunks()
        bp = tbl["bid_prices"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, self.levels)
        bq = tbl["bid_qtys"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, self.levels)
        ap = tbl["ask_prices"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, self.levels)
        aq = tbl["ask_qtys"].chunk(0).values.to_numpy(zero_copy_only=True).reshape(-1, self.levels)
        end = start + length
        bp = bp[start:end]; bq = bq[start:end]
        ap = ap[start:end]; aq = aq[start:end]
        # Channel order: bid_prices, bid_qtys, ask_prices, ask_qtys
        # Stack to (C=4*levels, T=length) so it matches the (channels-first,
        # time-last) convention used by PatchTST.
        x = np.concatenate([bp, bq, ap, aq], axis=1).astype(np.float32)  # (T, 4*L)
        return x.T  # (4*L, T)

    @staticmethod
    def _normalize_window(x: np.ndarray) -> np.ndarray:
        """RevIN-style per-channel normalization within window.
        Returns x normalized in-place style (returns NEW array).
        """
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-5
        return (x - mean) / std

    def __getitem__(self, _idx: int) -> MaskedLOBBatch:
        # File-weighted random pick
        f_idx = self.rng.choice(len(self.files), p=self.weights)
        path, n_rows = self.files[f_idx]
        max_start = n_rows - self.window_size
        start = int(self.rng.integers(0, max_start))
        x = self._read_window(path, start, self.window_size)        # (C, T) f32
        if self.normalize:
            x = self._normalize_window(x)

        # Random time-step mask (BERT-style).
        T = self.window_size
        n_mask = max(1, int(round(T * self.mask_ratio)))
        mask_idx = self.rng.choice(T, size=n_mask, replace=False)
        mask = np.zeros(T, dtype=bool)
        mask[mask_idx] = True

        target = x.copy()
        masked_input = x.copy()
        masked_input[:, mask] = 0.0   # zero-out masked positions

        return MaskedLOBBatch(
            input=torch.from_numpy(masked_input),
            target=torch.from_numpy(target),
            mask=torch.from_numpy(mask),
        )


def collate_masked(batch: list[MaskedLOBBatch]) -> MaskedLOBBatch:
    return MaskedLOBBatch(
        input=torch.stack([b.input for b in batch], dim=0),
        target=torch.stack([b.target for b in batch], dim=0),
        mask=torch.stack([b.mask for b in batch], dim=0),
    )
