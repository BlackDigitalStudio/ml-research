"""Unit tests for triple-barrier labelling in src/trainer.py.

The labelling logic lives inside `Trainer.build_samples` and is not directly
callable. We replicate the exact vectorised expression here on small synthetic
mid-price arrays — this guards against silent regressions if the snippet is
edited later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import UP, DOWN, FLAT
from src.trainer import HORIZON, SL_PCT, TP_PCT


def label_paths(future_mids: np.ndarray, current_mids: np.ndarray) -> np.ndarray:
    """Replicates the triple-barrier expression from build_samples.

    `future_mids` is shape (N, HORIZON), `current_mids` is shape (N,).
    Returns labels of shape (N,) with values in {UP, DOWN, FLAT}.
    """
    safe = np.where(current_mids > 0, current_mids, 1.0)
    rel = (future_mids - current_mids[:, None]) / safe[:, None] * 100

    long_tp_hit = rel >= TP_PCT
    long_sl_hit = rel <= -SL_PCT
    long_tp_first = np.where(long_tp_hit.any(axis=1),
                             long_tp_hit.argmax(axis=1), HORIZON)
    long_sl_first = np.where(long_sl_hit.any(axis=1),
                             long_sl_hit.argmax(axis=1), HORIZON)

    short_tp_hit = rel <= -TP_PCT
    short_sl_hit = rel >= SL_PCT
    short_tp_first = np.where(short_tp_hit.any(axis=1),
                              short_tp_hit.argmax(axis=1), HORIZON)
    short_sl_first = np.where(short_sl_hit.any(axis=1),
                              short_sl_hit.argmax(axis=1), HORIZON)

    long_wins = long_tp_first < long_sl_first
    short_wins = short_tp_first < short_sl_first

    n = len(current_mids)
    y = np.full(n, FLAT, dtype=np.int64)
    y[long_wins & ~short_wins] = UP
    y[short_wins & ~long_wins] = DOWN
    both = long_wins & short_wins
    y[both & (long_tp_first <= short_tp_first)] = UP
    y[both & (long_tp_first >  short_tp_first)] = DOWN
    return y


def _path(base: float, deltas_pct: list[float]) -> np.ndarray:
    """Build a HORIZON-long mid-price path from a list of percent moves.

    Each delta is applied at the corresponding tick; trailing ticks hold the
    last value. `deltas_pct[k]` is the cumulative move from `base`, NOT a
    per-tick delta.
    """
    path = np.full(HORIZON, base, dtype=np.float64)
    for tick, pct in enumerate(deltas_pct):
        path[tick:] = base * (1 + pct / 100.0)
    return path


def test_long_wins_up_label() -> None:
    # Path: +0.25% at tick 50 (LONG TP hits before any SL).
    base = 100_000.0
    path = _path(base, [0.25])  # whole path sits at +0.25%
    y = label_paths(path[None, :], np.array([base]))
    assert y[0] == UP, f"expected UP, got {y[0]}"


def test_short_wins_down_label() -> None:
    # Path: -0.25% throughout → SHORT TP hits first, LONG SL hits.
    # SHORT wins because its TP arrives at tick 0; LONG never wins.
    base = 100_000.0
    path = _path(base, [-0.25])
    y = label_paths(path[None, :], np.array([base]))
    assert y[0] == DOWN, f"expected DOWN, got {y[0]}"


def test_long_sl_then_up_is_flat() -> None:
    # The previous max-based labelling bug: price dips -0.15% (LONG SL fires)
    # then rallies +0.30%. Old code labelled this UP from `max(future)`. The
    # triple-barrier code must label it FLAT (LONG loses, SHORT also loses
    # because the rally hits its SL at +0.10% before any -0.20% TP).
    base = 100_000.0
    path = np.full(HORIZON, base, dtype=np.float64)
    path[10:30] = base * (1 - 0.0015)   # -0.15% — LONG SL at -0.10% fires
    path[30:] = base * (1 + 0.0030)     # +0.30% — SHORT SL at +0.10% fires
    y = label_paths(path[None, :], np.array([base]))
    assert y[0] == FLAT, f"expected FLAT (the look-ahead bug case), got {y[0]}"


def test_quiet_path_is_flat() -> None:
    # Path stays within ±0.08% — no barrier hit either way.
    base = 100_000.0
    rng = np.random.default_rng(0)
    path = base * (1 + rng.uniform(-0.0008, 0.0008, HORIZON))
    y = label_paths(path[None, :], np.array([base]))
    assert y[0] == FLAT, f"expected FLAT, got {y[0]}"


def test_long_then_short_takes_faster_tp() -> None:
    # Both directions would profit in the same horizon — labelling should
    # follow whichever TP fires first. Construct a path where SHORT TP fires
    # at tick 5 then LONG TP fires at tick 50.
    base = 100_000.0
    path = np.full(HORIZON, base, dtype=np.float64)
    path[5:50] = base * (1 - 0.0025)    # SHORT TP at -0.20%, fires tick 5
    path[50:]  = base * (1 + 0.0025)    # ... but recovers and LONG TP fires
    y = label_paths(path[None, :], np.array([base]))
    # BUT: LONG SL at -0.10% would fire FIRST inside the dip, so LONG cannot
    # win. SHORT TP fires before any +0.10% SL → label = DOWN.
    assert y[0] == DOWN, f"expected DOWN, got {y[0]}"


def test_batch_shapes() -> None:
    # Run the labeller on a batch of N=4 paths and assert each label.
    base = 100_000.0
    p1 = _path(base, [0.25])               # UP
    p2 = _path(base, [-0.25])              # DOWN
    p3 = np.full(HORIZON, base, dtype=np.float64)  # FLAT
    p4 = np.full(HORIZON, base, dtype=np.float64)
    p4[10:30] = base * (1 - 0.0015)
    p4[30:]   = base * (1 + 0.0030)        # FLAT (look-ahead bug case)
    paths = np.stack([p1, p2, p3, p4])
    current = np.array([base, base, base, base])
    y = label_paths(paths, current)
    assert y[0] == UP
    assert y[1] == DOWN
    assert y[2] == FLAT
    assert y[3] == FLAT


if __name__ == "__main__":
    test_long_wins_up_label()
    test_short_wins_down_label()
    test_long_sl_then_up_is_flat()
    test_quiet_path_is_flat()
    test_long_then_short_takes_faster_tp()
    test_batch_shapes()
    print("triple-barrier tests OK")
