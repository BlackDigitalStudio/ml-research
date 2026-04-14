#!/usr/bin/env python3
"""Honest OOS evaluation — fixes the two methodology bugs behind the
inflated WR numbers from B3 / D2 / FQI / grid_test_ensemble.

Bugs uncovered 2026-04-14:

  BUG 1 (label-artifact WR):
    The existing grid uses `pnl_val = target_pnl` as realised PnL. In
    Rust `label_from_outcomes`, target_pnl is defined as:
        - UP   sample → pnl_long   (positive)
        - DOWN sample → pnl_short  (positive)
        - FLAT sample → max(pnl_long, pnl_short) ≤ 0 (negative)
    i.e. target_pnl > 0 iff y != FLAT — *regardless of which direction
    the primary actually picked*. Taking `pnl_val[take] > 0` as the win
    condition therefore measures "fraction of taken samples that are
    non-FLAT", not "fraction of trades that made money".

  BUG 2 (stacker in-sample leak):
    `scripts/fix_stacker_classweight.py` retrains the balanced stacker
    on the ENTIRE val set (`m_full.fit(X_stack, y_val, ...)`). Its
    argmax on the walk-forward tail therefore ≡ y on the tail, so
    "primary_pred != y & y != FL" shows up as 0 wrong-direction cases.
    This makes any downstream metric that depends on stacker_soft on
    val look better than it would be for a stacker that had never seen
    those labels.

This script reruns `rust_bridge.simulate_labels` to get per-sample
`pnl_long` and `pnl_short`, then realises PnL from whichever direction
the primary actually picked. With that in hand, WR becomes "fraction of
realised trades with positive net PnL" — the real scalping metric.

Works on the v3 cache (mid_paths + entry_long/short persisted). Primary
is pluggable via --primary {proxy_imbalance | stacker_soft |
stacker_soft_balanced}; the stacker variants come from
val_predictions.npz and require --val-predictions to be provided.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402

UP, DOWN, FLAT = 0, 1, 2
INITIAL_CAPITAL = 50.0


def _load_v3(cache_dir: Path) -> dict:
    candidates = sorted(cache_dir.glob("samples_v3_*_mid_paths.npy"))
    if not candidates:
        raise FileNotFoundError(f"No v3 cache in {cache_dir}")
    prefix = str(candidates[-1])[: -len("_mid_paths.npy")]
    return {
        "prefix":       prefix,
        "X_feat":       np.load(f"{prefix}_X_feat.npy"),
        "y":            np.load(f"{prefix}_y.npy"),
        "target_pnl":   np.load(f"{prefix}_pnl.npy"),
        "mid_paths":    np.load(f"{prefix}_mid_paths.npy"),
        "entry_long":   np.load(f"{prefix}_entry_long.npy"),
        "entry_short":  np.load(f"{prefix}_entry_short.npy"),
    }


def _sim(c: dict, tp: float, sl: float, to: int,
          partial: bool, trailing: bool) -> dict:
    n = len(c["entry_long"])
    return rust_bridge.simulate_labels(
        entry_long=c["entry_long"], entry_short=c["entry_short"],
        mid_paths=c["mid_paths"],
        tp_pct=np.full(n, tp, dtype=np.float64),
        sl_pct=np.full(n, sl, dtype=np.float64),
        timeout_ticks=np.full(n, to, dtype=np.int64),
        partial_enabled=partial, trailing_enabled=trailing,
    )


def _proxy_imbalance(X_feat: np.ndarray, lo: float = 0.15,
                      hi: float = -0.15) -> np.ndarray:
    """+1 if imbalance > 0.15, -1 if < -0.15, else 0."""
    imb = X_feat[:, 1]
    d = np.zeros(len(imb), dtype=np.int8)
    d[imb > lo] = +1
    d[imb < hi] = -1
    return d


def _realise(direction: np.ndarray, pnl_long: np.ndarray,
              pnl_short: np.ndarray) -> np.ndarray:
    """Pick pnl_long or pnl_short per sample based on direction choice."""
    return np.where(direction == +1, pnl_long,
             np.where(direction == -1, pnl_short, 0.0))


def _metric(realised: np.ndarray, taken_mask: np.ndarray, label: str) -> dict:
    r = realised[taken_mask]
    n = int(taken_mask.sum())
    wr = float((r > 0).mean() * 100) if n else 0.0
    s = float(r.sum())
    eq = INITIAL_CAPITAL * np.cumprod(1.0 + realised / 100.0) if len(realised) else np.array([INITIAL_CAPITAL])
    net = 100 * (float(eq[-1]) / INITIAL_CAPITAL - 1)
    peaks = np.maximum.accumulate(eq) if len(eq) else np.array([INITIAL_CAPITAL])
    dd = float(((peaks - eq) / np.maximum(peaks, 1e-12)).max()) * 100 if len(eq) else 0.0
    print(f"  {label:35s}  n={n:>6d}  WR={wr:>5.1f}%  sum={s:>+8.2f}%  "
          f"net={net:>+7.2f}%  DD={dd:>5.1f}%")
    return {"label": label, "n_trades": n, "win_rate_pct": wr,
             "sum_pnl_pct": s, "net_return_pct": net, "max_dd_pct": dd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/scalper/scalper-bot/data/_cache")
    ap.add_argument("--tp", type=float, default=0.20)
    ap.add_argument("--sl", type=float, default=0.10)
    ap.add_argument("--timeout-ticks", type=int, default=600)
    ap.add_argument("--partial", action="store_true", default=True)
    ap.add_argument("--no-partial", dest="partial", action="store_false")
    ap.add_argument("--trailing", action="store_true", default=True)
    ap.add_argument("--no-trailing", dest="trailing", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--out",
                    default="/home/scalper/scalper-bot/models/honest_eval.json")
    args = ap.parse_args()

    print(f"[honest] loading v3 cache from {args.cache_dir}")
    c = _load_v3(Path(args.cache_dir))
    n = len(c["y"])
    print(f"[honest] n={n} samples")

    print(f"[honest] running sim_labels  tp={args.tp}  sl={args.sl}  "
          f"to={args.timeout_ticks}  partial={args.partial}  "
          f"trailing={args.trailing}")
    sim = _sim(c, args.tp, args.sl, args.timeout_ticks,
                args.partial, args.trailing)
    pnl_long  = sim["pnl_long"]
    pnl_short = sim["pnl_short"]
    y = sim["y"]

    # Sanity: the claim that pnl_long / pnl_short differ materially from
    # target_pnl for the WRONG direction is empirically checkable.
    print(f"[honest] sim outcomes: UP={int((y==UP).sum())} "
          f"DOWN={int((y==DOWN).sum())} FLAT={int((y==FLAT).sum())}")
    print(f"[honest] pnl_long  mean={pnl_long.mean():+.5f}  pos_frac={(pnl_long>0).mean():.3f}")
    print(f"[honest] pnl_short mean={pnl_short.mean():+.5f}  pos_frac={(pnl_short>0).mean():.3f}")

    # Tail window (we don't train anything here — strategies are fixed-rule).
    tail_lo = int(n * (1 - args.val_frac))
    tail = slice(tail_lo, n)
    print(f"\n[honest] tail={n - tail_lo} samples ({args.val_frac*100:.0f}% tail)")

    print(f"\n{'strategy':37s}     n      WR      sum       net      DD")
    results = []

    # ORACLE — always pick TB winner direction. Upper bound.
    oracle_dir = np.where(y == UP, +1,
                    np.where(y == DOWN, -1, 0)).astype(np.int8)
    r = _realise(oracle_dir, pnl_long, pnl_short)
    results.append(_metric(r[tail], (oracle_dir[tail] != 0), "ORACLE (TB winner)"))

    # ALWAYS LONG  — take fraction=1 on every sample as LONG.
    r = _realise(np.full(n, +1, dtype=np.int8), pnl_long, pnl_short)
    results.append(_metric(r[tail], np.ones(n, dtype=bool)[tail],
                             "Always LONG every tick"))

    # PROXY: imbalance threshold direction.
    prox = _proxy_imbalance(c["X_feat"])
    r = _realise(prox, pnl_long, pnl_short)
    results.append(_metric(r[tail], (prox[tail] != 0),
                             "Proxy: imbalance threshold"))

    # PROXY + label-artifact WR — show the buggy metric alongside for
    # direct comparison with prior B3 / D2 / FQI numbers.
    artifact_pnl = np.where((prox != 0), c["target_pnl"].astype(np.float64), 0.0)
    results.append(_metric(artifact_pnl[tail], (prox[tail] != 0),
                             "Proxy via target_pnl (BUGGY)"))

    # Write JSON.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "cache_prefix": c["prefix"],
            "config": {"tp": args.tp, "sl": args.sl,
                        "timeout_ticks": args.timeout_ticks,
                        "partial": args.partial, "trailing": args.trailing,
                        "val_frac": args.val_frac},
            "strategies": results,
        }, f, indent=2)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
