#!/usr/bin/env python3
"""Grid sweep using the v2 retargeted meta — finds (TP, SL, timeout, thr)
combos that produce positive net PnL on the eval tail.

Differs from grid_live.py: uses stacker_meta_v2.npz which trains meta on
realised PnL (`pnl > BE`) instead of classification correctness.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge                        # noqa: E402


CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
V2_PATH = Path("/home/scalper/scalper-bot/models/stacker_meta_v2.npz")
OUT = Path("/home/scalper/scalper-bot/models/grid_live_v5_retarget.json")

# Grid — slightly narrower than wide grid to keep runtime manageable.
TP_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SL_GRID = [0.10, 0.12, 0.15, 0.18]
TIMEOUT_GRID = [600, 900, 1200, 1500, 1800]
META_THR_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
MIN_PROB_GRID = [0.50, 0.55]
SPREAD_BPS_GRID = [0, 2]
FILL_PROB_GRID = [1.0, 0.8]
KELLY_FRAC = 0.25   # fixed for this sweep


def _load_cache():
    cand = sorted(CACHE_DIR.glob("samples_v3_*_mid_paths.npy"))
    prefix = str(cand[-1])[: -len("_mid_paths.npy")]
    return {
        "prefix": prefix,
        "y": np.load(f"{prefix}_y.npy"),
        "pnl": np.load(f"{prefix}_pnl.npy"),
        "mid_paths": np.load(f"{prefix}_mid_paths.npy"),
        "entry_long": np.load(f"{prefix}_entry_long.npy"),
        "entry_short": np.load(f"{prefix}_entry_short.npy"),
    }


def main():
    print("[grid_rt] loading cache + meta v2")
    c = _load_cache()
    d = np.load(V2_PATH, allow_pickle=False)
    stacker_soft = d["stacker_soft"]
    meta_prob = d["meta_prob"]
    n_tr = int(d["n_train"])
    N = c["y"].shape[0]

    primary_pred = stacker_soft.argmax(axis=-1)
    primary_max = stacker_soft.max(axis=-1)
    non_flat = primary_pred != 2       # FLAT=2

    print(f"[grid_rt] N={N:,}  tail={N - n_tr:,}  non_flat_tail={(non_flat[n_tr:]).sum():,}")

    # Evaluate TP × SL × timeout once, reuse for inner grid.
    rows = []
    t_start = time.monotonic()
    combos = list(itertools.product(TP_GRID, SL_GRID, TIMEOUT_GRID))
    print(f"[grid_rt] {len(combos)} outer configs × {len(META_THR_GRID)*len(MIN_PROB_GRID)*len(SPREAD_BPS_GRID)*len(FILL_PROB_GRID)} inner")

    rng = np.random.default_rng(42)
    for i_combo, (tp, sl, to_ticks) in enumerate(combos):
        tp_arr = np.full(N, tp, dtype=np.float64)
        sl_arr = np.full(N, sl, dtype=np.float64)
        to_arr = np.full(N, to_ticks, dtype=np.int64)
        out = rust_bridge.simulate_labels(
            c["entry_long"], c["entry_short"], c["mid_paths"],
            tp_arr, sl_arr, to_arr,
            commission_win_pct=0.04, commission_loss_pct=0.07,
            partial_enabled=True, trailing_enabled=True, fill_latency_ms=150.0,
        )
        pnl_long = out["pnl_long"].astype(np.float64)
        pnl_short = out["pnl_short"].astype(np.float64)

        # Directional realised PnL per sample at this outer config
        real = np.where(primary_pred == 0, pnl_long,
                np.where(primary_pred == 1, pnl_short, 0.0))

        for thr, min_p, spread_bps, fp in itertools.product(
            META_THR_GRID, MIN_PROB_GRID, SPREAD_BPS_GRID, FILL_PROB_GRID
        ):
            spread_cost = spread_bps / 100.0  # bps → pct
            gate = non_flat & (primary_max >= min_p) & (meta_prob >= thr)
            if not gate[n_tr:].any():
                continue

            # Fill Bernoulli drop — deterministic via seed
            fill_mask = rng.random(N) < fp if fp < 1.0 else np.ones(N, dtype=bool)

            take = gate & fill_mask
            real_net = (real - spread_cost) * KELLY_FRAC * take

            eval_mask = np.zeros(N, dtype=bool)
            eval_mask[n_tr:] = True
            real_eval = real_net[eval_mask]
            n_trades_eval = int(take[eval_mask].sum())
            if n_trades_eval < 10:
                continue
            trades = real_eval[take[eval_mask]]
            wr = float((trades > 0).mean())
            net = float(trades.sum())
            sharpe = float(trades.mean() / (trades.std() + 1e-9) * np.sqrt(len(trades)))
            equity_curve = np.cumsum(trades)
            max_dd = float(np.max(equity_curve.max() - equity_curve) if len(equity_curve) else 0)

            rows.append({
                "tp": tp, "sl": sl, "timeout": to_ticks,
                "meta_thr": thr, "min_prob": min_p,
                "spread_bps": spread_bps, "fill_prob": fp,
                "n_trades": n_trades_eval,
                "win_rate_pct": wr * 100,
                "net_pct": net,
                "max_dd_pct": max_dd,
                "sharpe": sharpe,
            })

        if (i_combo + 1) % 20 == 0:
            dt = time.monotonic() - t_start
            print(f"[grid_rt] combo {i_combo+1}/{len(combos)}  {dt:.1f}s elapsed  rows={len(rows)}")

    dt = time.monotonic() - t_start
    print(f"[grid_rt] done in {dt:.1f}s  total rows={len(rows)}")

    # Rank + save
    rows.sort(key=lambda r: -r["net_pct"])
    profitable = [r for r in rows if r["net_pct"] > 0]
    print(f"[grid_rt] profitable configs (net>0): {len(profitable)} / {len(rows)}")

    result = {
        "n_samples": N,
        "n_train": n_tr,
        "n_eval": N - n_tr,
        "top_by_net": rows[:50],
        "top_by_sharpe": sorted(rows, key=lambda r: -r["sharpe"])[:50],
        "n_rows": len(rows),
        "n_profitable": len(profitable),
    }
    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(f"[grid_rt] wrote {OUT}")

    if profitable:
        print("\n=== TOP 10 BY NET ===")
        print(f"{'tp':>5} {'sl':>5} {'to_s':>5} {'thr':>5} {'mp':>4} {'spb':>4} {'fp':>4} "
              f"{'n':>5} {'WR%':>6} {'net%':>7} {'DD%':>6} {'sharpe':>7}")
        # net_pct is already in percent units (sum of trade pnl_pct * kelly,
        # each trade pnl_pct is 0.15 = 0.15 %). Do not *100.
        for r in rows[:10]:
            print(f"{r['tp']:>5.2f} {r['sl']:>5.2f} {r['timeout']/10:>4.0f}s "
                  f"{r['meta_thr']:>5.2f} {r['min_prob']:>4.2f} {r['spread_bps']:>4.0f} "
                  f"{r['fill_prob']:>4.1f} {r['n_trades']:>5d} "
                  f"{r['win_rate_pct']:>5.1f}% {r['net_pct']:>+6.2f}% "
                  f"{r['max_dd_pct']:>5.2f}% {r['sharpe']:>+7.2f}")


if __name__ == "__main__":
    main()
