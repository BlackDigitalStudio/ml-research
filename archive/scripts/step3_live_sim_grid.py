#!/usr/bin/env python3
"""Step 3 — wire src.live_sim / rust_bridge.simulate_labels as the grid backend.

Replaces the precomputed-target_pnl grid approach (which hard-bakes one
TP/SL/partial/trailing config at build_samples time) with per-config
forward simulation. Each grid cell re-runs the Rust sim_labels binary
with that cell's exit-strategy parameters — unblocks B1 (partial_tp
toggle, trailing_stop toggle, time-based exit sweep, parameterised
commissions).

Requires the v3 cache schema (mid_paths + entry_long/short sidecars).

Usage:
    python3 scripts/step3_live_sim_grid.py \\
        --cache-dir /home/scalper/scalper-bot/data/_cache \\
        --out       /home/scalper/backups/pod/recover_v2/step3_grid.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402


UP, DOWN, FLAT = 0, 1, 2


def _load_v3_cache(cache_dir: Path, hours: int = 50) -> dict:
    """Find v3 cache — X_feat / y / target_pnl / mid / mid_paths / entries."""
    candidates = sorted(cache_dir.glob("samples_v3_*_mid_paths.npy"))
    if not candidates:
        raise FileNotFoundError(
            f"No v3 mid_paths.npy in {cache_dir} — rebuild cache with updated "
            f"trainer.py (CACHE_SCHEMA_VERSION=v3)."
        )
    newest = candidates[-1]
    prefix = str(newest)[: -len("_mid_paths.npy")]
    return {
        "prefix": prefix,
        "X_feat":        np.load(f"{prefix}_X_feat.npy"),
        "y":             np.load(f"{prefix}_y.npy"),
        "target_pnl":    np.load(f"{prefix}_pnl.npy"),
        "mid":           np.load(f"{prefix}_mid.npy"),
        "mid_paths":     np.load(f"{prefix}_mid_paths.npy"),
        "entry_long":    np.load(f"{prefix}_entry_long.npy"),
        "entry_short":   np.load(f"{prefix}_entry_short.npy"),
    }


def _run_sim(entry_long, entry_short, mid_paths, *, tp_pct: float,
              sl_pct: float, timeout_ticks: int, commission_win: float,
              commission_loss: float, partial: bool, trailing: bool) -> dict:
    n = len(entry_long)
    tp = np.full(n, tp_pct, dtype=np.float64)
    sl = np.full(n, sl_pct, dtype=np.float64)
    to = np.full(n, timeout_ticks, dtype=np.int64)
    out = rust_bridge.simulate_labels(
        entry_long=entry_long, entry_short=entry_short,
        mid_paths=mid_paths, tp_pct=tp, sl_pct=sl, timeout_ticks=to,
        commission_win_pct=commission_win, commission_loss_pct=commission_loss,
        partial_enabled=partial, trailing_enabled=trailing,
    )
    return out


def summarise(out: dict, proxy_dir: np.ndarray | None = None) -> dict:
    """Oracle + (optional) proxy-primary metrics.

    `proxy_dir` is an optional per-sample direction array with values in
    {-1=SHORT, 0=SKIP, +1=LONG}. When present we realise pnl_long or
    pnl_short by the proxy's pick — that's the real grid metric a noisy
    primary would see. Without it we only report the TB-winner oracle.
    """
    y = out["y"]
    pnl = out["target_pnl"]
    n = len(y)
    n_up = int((y == UP).sum())
    n_dn = int((y == DOWN).sum())
    n_fl = n - n_up - n_dn
    non_flat_mask = y != FLAT
    pnl_t = pnl[non_flat_mask]
    wr_nf = float((pnl_t > 0).mean() * 100) if len(pnl_t) else 0.0
    summary = {
        "n": n, "n_up": n_up, "n_dn": n_dn, "n_fl": n_fl,
        "non_flat_frac": float((non_flat_mask).mean()),
        "pnl_mean_all_pct": float(pnl.mean()),
        "pnl_mean_nonflat_pct": float(pnl_t.mean()) if len(pnl_t) else 0.0,
        "oracle_sum_nonflat_pct": float(pnl_t.sum()),
        "oracle_wr_pct": wr_nf,
    }
    if proxy_dir is not None:
        pnl_long = out["pnl_long"]
        pnl_short = out["pnl_short"]
        realised = np.where(proxy_dir == +1, pnl_long,
                              np.where(proxy_dir == -1, pnl_short, 0.0))
        take = proxy_dir != 0
        realised_t = realised[take]
        n_t = int(take.sum())
        summary["proxy_n_trades"] = n_t
        summary["proxy_wr_pct"] = float((realised_t > 0).mean() * 100) if n_t else 0.0
        summary["proxy_sum_pnl_pct"] = float(realised_t.sum())
        summary["proxy_mean_pnl_pct"] = float(realised_t.mean()) if n_t else 0.0
    return summary


def imbalance_proxy_direction(X_feat: np.ndarray,
                               long_thr: float = 0.15,
                               short_thr: float = -0.15) -> np.ndarray:
    """Simplest non-trivial primary: +1 if imbalance_ratio > 0.15, -1 if
    < -0.15, else skip. Uses feature column 1 (`imbalance_ratio`)."""
    imb = X_feat[:, 1]
    d = np.zeros(len(imb), dtype=np.int8)
    d[imb > long_thr] = +1
    d[imb < short_thr] = -1
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/scalper/scalper-bot/data/_cache")
    ap.add_argument("--out",
                    default="/home/scalper/scalper-bot/models/step3_livesim_grid.json")
    ap.add_argument("--tp-grid", default="0.10,0.15,0.20,0.30",
                    help="comma-separated TP%")
    ap.add_argument("--sl-grid", default="0.05,0.10,0.15,0.20",
                    help="comma-separated SL%")
    ap.add_argument("--timeout-grid", default="300,600,1200",
                    help="comma-separated timeout_ticks (100ms each)")
    ap.add_argument("--commission-loss-grid", default="0.05,0.07,0.10",
                    help="comma-separated commission_loss_pct values")
    args = ap.parse_args()

    print(f"[step3] loading v3 cache from {args.cache_dir}")
    c = _load_v3_cache(Path(args.cache_dir))
    n = len(c["y"])
    print(f"[step3] cache: N={n}, mid_paths={c['mid_paths'].shape}, "
          f"entry_long={c['entry_long'].shape}")

    tp_list = [float(x) for x in args.tp_grid.split(",")]
    sl_list = [float(x) for x in args.sl_grid.split(",")]
    to_list = [int(x) for x in args.timeout_grid.split(",")]
    cl_list = [float(x) for x in args.commission_loss_grid.split(",")]
    partial_list = [True, False]
    trailing_list = [True, False]

    configs = list(itertools.product(
        tp_list, sl_list, to_list, cl_list, partial_list, trailing_list,
    ))
    print(f"[step3] {len(configs)} configs to evaluate")

    # Proxy primary — imbalance-threshold direction picker. Gives a real
    # grid metric (non-oracle). Primaries on a pod run would replace this.
    proxy_dir = imbalance_proxy_direction(c["X_feat"])
    print(f"[step3] proxy primary trades: {int((proxy_dir != 0).sum())} "
          f"({100 * (proxy_dir != 0).mean():.1f}%)")

    t0 = time.time()
    rows = []
    for ci, (tp, sl, to, cl, partial, trailing) in enumerate(configs):
        if ci % 10 == 0:
            print(f"  [{ci+1:4d}/{len(configs)}]  "
                  f"tp={tp} sl={sl} to={to} cl={cl} "
                  f"partial={partial} trailing={trailing} "
                  f"(elapsed {time.time() - t0:.1f}s)")
        out = _run_sim(c["entry_long"], c["entry_short"], c["mid_paths"],
                        tp_pct=tp, sl_pct=sl, timeout_ticks=to,
                        commission_win=0.04, commission_loss=cl,
                        partial=partial, trailing=trailing)
        summ = summarise(out, proxy_dir=proxy_dir)
        rows.append({
            "tp_pct": tp, "sl_pct": sl, "timeout_ticks": to,
            "commission_loss_pct": cl,
            "partial_enabled": partial, "trailing_enabled": trailing,
            **summ,
        })

    dt = time.time() - t0
    print(f"\n[step3] DONE — {len(rows)} configs in {dt:.1f}s")

    # Rank by oracle_sum_nonflat_pct — the best config for TB-winner trades.
    rows_sorted = sorted(rows, key=lambda r: -r["oracle_sum_nonflat_pct"])
    print(f"\nTop 10 configs by oracle_sum (sum of non-FLAT target_pnl):")
    print(f"  {'tp':>5s} {'sl':>5s} {'to':>5s} {'cl':>5s} {'part':>4s} {'trail':>5s}  "
          f"{'n_nf':>6s} {'nf%':>5s} {'sumPnl%':>8s} {'wr%':>5s}")
    for r in rows_sorted[:10]:
        print(f"  {r['tp_pct']:>5.2f} {r['sl_pct']:>5.2f} {r['timeout_ticks']:>5d} "
              f"{r['commission_loss_pct']:>5.2f} "
              f"{str(r['partial_enabled'])[:1]:>4s} "
              f"{str(r['trailing_enabled'])[:1]:>5s}  "
              f"{r['n_up']+r['n_dn']:>6d} {r['non_flat_frac']*100:>4.1f}%  "
              f"{r['oracle_sum_nonflat_pct']:>+8.2f} {r['oracle_wr_pct']:>5.1f}")

    # Isolated ablations — hold other params at defaults.
    def _cell(rs, tp, sl, to, cl, p, tr):
        for r in rs:
            if (r["tp_pct"] == tp and r["sl_pct"] == sl
                and r["timeout_ticks"] == to and r["commission_loss_pct"] == cl
                and r["partial_enabled"] == p and r["trailing_enabled"] == tr):
                return r
        return None

    print(f"\nAblations at tp=0.20 sl=0.10 timeout=600 cl=0.07:")
    for p, tr in [(True, True), (True, False), (False, True), (False, False)]:
        r = _cell(rows, 0.20, 0.10, 600, 0.07, p, tr)
        if r is None:
            continue
        print(f"  partial={p}  trailing={tr}  ->  "
              f"oracle sum={r['oracle_sum_nonflat_pct']:+.2f}%  "
              f"|  proxy n={r.get('proxy_n_trades', 0)} "
              f"WR={r.get('proxy_wr_pct', 0):.1f}% "
              f"sum={r.get('proxy_sum_pnl_pct', 0):+.2f}% "
              f"mean={r.get('proxy_mean_pnl_pct', 0):+.4f}%")

    # Top 10 by PROXY sum (realistic grid ranking).
    proxy_rows = sorted(rows, key=lambda r: -r.get("proxy_sum_pnl_pct", 0))[:10]
    print(f"\nTop 10 configs by proxy_sum (imbalance-threshold primary):")
    print(f"  {'tp':>5s} {'sl':>5s} {'to':>5s} {'cl':>5s} {'part':>4s} {'trail':>5s}  "
          f"{'n':>5s} {'WR%':>5s} {'sumPnl%':>8s} {'meanPnl%':>9s}")
    for r in proxy_rows:
        print(f"  {r['tp_pct']:>5.2f} {r['sl_pct']:>5.2f} {r['timeout_ticks']:>5d} "
              f"{r['commission_loss_pct']:>5.2f} "
              f"{str(r['partial_enabled'])[:1]:>4s} "
              f"{str(r['trailing_enabled'])[:1]:>5s}  "
              f"{r.get('proxy_n_trades',0):>5d} {r.get('proxy_wr_pct',0):>4.1f}% "
              f"{r.get('proxy_sum_pnl_pct',0):>+8.2f} "
              f"{r.get('proxy_mean_pnl_pct',0):>+9.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"configs": rows, "cache_prefix": c["prefix"],
                    "elapsed_s": dt}, f, indent=2)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
