#!/usr/bin/env python3
"""grid_live — direction-aware strategy grid for live-deploy selection.

Replaces scripts/grid_test_ensemble.py. The core fix: realised PnL is
computed from `pnl_long[i]` or `pnl_short[i]` (from Rust simulate_labels)
according to the direction *the primary actually picked*, not from the
TB-winner `target_pnl`.

What the grid does — in order

  1. Load primary softmaxes (inferred once by scripts/infer_primaries_v3.py
     on the v3 cache; can be rerun when a pod retrains the primaries).
  2. Walk-forward split — first 75% of samples train, tail 25% eval.
  3. Train L2 XGBoost stacker on train only; inference over full series.
  4. Train XGBoost meta (Lopez-de-Prado) on train only; inference over full.
  5. For each (TP, SL, timeout, partial, trailing) outer combo run Rust
     simulate_labels ONCE, caching pnl_long / pnl_short. Inner sweeps over
     Kelly / meta_thr / min_prob / spread_cost / fill_prob re-use that
     cache with no extra sim cost.
  6. For each full config, realise per-trade PnL as
         primary_pred == UP   → pnl_long  - spread_cost_pct
         primary_pred == DOWN → pnl_short - spread_cost_pct
         primary_pred == FLAT → 0
     multiply by Kelly fraction, apply post-only fill probability
     (Bernoulli drop), compound into equity, compute WR / Sharpe / DD.
  7. Rank by user-selected metric, write JSON + top-K printout.

Realism knobs worth sweeping

  * spread_cost_bps — additional bps lost to spread on round-trip (on top
    of commissions; default grid 0 / 2 / 4 bps).
  * fill_prob — probability a LIMIT GTX entry actually fills before the
    2-sec timeout (default 1.0 / 0.7 / 0.5). Drops are Bernoulli; we fix
    a deterministic seed per config so fills are replayable.
  * timeout_ticks — already inside Rust sim; sweep {300, 600, 1200}.
  * partial_enabled, trailing_enabled — Rust sim toggles.

Output

  JSON with `top_by_net`, `top_by_sharpe`, `top_by_dd_adj`, and the full
  `rows` list. Use this to pick a deploy candidate, then re-verify on an
  independent test slice (not in this script's scope).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402
from src.models.meta_label import MetaConfig, build_meta_dataset, train_meta  # noqa: E402

UP, DOWN, FLAT = 0, 1, 2
INITIAL_CAPITAL = 50.0


def _load_v3(cache_dir: Path) -> dict:
    cand = sorted(cache_dir.glob("samples_v3_*_mid_paths.npy"))
    if not cand:
        raise FileNotFoundError(f"No v3 cache in {cache_dir}")
    prefix = str(cand[-1])[: -len("_mid_paths.npy")]
    return {
        "prefix":       prefix,
        "X_feat":       np.load(f"{prefix}_X_feat.npy"),
        "y":            np.load(f"{prefix}_y.npy"),
        "mid_paths":    np.load(f"{prefix}_mid_paths.npy"),
        "entry_long":   np.load(f"{prefix}_entry_long.npy"),
        "entry_short":  np.load(f"{prefix}_entry_short.npy"),
    }


def _walk_forward_stacker_meta(primary_softs: list[np.ndarray], y: np.ndarray,
                                 pnl_for_meta: np.ndarray, n_tr: int,
                                 seed: int = 42):
    """Train XGBoost stacker + meta on the FIRST n_tr samples only.

    We intentionally skip the full-val stacker retraining that caused
    the leak in fix_stacker_classweight.py: stacker is trained only on
    the first n_tr samples. Returns inference outputs for the FULL
    series so the grid can sweep on any tail slice.

    pnl_for_meta is the pnl signal meta will regress against (we use
    pnl_long_over_short_by_tb_winner === target_pnl for label
    construction only — meta just needs the 0/1 "profitable non-FL" tag).
    """
    import xgboost as xgb

    # Stacker inputs — each primary's (soft + summaries) concat.
    def _stack_inputs(softs: list[np.ndarray]) -> np.ndarray:
        parts = []
        for s in softs:
            s = s.astype(np.float32)
            p_max = s.max(axis=-1, keepdims=True)
            p_marg = (np.sort(s, axis=-1)[:, -1]
                       - np.sort(s, axis=-1)[:, -2])[:, None]
            p_ent = -(s * np.log(s.clip(1e-9))).sum(axis=-1, keepdims=True)
            parts.append(np.concatenate([s, p_max, p_marg, p_ent], axis=-1))
        return np.concatenate(parts, axis=-1).astype(np.float32)

    X_stack = _stack_inputs(primary_softs)
    # Class-balanced sample weights on train slice — exposes non-FL signal.
    y_tr = y[:n_tr]
    classes, counts = np.unique(y_tr, return_counts=True)
    freq = counts / counts.sum()
    inv = {int(c): 1.0 / f for c, f in zip(classes, freq)}
    w_tr = np.array([inv[int(yi)] for yi in y_tr], dtype=np.float32)
    w_tr /= w_tr.mean()

    stk = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    stk.fit(X_stack[:n_tr], y_tr, sample_weight=w_tr, verbose=False)
    stacker_soft = stk.predict_proba(X_stack).astype(np.float32)

    # Meta — trained on train split only.
    X_m_tr, y_m_tr, w_m_tr = build_meta_dataset(
        primary_softmax=stacker_soft[:n_tr],
        y_true=y[:n_tr],
        target_pnl=pnl_for_meta[:n_tr],
    )
    meta, meta_metrics = train_meta(X_m_tr, y_m_tr, w_m_tr,
                                      cfg=MetaConfig(), val_frac=0.2, seed=seed)

    # Meta-prob inference over full series (only on non-FLAT primary picks).
    primary_pred = stacker_soft.argmax(axis=-1)
    non_flat = primary_pred != FLAT
    p_max = stacker_soft.max(axis=-1, keepdims=True)
    p_marg = (np.sort(stacker_soft, axis=-1)[:, -1]
               - np.sort(stacker_soft, axis=-1)[:, -2])[:, None]
    p_ent = -(stacker_soft * np.log(stacker_soft.clip(1e-9))).sum(axis=-1, keepdims=True)
    X_meta = np.concatenate([
        stacker_soft, p_max, p_marg, p_ent,
        (primary_pred == UP).astype(np.float32)[:, None],
    ], axis=-1).astype(np.float32)
    meta_prob = np.zeros(len(y), dtype=np.float32)
    if non_flat.any():
        meta_prob[non_flat] = meta.predict_proba(X_meta[non_flat])[:, 1]

    return stacker_soft, meta_prob, meta_metrics


def _kelly_fraction(p: float, win_pct: float, loss_pct: float,
                     cap: float) -> float:
    """Classic Kelly: f = (b·p - (1-p)) / b  where b = win/|loss|."""
    if win_pct <= 0 or loss_pct <= 0:
        return 0.0
    b = win_pct / loss_pct
    f = (b * p - (1.0 - p)) / b
    if not math.isfinite(f) or f <= 0:
        return 0.0
    return min(f, cap)


def _realise_per_config(
    sim: dict, primary_pred: np.ndarray, primary_max: np.ndarray,
    meta_prob: np.ndarray, *,
    min_prob: float, meta_thr: float, kelly_frac: float,
    kelly_cap: float, tp_pct: float, sl_pct: float,
    spread_cost_pct: float, fill_prob: float, seed: int,
) -> dict:
    """Realise per-trade PnL for one (kelly, meta_thr, min_prob, spread,
    fill_prob) inner config, given an outer (TP/SL/timeout/partial/trail)
    simulation already cached in `sim`."""
    pnl_long = sim["pnl_long"]
    pnl_short = sim["pnl_short"]
    n = len(pnl_long)

    non_flat = primary_pred != FLAT
    gate = non_flat & (primary_max >= min_prob) & (meta_prob >= meta_thr)

    # Direction-aware realised PnL (before sizing). Subtract spread cost
    # once per entry + exit round-trip.
    real = np.where(primary_pred == UP,   pnl_long,
              np.where(primary_pred == DOWN, pnl_short, 0.0))
    real = real - spread_cost_pct  # flat cost per intended trade

    # Kelly sizing per-sample — cap so we stay under max leverage.
    sizes = np.zeros(n, dtype=np.float64)
    if kelly_frac > 0:
        # Per-sample p = primary_max clipped
        p = np.clip(primary_max.astype(np.float64), 0.01, 0.99)
        b = tp_pct / max(sl_pct, 1e-6)
        raw_f = (b * p - (1.0 - p)) / b
        raw_f = np.clip(raw_f, 0.0, kelly_cap)
        sizes = kelly_frac * raw_f

    # Post-only fill Bernoulli drop.
    rng = np.random.default_rng(seed)
    fill_mask = rng.random(n) < fill_prob if fill_prob < 1.0 else np.ones(n, dtype=bool)
    take = gate & fill_mask

    # Zero out everything not taken.
    realised = np.where(take, sizes * real, 0.0)
    if not take.any():
        return {
            "n_trades": 0, "win_rate_pct": 0.0,
            "net_return_pct": 0.0, "sum_pnl_pct": 0.0,
            "max_dd_pct": 0.0, "sharpe": 0.0, "mean_size": 0.0,
        }

    ret = realised / 100.0
    eq = INITIAL_CAPITAL * np.cumprod(1.0 + ret)
    final_eq = float(eq[-1])
    peaks = np.maximum.accumulate(eq)
    dd = float(((peaks - eq) / np.maximum(peaks, 1e-12)).max()) * 100

    taken_pnl = realised[take]
    n_t = int(take.sum())
    wr = float((taken_pnl > 0).mean() * 100)
    sharpe = float(taken_pnl.mean() / (taken_pnl.std() + 1e-9)) if n_t > 1 else 0.0
    mean_sz = float(sizes[take].mean())

    return {
        "n_trades": n_t,
        "win_rate_pct": wr,
        "net_return_pct": 100 * (final_eq / INITIAL_CAPITAL - 1),
        "sum_pnl_pct": float(taken_pnl.sum()),
        "max_dd_pct": dd,
        "sharpe": sharpe,
        "mean_size": mean_sz,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/home/scalper/scalper-bot/data/_cache")
    ap.add_argument("--primary-softs",
                    default="/home/scalper/scalper-bot/models/primary_softs_v3.npz")
    ap.add_argument("--out",
                    default="/home/scalper/scalper-bot/models/grid_live_results.json")
    ap.add_argument("--train-frac", type=float, default=0.75,
                    help="fraction used for stacker+meta training "
                          "(tail = 1 - train_frac is the OOS eval slice)")
    # Outer: resimulated per config (expensive)
    ap.add_argument("--tp-grid", default="0.10,0.15,0.20,0.30,0.40")
    ap.add_argument("--sl-grid", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--timeout-grid", default="300,600,1200")
    ap.add_argument("--partial-grid", default="true,false")
    ap.add_argument("--trailing-grid", default="true,false")
    # Inner: reused per outer (cheap)
    ap.add_argument("--kelly-grid", default="0.10,0.25,0.50,1.00")
    ap.add_argument("--meta-thr-grid", default="0.50,0.60,0.70,0.80")
    ap.add_argument("--min-prob-grid", default="0.50,0.55,0.60")
    ap.add_argument("--spread-bps-grid", default="0,2,4",
                    help="extra basis points on top of commissions")
    ap.add_argument("--fill-prob-grid", default="1.00,0.80,0.60",
                    help="post-only fill probability per trade")
    ap.add_argument("--kelly-cap", type=float, default=19.0,
                    help="20× leverage × 95% margin cap")
    ap.add_argument("--min-trades", type=int, default=30,
                    help="rank filter — configs with fewer trades are "
                          "shown but ranked separately")
    ap.add_argument("--top-k", type=int, default=25)
    args = ap.parse_args()

    # === Load ===
    print(f"[grid_live] loading v3 cache and primary softmaxes ...")
    c = _load_v3(Path(args.cache_dir))
    softs_npz = np.load(args.primary_softs, allow_pickle=False)
    primary_keys = sorted(k for k in softs_npz.files if k.startswith("soft_"))
    primary_softs = [softs_npz[k] for k in primary_keys]
    if len(primary_softs) == 0:
        raise SystemExit("no primary softs")
    print(f"[grid_live] primaries: {primary_keys}")

    # Sanity: softs align with cache y.
    y_npz = softs_npz["y"]
    assert len(y_npz) == len(c["y"]), "primary softs len mismatch"
    assert np.array_equal(y_npz, c["y"]), "primary softs y mismatch"

    n = len(c["y"])
    n_tr = int(n * args.train_frac)
    print(f"[grid_live] N={n}, train {n_tr}, tail {n - n_tr}")

    # === Stacker + meta, walk-forward ===
    print(f"[grid_live] training walk-forward stacker + meta on first {n_tr}")
    pnl_proxy = c["entry_long"] * 0.0 + c["X_feat"][:, 0] * 0.0  # placeholder
    # For meta we need pnl aligned with y. Use the cache target_pnl stored
    # by the trainer (positive for non-FL, negative for FL) — meta label
    # def: non-FL primary AND positive TB pnl.
    pnl_for_meta = np.load(f"{c['prefix']}_pnl.npy")
    t0 = time.time()
    stacker_soft, meta_prob, meta_metrics = _walk_forward_stacker_meta(
        primary_softs, c["y"], pnl_for_meta, n_tr=n_tr,
    )
    dt = time.time() - t0
    primary_pred = stacker_soft.argmax(axis=-1)
    primary_max  = stacker_soft.max(axis=-1)
    non_fl_tail = int((primary_pred[n_tr:] != FLAT).sum())
    dir_correct_tail = int((primary_pred[n_tr:] == c["y"][n_tr:]).sum())
    wrong_nf = int(((primary_pred[n_tr:] != c["y"][n_tr:])
                     & (primary_pred[n_tr:] != FLAT)
                     & (c["y"][n_tr:] != FLAT)).sum())
    print(f"[grid_live] stacker+meta trained in {dt:.1f}s | tail primary: "
          f"non_fl={non_fl_tail} dir_correct={dir_correct_tail} "
          f"wrong_dir_nfnf={wrong_nf}  (wrong>0 ⇒ no leak)")
    print(f"[grid_live] meta AUC={meta_metrics['val_auc']:.3f} "
          f"prec={meta_metrics['val_precision']:.3f}")

    # === Outer simulations ===
    tp_list = [float(x) for x in args.tp_grid.split(",")]
    sl_list = [float(x) for x in args.sl_grid.split(",")]
    to_list = [int(x) for x in args.timeout_grid.split(",")]
    par_list = [x.lower() == "true" for x in args.partial_grid.split(",")]
    tr_list  = [x.lower() == "true" for x in args.trailing_grid.split(",")]

    outer = list(itertools.product(tp_list, sl_list, to_list, par_list, tr_list))
    print(f"[grid_live] {len(outer)} outer sims to run")
    sim_cache = {}
    t0 = time.time()
    for oi, (tp, sl, to, par, tr) in enumerate(outer):
        ncfg = len(c["entry_long"])
        sim = rust_bridge.simulate_labels(
            entry_long=c["entry_long"], entry_short=c["entry_short"],
            mid_paths=c["mid_paths"],
            tp_pct=np.full(ncfg, tp, dtype=np.float64),
            sl_pct=np.full(ncfg, sl, dtype=np.float64),
            timeout_ticks=np.full(ncfg, to, dtype=np.int64),
            partial_enabled=par, trailing_enabled=tr,
        )
        sim_cache[(tp, sl, to, par, tr)] = sim
        if (oi + 1) % 20 == 0 or oi + 1 == len(outer):
            print(f"  sim [{oi+1}/{len(outer)}]  elapsed {time.time() - t0:.1f}s")
    print(f"[grid_live] outer sims done in {time.time() - t0:.1f}s")

    # === Inner sweep ===
    kelly_list = [float(x) for x in args.kelly_grid.split(",")]
    mthr_list  = [float(x) for x in args.meta_thr_grid.split(",")]
    mprob_list = [float(x) for x in args.min_prob_grid.split(",")]
    spread_list = [float(x) / 100.0 for x in args.spread_bps_grid.split(",")]  # bps → pct
    fill_list   = [float(x) for x in args.fill_prob_grid.split(",")]

    # Restrict eval to the tail (no train leakage into scoring).
    tail = slice(n_tr, n)
    tail_primary_pred = primary_pred[tail]
    tail_primary_max  = primary_max[tail]
    tail_meta_prob    = meta_prob[tail]

    rows = []
    t0 = time.time()
    total_configs = len(outer) * len(kelly_list) * len(mthr_list) * \
                     len(mprob_list) * len(spread_list) * len(fill_list)
    print(f"[grid_live] inner sweep — {total_configs} total configs")

    for (tp, sl, to, par, tr), sim in sim_cache.items():
        # slice sim outputs to tail
        sim_tail = {
            "pnl_long":  sim["pnl_long"][tail],
            "pnl_short": sim["pnl_short"][tail],
        }
        for kelly, mthr, mprob, spread, fp in itertools.product(
            kelly_list, mthr_list, mprob_list, spread_list, fill_list,
        ):
            m = _realise_per_config(
                sim_tail, tail_primary_pred, tail_primary_max, tail_meta_prob,
                min_prob=mprob, meta_thr=mthr, kelly_frac=kelly,
                kelly_cap=args.kelly_cap, tp_pct=tp, sl_pct=sl,
                spread_cost_pct=spread, fill_prob=fp, seed=42,
            )
            rows.append({
                "tp": tp, "sl": sl, "timeout": to, "partial": par, "trailing": tr,
                "kelly": kelly, "meta_thr": mthr, "min_prob": mprob,
                "spread_bps": spread * 100, "fill_prob": fp,
                **m,
            })

    print(f"[grid_live] scored {len(rows)} configs in {time.time() - t0:.1f}s")

    # === Rank ===
    profitable = [r for r in rows if r["n_trades"] >= args.min_trades
                                      and r["net_return_pct"] > 0]
    print(f"[grid_live] profitable (n≥{args.min_trades}, net>0): {len(profitable)}")

    def _sorted(metric, desc=True):
        sign = -1 if desc else 1
        return sorted([r for r in rows if r["n_trades"] >= args.min_trades],
                       key=lambda r: sign * r[metric])[:args.top_k]

    top_net = _sorted("net_return_pct")
    top_sharpe = _sorted("sharpe")
    # DD-adjusted: prefer configs where net is high and DD is low.
    for r in rows:
        r["dd_adj_score"] = r["net_return_pct"] / (1.0 + r["max_dd_pct"])
    top_dd_adj = _sorted("dd_adj_score")

    def _print(title, top):
        print(f"\n=== {title} (top {min(args.top_k, len(top))}) ===")
        print(f"  {'tp':>5s} {'sl':>5s} {'to':>5s} "
              f"{'par':>4s} {'trl':>4s} {'kel':>5s} {'mthr':>5s} "
              f"{'mprb':>5s} {'sprd':>5s} {'fill':>4s}  "
              f"{'n':>5s} {'WR%':>5s} {'net%':>8s} {'DD%':>6s} {'Shp':>5s}")
        for r in top:
            print(f"  {r['tp']:>5.2f} {r['sl']:>5.2f} {r['timeout']:>5d} "
                  f"{str(r['partial'])[:1]:>4s} {str(r['trailing'])[:1]:>4s} "
                  f"{r['kelly']:>5.2f} {r['meta_thr']:>5.2f} "
                  f"{r['min_prob']:>5.2f} {r['spread_bps']:>5.1f} "
                  f"{r['fill_prob']:>4.2f}  "
                  f"{r['n_trades']:>5d} {r['win_rate_pct']:>4.1f}% "
                  f"{r['net_return_pct']:>+8.2f} {r['max_dd_pct']:>5.1f}% "
                  f"{r['sharpe']:>5.2f}")

    _print("by net_return", top_net[:10])
    _print("by sharpe",      top_sharpe[:10])
    _print("by dd_adj",      top_dd_adj[:10])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_samples": n, "n_train": n_tr, "n_eval": n - n_tr,
            "primary_keys": primary_keys,
            "meta_metrics": meta_metrics,
            "n_rows": len(rows),
            "n_profitable": len(profitable),
            "top_by_net": top_net,
            "top_by_sharpe": top_sharpe,
            "top_by_dd_adj": top_dd_adj,
        }, f, indent=2)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
