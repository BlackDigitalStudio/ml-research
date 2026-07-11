#!/usr/bin/env python3
"""HD2 -> grid_sim: realistic execution sweep of the Mamba signal (Modal).

Builds the Rust grid_sim binary in the image, then sweeps a RICH execution grid
on the LTC OOS Mamba signal. Entries = Mamba decision points (logit-change rows
of the 10s rlseries); forward path = 10s bars (30min=180 bars), so timeout is in
10s bars. Direction/confidence from the H1800 (30min) logit, median-centred
(matches cap_edge's sign(logit-median)). Costs = TAKER (0.07 win / 0.10 loss).

SWEEP (pinned per user: kelly=1.0, fill_prob=1.0, spread=0, taker fees):
  outer  : tp x sl x timeout{1,3,10,30min} x partial{0,1} x trailing{0,1}
  inner  : min_prob curve (to read the selectivity ~10 trades/day operating pt)
-> tens of thousands of combos (budget 1M; grid_sim eats it in seconds).

Reports: top configs by net_return among those near ~10 trades/day, + the
selectivity curve. Frictionless-fill but REAL taker fees -> first realistic read.

  modal run scripts/hd2_grid_modal.py            # build + sweep + report (detached ok)
"""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent
IMG = (
    modal.Image.from_registry("rust:slim", add_python="3.11")   # >=1.85 for edition2024 deps
    .add_local_dir(str(REPO / "rust_ingest"), "/rust_ingest", copy=True)
    .run_commands("cd /rust_ingest && cargo build --release --bin grid_sim")
    .pip_install("numpy==1.26.4")
)
VOL = modal.Volume.from_name("hd2-cache")
MNT = "/cache"
GRID_BIN = "/rust_ingest/target/release/grid_sim"
SERIES = f"{MNT}/results/hd2_pool/POOL_reg_d0.1_wd0.001_s0.rlseries.npz"
H_BARS = 180          # 30 min @ 10s bars (max forward path / max timeout)
app = modal.App("hd2-grid")


@app.function(image=IMG, cpu=8.0, timeout=3600, volumes={MNT: VOL})
def run_grid():
    import numpy as np, json, subprocess, os, itertools
    VOL.reload()
    d = np.load(SERIES)
    ts, mid, logits, day = (d["ts"], d["mid"].astype(np.float64),
                            d["logits"].astype(np.float64), d["day"])
    n = len(ts)
    # entries = decision points = rows where the (ffilled) logit vector changes
    chg = np.zeros(n, bool); chg[0] = True
    chg[1:] = np.any(logits[1:] != logits[:-1], axis=1) | (day[1:] != day[:-1])
    idx = np.where(chg)[0]
    # keep only entries with H_BARS ahead inside the SAME day
    nxt = idx + H_BARS
    ok = (nxt < n) & (day[np.clip(nxt, 0, n - 1)] == day[idx])
    keep = idx[ok]
    entry = mid[keep]
    paths = np.stack([mid[i:i + H_BARS] for i in keep]).astype(np.float64)  # (m,180)
    l30 = logits[keep, 2]                          # H1800 (30 min) logit
    l30c = l30 - np.median(l30)                    # median-centre (cap_edge convention)
    pred = np.where(l30c >= 0.0, 0, 1).astype(np.int64)      # 0 UP, 1 DN
    # IC logits are uncalibrated/saturated (sigmoid->~0/1), so use the confidence
    # RANK-percentile as max_prob: then min_prob directly = selectivity quantile
    # (gate keeps top (1-min_prob) fraction). 10 trades/day ~ min_prob 0.975.
    conf = np.abs(l30c)
    max_prob = ((np.argsort(np.argsort(conf)) + 1.0) / len(conf)).astype(np.float64)
    n_days = int(len(np.unique(day)))
    m = len(keep)
    print(f"grid prep: {m} entries (dp), {n_days} OOS days, "
          f"max_prob[min/med/max]={max_prob.min():.3f}/{np.median(max_prob):.3f}/{max_prob.max():.3f}")

    # ── RICH outer grid (pinned: kelly=1, fill=1, spread=0, taker fees) ──
    tps = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    sls = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    tos = [6, 18, 60, 180]                         # 1,3,10,30 min in 10s bars
    configs = [{"tp": tp, "sl": sl, "to": to, "par": par, "tr": tr}
               for tp in tps for sl in sls for to in tos
               for par in (False, True) for tr in (False, True)]
    min_probs = [round(x, 4) for x in np.arange(0.90, 0.9955, 0.0025)]  # selectivity quantile -> ~40..2/day
    print(f"grid: {len(configs)} outer x {len(min_probs)} min_prob = "
          f"{len(configs) * len(min_probs)} inner combos")

    td = "/tmp/grid"; os.makedirs(td, exist_ok=True)
    np.save(f"{td}/el.npy", entry); np.save(f"{td}/es.npy", entry)
    np.save(f"{td}/mid.npy", paths)
    np.save(f"{td}/pred.npy", pred); np.save(f"{td}/mp.npy", max_prob)
    json.dump(configs, open(f"{td}/configs.json", "w"))
    inner_out = f"{td}/inner.json"
    cmd = [GRID_BIN,
           "--entry-long", f"{td}/el.npy", "--entry-short", f"{td}/es.npy",
           "--mid-paths", f"{td}/mid.npy", "--configs", f"{td}/configs.json",
           "--commission-win-pct", "0.07", "--commission-loss-pct", "0.10",
           "--fill-latency-ms", "150.0", "--out-prefix", f"{td}/out",
           "--pred", f"{td}/pred.npy", "--max-prob", f"{td}/mp.npy",
           "--holdout-start", "0", "--n-eff-days", str(float(n_days)),
           "--inner-min-probs", ",".join(str(x) for x in min_probs),
           "--inner-spreads", "0.0", "--inner-fill-probs", "1.0",
           "--inner-kelly-fracs", "1.0", "--inner-initial-capital", "100.0",
           "--inner-out", inner_out]
    print("running grid_sim ..."); subprocess.run(cmd, check=True)
    res = json.load(open(inner_out))
    print(f"grid_sim returned {len(res)} inner results")

    # ~10 trades/day operating band, rank by net_return
    band = [r for r in res if 6.0 <= r["trades_per_day"] <= 16.0]
    band.sort(key=lambda r: r["ev_per_trade_pct"], reverse=True)   # per-trade edge (Kelly-agnostic)
    os.makedirs(f"{MNT}/results/grid", exist_ok=True)
    json.dump({"n_entries": m, "n_days": n_days, "all": res, "band_10pd_top": band[:50]},
              open(f"{MNT}/results/grid/mamba_grid.json", "w"), default=float)
    VOL.commit()
    print(f"\n=== TOP configs @ ~10 trades/day (taker 0.07/0.10, kelly=1) — n={len(band)} in band ===")
    print(f"{'tp':>5} {'sl':>5} {'to':>4} {'par':>3} {'tr':>3} {'minp':>5} "
          f"{'t/day':>6} {'WR%':>6} {'net%':>9} {'ev/tr%':>7} {'sharpe':>7} {'maxDD%':>7}")
    for r in band[:25]:
        print(f"{r['tp']:5.2f} {r['sl']:5.2f} {int(r['timeout']):4d} "
              f"{int(r['partial']):3d} {int(r['trailing']):3d} {r['min_prob']:5.2f} "
              f"{r['trades_per_day']:6.1f} {r['win_rate_pct']:6.1f} {r['net_return_pct']:9.2f} "
              f"{r['ev_per_trade_pct']:7.3f} {r['sharpe']:7.2f} {r['max_dd_pct']:7.1f}")
    return {"n_band": len(band), "top": band[:5]}


@app.local_entrypoint()
def main():
    h = run_grid.spawn()
    print(f"HD2 grid_sim SWEEP spawned {h.object_id} (detached -> /cache/results/grid/mamba_grid.json)")
