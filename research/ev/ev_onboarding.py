"""EV-A (HDATA cell EV-A): perp-onboarding event-study on Binance USD-M 1m klines.
Zero new data -- reads free_v1/klines_1m/*.parquet (open_time, close only) from GCS
via column projection. Event = a symbol's FIRST 1m candle (perp onboarding).

Surfaces (CLAUDE.md rule 1 -- surface, not verdict):
  (1) post-listing DRIFT: r_h = log(close[t0+h]/close[t0]) for h in {1,5,15,60}m
      (mean/median/frac>0/med|r|/clear_cost@0.13%).
  (2) early MOMENTUM/REVERSAL predictability: rank_IC(early[0,k], next[k,2k]) +
      dir_ret = sign(early)*next, for k in {5,15,30}m. Honest: early window is
      fully observed before entry at t0+k.

true-onboarding set = t0 > global_data_start + 7d (excludes left-censored syms
whose first candle is just the 2yr window start, not a real listing). The
censored set is reported as a built-in PLACEBO (arbitrary anchor -> expect ~0).
"""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJ = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
PREFIX = "free_v1/klines_1m/"
COST = 0.0013
OFFS = [0, 1, 5, 10, 15, 30, 60]   # minutes from t0 at which we need close
DRIFT_H = [1, 5, 15, 60]
MOM_K = [5, 15, 30]
TOL_MS = 120_000                   # accept a bar within +/-2min of target (gap tolerance)
MIN_MS = 60_000

fs = pafs.GcsFileSystem()
names = [b.name for b in storage.Client(project=PROJ).list_blobs(BUCKET, prefix=PREFIX)
         if b.name.endswith(".parquet")]
print(f"files={len(names)}", flush=True)


def closes_at_offsets(name):
    try:
        pf = pq.ParquetFile(fs.open_input_file(f"{BUCKET}/{name}"))
        t = pf.read(columns=["open_time", "close"])
        ot = pd.to_numeric(pd.Series(t.column("open_time").to_numpy()), errors="coerce").to_numpy()
        cl = pd.to_numeric(pd.Series(t.column("close").to_numpy()), errors="coerce").to_numpy()
    except Exception:
        return None
    m = np.isfinite(ot) & np.isfinite(cl)
    ot, cl = ot[m], cl[m]
    if len(ot) < 2:
        return None
    o = np.argsort(ot, kind="stable")
    ot, cl = ot[o], cl[o]
    uq = np.concatenate(([True], np.diff(ot) > 0))
    ot, cl = ot[uq], cl[uq]
    t0 = ot[0]
    out = {}
    for off in OFFS:
        tt = t0 + off * MIN_MS
        j = np.searchsorted(ot, tt, "left")
        if j < len(ot) and abs(ot[j] - tt) <= TOL_MS:
            out[off] = cl[j]
        elif j > 0 and abs(ot[j - 1] - tt) <= TOL_MS:
            out[off] = cl[j - 1]
        else:
            out[off] = np.nan
    return name.split("/")[-1].replace(".parquet", ""), int(t0), out


recs, done = [], 0
with ThreadPoolExecutor(max_workers=32) as ex:
    futs = [ex.submit(closes_at_offsets, n) for n in names]
    for f in as_completed(futs):
        done += 1
        r = f.result()
        if r:
            recs.append(r)
        if done % 50 == 0:
            print(f"  ...{done}/{len(names)}", flush=True)
print(f"read symbols={len(recs)} of {len(names)}", flush=True)

t0s = np.array([r[1] for r in recs])
C = {off: np.array([r[2][off] for r in recs], float) for off in OFFS}
gmin = int(t0s.min())
true_onb = t0s > gmin + 7 * 86400 * 1000
print(f"global_data_start={pd.to_datetime(gmin, unit='ms', utc=True)} "
      f"true_onboardings={int(true_onb.sum())} censored={int((~true_onb).sum())}", flush=True)


def auc(score, label):
    label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_se(fn, *arrs, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(arrs[0])
    out = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            out.append(fn(*[a[idx] for a in arrs]))
        except Exception:
            pass
    return float(np.nanstd(out)) if out else float("nan")


def report(mask, label, store):
    c0 = C[0][mask]
    d = {"n": int(mask.sum()), "drift": {}, "momentum": {}}
    print(f"\n=== {label} (n={int(mask.sum())}) ===")
    print("  -- post-listing DRIFT (anchor = first-candle close) --")
    for h in DRIFT_H:
        r = np.log(C[h][mask] / c0)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        row = {"mean_bp": np.mean(r) * 1e4, "med_bp": np.median(r) * 1e4,
               "frac_pos": np.mean(r > 0), "med_abs_pct": np.median(np.abs(r)) * 100,
               "clear_cost": np.mean(np.abs(r) > COST), "n": len(r)}
        d["drift"][h] = {k: round(float(v), 5) for k, v in row.items()}
        print(f"  h={h:>3}m mean={row['mean_bp']:+.2f}bp med={row['med_bp']:+.2f}bp "
              f"frac>0={row['frac_pos']:.3f} med|r|={row['med_abs_pct']:.3f}% "
              f"clear_cost={row['clear_cost']:.3f} n={row['n']}")
    print("  -- early MOMENTUM/REVERSAL: rank_IC(early[0,k], next[k,2k]) --")
    for k in MOM_K:
        early = np.log(C[k][mask] / c0)
        nxt = np.log(C[2 * k][mask] / C[k][mask])
        ok = np.isfinite(early) & np.isfinite(nxt)
        early, nxt = early[ok], nxt[ok]
        if len(early) < 10:
            continue
        ric = auc(early, nxt > 0) - 0.5
        dr = np.sign(early) * nxt
        row = {"rank_IC": ric, "rank_IC_se": boot_se(lambda a, b: auc(a, b > 0) - 0.5, early, nxt),
               "dir_ret_bp": np.mean(dr) * 1e4, "dir_ret_se_bp": boot_se(lambda x: np.mean(x) * 1e4, dr),
               "win": np.mean(dr > 0), "med_next_pct": np.median(np.abs(nxt)) * 100, "n": len(early)}
        d["momentum"][k] = {kk: round(float(vv), 5) for kk, vv in row.items()}
        print(f"  k={k:>3}m rank_IC={row['rank_IC']:+.4f}(+/-{row['rank_IC_se']:.4f}) "
              f"dir_ret={row['dir_ret_bp']:+.2f}bp(+/-{row['dir_ret_se_bp']:.2f}) "
              f"win={row['win']:.3f} med|next|={row['med_next_pct']:.3f}% n={row['n']}")
    store[label] = d


store = {"n_read": len(recs), "global_data_start_ms": gmin,
         "n_true": int(true_onb.sum()), "n_censored": int((~true_onb).sum()),
         "cost": COST, "offsets_min": OFFS}
report(true_onb, "TRUE_perp_onboardings", store)
report(~true_onb, "CENSORED_placebo", store)
json.dump(store, open("C:/Dev/sb-data-poc/ev_onboarding.json", "w"), indent=2)
print("\nEV_A_ONBOARDING_DONE", flush=True)
