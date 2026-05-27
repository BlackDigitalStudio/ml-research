#!/usr/bin/env python3
"""A x C compatibility: is the move DIRECTION predictable on the windows that
clear the cost floor?

Reuses analysis.py (read_book / read_trades / grid_day / list_days / rank_ic).
All on exchange `timestamp`. Pools (OBI_t, forward_return_h) across 30 days
per symbol.

Three views per symbol x horizon:
  1. Conditional rank-IC: IC(OBI, r_h) on subset |r_h| >= floor.
     (outcome-conditioned -> look-ahead; diagnostic only, not deployable).
  2. Conditional directional accuracy on the same subset:
     mean( sign(r_h) == sign(OBI) ).
  3. Signal-conditioned (deployable, no look-ahead): windows with |OBI| in its
     top decile -> mean directional capture sign(OBI)*r_h (bp), directional
     hit-rate, and fraction where sign(OBI)*r_h >= floor.
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import analysis as a

warnings.simplefilter("ignore")

HZ = [2, 5, 10, 15, 30, 45, 60]
FLOORS = [7, 10, 13]
OBI_TOP_Q = 0.90  # top-decile by |OBI|


def process(sym):
    days = a.list_days(sym)
    obi_parts, r_parts = [], {h: [] for h in HZ}
    for day in days:
        try:
            book = a.read_book(sym, day)
            tr = a.read_trades(sym, day)
        except Exception:
            continue
        g = a.grid_day(book, tr)
        mid = g["mid"]
        obi_parts.append(g["obi"].to_numpy())
        for h in HZ:
            k = max(1, round(h / a.STEP_S))
            r = (mid.shift(-k) / mid - 1.0) * 1e4
            r_parts[h].append(r.to_numpy())

    obi = np.concatenate(obi_parts)
    res = {"symbol": sym, "n_days_ok": len(obi_parts), "by_h": {}}
    for h in HZ:
        r = np.concatenate(r_parts[h])
        valid = np.isfinite(obi) & np.isfinite(r)
        o, rr = obi[valid], r[valid]
        sgo = np.sign(o)
        d = {"n_valid": int(valid.sum()), "ic_all": _ic(o, rr), "floors": {}}
        for f in FLOORS:
            mF = np.abs(rr) >= f
            nz = mF & (sgo != 0)
            d["floors"][f] = {
                "n": int(mF.sum()),
                "cond_ic": _ic(o[mF], rr[mF]),
                "dir_acc": float(np.mean(np.sign(rr[nz]) == sgo[nz])) if nz.sum() else float("nan"),
            }
        # signal-conditioned (deployable)
        thr = float(np.quantile(np.abs(o), OBI_TOP_Q))
        hi = np.abs(o) >= thr
        cap = sgo[hi] * rr[hi]            # directional capture in bp
        d["signal_top_decile"] = {
            "obi_abs_thr": thr,
            "n_hi": int(hi.sum()),
            "capture_bp_mean": float(np.mean(cap)),
            "capture_bp_median": float(np.median(cap)),
            "hit_rate": float(np.mean(cap > 0)),
            "frac_clear": {f: float(np.mean(cap >= f)) for f in FLOORS},
        }
        res["by_h"][h] = d
    return res


def _ic(x, y):
    if len(x) < 100 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    xr = x.argsort().argsort().astype(float)
    yr = y.argsort().argsort().astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process, s): s for s in a.SYMBOLS}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                results[s] = fut.result()
                print(f"[done] {s}", flush=True)
            except Exception as e:
                print(f"[FAIL] {s}: {e}", flush=True)
    with open("results_axc.json", "w") as fh:
        json.dump(results, fh, indent=2)

    syms = [s for s in a.SYMBOLS if s in results]

    print("\n===== 1+2. CONDITIONAL on |r_h|>=floor : cond_IC / dir_acc / n (floor=10bp) =====")
    print(f"{'sym':5} " + " ".join(f"{h:>16}" for h in HZ))
    for s in syms:
        cells = []
        for h in HZ:
            c = results[s]["by_h"][h]["floors"][10]
            cells.append(f"{c['cond_ic']:+.2f}/{c['dir_acc']:.2f}/{c['n']//1000}k")
        print(f"{s:5} " + " ".join(f"{x:>16}" for x in cells))

    print("\n===== same, floor=13bp =====")
    print(f"{'sym':5} " + " ".join(f"{h:>16}" for h in HZ))
    for s in syms:
        cells = []
        for h in HZ:
            c = results[s]["by_h"][h]["floors"][13]
            cells.append(f"{c['cond_ic']:+.2f}/{c['dir_acc']:.2f}/{c['n']//1000}k")
        print(f"{s:5} " + " ".join(f"{x:>16}" for x in cells))

    print("\n===== 3. SIGNAL-CONDITIONED top-decile |OBI| : mean directional capture (bp) =====")
    print(f"{'sym':5} " + " ".join(f"{h:>8}" for h in HZ))
    for s in syms:
        print(f"{s:5} " + " ".join(f"{results[s]['by_h'][h]['signal_top_decile']['capture_bp_mean']:8.2f}" for h in HZ))

    print("\n===== 3b. top-decile |OBI| : directional hit-rate =====")
    print(f"{'sym':5} " + " ".join(f"{h:>8}" for h in HZ))
    for s in syms:
        print(f"{s:5} " + " ".join(f"{results[s]['by_h'][h]['signal_top_decile']['hit_rate']:8.3f}" for h in HZ))

    print("\n===== 3c. top-decile |OBI| : frac where directional capture >= 13bp =====")
    print(f"{'sym':5} " + " ".join(f"{h:>8}" for h in HZ))
    for s in syms:
        print(f"{s:5} " + " ".join(f"{results[s]['by_h'][h]['signal_top_decile']['frac_clear'][13]:8.3f}" for h in HZ))
    print("\nDONE")


if __name__ == "__main__":
    main()
