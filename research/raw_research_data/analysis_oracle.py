#!/usr/bin/env python3
"""Selective perfect-foresight ceiling: an oracle trades ONLY windows whose move
clears the cost floor, capturing the tail rather than the mean.

Per symbol x horizon x floor (maker=4bp, taker=10bp+spread):
  freq        = P(|r_h| >= c)                          trade frequency
  cond_mean   = E[|r_h| | |r_h| >= c]                  avg size of traded moves
  net_trade   = cond_mean - c                          avg net per traded window (bp)
  trades_day  = freq * (86400/h)                       non-overlapping trades/day
  bp_day      = E[(|r_h|-c)+] * (86400/h)              selective ceiling, bp/day

Only mid is needed -> reads book L0 (no trades). All on exchange `timestamp`.
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import analysis as a

warnings.simplefilter("ignore")

HZ = [5, 15, 30, 60]


def process(sym):
    days = a.list_days(sym)
    R = {h: [] for h in HZ}
    sp = []
    nd = 0
    for day in days:
        try:
            b = a.read_book(sym, day)
        except Exception:
            continue
        mid = b["mid"].resample(a.GRID).last().ffill(limit=a.FFILL_LIMIT)
        sp.append(float(np.nanmedian(b["spread_bp"].to_numpy())))
        for h in HZ:
            k = max(1, round(h / a.STEP_S))
            r = (mid.shift(-k) / mid - 1.0).abs() * 1e4
            R[h].append(r.dropna().to_numpy())
        nd += 1
    spread = float(np.nanmedian(sp))
    res = {"symbol": sym, "spread": spread, "n_days": nd, "h": {}}
    for h in HZ:
        arr = np.concatenate(R[h])
        slots = 86400.0 / h
        d = {"n": int(len(arr))}
        for name, c in (("maker", 4.0), ("taker", 10.0 + spread)):
            clear = arr >= c
            freq = float(clear.mean())
            cond = float(arr[clear].mean()) if clear.any() else float("nan")
            eplus = float(np.clip(arr - c, 0.0, None).mean())
            d[name] = {"floor": c, "freq": freq, "cond_mean": cond,
                       "net_trade": cond - c, "trades_day": freq * slots,
                       "bp_day": eplus * slots}
        res["h"][h] = d
    return res


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
    with open("results_oracle.json", "w") as fh:
        json.dump(results, fh, indent=2)

    syms = [s for s in a.SYMBOLS if s in results]
    for floor in ("maker", "taker"):
        for h in (30, 60):
            print(f"\n===== SELECTIVE ORACLE | floor={floor} | h={h}s "
                  f"(sorted by bp/day) =====")
            print(f"{'sym':5} {'floorBp':>7} {'freq%':>6} {'condMove':>9} "
                  f"{'net/trade':>9} {'trades/d':>9} {'bp/day':>9}")
            rows = []
            for s in syms:
                d = results[s]["h"][h][floor]
                rows.append((s, results[s]["h"][h][floor]["floor"], d))
            rows.sort(key=lambda x: x[2]["bp_day"], reverse=True)
            for s, fbp, d in rows:
                print(f"{s:5} {fbp:7.2f} {d['freq']*100:6.2f} {d['cond_mean']:9.2f} "
                      f"{d['net_trade']:9.2f} {d['trades_day']:9.0f} {d['bp_day']:9.0f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
