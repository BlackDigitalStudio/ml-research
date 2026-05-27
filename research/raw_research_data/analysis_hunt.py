#!/usr/bin/env python3
"""Hunt for a DEPLOYABLE 'predictable AND large enough' entry condition.

Every condition is computable at decision time t (no look-ahead): signal
strength, multi-level book imbalance, trade-flow agreement, trailing realized
volatility regime, spread. For each (symbol, condition, horizon) measure the
directional capture sign(signal)*r_h, hit-rate, firing frequency, net vs the
maker (4 bp) and taker (10 bp + spread) round-trip floors, and per-day
stability (mean / sd across the 30 days).

Reuses analysis.py for read_dir / read_trades / list_days / constants.
All on exchange `timestamp`.
"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import analysis as a

warnings.simplefilter("ignore")

HZ = [15, 30, 60]
BOOK_COLS = (["timestamp", "bid_0_price", "ask_0_price"]
             + [f"bid_{i}_size" for i in range(20)]
             + [f"ask_{i}_size" for i in range(20)])


def read_book_multi(sym, day):
    df = a.read_dir("book", sym, day, BOOK_COLS)
    df = df[(df["bid_0_price"] > 0) & (df["ask_0_price"] > df["bid_0_price"])]
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
    df["mid"] = (df["bid_0_price"] + df["ask_0_price"]) / 2.0
    df["spread_bp"] = (df["ask_0_price"] - df["bid_0_price"]) / df["mid"] * 1e4
    bsz = df[[f"bid_{i}_size" for i in range(20)]].to_numpy()
    asz = df[[f"ask_{i}_size" for i in range(20)]].to_numpy()
    b1, a1 = bsz[:, 0], asz[:, 0]
    b5, a5 = bsz[:, :5].sum(1), asz[:, :5].sum(1)
    b20, a20 = bsz.sum(1), asz.sum(1)
    df["obi1"] = (b1 - a1) / (b1 + a1)
    df["obi5"] = (b5 - a5) / (b5 + a5)
    df["obi20"] = (b20 - a20) / (b20 + a20)
    return df.set_index("ts").sort_index()


def build_grid(sym, day):
    book = read_book_multi(sym, day)
    tr = a.read_trades(sym, day)
    g = pd.DataFrame()
    g["mid"] = book["mid"].resample(a.GRID).last().ffill(limit=a.FFILL_LIMIT)
    for c in ["obi1", "obi5", "obi20", "spread_bp"]:
        g[c] = book[c].resample(a.GRID).last().ffill(limit=a.FFILL_LIMIT)
    g["tfi"] = tr["signed"].resample(a.GRID).sum().reindex(g.index).fillna(0.0)
    lr = np.log(g["mid"]).diff()
    g["rv"] = lr.rolling(240, min_periods=60).std() * 1e4  # bp per 250ms step, past-only
    for h in HZ:
        k = max(1, round(h / a.STEP_S))
        g[f"r{h}"] = (g["mid"].shift(-k) / g["mid"] - 1.0) * 1e4
    return g


def conditions(F, thr):
    """Return dict name -> (mask, direction_sign). F: dict of feature arrays."""
    o1, o5, o20, tfi, rv = F["obi1"], F["obi5"], F["obi20"], F["tfi"], F["rv"]
    out = {}
    out["obi1_top1"] = (np.abs(o1) >= thr["o1_99"], np.sign(o1))
    out["obi1_top0.1"] = (np.abs(o1) >= thr["o1_999"], np.sign(o1))
    out["obi20_top1"] = (np.abs(o20) >= thr["o20_99"], np.sign(o20))
    out["obi20_top0.1"] = (np.abs(o20) >= thr["o20_999"], np.sign(o20))
    out["agree_obi5_tfi"] = (
        (np.sign(o5) == np.sign(tfi)) & (tfi != 0)
        & (np.abs(o5) >= thr["o5_90"]) & (np.abs(tfi) >= thr["tfi_90"]),
        np.sign(o5))
    out["hivol_obi5"] = (
        (rv >= thr["rv_75"]) & (np.abs(o5) >= thr["o5_90"]), np.sign(o5))
    out["hivol_top_obi5"] = (
        (rv >= thr["rv_90"]) & (np.abs(o5) >= thr["o5_99"]), np.sign(o5))
    return out


def process(sym):
    days = a.list_days(sym)
    feats = {k: [] for k in ["obi1", "obi5", "obi20", "tfi", "rv", "spread_bp"]}
    rets = {h: [] for h in HZ}
    dayix = []
    ndays = 0
    for di, day in enumerate(days):
        try:
            g = build_grid(sym, day)
        except Exception:
            continue
        for k in feats:
            feats[k].append(g[k].to_numpy())
        for h in HZ:
            rets[h].append(g[f"r{h}"].to_numpy())
        dayix.append(np.full(len(g), di))
        ndays += 1

    F = {k: np.concatenate(v) for k, v in feats.items()}
    R = {h: np.concatenate(rets[h]) for h in HZ}
    day = np.concatenate(dayix)

    base = np.isfinite(F["obi1"]) & np.isfinite(F["obi5"]) & np.isfinite(F["obi20"]) \
        & np.isfinite(F["rv"]) & np.isfinite(F["spread_bp"])
    tfi_nz = F["tfi"][F["tfi"] != 0]
    thr = {
        "o1_99": np.quantile(np.abs(F["obi1"][base]), 0.99),
        "o1_999": np.quantile(np.abs(F["obi1"][base]), 0.999),
        "o20_99": np.quantile(np.abs(F["obi20"][base]), 0.99),
        "o20_999": np.quantile(np.abs(F["obi20"][base]), 0.999),
        "o5_90": np.quantile(np.abs(F["obi5"][base]), 0.90),
        "o5_99": np.quantile(np.abs(F["obi5"][base]), 0.99),
        "tfi_90": np.quantile(np.abs(tfi_nz), 0.90) if len(tfi_nz) else np.inf,
        "rv_75": np.quantile(F["rv"][base], 0.75),
        "rv_90": np.quantile(F["rv"][base], 0.90),
    }
    spread_med = float(np.median(F["spread_bp"][base]))
    conds = conditions(F, thr)

    res = {"symbol": sym, "n_days": ndays, "spread_med_bp": spread_med, "cells": {}}
    for cname, (cmask, dsign) in conds.items():
        for h in HZ:
            m = base & cmask & (dsign != 0) & np.isfinite(R[h])
            n = int(m.sum())
            if n < 200:
                continue
            cap = dsign[m] * R[h][m]
            # per-day means for stability
            dd = day[m]
            uniq = np.unique(dd)
            per_day = np.array([cap[dd == u].mean() for u in uniq])
            res["cells"][f"{cname}@{h}"] = {
                "cond": cname, "h": h, "n": n, "fires_per_day": n / ndays,
                "cap_mean": float(cap.mean()), "cap_median": float(np.median(cap)),
                "hit": float(np.mean(cap > 0)),
                "sd_days": float(per_day.std()),
                "frac_pos_days": float(np.mean(per_day > 0)),
                "net_maker": float(cap.mean() - 4.0),
                "net_taker": float(cap.mean() - 10.0 - spread_med),
            }
    return res


def main():
    syms = a.SYMBOLS
    if len(sys.argv) > 1:
        syms = sys.argv[1:]
    results = {}
    with ProcessPoolExecutor(max_workers=min(8, len(syms))) as ex:
        futs = {ex.submit(process, s): s for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                results[s] = fut.result()
                print(f"[done] {s}: {results[s]['n_days']} days, "
                      f"{len(results[s]['cells'])} cells", flush=True)
            except Exception as e:
                import traceback
                print(f"[FAIL] {s}: {e}\n{traceback.format_exc()}", flush=True)

    with open("results_hunt.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # flat leaderboard by mean directional capture
    flat = []
    for s, r in results.items():
        for cell in r["cells"].values():
            flat.append((s, cell))
    flat.sort(key=lambda x: x[1]["cap_mean"], reverse=True)

    print("\n===== LEADERBOARD: top cells by mean directional capture (bp) =====")
    print(f"{'sym':5} {'condition':16} {'h':>3} {'cap':>6} {'med':>6} {'hit':>5} "
          f"{'sd_d':>6} {'pos_d':>6} {'fire/d':>9} {'netMkr':>7} {'netTkr':>7}")
    for s, c in flat[:25]:
        print(f"{s:5} {c['cond']:16} {c['h']:>3} {c['cap_mean']:6.2f} "
              f"{c['cap_median']:6.2f} {c['hit']:5.3f} {c['sd_days']:6.2f} "
              f"{c['frac_pos_days']:6.2f} {c['fires_per_day']:9.0f} "
              f"{c['net_maker']:7.2f} {c['net_taker']:7.2f}")

    # capture matrix at h=60
    conds_order = ["obi1_top1", "obi1_top0.1", "obi20_top1", "obi20_top0.1",
                   "agree_obi5_tfi", "hivol_obi5", "hivol_top_obi5"]
    print("\n===== mean capture (bp) @ h=60 : rows=symbol, cols=condition =====")
    print(f"{'sym':5} " + " ".join(f"{c[:12]:>13}" for c in conds_order))
    for s in syms:
        if s not in results:
            continue
        row = []
        for c in conds_order:
            cell = results[s]["cells"].get(f"{c}@60")
            row.append(f"{cell['cap_mean']:13.2f}" if cell else f"{'-':>13}")
        print(f"{s:5} " + " ".join(row))
    print("\nDONE")


if __name__ == "__main__":
    main()
