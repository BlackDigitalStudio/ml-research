#!/usr/bin/env python3
"""Per-symbol VOLATILITY inventory of the sub-60s forward move rH (book-mid logret, bp).

Motivates a per-symbol vol-adaptive Model-A gate: a FIXED |rH60|>=13bp threshold means
very different things per symbol (rare tail for BTC, common for DOGE) -> the gate target
is not comparable across symbols. Reports, per symbol x horizon {15,30,45,60}s:
  std(rH), mean|rH|, |rH| quantiles {p50,p75,p90,p95,p99}, base rate P(|rH|>=13bp),
  and the per-symbol thresholds that hit canonical non-flat rates {10%,5%,2%}.
Light read (rH + valid only). Run on VM.
"""
import argparse, io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; FEATS = "feats_sub60"
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
HS = [15, 30, 45, 60]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def load_sym(sym, stride, workers, day_stride=1):
    blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{FEATS}/{sym}/") if b.name.endswith(".npz"))
    blobs = blobs[::day_stride]   # subsample days (distribution stats robust; light read while build runs)
    acc = {H: [] for H in HS}
    def fetch(n): return np.load(io.BytesIO(bk.blob(n).download_as_bytes()))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(fetch, blobs):
            n = len(d["td"])
            if n < 200:
                continue
            sel = np.arange(0, n, stride)
            for H in HS:
                r = d[f"rH_{H}"].astype(np.float64)[sel]; v = d[f"valid_{H}"].astype(bool)[sel]
                acc[H].append(r[v & np.isfinite(r)])
    return len(blobs), {H: np.concatenate(acc[H]) if acc[H] else np.array([]) for H in HS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=SYMS)
    ap.add_argument("--stride", type=int, default=16)   # within-day subsample (distribution stats robust)
    ap.add_argument("--day-stride", type=int, default=10)  # subsample days -> light read while build runs
    ap.add_argument("--workers", type=int, default=3)   # gentle: maker build is running in parallel
    a = ap.parse_args()
    print(f"VOLATILITY inventory | rH = fwd book-mid logret (bp) | within-day stride={a.stride}\n")
    print(f"{'SYMBOL':6s} {'H':>3s} {'n':>9s} {'std':>7s} {'mean|r|':>8s} "
          f"{'p50':>6s} {'p75':>6s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'P(>=13bp)':>9s} "
          f"{'thr@10%':>8s} {'thr@5%':>8s} {'thr@2%':>8s}")
    rows = {}
    for sym in a.symbols:
        try:
            ndays, R = load_sym(sym, a.stride, a.workers, a.day_stride)
        except Exception as e:
            print(f"{sym}: ERR {e}"); continue
        rows[sym] = {}
        for H in HS:
            r = R[H]
            if len(r) < 1000:
                print(f"{sym.split('-')[0]:6s} {H:3d} {len(r):9d}  (too few)"); continue
            ar = np.abs(r)
            q = np.quantile(ar, [0.50, 0.75, 0.90, 0.95, 0.99])
            base13 = float((ar >= 13.0).mean())
            thr = {p: float(np.quantile(ar, 1 - p)) for p in (0.10, 0.05, 0.02)}
            rows[sym][H] = {"n": int(len(r)), "std": float(r.std()), "mean_abs": float(ar.mean()),
                            "q": q.tolist(), "base13": base13, "thr": thr}
            print(f"{sym.split('-')[0]:6s} {H:3d} {len(r):9d} {r.std():7.2f} {ar.mean():8.2f} "
                  f"{q[0]:6.2f} {q[1]:6.2f} {q[2]:7.2f} {q[3]:7.2f} {q[4]:7.2f} {base13*100:8.2f}% "
                  f"{thr[0.10]:8.2f} {thr[0.05]:8.2f} {thr[0.02]:8.2f}")
        print()
    # summary: at H=60, how the FIXED 13bp gate maps to wildly different base rates
    print("=== H=60s: fixed-13bp base rate vs per-symbol vol (the gate-comparability problem) ===")
    for sym in a.symbols:
        if sym in rows and 60 in rows[sym]:
            d = rows[sym][60]
            print(f"  {sym.split('-')[0]:6s} std={d['std']:6.2f}bp  P(|rH60|>=13bp)={d['base13']*100:5.2f}%  "
                  f"(13bp = {13.0/d['std']:.2f} sigma; thr for 10% nonflat = {d['thr'][0.10]:.1f}bp)")


if __name__ == "__main__":
    main()
