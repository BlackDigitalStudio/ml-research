#!/usr/bin/env python3
"""Sub-60s feature ORCHESTRATOR (runs on the GCP VM, in-region, ADC).

Per (symbol, day): build a DENSE 1s decision grid from the raw book timestamps,
run the CORRECTED Rust feature_builder (reads raw flat schema, ETH lead-lag cols
repurposed to clean point-to-point log-returns) on raw book+trades+funding+ETH,
then attach forward mid-return labels at H in {15,30,45,60}s and upload one
compressed npz to gs://{BUCKET}/feats_sub60/{sym}/{day}.npz.

  npz: X (n,56) f32, feat_cols, td (n,) i64 ns, mid (n,) f64,
       rH_{H} (n,) f32 bp, valid_{H} (n,) bool, meta json

Idempotent (skips existing). Same-region => egress free.
  python subs60_orch.py --workers 7
  python subs60_orch.py --symbols SOL-USDT-PERP --days 2025-08-15   # one cell
"""
import argparse, io, json, os, subprocess, tempfile, time
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"
BUCKET = "market-data-0998ac51"
OUT = "feats_sub60"
FB = "/tmp/feature_builder"
NS = 1_000_000_000
GRID_S = 1.0
HS = [15, 30, 45, 60]
SYMS = ["BNB-USDT-PERP", "BTC-USDT-PERP", "DOGE-USDT-PERP", "ETH-USDT-PERP",
        "LINK-USDT-PERP", "LTC-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
ETH = "ETH-USDT-PERP"
_bk = None


def bucket():
    global _bk
    if _bk is None:
        _bk = storage.Client(project=PROJ).bucket(BUCKET)
    return _bk


def _first(pref):
    for b in bucket().client.list_blobs(bucket(), prefix=pref):
        if b.name.endswith(".parquet"):
            return b.name
    return None


def _dl(pref, dest):
    n = _first(pref)
    if n is None:
        return None
    bucket().blob(n).download_to_filename(dest)
    return dest


def _book_ts_mid(path):
    t = pq.read_table(path, columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts = t["timestamp"].to_numpy().astype(np.int64)
    mid = 0.5 * (t["bid_0_price"].to_numpy().astype(np.float64)
                 + t["ask_0_price"].to_numpy().astype(np.float64))
    o = np.argsort(ts, kind="stable")
    return ts[o], mid[o]


def build_cell(sym, day, td_dir, eth_path):
    out_blob = f"{OUT}/{sym}/{day}.npz"
    if bucket().blob(out_blob).exists():
        return "skip"
    with tempfile.TemporaryDirectory(dir=td_dir) as td:
        book = _dl(f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/",
                   os.path.join(td, "b.parquet"))
        tr = _dl(f"raw/trades/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/",
                 os.path.join(td, "t.parquet"))
        if book is None or tr is None:
            return "no_input"
        fund = _dl(f"raw/funding/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/",
                   os.path.join(td, "f.parquet"))
        liq = _dl(f"raw/liquidations/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/",
                  os.path.join(td, "l.parquet"))
        oi = _dl(f"raw/open_interest/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/",
                 os.path.join(td, "o.parquet"))
        bts, mid = _book_ts_mid(book)
        if len(bts) < 2000:
            return "short"
        grid = np.arange(bts[0] + 120 * NS, bts[-1] - 70 * NS, int(GRID_S * NS), dtype=np.int64)
        idx = np.unique(np.clip(np.searchsorted(bts, grid, "right") - 1, 0, len(bts) - 1)).astype(np.int64)
        if len(idx) < 200:
            return "short_grid"
        ipath = os.path.join(td, "idx.npy"); np.save(ipath, idx)
        opath = os.path.join(td, "feat.npy")
        cmd = [FB, "--depth", book, "--indices", ipath, "--out", opath, "--trades", tr]
        if fund:
            cmd += ["--funding", fund]
        if eth_path:
            cmd += ["--eth", eth_path]
        if liq:
            cmd += ["--liquidations", liq]
        if oi:
            cmd += ["--open-interest", oi]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return f"FB_ERR {r.stderr.strip()[-80:]}"
        X = np.load(opath).astype(np.float32)
        td_ns = bts[idx]; mid_td = mid[idx]
        res = {"X": X, "td": td_ns.astype(np.int64), "mid": mid_td.astype(np.float64)}
        for H in HS:
            rf = np.clip(np.searchsorted(bts, td_ns + H * NS, "right") - 1, 0, len(bts) - 1)
            el = (bts[rf] - td_ns) / NS
            rH = np.log(mid[rf] / np.where(mid_td > 0, mid_td, np.nan)) * 1e4
            gap = (np.abs(el - H) > 5) | ~np.isfinite(rH)
            res[f"rH_{H}"] = np.nan_to_num(rH, nan=0.0).astype(np.float32)
            res[f"valid_{H}"] = (~gap).astype(bool)
        res["meta"] = json.dumps({"sym": sym, "day": day, "n": int(len(idx)),
                                  "grid_s": GRID_S, "Hs": HS, "n_feat": int(X.shape[1])})
        tmp = os.path.join(td, "out.npz")
        np.savez_compressed(tmp, **res)
        bucket().blob(out_blob).upload_from_filename(tmp)
    return "built"


def _worker(day):
    try:
        with tempfile.TemporaryDirectory() as td_dir:
            eth_path = _dl(f"raw/trades/exchange=BINANCE_FUTURES/symbol={ETH}/dt={day}/",
                           os.path.join(td_dir, "eth.parquet"))
            n = 0; states = {}
            for sym in SYMS:
                st = build_cell(sym, day, td_dir, eth_path)
                states[st] = states.get(st, 0) + 1
                if st == "built":
                    n += 1
            return (day, n, states)
    except Exception as e:
        return (day, 0, {f"ERR {type(e).__name__}: {str(e)[:80]}": 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--start", default=None)  # YYYY-MM-DD inclusive
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    global SYMS
    if a.symbols:
        SYMS = a.symbols
    if a.days:
        days = sorted(a.days)
    else:
        it = bucket().client.list_blobs(bucket(), prefix=f"raw/book/exchange=BINANCE_FUTURES/symbol={ETH}/", delimiter="/")
        for _ in it:
            pass
        days = sorted(p.split("dt=")[1].rstrip("/") for p in it.prefixes)
        if a.start:
            days = [d for d in days if d >= a.start]
        if a.end:
            days = [d for d in days if d <= a.end]
    print(f"syms={SYMS}\n{len(days)} days {days[0]}..{days[-1]} workers={a.workers} -> gs://{BUCKET}/{OUT}/", flush=True)
    t0 = time.time(); tot = 0; agg = {}
    with mp.Pool(a.workers) as pool:
        for i, (day, n, st) in enumerate(pool.imap_unordered(_worker, days, chunksize=1)):
            tot += n
            for k, v in st.items():
                agg[k] = agg.get(k, 0) + v
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(days)} days built={tot} {time.time()-t0:.0f}s {agg}", flush=True)
    print(f"DONE built={tot} sym-days in {time.time()-t0:.0f}s  states={agg}", flush=True)


if __name__ == "__main__":
    main()
