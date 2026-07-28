#!/usr/bin/env python3
"""USDC FILL LAYER — 3-way decomposition of the maker fill error (OPS-EXEC rev17).

The deployed policy computes features/scores on the USDT book (correct, unchanged) but
EXECUTES on USDC. Every historical cell simulated fills on USDT with a model that fills
unconditionally on a gap past our level. This runner separates the two errors:

  (a) frozen model on the USDT book  = what the cells used   (existing _recev_* artifacts)
  (b) frozen model on the USDC book  = VENUE error alone     (this run, default flags)
  (c) strict model on the USDC book  = venue + MODEL error   (this run, --strict-entry-fill)

Scores are NOT recomputed: the 3s decision grid is calendar-anchored at UTC midnight on
both venues, so USDC rows join to the existing USDT scores by sample timestamp.

Env: SYM(DOGE) DAYS_FROM(20260628) DAYS_TO(20260728) SCORE_PREFIX(_recev_h150anch2_DOGE)
     OUT_PREFIX(_usdcfill_DOGE) WORKDIR BINS
Run on hd2 (needs the rebuilt husdc binaries with --emit-level-flow/--strict-entry-fill).
"""
import io, os, subprocess, sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"
MKT = "market-data-0998ac51"
REC = "recorder-data-asia-0998ac51"
RB = "chronos/scalper-recorder/binance_futures"

SYM = os.environ.get("SYM", "DOGE")
SYMC = SYM + "USDC"
D_FROM = os.environ.get("DAYS_FROM", "20260628")
D_TO = os.environ.get("DAYS_TO", "20260728")
SCORE_PREFIX = os.environ.get("SCORE_PREFIX", f"research_runs/_recev_h150anch2_{SYM}")
OUT_PREFIX = os.environ.get("OUT_PREFIX", f"research_runs/_usdcfill_{SYM}")
TD = os.environ.get("WORKDIR", f"/home/delmi/usdcfill_{SYM}")
BINS = os.environ.get("BINS", "/home/delmi/research_bins")
BS = f"{BINS}/husdc_target/release/build_samples"
GRID = f"{BINS}/husdc_target/release/grid_sim_exitdbg"

LV = 20; W = 50; H = 6000; STEP_S = 3.0; NS = 1_000_000_000
ENTRY_MS = 60_000; CHASE_MS = 300_000; HOLD_MS = 150_000

os.makedirs(TD, exist_ok=True)
cl = storage.Client(project=PROJ); mkt = cl.bucket(MKT); rec = cl.bucket(REC)


def days():
    a = datetime.strptime(D_FROM, "%Y%m%d"); b = datetime.strptime(D_TO, "%Y%m%d")
    out = []
    while a <= b:
        out.append(a.strftime("%Y%m%d")); a += timedelta(days=1)
    return out


def book(day, out):
    cols = ["timestamp", "receipt_timestamp", "sequence_number"] + \
           [f"{s}_{i}_{f}" for i in range(LV) for s in ("bid", "ask") for f in ("price", "size")]
    acc = {c: [] for c in cols}
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{SYMC}/depth_snapshot/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/d.parquet")
        df = pq.read_table(f"{TD}/d.parquet", columns=["exchange_event_ts_us", "local_ts_us",
                           "bid_prices", "bid_qtys", "ask_prices", "ask_qtys"]).to_pandas()
        df = df[df["exchange_event_ts_us"].notna()]
        if not len(df):
            continue
        acc["timestamp"].append((df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
        acc["receipt_timestamp"].append((df["local_ts_us"].astype("int64") * 1000).to_numpy())
        acc["sequence_number"].append(np.zeros(len(df), np.int64))
        for i in range(LV):
            for s, pc, qc in (("bid", "bid_prices", "bid_qtys"), ("ask", "ask_prices", "ask_qtys")):
                acc[f"{s}_{i}_price"].append(df[pc].apply(lambda x, i=i: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
                acc[f"{s}_{i}_size"].append(df[qc].apply(lambda x, i=i: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
    if not acc["timestamp"]:
        return 0
    ar = {c: np.concatenate(acc[c]) for c in cols}
    o = np.argsort(ar["timestamp"], kind="stable"); ar = {c: ar[c][o] for c in cols}
    ar["sequence_number"] = np.arange(len(ar["timestamp"]), dtype=np.int64)
    pq.write_table(pa.table(ar), out); return len(ar["timestamp"])


def trades(day, out):
    ts, px, qt, bm = [], [], [], []
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{SYMC}/agg_trade/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/t.parquet")
        df = pq.read_table(f"{TD}/t.parquet", columns=["exchange_event_ts_us", "price", "qty", "is_buyer_maker"]).to_pandas()
        df = df[df["exchange_event_ts_us"].notna()]
        if not len(df):
            continue
        ts.append((df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
        px.append(df["price"].astype(float).to_numpy())
        qt.append(df["qty"].astype(float).to_numpy())
        bm.append(df["is_buyer_maker"].astype(bool).to_numpy())
    if not ts:
        return 0
    ts = np.concatenate(ts); px = np.concatenate(px); qt = np.concatenate(qt); bm = np.concatenate(bm)
    o = np.argsort(ts, kind="stable")
    t = pa.table({"side": np.where(bm[o], "sell", "buy"), "amount": qt[o], "price": px[o],
                  "id": np.arange(len(ts), dtype=np.int64), "timestamp": ts[o],
                  "receipt_timestamp": ts[o]})
    pq.write_table(t, out); return t.num_rows


import json
json.dump([{"tp": 50.0, "sl": 50.0, "to": 282, "to_ms": float(HOLD_MS), "par": False, "tr": False}],
          open(f"{TD}/cfg.json", "w"))

done = {b.name.split("/")[-1][2:10] for b in mkt.client.list_blobs(mkt, prefix=f"{OUT_PREFIX}/D_")
        if b.name.endswith(".npz")}
for day in days():
    if day in done:
        print(f"{day}: persisted, skip", flush=True); continue
    nb = book(day, f"{TD}/book.parquet")
    nt = trades(day, f"{TD}/trades.parquet")
    if not nb or not nt:
        print(f"{day}: book={nb} tr={nt} skip", flush=True); continue
    bt = pq.read_table(f"{TD}/book.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    n = len(bt)
    if n < W + H + 100:
        print(f"{day}: thin {n}", flush=True); continue
    mid0 = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()) * NS
    grid = np.arange(mid0, bt[-1], int(STEP_S * NS)); grid = grid[grid >= bt[0]]
    ends = np.unique(np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1))
    ends = ends[(ends >= W - 1) & (ends < n - H - 1)].astype(np.int64)
    if len(ends) < 50:
        print(f"{day}: few-ends {len(ends)}", flush=True); continue
    np.save(f"{TD}/ends.npy", ends)
    od = f"{TD}/bs"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(f"{od}/{f}")
    r = subprocess.run([BS, "--depth", f"{TD}/book.parquet", "--trades", f"{TD}/trades.parquet",
                        "--out-dir", od, "--window", str(W), "--horizon", str(H),
                        "--sample-ends", f"{TD}/ends.npy", "--skip-xlob", "--emit-level-flow"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{day}: BS fail {r.stderr[-200:]}", flush=True); continue
    base = [GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
            "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
            "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
            "--entry-q", f"{od}/entry_q.npy", "--configs", f"{TD}/cfg.json",
            "--queue-mult", "1.0", "--exit-queue-mult", "1.0", "--ts-paths", f"{od}/ts_paths.npy",
            "--sample-ts", f"{od}/sample_ts.npy", "--entry-window-ms", str(ENTRY_MS),
            "--chase-ms", str(CHASE_MS), "--entry-window-ticks", "120", "--maker-offset-frac", "0",
            "--commission-win-pct", "0", "--commission-loss-pct", "0"]
    res = {}
    for tag, extra in (("frozen", []),
                       ("strict", ["--strict-entry-fill", "--level-flow-paths", f"{od}/flow_lvl_paths.npy"])):
        g = f"{TD}/{tag}"
        rr = subprocess.run(base + ["--out-prefix", g] + extra, capture_output=True, text=True)
        if rr.returncode != 0:
            print(f"{day}: GRID {tag} fail {rr.stderr[-200:]}", flush=True); res = None; break
        res[tag] = dict(
            netl=np.load(f"{g}_pnl_long.npy")[0] * 100.0,
            nets=np.load(f"{g}_pnl_short.npy")[0] * 100.0,
            FL=np.load(f"{g}_filled_long.npy").astype(bool),
            FS=np.load(f"{g}_filled_short.npy").astype(bool))
    if res is None:
        continue
    sts = np.load(f"{od}/sample_ts.npy").astype(np.int64)   # ms, joins to the USDT grid
    buf = io.BytesIO()
    np.savez(buf, sample_ts=sts,
             netl_f=res["frozen"]["netl"].astype(np.float32), nets_f=res["frozen"]["nets"].astype(np.float32),
             FL_f=res["frozen"]["FL"], FS_f=res["frozen"]["FS"],
             netl_s=res["strict"]["netl"].astype(np.float32), nets_s=res["strict"]["nets"].astype(np.float32),
             FL_s=res["strict"]["FL"], FS_s=res["strict"]["FS"])
    mkt.blob(f"{OUT_PREFIX}/D_{day}.npz").upload_from_string(buf.getvalue())
    fr = float((res["frozen"]["FL"] | res["frozen"]["FS"]).mean())
    sr = float((res["strict"]["FL"] | res["strict"]["FS"]).mean())
    print(f"{day}: samp {len(sts)} | fill frozen {fr:.3f} strict {sr:.3f} | "
          f"netl frozen {np.nanmean(np.where(res['frozen']['FL'], res['frozen']['netl'], np.nan)):+.2f} "
          f"strict {np.nanmean(np.where(res['strict']['FL'], res['strict']['netl'], np.nan)):+.2f} -> saved", flush=True)
print(f"[{SYM}] done", flush=True)
