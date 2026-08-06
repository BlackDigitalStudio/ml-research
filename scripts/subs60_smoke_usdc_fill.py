#!/usr/bin/env python3
"""Smoke + first measurement of the rev16 fill fix, end to end on one real day.

Builds the depth/trades parquet for a SYMBOL (works for the USDC venue too), runs
build_samples with --emit-level-flow, then grid_sim_exitdbg twice on the SAME samples:
  A) frozen model            (library entry: unconditional gap-through)
  B) --strict-entry-fill     (price-resolved queue model)
and reports the fill rates + 150s maker PnL under each.

Env: SYMR (DOGEUSDC), DAY (20260723), WORKDIR.
"""
import io, os, subprocess
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; REC = "recorder-data-asia-0998ac51"
RB = "chronos/scalper-recorder/binance_futures"
SYMR = os.environ.get("SYMR", "DOGEUSDC"); DAY = os.environ.get("DAY", "20260723")
TD = os.environ.get("WORKDIR", "/home/delmi/smoke_lvl"); os.makedirs(TD, exist_ok=True)
BS = "/home/delmi/research_bins/husdc_target/release/build_samples"
GRID = "/home/delmi/research_bins/husdc_target/release/grid_sim_exitdbg"
LV = 20; W = 50; H = 6000; STEP_S = 3.0; NS = 1_000_000_000
ENTRY_MS = 60_000; CHASE_MS = 300_000; HOLD_MS = 150_000
rec = storage.Client(project=PROJ).bucket(REC)


def book(out):
    cols = ["timestamp", "receipt_timestamp", "sequence_number"] + \
           [f"{s}_{i}_{f}" for i in range(LV) for s in ("bid", "ask") for f in ("price", "size")]
    acc = {c: [] for c in cols}
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{SYMR}/depth_snapshot/{DAY}"):
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
    ar = {c: np.concatenate(acc[c]) for c in cols}
    o = np.argsort(ar["timestamp"], kind="stable"); ar = {c: ar[c][o] for c in cols}
    ar["sequence_number"] = np.arange(len(ar["timestamp"]), dtype=np.int64)
    pq.write_table(pa.table(ar), out); return len(ar["timestamp"])


def trades(out):
    ts, px, qt, bm = [], [], [], []
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{SYMR}/agg_trade/{DAY}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/t.parquet")
        df = pq.read_table(f"{TD}/t.parquet", columns=["exchange_event_ts_us", "price", "qty", "is_buyer_maker"]).to_pandas()
        df = df[df["exchange_event_ts_us"].notna()]
        ts.append((df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
        px.append(df["price"].astype(float).to_numpy()); qt.append(df["qty"].astype(float).to_numpy())
        bm.append(df["is_buyer_maker"].astype(bool).to_numpy())
    ts = np.concatenate(ts); px = np.concatenate(px); qt = np.concatenate(qt); bm = np.concatenate(bm)
    o = np.argsort(ts, kind="stable")
    t = pa.table({"side": np.where(bm[o], "sell", "buy"), "amount": qt[o], "price": px[o],
                  "id": np.arange(len(ts), dtype=np.int64), "timestamp": ts[o],
                  "receipt_timestamp": ts[o]})
    pq.write_table(t, out); return t.num_rows


nb = book(f"{TD}/book.parquet"); nt = trades(f"{TD}/trades.parquet")
bt = pq.read_table(f"{TD}/book.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
n = len(bt)
print(f"{SYMR} {DAY}: book {nb} rows, trades {nt}", flush=True)
from datetime import datetime as _dt, timezone as _tz
mid0 = int(_dt.strptime(DAY, "%Y%m%d").replace(tzinfo=_tz.utc).timestamp()) * NS
grid = np.arange(mid0, bt[-1], int(STEP_S * NS)); grid = grid[grid >= bt[0]]
ends = np.unique(np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1))
ends = ends[(ends >= W - 1) & (ends < n - H - 1)].astype(np.int64)
np.save(f"{TD}/ends.npy", ends)
print(f"samples: {len(ends)}", flush=True)
od = f"{TD}/bs"; os.makedirs(od, exist_ok=True)
for f in os.listdir(od):
    os.remove(f"{od}/{f}")
r = subprocess.run([BS, "--depth", f"{TD}/book.parquet", "--trades", f"{TD}/trades.parquet",
                    "--out-dir", od, "--window", str(W), "--horizon", str(H),
                    "--sample-ends", f"{TD}/ends.npy", "--skip-xlob", "--emit-level-flow"],
                   capture_output=True, text=True)
print("build_samples rc", r.returncode, r.stderr.strip().splitlines()[-2:] if r.stderr else "", flush=True)
assert r.returncode == 0
assert os.path.exists(f"{od}/flow_lvl_paths.npy"), "flow_lvl_paths.npy not written"
lv = np.load(f"{od}/flow_lvl_paths.npy", mmap_mode="r")
fl = np.load(f"{od}/flow_paths.npy", mmap_mode="r")
print(f"flow_lvl_paths {lv.shape} | level-resolved share of total flow: "
      f"sell {lv[:,:,0].sum()/max(fl[:,:,1].sum(),1):.3f}  buy {lv[:,:,1].sum()/max(fl[:,:,0].sum(),1):.3f}", flush=True)

import json
json.dump([{"tp": 50.0, "sl": 50.0, "to": 282, "to_ms": float(HOLD_MS), "par": False, "tr": False}],
          open(f"{TD}/cfg.json", "w"))
base = [GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
        "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
        "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
        "--entry-q", f"{od}/entry_q.npy", "--configs", f"{TD}/cfg.json",
        "--queue-mult", "1.0", "--exit-queue-mult", "1.0", "--ts-paths", f"{od}/ts_paths.npy",
        "--sample-ts", f"{od}/sample_ts.npy", "--entry-window-ms", str(ENTRY_MS),
        "--chase-ms", str(CHASE_MS), "--entry-window-ticks", "120", "--maker-offset-frac", "0",
        "--commission-win-pct", "0", "--commission-loss-pct", "0"]
for tag, extra in (("A_frozen", []),
                   ("B_strict", ["--strict-entry-fill", "--level-flow-paths", f"{od}/flow_lvl_paths.npy"])):
    g = f"{TD}/{tag}"
    rr = subprocess.run(base + ["--out-prefix", g] + extra, capture_output=True, text=True)
    print(f"{tag} rc {rr.returncode} {rr.stderr.strip()[-160:] if rr.returncode else ''}", flush=True)
    assert rr.returncode == 0
    FL = np.load(f"{g}_filled_long.npy").astype(bool); FS = np.load(f"{g}_filled_short.npy").astype(bool)
    PL = np.load(f"{g}_pnl_long.npy")[0] * 100.0; PS = np.load(f"{g}_pnl_short.npy")[0] * 100.0
    print(f"   {tag}: long fill {FL.mean():.3f} short fill {FS.mean():.3f} | "
          f"netl(filled) {np.nanmean(np.where(FL,PL,np.nan)):+.2f}bp  nets(filled) {np.nanmean(np.where(FS,PS,np.nan)):+.2f}bp", flush=True)
