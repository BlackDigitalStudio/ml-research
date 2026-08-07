#!/usr/bin/env python3
"""HD3 rev13 prep: one recorder day -> fb_incr_harness inputs (book/trades/fund/liq/oi/eth
parquets + grid indices). Functions copied VERBATIM from the frozen
subs60_recorder_ev_h150.py (book_cl/trades_cl/liq_cl/funding_cl/oi_cl + the midnight
3s grid); FUNDING_MODE env passes through (run once per mode). No BS/GRID/scoring —
feature-parity harness needs only the inputs.
Env: DAY (YYYYMMDD), SYMR (DOGEUSDT), TD (workdir), FUNDING_MODE."""
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

SYMR = os.environ.get("SYMR", "DOGEUSDT")
DAY = os.environ["DAY"]
TD = os.environ.get("TD", "/home/delmi/parity_fund")
os.makedirs(TD, exist_ok=True)
LV = 20; NS = 1_000_000_000; W = 50; H = 6000; STEP_S = 3.0
RB = "chronos/scalper-recorder/binance_futures"
rec = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("recorder-data-asia-0998ac51")
FUNDING_MODE = os.environ.get("FUNDING_MODE", "anchor")


def dl_rec(sym, stream, day, cols):
    rows = {c: [] for c in cols}; got = False
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{sym}/{stream}/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/r.parquet")
        df = pq.read_table(f"{TD}/r.parquet").to_pandas(); got = True
        for c in cols:
            rows[c].append(df[c].to_numpy() if c in df.columns else np.full(len(df), np.nan))
    if not got:
        return None
    return {c: (np.concatenate(rows[c]) if rows[c] else np.array([])) for c in cols}


def book_cl(sym, day, out):
    cols = ["timestamp", "receipt_timestamp", "sequence_number"] + [f"{s}_{i}_{f}" for i in range(LV) for s in ("bid", "ask") for f in ("price", "size")]
    acc = {c: [] for c in cols}
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{sym}/depth_snapshot/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/d.parquet")
        df = pq.read_table(f"{TD}/d.parquet", columns=["exchange_event_ts_us", "local_ts_us", "bid_prices", "bid_qtys", "ask_prices", "ask_qtys"]).to_pandas()
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
    ar = {c: np.concatenate(acc[c]) for c in cols}; o = np.argsort(ar["timestamp"], kind="stable"); ar = {c: ar[c][o] for c in cols}
    ar["sequence_number"] = np.arange(len(ar["timestamp"]), dtype=np.int64)
    pq.write_table(pa.table({c: ar[c] for c in cols}), out); return len(ar["timestamp"])


def trades_cl(sym, day, out, stream="agg_trade"):
    d = dl_rec(sym, stream, day, ["exchange_event_ts_us", "local_ts_us", "trade_id", "price", "qty", "is_buyer_maker"])
    if d is None or not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    tid = d["trade_id"][m]
    _, ui = np.unique(tid.astype(np.int64), return_index=True); ui = np.sort(ui)
    t = pa.table({"side": np.where(d["is_buyer_maker"][m][ui].astype(bool), "sell", "buy"),
                  "amount": d["qty"][m][ui].astype(np.float64), "price": d["price"][m][ui].astype(np.float64),
                  "id": tid[ui].astype(np.int64), "timestamp": (d["exchange_event_ts_us"][m][ui].astype(np.int64) * 1000),
                  "receipt_timestamp": (d["local_ts_us"][m][ui].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


def liq_cl(sym, day, out):
    d = dl_rec(sym, "liquidation", day, ["exchange_event_ts_us", "local_ts_us", "side", "original_qty", "price"])
    if d is None or not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    if not m.any():
        return 0
    n = int(m.sum())
    t = pa.table({"side": np.array([str(s).lower() for s in d["side"][m]]), "quantity": d["original_qty"][m].astype(np.float64),
                  "price": d["price"][m].astype(np.float64), "id": np.arange(n, dtype=np.int64), "status": np.array(["filled"] * n),
                  "timestamp": (d["exchange_event_ts_us"][m].astype(np.int64) * 1000), "receipt_timestamp": (d["local_ts_us"][m].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


def funding_cl(sym, day, out):
    d = dl_rec(sym, "mark_price", day, ["exchange_event_ts_us", "local_ts_us", "funding_rate", "mark_price"])
    if d is None or not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    if FUNDING_MODE == "anchor":
        ets = d["exchange_event_ts_us"][m].astype(np.int64)
        fr = np.nan_to_num(d["funding_rate"][m].astype(np.float64))
        i = int(np.argmin(ets))
        t = pa.table({"funding_rate": np.array([fr[i]], np.float64),
                      "mark_price": np.array([0.0], np.float64),
                      "timestamp": np.array([1], np.int64)})
    else:
        t = pa.table({"funding_rate": np.nan_to_num(d["funding_rate"][m].astype(np.float64)), "mark_price": d["mark_price"][m].astype(np.float64),
                      "timestamp": (d["exchange_event_ts_us"][m].astype(np.int64) // 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


def oi_cl(sym, day, out):
    d = dl_rec(sym, "derivatives_poll", day, ["local_ts_us", "open_interest"])
    if d is None or not len(d["local_ts_us"]):
        return 0
    m = ~np.isnan(d["local_ts_us"].astype(float)) & ~np.isnan(d["open_interest"].astype(float))
    if not m.any():
        return 0
    t = pa.table({"open_interest": d["open_interest"][m].astype(np.float64),
                  "timestamp": (d["local_ts_us"][m].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


nb = book_cl(SYMR, DAY, f"{TD}/book.parquet")
ntr = trades_cl(SYMR, DAY, f"{TD}/trades.parquet")
nliq = liq_cl(SYMR, DAY, f"{TD}/liq.parquet")
nfd = funding_cl(SYMR, DAY, f"{TD}/fund.parquet")
noi = oi_cl(SYMR, DAY, f"{TD}/oi.parquet")
neth = trades_cl("ETHUSDT", DAY, f"{TD}/eth.parquet")
bt = pq.read_table(f"{TD}/book.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
n = len(bt)
from datetime import datetime as _dt, timezone as _tz
mid0 = int(_dt.strptime(DAY, "%Y%m%d").replace(tzinfo=_tz.utc).timestamp()) * NS
grid = np.arange(mid0, bt[-1], int(STEP_S * NS))
grid = grid[grid >= bt[0]]
ends = np.unique(np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1))
ends = ends[(ends >= W - 1) & (ends < n - H - 1)].astype(np.int64)
np.save(f"{TD}/ends.npy", ends)
print(f"{DAY} {SYMR} mode={FUNDING_MODE}: book={nb} tr={ntr} liq={nliq} fund={nfd} oi={noi} eth={neth} ends={len(ends)}", flush=True)
