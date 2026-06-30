#!/usr/bin/env python3
"""Derive cryptolake's book downsampling rule from the 2026-06-05 overlap. Both cryptolake `book`
and chronos depth carry Binance exchange-event time, so cl's 1.3/s snapshots should be a SUBSET of
chronos's ~9/s events. Find which events cl kept and test whether a reproducible, live-applicable
rule predicts 'kept' (time-interval? mid-move threshold? size churn?). If yes -> live can reproduce
cl sampling. If not -> cl sampling is infra-bound and irreproducible -> dense-trained model needed.
"""
import numpy as np, pyarrow.parquet as pq
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; TICK = 0.00001
bk = storage.Client(project=PROJ).bucket(BUCKET); DAY = "2026-06-05"

# cryptolake book (kept subset)
n = next(b.name for b in bk.client.list_blobs(bk, prefix=f"raw/book/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={DAY}/") if b.name.endswith(".parquet"))
bk.blob(n).download_to_filename("/tmp/cl.parquet")
ct = pq.read_table("/tmp/cl.parquet", columns=["timestamp", "bid_0_price", "ask_0_price"]).to_pandas().sort_values("timestamp")
cl_us = (ct["timestamp"].to_numpy().astype(np.int64) // 1000)
print(f"cryptolake snapshots: {len(cl_us)}", flush=True)

# chronos depth (full 9/s event stream)
ch_us, ch_mid = [], []
for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/depth_snapshot/"):
    if not b.name.endswith(".parquet"):
        continue
    bk.blob(b.name).download_to_filename("/tmp/ch.parquet")
    df = pq.read_table("/tmp/ch.parquet", columns=["exchange_event_ts_us", "bid_prices", "ask_prices"]).to_pandas()
    df = df[df["exchange_event_ts_us"].notna()]
    ch_us.append(df["exchange_event_ts_us"].astype("int64").to_numpy())
    ch_mid.append((df["bid_prices"].apply(lambda x: float(x[0])) + df["ask_prices"].apply(lambda x: float(x[0]))).to_numpy() / 2)
ch_us = np.concatenate(ch_us); ch_mid = np.concatenate(ch_mid)
o = np.argsort(ch_us); ch_us, ch_mid = ch_us[o], ch_mid[o]
# restrict chronos to cl window
m = (ch_us >= cl_us[0] - 1000) & (ch_us <= cl_us[-1] + 1000); ch_us, ch_mid = ch_us[m], ch_mid[m]
print(f"chronos events in window: {len(ch_us)} ({len(ch_us)/(len(cl_us))::.1f}x cl)" if False else f"chronos events in window: {len(ch_us)} ({len(ch_us)/max(len(cl_us),1):.1f}x cl)", flush=True)

# exact-ish match: each cl snapshot -> nearest chronos event
j = np.clip(np.searchsorted(ch_us, cl_us), 1, len(ch_us) - 1)
jn = np.where(np.abs(ch_us[j - 1] - cl_us) < np.abs(ch_us[j] - cl_us), j - 1, j)
dt = np.abs(ch_us[jn] - cl_us)
exact = (dt < 2000).mean()  # within 2ms = same event
print(f"cl snapshots matching a chronos event within 2ms: {100*exact:.1f}% (-> cl is a subset of chronos events)", flush=True)
kept = np.zeros(len(ch_us), bool); kept[jn[dt < 50000]] = True
print(f"kept chronos events: {int(kept.sum())} / {len(ch_us)} = {100*kept.mean():.1f}%", flush=True)

# --- is the spacing time-based or event/move-based? ---
kept_idx = np.where(kept)[0]
ivl = np.diff(ch_us[kept_idx]) / 1000.0  # ms between kept
print(f"\n[kept interval] med {np.median(ivl):.0f}ms p10 {np.percentile(ivl,10):.0f} p90 {np.percentile(ivl,90):.0f}", flush=True)

# mid-move since last kept, measured at kept vs dropped events
last_kept_mid = np.empty(len(ch_us)); lk = ch_mid[0]
for i in range(len(ch_us)):
    last_kept_mid[i] = lk
    if kept[i]:
        lk = ch_mid[i]
move_ticks = np.abs(ch_mid - last_kept_mid) / TICK
print(f"|mid-move since last kept| ticks: at KEPT events med {np.median(move_ticks[kept]):.2f} p90 {np.percentile(move_ticks[kept],90):.2f}", flush=True)
print(f"                                  at DROPPED events med {np.median(move_ticks[~kept]):.2f} p90 {np.percentile(move_ticks[~kept],90):.2f}", flush=True)

# --- test a reproducible rule: greedy keep when |mid-move| >= tau OR >= max_gap ms; match cl's kept set ---
best = None
for tau in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for maxgap in [1000, 2000, 5000]:
        sim = np.zeros(len(ch_us), bool); lkm = ch_mid[0]; lkt = ch_us[0]; sim[0] = True
        for i in range(1, len(ch_us)):
            if abs(ch_mid[i] - lkm) / TICK >= tau or (ch_us[i] - lkt) / 1000.0 >= maxgap:
                sim[i] = True; lkm = ch_mid[i]; lkt = ch_us[i]
        inter = (sim & kept).sum(); union = (sim | kept).sum(); jac = inter / max(union, 1)
        rate = sim.mean() * len(ch_us) / ((ch_us[-1] - ch_us[0]) / 1e6)
        if best is None or jac > best[0]:
            best = (jac, tau, maxgap, rate)
print(f"\n[best reproducible rule vs cl-kept] Jaccard {best[0]:.2f} (tau={best[1]}tick, maxgap={best[2]}ms, rate {best[3]:.2f}/s)", flush=True)
print(f"  -> {'REPRODUCIBLE (rule predicts cl kept-set)' if best[0]>0.6 else 'NOT reproducible (cl kept-set not explained by mid-move/interval -> infra-bound)'}", flush=True)
