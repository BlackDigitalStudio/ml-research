#!/usr/bin/env python3
"""Deploy-candidate labels: honest TIME-BASED maker cycle at the qm1-equivalent horizon —
entry window 60s from decision, holds {90,150,240}s FROM FILL, pegged chase 300s, always-last
both legs, 0 fee, 3s time-uniform grid. Features are NOT recomputed: F/rH/btc/tod merged from
maker_labels_tb3s_full dailies by exact ts (same grid; larger H trims a few day-tail rows).
Output: research_runs/maker_labels_tb3s_h150/daily/DOGE_{day}.npz -> COMBINE via
subs60_build_tb3s_labels.py (OUTSUB=research_runs/maker_labels_tb3s_h150)."""
import io, json, os, subprocess, time
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RAWB = "raw/book/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP"
RAWT = "raw/trades/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP"
BS = "/tmp/husdc_target/release/build_samples"; GRID = "/tmp/husdc_target/release/grid_sim_exitdbg"
SRC = "research_runs/maker_labels_tb3s_full/daily"; DST = "research_runs/maker_labels_tb3s_h150/daily"
NS = 1_000_000_000; W = 50; H = 1500; STEP_S = 3.0
ENTRY_MS = 60_000; CHASE_MS = 300_000; HOLDS_S = [90.0, 150.0, 240.0]
TD = "/home/delmi/tb3s_h150"; os.makedirs(TD, exist_ok=True)
bk = storage.Client(project=PROJ).bucket(BUCKET)


def dl(prefix, dst):
    n = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not n:
        return False
    bk.blob(n).download_to_filename(dst); return True


days = sorted(b.name.split("_")[-1][:-4] for b in bk.client.list_blobs(bk, prefix=f"{SRC}/DOGE_") if b.name.endswith(".npz"))
done = {b.name.split("_")[-1][:-4] for b in bk.client.list_blobs(bk, prefix=f"{DST}/DOGE_") if b.name.endswith(".npz")}
todo = [d for d in days if d not in done]
print(f"[h150] {len(days)} days, {len(todo)} to do | entry {ENTRY_MS}ms holds {HOLDS_S}s chase {CHASE_MS}ms H {H}", flush=True)
cfgs = [{"tp": 50.0, "sl": 50.0, "to": 282, "to_ms": h * 1000.0, "par": False, "tr": False} for h in HOLDS_S]
json.dump(cfgs, open(f"{TD}/cfg.json", "w"))

for i, day in enumerate(todo):
    t0 = time.time()
    bp, tp = f"{TD}/b.parquet", f"{TD}/t.parquet"
    if not (dl(f"{RAWB}/dt={day}/", bp) and dl(f"{RAWT}/dt={day}/", tp)):
        print(f"  {day}: no-raw", flush=True); continue
    tt = pq.read_table(tp)
    ids = tt["id"].to_numpy()
    if len(ids) / max(len(np.unique(ids)), 1) > 1.001:
        _, ui = np.unique(ids, return_index=True)
        pq.write_table(tt.take(np.sort(ui)), tp)
    ts_ns = pq.read_table(bp, columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    n = len(ts_ns)
    if n < W + H + 100:
        print(f"  {day}: thin {n}", flush=True); continue
    grid = np.arange(ts_ns[0], ts_ns[-1], int(STEP_S * NS))
    ends = np.unique(np.clip(np.searchsorted(ts_ns, grid, "right") - 1, 0, n - 1))
    ends = ends[(ends >= W - 1) & (ends < n - H - 1)].astype(np.int64)
    if len(ends) < 100:
        print(f"  {day}: few-ends", flush=True); continue
    np.save(f"{TD}/ends.npy", ends)
    od = f"{TD}/bs"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(f"{od}/{f}")
    r = subprocess.run([BS, "--depth", bp, "--trades", tp, "--out-dir", od, "--window", str(W),
                        "--horizon", str(H), "--sample-ends", f"{TD}/ends.npy", "--skip-xlob"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {day}: BS-fail {r.stderr[-150:]}", flush=True); continue
    se = np.load(f"{od}/end_indices.npy").astype(np.int64)
    g = f"{TD}/g"
    rr = subprocess.run([GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
                         "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
                         "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
                         "--entry-q", f"{od}/entry_q.npy", "--configs", f"{TD}/cfg.json", "--out-prefix", g,
                         "--queue-mult", "1.0", "--exit-queue-mult", "1.0",
                         "--ts-paths", f"{od}/ts_paths.npy", "--sample-ts", f"{od}/sample_ts.npy",
                         "--entry-window-ms", str(ENTRY_MS), "--chase-ms", str(CHASE_MS),
                         "--entry-window-ticks", "120", "--maker-offset-frac", "0",
                         "--commission-win-pct", "0", "--commission-loss-pct", "0"],
                        capture_output=True, text=True)
    if rr.returncode != 0:
        print(f"  {day}: GRID-fail {rr.stderr[-200:]}", flush=True); continue
    PL = np.load(f"{g}_pnl_long.npy"); PS = np.load(f"{g}_pnl_short.npy")
    FL = np.load(f"{g}_filled_long.npy"); FS = np.load(f"{g}_filled_short.npy")
    dtd = ts_ns[se]
    # merge F/rH from tb3s_full daily by exact ts (new ends subset of old due to bigger H)
    z = np.load(io.BytesIO(bk.blob(f"{SRC}/DOGE_{day}.npz").download_as_bytes()))
    ots = z["ts"].astype(np.int64)
    pos = np.searchsorted(ots, dtd)
    ok = (pos < len(ots)) & (ots[np.clip(pos, 0, len(ots) - 1)] == dtd)
    if ok.mean() < 0.95:
        print(f"  {day}: ts-merge only {100*ok.mean():.0f}%", flush=True); continue
    ki = np.where(ok)[0]; oi = pos[ki]
    buf = io.BytesIO()
    np.savez_compressed(buf, F=z["F"][oi], rH15=z["rH15"][oi], rH30=z["rH30"][oi], rH60=z["rH60"][oi],
                        ts=dtd[ki], pnl_long=PL[:, ki].astype(np.float32), pnl_short=PS[:, ki].astype(np.float32),
                        FL=FL[ki].astype(np.uint8), FS=FS[ki].astype(np.uint8))
    bk.blob(f"{DST}/DOGE_{day}.npz").upload_from_string(buf.getvalue())
    if i % 40 == 0 or i == 0:
        fl = FL[ki].astype(bool)
        print(f"  {day} ({i+1}/{len(todo)}): n={len(ki)} fill={fl.mean():.2f} "
              f"netl150(filled)={np.nanmean(np.where(fl, PL[1, ki]*100, np.nan)):+.2f}bp [{time.time()-t0:.0f}s]", flush=True)
print("[h150 build DONE]", flush=True)
