#!/usr/bin/env python3
"""Prototype the PEGGED maker exit (no taker, always-last queue) and sweep exit_queue_mult.
Replicate build_day for a few days: download raw -> build_samples (once/day) -> grid_sim_exitdbg
(30s config, entry qm0) for exit_queue_mult in {0,1,2,4}. Report passive-fill% vs ran-out and the
gross EV (0 fee) per side. Tells us if a maker-only pegged exit survives, and how it depends on the
always-last queue depth (the only realism lever).
"""
import subprocess, os, json
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")
BS = "/tmp/husdc/rust_ingest/target/release/build_samples"
GRID = "/tmp/edbg_target/release/grid_sim_exitdbg"
RAWB = "raw/book/exchange=BINANCE_FUTURES"; RAWT = "raw/trades/exchange=BINANCE_FUTURES"
WINDOW, H_TICKS, ENTRY_WIN, TARGET = 50, 700, 120, 40000
CFG = [{"tp": 50.0, "sl": 50.0, "to": 282, "par": False, "tr": False}]  # 30s hold, CFGIDX=1
PAIRS = [("BTC-USDT-PERP", "2025-09-15"), ("ETH-USDT-PERP", "2025-11-10"), ("DOGE-USDT-PERP", "2025-09-15")]
EXIT_QMS = [0.0, 1.0, 2.0, 4.0]


def dl(prefix, dst):
    name = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not name:
        return False
    bk.blob(name).download_to_filename(dst); return True


for sym, day in PAIRS:
    od = "/dev/shm/probe_od"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    bp, tp = "/dev/shm/b.parquet", "/dev/shm/t.parquet"
    if not (dl(f"{RAWB}/symbol={sym}/dt={day}/", bp) and dl(f"{RAWT}/symbol={sym}/dt={day}/", tp)):
        print(f"\n===== {sym} {day}: no raw ====="); continue
    nrows = pq.ParquetFile(bp).metadata.num_rows
    step = max(1, -(-nrows // TARGET))
    r = subprocess.run([BS, "--depth", bp, "--trades", tp, "--out-dir", od, "--window", str(WINDOW),
                        "--horizon", str(H_TICKS), "--step", str(step), "--max-samples", "2000000"],
                       capture_output=True, text=True)
    for f in (bp, tp):
        try: os.remove(f)
        except OSError: pass
    if r.returncode != 0:
        print(f"\n===== {sym} {day}: BS fail: {r.stderr[-200:]} ====="); continue
    json.dump(CFG, open("/dev/shm/cfg.json", "w"))
    print(f"\n===== {sym} {day} (nrows={nrows}, step={step}) =====")
    print(f"  {'exitQM':>6} {'pass%L':>7} {'pass%S':>7} {'EVl(bp)':>8} {'EVs(bp)':>8} {'EVavg':>7}")
    for qm in EXIT_QMS:
        g = "/dev/shm/g"
        cmd = [GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
               "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
               "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
               "--entry-q", f"{od}/entry_q.npy", "--configs", "/dev/shm/cfg.json", "--out-prefix", g,
               "--queue-mult", "0", "--exit-queue-mult", str(qm), "--entry-window-ticks", str(ENTRY_WIN),
               "--maker-offset-frac", "0", "--commission-win-pct", "0", "--commission-loss-pct", "0"]
        r2 = subprocess.run(cmd, capture_output=True, text=True)
        # parse passive-fill % from stderr histogram
        def passpct(side):
            tot = fill = 0
            grab = False
            for ln in r2.stderr.splitlines():
                if f"EXIT-REASON [{side}]" in ln:
                    grab = True; continue
                if grab:
                    if "PassiveFill" in ln:
                        fill = float(ln.split("(")[-1].split("%")[0])
                    if "EXIT-REASON" in ln or "filled" in ln and side not in ln:
                        break
            return fill
        pl = np.load(f"{g}_pnl_long.npy")[0]; ps = np.load(f"{g}_pnl_short.npy")[0]
        fl = np.load(f"{g}_filled_long.npy").astype(bool); fs = np.load(f"{g}_filled_short.npy").astype(bool)
        evl = float(np.nanmean(pl[fl]) * 100); evs = float(np.nanmean(ps[fs]) * 100)
        print(f"  {qm:>6.1f} {passpct('LONG'):>7.1f} {passpct('SHORT'):>7.1f} {evl:>+8.2f} {evs:>+8.2f} {(evl+evs)/2:>+7.2f}")
