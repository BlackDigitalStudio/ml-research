#!/usr/bin/env python3
"""TB3S label build: HONEST time-based maker labels on the Cryptolake year, ~3s decision cadence.

Fixes the two sim-vs-live honesty gaps found in the 2026-07-05 optimism audit:
  1. TIME-BASED windows. CL books are event-driven (1.4-2.3 snaps/s), so the old
     tick-count windows ("282 ticks = 30s") actually priced 2-3-MINUTE holds. Here the
     maker sim (grid_sim_exitdbg --ts-paths) uses wall-clock ms: entry window 12.8s from
     the decision ts, hold {15,30,60}s FROM THE FILL ts, pegged-exit chase 300s (honest
     chase — recorder study: ~98.6% of pegged exits fill within 5 min; residual ran-out
     marked at touch).
  2. ~3s DECISION CADENCE, TIME-UNIFORM (live decides on wall clock, not on book events):
     decision ticks = last book tick <= each point of a 3s grid (deduped).

Everything else mirrors the deployed robust2 label build: husdc build_samples (always-last
maker entry at touch, qm=1.0), exit_queue_mult=1.0, 0 commission (DOGEUSDC maker 0%),
robust feature_builder X64 (vol 1s-grid, trade-dedup) at the SAME decision ticks, real
btc_lead from feats_sub60 BTC mids, ToD4. rH15/30/60 TIME-based from book mid (+-5s tol).

Per-day npz -> gs://.../research_runs/maker_labels_tb3s/daily/DOGE_{day}.npz (resumable;
PARITY/NSHARD env for 2-worker sharding). COMBINE=1 -> single DOGE.npz in the
subs60_xgb_optuna_ic schema: F,rH30,rH15,rH60,day,ts,pnl_long(NC,1,N),pnl_short,
fill_long(1,N),fill_short,feat_names,meta.

Env: START,END (day range), STEP_S(3), H_TICKS(1100), ENTRY_MS(12800), CHASE_MS(300000),
HOLDS_S(15,30,60), WORKDIR, PARITY,NSHARD, COMBINE. Run on hd2-feats-003.
"""
import io, json, os, subprocess, time
import numpy as np
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMF = os.environ.get("SYMF", "DOGE-USDT-PERP"); SYMK = SYMF.split("-")[0]
RAWB = f"raw/book/exchange=BINANCE_FUTURES/symbol={SYMF}"
RAWT = f"raw/trades/exchange=BINANCE_FUTURES/symbol={SYMF}"
# FULLFEAT=1: feed FB the full input set (funding/liquidations/open-interest/ETH-trades) —
# without it cols 13,14-16/55,56-60 are silently zero (the robust-rebuild omission).
FULLFEAT = os.environ.get("FULLFEAT", "") == "1"
ETHF = "ETH-USDT-PERP"
FB = os.environ.get("FB_BIN", "/tmp/fb_target/release/feature_builder")          # robust feats
BS = os.environ.get("BS_BIN", "/tmp/husdc_target/release/build_samples")         # husdc + time-mode
GRID = os.environ.get("GRID_BIN", "/tmp/husdc_target/release/grid_sim_exitdbg")  # pegged exit + time-mode
OUT = "research_runs/maker_labels_tb3s"
NS = 1_000_000_000
START = os.environ.get("START", "2025-05-09"); END = os.environ.get("END", "2026-06-02")
STEP_S = float(os.environ.get("STEP_S", "3"))
W = 50; H_TICKS = int(os.environ.get("H_TICKS", "1100"))
ENTRY_MS = int(os.environ.get("ENTRY_MS", "12800"))
CHASE_MS = int(os.environ.get("CHASE_MS", "300000"))
HOLDS_S = [float(x) for x in os.environ.get("HOLDS_S", "15,30,60").split(",")]
QM = 1.0; EXIT_QM = 1.0
# LEGACY=1: pipeline-validation cell — reproduce the OLD tick-window semantics through this
# exact build path (tick-strided ~TICK_TPD/day grid, entry 120 ticks, holds in TICKS
# [141,282,563], chase to path end, H=700). Target: reproduce the recorded ~+3.4..+4.3
# year EV; validates the new pipeline end-to-end, attributing the tb3s delta to semantics.
LEGACY = os.environ.get("LEGACY", "") == "1"
TICK_TPD = int(os.environ.get("TICK_TPD", "5950"))
OUT = os.environ.get("OUTSUB", OUT)
PARITY = int(os.environ.get("PARITY", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
COMBINE = os.environ.get("COMBINE", "") == "1"
TD = os.environ.get("WORKDIR", "/home/delmi/tb3s"); os.makedirs(TD, exist_ok=True)  # DISK, not /dev/shm (logind RemoveIPC wipes shm)
NAMES = [f"x{c}" for c in range(64)] + ["btc_ret5", "btc_ret30", "btc_ret60", "sin_h", "cos_h", "sin_f8", "cos_f8"]

cl = storage.Client(project=PROJ); bk = cl.bucket(BUCKET)


def log(s):
    print(s, flush=True)


def dl_raw(prefix, dst):
    name = next((b.name for b in cl.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not name:
        return False
    bk.blob(name).download_to_filename(dst); return True


def load_btc_mid():
    blobs = sorted(b.name for b in cl.list_blobs(bk, prefix="feats_sub60/BTC-USDT-PERP/") if b.name.endswith(".npz"))
    def fetch(n):
        return np.load(io.BytesIO(bk.blob(n).download_as_bytes()))
    tds, mids = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for d in ex.map(fetch, blobs):
            tds.append(d["td"].astype(np.int64)); mids.append(d["mid"].astype(np.float64))
    td = np.concatenate(tds); mid = np.concatenate(mids); o = np.argsort(td, kind="stable")
    return td[o], mid[o]


def feat71(dtd, X, bt, bm):
    nb = len(bt); i = np.clip(np.searchsorted(bt, dtd, "right") - 1, 0, nb - 1)
    bl = []
    for Wd in (5, 30, 60):
        j = np.clip(np.searchsorted(bt, dtd - int(Wd * NS), "right") - 1, 0, nb - 1)
        a = bm[j]; b = bm[i]
        bl.append(np.where((a > 0) & (b > 0), np.log(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0)), 0.0) * 1e4)
    h = ((dtd / NS) % 86400.0) / 3600.0; hf = h % 8.0
    tod = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * hf / 8), np.cos(2 * np.pi * hf / 8)]
    return np.concatenate([X, np.stack(bl, 1), np.stack(tod, 1)], axis=1).astype(np.float32)


def rh_time(ts_ns, mid, ends, hs):
    """Time-based fwd log-return over hs seconds from each decision tick (bp; NaN if no tick within +-5s)."""
    tgt = ts_ns[ends] + int(hs * NS)
    rf = np.clip(np.searchsorted(ts_ns, tgt, "left"), 0, len(ts_ns) - 1)
    ok = (np.abs(ts_ns[rf] - tgt) <= 5 * NS) & (mid[rf] > 0) & (mid[ends] > 0)
    r = np.full(len(ends), np.nan, np.float32)
    r[ok] = (np.log(mid[rf][ok] / mid[ends][ok]) * 1e4).astype(np.float32)
    return r


def day_list():
    days = sorted(set(b.name.split("dt=")[1][:10] for b in cl.list_blobs(bk, prefix=RAWB + "/")
                      if "dt=" in b.name and b.name.endswith(".parquet")))
    return [d for d in days if START <= d <= END]


def process_day(day, bt, bm):
    bp, tp = f"{TD}/b.parquet", f"{TD}/t.parquet"
    if not (dl_raw(f"{RAWB}/dt={day}/", bp) and dl_raw(f"{RAWT}/dt={day}/", tp)):
        return "no-raw"
    t0 = time.time()
    # dedup CL trade triplication by id (recent-CL-days artifact; no-op dup=1.00 on clean days).
    # build_samples aggregates flow straight from this parquet — inflated flow = optimistic fills.
    tt = pq.read_table(tp)
    ids = tt["id"].to_numpy()
    dup = len(ids) / max(len(np.unique(ids)), 1)
    if dup > 1.001:
        _, ui = np.unique(ids, return_index=True)
        pq.write_table(tt.take(np.sort(ui)), tp)
    tbl = pq.read_table(bp, columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts_ns = tbl["timestamp"].to_numpy().astype(np.int64)
    b0 = tbl["bid_0_price"].to_numpy().astype(np.float64); a0 = tbl["ask_0_price"].to_numpy().astype(np.float64)
    mid = np.where((b0 > 0) & (a0 > 0), 0.5 * (b0 + a0), 0.0)
    n = len(ts_ns)
    if n < W + H_TICKS + 100:
        return f"thin-book n={n}"
    if LEGACY:
        # tick-strided grid like the old feats/build pipeline (~TICK_TPD decisions/day)
        stp = max(1, (n - W - H_TICKS) // TICK_TPD)
        ends = np.arange(W - 1, n - H_TICKS - 1, stp, dtype=np.int64)
    else:
        # time-uniform 3s decision grid -> last tick <= grid point (live wall-clock cadence)
        grid = np.arange(ts_ns[0], ts_ns[-1], int(STEP_S * NS))
        ends = np.unique(np.clip(np.searchsorted(ts_ns, grid, "right") - 1, 0, n - 1))
        ends = ends[(ends >= W - 1) & (ends < n - H_TICKS - 1)].astype(np.int64)
    if len(ends) < 100:
        return f"few-ends {len(ends)}"
    np.save(f"{TD}/ends.npy", ends)
    od = f"{TD}/bs"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(f"{od}/{f}")
    r = subprocess.run([BS, "--depth", bp, "--trades", tp, "--out-dir", od, "--window", str(W),
                        "--horizon", str(H_TICKS), "--sample-ends", f"{TD}/ends.npy", "--skip-xlob"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"BS-fail:{r.stderr[-200:]}"
    t_bs = time.time() - t0
    # sample_ends after build_samples' own validity filter (starts sorted+deduped => order preserved)
    se = np.load(f"{od}/end_indices.npy").astype(np.int64)
    if LEGACY:
        cfgs = [{"tp": 50.0, "sl": 50.0, "to": t, "par": False, "tr": False} for t in (141, 282, 563)]
    else:
        cfgs = [{"tp": 50.0, "sl": 50.0, "to": 282, "to_ms": hs * 1000.0, "par": False, "tr": False} for hs in HOLDS_S]
    json.dump(cfgs, open(f"{TD}/cfg.json", "w"))
    g = f"{TD}/g"; t1 = time.time()
    cmd = [GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
           "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
           "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
           "--entry-q", f"{od}/entry_q.npy", "--configs", f"{TD}/cfg.json", "--out-prefix", g,
           "--queue-mult", str(QM), "--exit-queue-mult", str(EXIT_QM),
           "--entry-window-ticks", "120", "--maker-offset-frac", "0",
           "--commission-win-pct", "0", "--commission-loss-pct", "0"]
    if not LEGACY:
        cmd += ["--ts-paths", f"{od}/ts_paths.npy", "--sample-ts", f"{od}/sample_ts.npy",
                "--entry-window-ms", str(ENTRY_MS), "--chase-ms", str(CHASE_MS)]
    rr = subprocess.run(cmd, capture_output=True, text=True)
    if rr.returncode != 0:
        return f"GRID-fail:{rr.stderr[-250:]}"
    t_gs = time.time() - t1
    PL = np.load(f"{g}_pnl_long.npy"); PS = np.load(f"{g}_pnl_short.npy")            # (NC,N) pct
    FL = np.load(f"{g}_filled_long.npy"); FS = np.load(f"{g}_filled_short.npy")      # (N,) u8
    # features at the same decision ticks
    t2 = time.time()
    np.save(f"{TD}/idx.npy", se)
    fcmd = [FB, "--depth", bp, "--indices", f"{TD}/idx.npy", "--out", f"{TD}/f.npy", "--trades", tp]
    if FULLFEAT:
        for stream, symq, flag in (("funding", SYMF, "--funding"), ("liquidations", SYMF, "--liquidations"),
                                   ("open_interest", SYMF, "--open-interest"), ("trades", ETHF, "--eth")):
            fpth = f"{TD}/{flag.strip('-')}.parquet"
            if dl_raw(f"raw/{stream}/exchange=BINANCE_FUTURES/symbol={symq}/dt={day}/", fpth):
                fcmd += [flag, fpth]
    fr = subprocess.run(fcmd, capture_output=True, text=True)
    if fr.returncode != 0:
        return f"FB-fail:{fr.stderr[-200:]}"
    t_fb = time.time() - t2
    X = np.load(f"{TD}/f.npy").astype(np.float32)
    if len(X) != len(se):
        return f"FB-len {len(X)} != {len(se)}"
    dtd = ts_ns[se]
    F = feat71(dtd, X, bt, bm)
    r15 = rh_time(ts_ns, mid, se, 15.0); r30 = rh_time(ts_ns, mid, se, 30.0); r60 = rh_time(ts_ns, mid, se, 60.0)
    buf = io.BytesIO()
    np.savez_compressed(buf, F=F, rH15=r15, rH30=r30, rH60=r60, ts=dtd,
                        pnl_long=PL.astype(np.float32), pnl_short=PS.astype(np.float32),
                        FL=FL.astype(np.uint8), FS=FS.astype(np.uint8))
    bk.blob(f"{OUT}/daily/{SYMK}_{day}.npz").upload_from_string(buf.getvalue())
    fl = FL.astype(bool); fill_r = float(fl.mean())
    i30 = min(range(len(HOLDS_S)), key=lambda i: abs(HOLDS_S[i] - 30.0))
    nl30 = PL[i30] * 100.0
    return (f"ok n={len(se)} book={n} dens={n/max((ts_ns[-1]-ts_ns[0])/NS,1):.2f}/s dup={dup:.2f} "
            f"fill={fill_r:.2f} netl30(filled)={np.nanmean(np.where(fl, nl30, np.nan)):+.2f}bp "
            f"[bs={t_bs:.0f}s gs={t_gs:.0f}s fb={t_fb:.0f}s]")


def combine():
    blobs = sorted(b.name for b in cl.list_blobs(bk, prefix=f"{OUT}/daily/{SYMK}_") if b.name.endswith(".npz"))
    log(f"[combine] {len(blobs)} day files")
    aF, a15, a30, a60, aTs, aPL, aPS, aFL, aFS, aDay = [], [], [], [], [], [], [], [], [], []
    for di, nm in enumerate(blobs):
        z = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
        m = np.isfinite(z["rH30"])          # drop rows without a valid 30s target (day tail)
        aF.append(z["F"][m]); a15.append(z["rH15"][m]); a30.append(z["rH30"][m]); a60.append(z["rH60"][m])
        aTs.append(z["ts"][m]); aPL.append(z["pnl_long"][:, m]); aPS.append(z["pnl_short"][:, m])
        aFL.append(z["FL"][m]); aFS.append(z["FS"][m]); aDay.append(np.full(int(m.sum()), di, np.int32))
        if di % 40 == 0:
            log(f"  {di}/{len(blobs)} {nm.split('/')[-1]}")
    F = np.concatenate(aF); day = np.concatenate(aDay); ts = np.concatenate(aTs)
    PL = np.concatenate(aPL, 1)[:, None, :]; PS = np.concatenate(aPS, 1)[:, None, :]   # (NC,1,N)
    FL = np.concatenate(aFL)[None, :]; FS = np.concatenate(aFS)[None, :]               # (1,N)
    meta = {"symbol": SYMF, "n": int(len(F)), "n_days": len(blobs),
            "step_s": STEP_S, "entry_ms": ENTRY_MS, "chase_ms": CHASE_MS, "holds_s": HOLDS_S,
            "queue_mults": [QM], "exit_qm": EXIT_QM, "H_ticks": H_TICKS, "window": W,
            "cfgs": [{"tp": 50.0, "sl": 50.0, "to_ms": h * 1000.0} for h in HOLDS_S],
            "maker_rt_fee_pct": 0.0, "time_based": not LEGACY, "legacy_ticks": LEGACY,
            "note": "honest time-based windows: entry 12.8s from decision, hold from FILL, chase 300s"}
    buf = io.BytesIO()
    np.savez_compressed(buf, F=F, rH30=np.concatenate(a30), rH15=np.concatenate(a15), rH60=np.concatenate(a60),
                        day=day, ts=ts, pnl_long=PL, pnl_short=PS, fill_long=FL, fill_short=FS,
                        feat_names=np.array(NAMES), meta=np.array(json.dumps(meta)))
    bk.blob(f"{OUT}/{SYMK}.npz").upload_from_string(buf.getvalue())
    log(f"[saved] gs://{BUCKET}/{OUT}/{SYMK}.npz N={len(F)} days={len(blobs)} ({buf.tell()/1e6:.0f}MB)")


def main():
    if COMBINE:
        combine(); return
    days = day_list()
    days = [d for i, d in enumerate(sorted(days)) if i % NSHARD == PARITY]
    done = {b.name.split("_")[-1][:-4] for b in cl.list_blobs(bk, prefix=f"{OUT}/daily/{SYMK}_") if b.name.endswith(".npz")}
    todo = [d for d in days if d not in done]
    log(f"[tb3s] shard {PARITY}/{NSHARD}: {len(days)} days, {len(todo)} to do | step {STEP_S}s "
        f"entry {ENTRY_MS}ms holds {HOLDS_S}s chase {CHASE_MS}ms H {H_TICKS}")
    log("[load BTC mids]"); bt, bm = load_btc_mid(); log(f"[BTC {len(bt)} ticks]")
    t0 = time.time()
    for i, day in enumerate(todo):
        try:
            st = process_day(day, bt, bm)
        except Exception as e:
            st = f"EXC {type(e).__name__}: {e}"
        log(f"  {day}: {st} [{(time.time()-t0)/60:.0f}m elapsed, {i+1}/{len(todo)}]")
    log("[DONE]")


if __name__ == "__main__":
    main()
