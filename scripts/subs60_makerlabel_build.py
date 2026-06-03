#!/usr/bin/env python3
"""Build the MAKER-REALISTIC (adverse-selection) label dataset for XGBoost A/B.

For every feats_sub60 decision point (at --feat-stride) we attach a REALISTIC maker
P&L label: a resting limit order that fills ONLY when realized taker flow reaches our
level (touch/queue) and MISSES when price runs away (the favorable runaways) -> adverse
selection emerges from the path, not a parameter. (HUSDC tooling, MAKER_SIM.md.)

Per (symbol, day):
  1. download raw book+trades parquet
  2. build_samples (native flat raw L2) -> maker arrays (book_paths, flow_paths, entry_q,
     entry_book, mid_paths, entry_long/short, sample_ts) at a dense tick grid (--step ticks)
  3. grid_sim MAKER mode (configs x queue-mults) -> pnl_long/short (NaN on MISS) + filled masks
  4. load feats_sub60 day npz; for each valid decision point at --feat-stride, match its
     ts(ns) to the nearest build_sample within --match-tol-ms; gather that sample's maker
     pnl/fill, and the EXACT engineered feat71 (X64 + signed BTC-lead{5,30,60}s + ToD4).
Saves a compact per-symbol npz -> gs://.../research_runs/maker_labels/{SYM}.npz with:
  F(N,71) f32 | rH60(N) f32 | day(N) i32 | ts(N) i64(ns) |
  pnl_long/pnl_short (NC,QM,N) f32 NaN=miss | fill_long/fill_short (QM,N) u8 |
  cfgs(json), queue_mults, feat_names, maker_rt_fee_pct=0.04
Labels are GROSS bp (commission 0 in sim); apply maker RT fee downstream.

Run on VM (8 vCPU). Probe:  python3 subs60_makerlabel_build.py --symbols SOL-USDT-PERP --max-days 2 --probe
"""
import argparse, io, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
FEATS = "feats_sub60"; OUT = "research_runs/maker_labels"
RAWB = "raw/book/exchange=BINANCE_FUTURES"; RAWT = "raw/trades/exchange=BINANCE_FUTURES"
BS = "/tmp/husdc/rust_ingest/target/release/build_samples"
GRID = "/tmp/husdc/rust_ingest/target/release/grid_sim"
NS = 1_000_000_000
H_TICKS = 700          # forward ticks built (~74s at ~106ms/tick) -> room past 60s
TO_TICKS = 563         # 60s timeout at ~106ms/tick
ENTRY_WIN = 120        # max forward ticks to wait for a maker fill before MISS (~12s)
WINDOW = 50            # build_samples lookback window for X_lob (unused here but required)
MAKER_RT_FEE_PCT = 0.04  # standard Binance maker round-trip (0.02%/side); NO VIP tier exists
# label configs: hold-60s (pure timeout, maker analog of rH60) + a FINE per-symbol R:R discovery grid
# (SL x TP, RR in [0.9,13] -> no inverted/garbage configs; grid_sim sweeps all of them cheaply per day).
_SLS = [0.05, 0.08, 0.10, 0.13]
_TPS = [0.10, 0.13, 0.16, 0.20, 0.26, 0.34, 0.45, 0.60]
CFGS = ([{"tp": 50.0, "sl": 50.0, "to": TO_TICKS, "par": False, "tr": False}] +          # hold-60s
        [{"tp": tp, "sl": sl, "to": TO_TICKS, "par": False, "tr": False}
         for sl in _SLS for tp in _TPS if 0.9 <= tp / sl <= 13.0])                        # R:R grid (32 cfgs)
bk = storage.Client(project=PROJ).bucket(BUCKET)

NAMES = ([f"x{c}" for c in range(64)] +
         ["btc_ret5", "btc_ret30", "btc_ret60", "sin_h", "cos_h", "sin_f8", "cos_f8"])


def log(s): print(s, flush=True)


def load_btc_mid(workers=8, max_days=None):
    blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{FEATS}/BTC-USDT-PERP/")
                   if b.name.endswith(".npz"))
    if max_days and len(blobs) > max_days:
        blobs = blobs[::max(1, len(blobs)//max_days)][:max_days]
    def fetch(n): return np.load(io.BytesIO(bk.blob(n).download_as_bytes()))
    tds, mids = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(fetch, blobs):
            tds.append(d["td"].astype(np.int64)); mids.append(d["mid"].astype(np.float64))
    td = np.concatenate(tds); mid = np.concatenate(mids); o = np.argsort(td, kind="stable")
    return td[o], mid[o]


def feat71(dtd, X, bt, bm):
    nb = len(bt); i = np.clip(np.searchsorted(bt, dtd, "right")-1, 0, nb-1)
    bl = []
    for W in (5, 30, 60):
        j = np.clip(np.searchsorted(bt, dtd-int(W*NS), "right")-1, 0, nb-1)
        a = bm[j]; b = bm[i]
        bl.append(np.where((a > 0) & (b > 0), np.log(np.where(a > 0, b/np.where(a > 0, a, 1.0), 1.0)), 0.0)*1e4)
    h = ((dtd/NS) % 86400.0)/3600.0; hf = h % 8.0
    tod = [np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), np.sin(2*np.pi*hf/8), np.cos(2*np.pi*hf/8)]
    return np.concatenate([X, np.stack(bl, 1), np.stack(tod, 1)], axis=1).astype(np.float32)


def dl_raw(prefix, dst):
    name = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not name:
        return False
    bk.blob(name).download_to_filename(dst); return True


def build_day(tmp, sym, day, target_samples, maxs):
    od = os.path.join(tmp, "bs"); os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    bp, tp = f"{tmp}/b.parquet", f"{tmp}/t.parquet"
    if not (dl_raw(f"{RAWB}/symbol={sym}/dt={day}/", bp) and dl_raw(f"{RAWT}/symbol={sym}/dt={day}/", tp)):
        return None, "no-raw"
    # adaptive tick-stride from book snapshot count -> target ~uniform build samples/day
    import pyarrow.parquet as pq
    nrows = pq.ParquetFile(bp).metadata.num_rows
    step = max(1, -(-nrows // target_samples))   # ceil(nrows/target); sparse days -> step=1 (densest)
    r = subprocess.run([BS, "--depth", bp, "--trades", tp, "--out-dir", od,
                        "--window", str(WINDOW), "--horizon", str(H_TICKS),
                        "--step", str(step), "--max-samples", str(maxs)],
                       capture_output=True, text=True)
    for pf in (bp, tp):
        try: os.remove(pf)
        except OSError: pass
    if r.returncode != 0:
        return None, f"BS-fail:{r.stderr[-160:]}"
    # drop heavy unused outputs to stay disk-safe
    for u in ("X_lob", "top5_bid", "top5_ask", "sample_starts", "end_indices", "mid"):
        try: os.remove(f"{od}/{u}.npy")
        except OSError: pass
    return od, "ok"


def grid_maker(tmp, od, qms):
    """Run grid_sim maker for each queue-mult. Returns pnl_long/short (NC,QM,N), fill (QM,N)."""
    json.dump(CFGS, open(f"{tmp}/cfg.json", "w"))
    pls, pss, fls, fss = [], [], [], []
    for qm in qms:
        g = f"{tmp}/g{qm}"
        cmd = [GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
               "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
               "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
               "--entry-q", f"{od}/entry_q.npy", "--configs", f"{tmp}/cfg.json", "--out-prefix", g,
               "--queue-mult", str(qm), "--entry-window-ticks", str(ENTRY_WIN), "--maker-offset-frac", "0",
               "--commission-win-pct", "0", "--commission-loss-pct", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None, f"GRID-fail qm{qm}:{r.stderr[-200:]}"
        pls.append(np.load(f"{g}_pnl_long.npy")); pss.append(np.load(f"{g}_pnl_short.npy"))
        fls.append(np.load(f"{g}_filled_long.npy")); fss.append(np.load(f"{g}_filled_short.npy"))
        for suf in ("pnl_long", "pnl_short", "filled_long", "filled_short"):
            try: os.remove(f"{g}_{suf}.npy")
            except OSError: pass
    # (NC,N) per qm -> stack to (NC,QM,N) ; fill (N,) per qm -> (QM,N)
    PL = np.stack(pls, 1).astype(np.float32); PS = np.stack(pss, 1).astype(np.float32)
    FL = np.stack(fls, 0).astype(np.uint8); FS = np.stack(fss, 0).astype(np.uint8)
    return (PL, PS, FL, FS), "ok"


def process_symbol(sym, bt, bm, a):
    symk = sym.split("-")[0]
    blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{FEATS}/{sym}/") if b.name.endswith(".npz"))
    if a.start:
        blobs = [b for b in blobs if b.split("/")[-1][:-4] >= a.start]
    if a.end:
        blobs = [b for b in blobs if b.split("/")[-1][:-4] <= a.end]
    if a.max_days and len(blobs) > a.max_days:
        blobs = blobs[:a.max_days] if a.probe else blobs[::max(1, len(blobs)//a.max_days)][:a.max_days]
    log(f"=== {sym}: {len(blobs)} days, target_samp={a.target_samples} feat_stride={a.feat_stride} qms={a.queue_mults} ===")
    scratch = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"   # RAM-backed scratch -> no disk-IO thrash under parallelism
    tmp = tempfile.mkdtemp(prefix="ml_", dir=scratch)
    accF, accR, accDay, accTs = [], [], [], []
    accPL, accPS, accFL, accFS = [], [], [], []
    nmatch_tot = 0; nfeat_tot = 0; nday = 0; t_dl = t_bs = t_gs = 0.0
    try:
        for di, nm in enumerate(blobs):
            day = nm.split("/")[-1].replace(".npz", "")
            # feats day
            d = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
            td = d["td"].astype(np.int64); n = len(td)
            if n < 200:
                continue
            sel = np.arange(0, n, a.feat_stride)
            v = d[f"valid_60"].astype(bool)[sel]
            sel = sel[v]
            if len(sel) < 20:
                continue
            ftd = td[sel]; X = d["X"].astype(np.float32)[sel]; rH = d["rH_60"].astype(np.float32)[sel]
            F = feat71(ftd, X, bt, bm)
            nfeat_tot += len(sel)
            # build_samples
            tA = time.time(); od, st = build_day(tmp, sym, day, a.target_samples, a.max_samples)
            if od is None:
                log(f"  {day}: {st}"); continue
            t_bs += time.time() - tA
            sts = np.load(f"{od}/sample_ts.npy").astype(np.int64) * 1_000_000   # ms -> ns
            if len(sts) < 5:
                log(f"  {day}: too few build samples ({len(sts)})"); continue
            # grid_sim maker
            tA = time.time(); G, st = grid_maker(tmp, od, a.queue_mults)
            if G is None:
                log(f"  {day}: {st}"); continue
            t_gs += time.time() - tA
            PL, PS, FL, FS = G  # (NC,QM,Ns),(NC,QM,Ns),(QM,Ns),(QM,Ns)
            # match each feats point to nearest build_sample within tol
            pos = np.clip(np.searchsorted(sts, ftd), 0, len(sts)-1)
            sel_feat, sel_bs = [], []
            for fi, (gd, p) in enumerate(zip(ftd, pos)):
                cand = [c for c in (p-1, p) if 0 <= c < len(sts)]
                best = min(cand, key=lambda c: abs(sts[c]-gd))
                if abs(sts[best]-gd) <= a.match_tol_ms * 1_000_000:
                    sel_feat.append(fi); sel_bs.append(best)
            if not sel_feat:
                log(f"  {day}: 0 matched (build n={len(sts)})"); continue
            sf = np.array(sel_feat); sbs = np.array(sel_bs)
            accF.append(F[sf]); accR.append(rH[sf]); accDay.append(np.full(len(sf), di, np.int32)); accTs.append(ftd[sf])
            accPL.append(PL[:, :, sbs]); accPS.append(PS[:, :, sbs])
            accFL.append(FL[:, sbs]); accFS.append(FS[:, sbs])
            nmatch_tot += len(sf); nday += 1
            if a.probe and nday == 1:   # label sanity on hold-60s, touch (cfg0,qm0)
                pl0 = PL[0, 0, sbs]; ps0 = PS[0, 0, sbs]; fl0 = FL[0, sbs].astype(bool)
                nf = (np.abs(rH[sf]) >= 13.0)
                log(f"    [label sanity hold60s/touch] pl_long: filled={fl0.mean():.2f} "
                    f"miss(NaN)={np.isnan(pl0).mean():.2f} | filled pl_long mean={np.nanmean(pl0)*100:+.2f}bp "
                    f"sd={np.nanstd(pl0)*100:.1f}bp | nonflat(|rH60|>=13bp)={nf.mean():.3f} "
                    f"| on nonflat: pl_long={np.nanmean(pl0[nf])*100:+.2f} pl_short={np.nanmean(ps0[nf])*100:+.2f}bp")
            for f in os.listdir(od):
                os.remove(os.path.join(od, f))
            if a.probe or (di % 25 == 0):
                fillr = float(FL[0].mean())
                log(f"  {day}: build_n={len(sts)} feat_n={len(sel)} matched={len(sf)} "
                    f"({100*len(sf)/max(len(sel),1):.0f}%) fill_long_touch={fillr:.2f} "
                    f"[t_bs={t_bs:.0f}s t_gs={t_gs:.0f}s]")
        if nmatch_tot < 20:
            log(f"  {sym}: too few matched ({nmatch_tot}); skip save"); return None
        F = np.concatenate(accF); rH = np.concatenate(accR); day = np.concatenate(accDay); ts = np.concatenate(accTs)
        PL = np.concatenate(accPL, 2); PS = np.concatenate(accPS, 2)
        FL = np.concatenate(accFL, 1); FS = np.concatenate(accFS, 1)
        meta = {"symbol": sym, "n": int(len(F)), "n_days": int(nday), "feat_stride": a.feat_stride,
                "target_samples": a.target_samples, "queue_mults": list(a.queue_mults), "cfgs": CFGS,
                "H_ticks": H_TICKS, "to_ticks": TO_TICKS, "entry_win_ticks": ENTRY_WIN,
                "maker_rt_fee_pct": MAKER_RT_FEE_PCT, "match_tol_ms": a.match_tol_ms,
                "feat_match_rate": float(nmatch_tot/max(nfeat_tot, 1)),
                "t_bs_s": round(t_bs, 1), "t_gs_s": round(t_gs, 1)}
        log(f"[{sym}] N={len(F)} days={nday} match_rate={meta['feat_match_rate']:.2f} "
            f"PL{PL.shape} | t_bs={t_bs:.0f}s t_gs={t_gs:.0f}s")
        if not a.probe:
            buf = io.BytesIO()
            np.savez_compressed(buf, F=F, rH60=rH, day=day, ts=ts, pnl_long=PL, pnl_short=PS,
                                fill_long=FL, fill_short=FS, feat_names=np.array(NAMES),
                                meta=np.array(json.dumps(meta)))
            bk.blob(f"{OUT}/{symk}.npz").upload_from_string(buf.getvalue())
            log(f"[saved] gs://{BUCKET}/{OUT}/{symk}.npz ({buf.tell()/1e6:.1f}MB)")
        return meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--start", default=None)   # YYYY-MM-DD inclusive day filter on feats_sub60 blobs
    ap.add_argument("--end", default=None)
    ap.add_argument("--target-samples", type=int, default=40000)  # adaptive tick-stride -> ~this many build samples/day
    ap.add_argument("--feat-stride", type=int, default=8)   # feats decision-point stride (matches xgb_optuna)
    ap.add_argument("--max-samples", type=int, default=2000000)  # high -> our adaptive step wins (no auto-override)
    ap.add_argument("--queue-mults", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--match-tol-ms", type=float, default=2500.0)
    ap.add_argument("--out-sub", default="maker_labels")    # output subdir (use a distinct one to not clobber)
    ap.add_argument("--probe", action="store_true")         # don't upload; verbose per-day
    a = ap.parse_args()
    global OUT; OUT = f"research_runs/{a.out_sub}"
    t0 = time.time()
    log(f"[out={OUT}] [cfgs={len(CFGS)}] [queue_mults={a.queue_mults}]")
    log(f"[load BTC mid for btc-lead]"); bt, bm = load_btc_mid(8, a.max_days if a.probe else None)
    log(f"[BTC: {len(bt)} ticks]")
    metas = []
    for sym in a.symbols:
        m = process_symbol(sym, bt, bm, a)
        if m: metas.append(m)
    log(f"\n[DONE] {len(metas)} symbols in {time.time()-t0:.0f}s")
    for m in metas:
        log(f"  {m['symbol']}: N={m['n']} days={m['n_days']} match={m['feat_match_rate']:.2f} "
            f"t_bs={m['t_bs_s']}s t_gs={m['t_gs_s']}s")


if __name__ == "__main__":
    main()
