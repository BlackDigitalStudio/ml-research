#!/usr/bin/env python3
"""Stage-2a: grid_sim on the GRU cascade, per symbol, to find best objective/R:R
for the Stage-2 fine-tune + the live strategy.

Pipeline (reuse: mamba2_cascade for inference, grid_sim.rs for the sweep):
  1. load GRU A_{sym}.best.pt (vol-gate) + B_pool.best.pt (direction).
  2. inference on TEST windows (same purged split): A_logit (gate) + B_logit (side)
     + forward 1s mid-path (from feats_sub60 mid).
  3. gate top-q by A; side = sign(B). entry_long/short + mid_paths for gated windows.
  4. ~100k configs {tp,sl,to}; NO Kelly (unit=1), par:false, tr:false.
  5. CHUNK configs -> grid_sim (commission 0 = GROSS) -> aggregate directed gross-EV
     per config in Python (discard per-sample -> 100k configs fit).
  6. post-hoc net per config for 3 RT tiers: taker-taker 0.10 / maker-taker 0.07 /
     maker-maker 0.04. -> best {tp,sl,to}+EV+WR+trd/day per symbol per tier -> GCS.

Run on VM:  python3 subs60_gru_gridsim.py --symbols DOGE-USDT-PERP ETH-USDT-PERP LINK-USDT-PERP
"""
import argparse, io, json, os, shutil, subprocess, tempfile, sys, traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mamba2_cascade as mc

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
CACHE = "hd2_sub60_cache"; FEATS = "feats_sub60"; MODELS = "gru_models"; OUTP = "research_runs/gru_gridsim"
H = 60; THR = 13.0; NS = 1_000_000_000
STEP_MS = 100; NSTEP = 1200    # 100ms forward path, 120s (>= max_to 90s + 30s margin)
OFFS_NS = (np.arange(1, NSTEP + 1) * STEP_MS * 1_000_000).astype(np.int64)
GRID = "/tmp/rust_ingest/target/release/grid_sim"


def book_ts_mid(buf):
    """raw book parquet -> (ts ns sorted, mid abs). Reused from subs60_finepath."""
    import pyarrow.parquet as pq
    t = pq.read_table(io.BytesIO(buf), columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts = t["timestamp"].to_numpy().astype(np.int64)
    mid = 0.5*(t["bid_0_price"].to_numpy().astype(np.float64) + t["ask_0_price"].to_numpy().astype(np.float64))
    o = np.argsort(ts, kind="stable"); return ts[o], mid[o]
TOP3 = ["DOGE-USDT-PERP", "ETH-USDT-PERP", "LINK-USDT-PERP"]   # sym_id order used in training pool
TIERS = {"taker_taker": 0.10, "maker_taker": 0.07, "maker_maker": 0.04}   # RT % round-trip
bk = storage.Client(project=PROJ).bucket(BUCKET)
DEV = "cpu"


def load_gru(tag, tmp):
    """Download {tag}.best.pt from the Modal-built artifact on GCS-mirror? No — it's
    on the Modal Volume. We mirror it to GCS first (caller ensures). Load -> model."""
    p = os.path.join(tmp, f"{tag}.best.pt")
    bk.blob(f"{MODELS}/{tag}.best.pt").download_to_filename(p)
    st = torch.load(p, map_location=DEV)
    F = st["F"]; cfg = st["cfg"]; sd = st["model"]
    n_sym = sd["sym.weight"].shape[0] if "sym.weight" in sd else 0
    model = mc.Cascade2Stream(F, cfg["cell"], cfg["d1"], cfg["n1"], cfg["d2"], cfg["n2"],
                              n_sym=n_sym, dropout=cfg.get("dropout", 0.1)).to(DEV)
    model.load_state_dict(sd); model.eval()
    return (model, cfg,
            torch.tensor(st["lob_mu"]).to(DEV), torch.tensor(st["lob_sd"]).to(DEV),
            torch.tensor(st["ft_mu"]).to(DEV), torch.tensor(st["ft_sd"]).to(DEV))


def _periods(n, L):
    return [(s, min(s + L, n)) for s in range(0, n, L)]


@torch.no_grad()
def predict(model, cfg, lob_mu, lob_sd, ft_mu, ft_sd, lob, t0, feat, dp, sym_id):
    """Batched LOB forward -> logit per decision in dp."""
    L = cfg["L"]; per = _periods(len(lob), L); pidx = (t0[dp] // L)
    out = torch.empty(len(dp), device=DEV)
    groups = {}
    for di, k in enumerate(pidx.tolist()):
        groups.setdefault(k, []).append(di)
    items = [(k, per[k][0], per[k][1], np.asarray(v)) for k, v in groups.items()]
    full = [g for g in items if g[2] - g[1] == L]; tail = [g for g in items if g[2] - g[1] != L]
    sid = torch.full((len(dp),), sym_id, dtype=torch.long, device=DEV) if sym_id is not None else None

    def emit(batch):
        xs = np.stack([lob[a:b] for _, a, b, _ in batch]).astype(np.float32)
        x = (torch.from_numpy(xs).to(DEV) - lob_mu) / lob_sd
        h1all = model.encode_lob(x)
        for row, (k, a, b, sel) in enumerate(batch):
            pos = torch.from_numpy((t0[dp[sel]] - a)).to(DEV)
            h1 = h1all[row, pos]
            fb = (torch.from_numpy(feat[dp[sel]]).to(DEV) - ft_mu) / ft_sd
            h2 = model.encode_feat(fb[None])[0]
            out[sel] = model.head_logit(h1, h2, sid[sel] if sid is not None else None)
    KB = 32
    for i in range(0, len(full), KB):
        emit(full[i:i + KB])
    for g in tail:
        emit([g])
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=TOP3)
    ap.add_argument("--score-stride", type=int, default=5)   # score every ~5s, then gate
    ap.add_argument("--gate-pct", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    ap.add_argument("--n-configs", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--day-stride", type=int, default=2)   # subsample test days (raw-book dl)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reanalyze", action="store_true")    # load cached pass-1 (skip dl+inference)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="gg_", dir="/tmp")
    print(f"[load B_pool]"); modelB = load_gru("B_pool_gru", tmp)

    # ~100k bracket grid: tp x sl x to(100ms ticks)  (no Kelly/unit=1, par:false, tr:false)
    # timeout <=60s per user. + hold-to-timeout configs (tp=sl=50% never hit) as baseline.
    tps = np.round(np.linspace(0.03, 0.50, 160), 4)     # TP % 0.03..0.50
    sls = np.round(np.linspace(0.02, 0.30, 158), 4)     # SL % 0.02..0.30
    tos = [150, 300, 450, 600]                            # timeout in 100ms ticks = 15/30/45/60s
    bracket_cfgs = [{"tp": float(tp), "sl": float(sl), "to": int(to), "par": False, "tr": False}
                    for to in tos for tp in tps for sl in sls]
    hold_cfgs = [{"tp": 50.0, "sl": 50.0, "to": int(to), "par": False, "tr": False} for to in tos]
    cfgs = bracket_cfgs + hold_cfgs
    NB = len(bracket_cfgs); NH = len(hold_cfgs)
    TP = np.array([c["tp"] for c in bracket_cfgs]); SL = np.array([c["sl"] for c in bracket_cfgs])
    TO = np.array([c["to"] for c in bracket_cfgs]); RR = TP / SL
    gmax = max(a.gate_pct); CACHE_PCT = max(2.0, gmax)   # cache superset >= largest gate for re-slice
    print(f"[configs] {NB} bracket (tp x sl x to = {len(tps)}x{len(sls)}x{len(tos)}, <=60s) + {NH} hold")

    for sym in a.symbols:
        sid = TOP3.index(sym) if sym in TOP3 else 0
        symk = sym.split('-')[0]
        print(f"\n=== {sym} (sym_id={sid}) ===", flush=True)
        cblob = f"{OUTP}/cache/{symk}_cache.npz"   # reanalysis cache: superset paths + full conviction

        if a.reanalyze:
            print(f"  [reanalyze] loading {cblob}", flush=True)
            cz = np.load(io.BytesIO(bk.blob(cblob).download_as_bytes()))
            Asup = cz["Alog"]; Bsup = cz["Blog"]; ENTsup = cz["ENT"].astype(np.float64)
            MPsup = cz["mp"].astype(np.float64)
            n_scored = int(cz["n_scored"]); nted = int(cz["nted"]); cache_pct = float(cz["cache_pct"])
            print(f"  superset {len(Asup)} | n_scored={n_scored} nted={nted} cache_pct={cache_pct}", flush=True)
        else:
            modelA = load_gru(f"A_{symk}_gru", tmp)
            cblobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{CACHE}/{sym}/") if b.name.endswith(".npz"))
            if a.max_days:
                cblobs = cblobs[:a.max_days]
            nd = len(cblobs); te_days = cblobs[int(nd * 0.68):][::a.day_stride]   # purged test, strided
            # PASS 1: score all windows; cache book mids to DISK; keep light 1D arrays + refs.
            Alog, Blog, rH, ENT, REF = [], [], [], [], []
            bkdir = os.path.join(tmp, f"bk_{symk}"); os.makedirs(bkdir, exist_ok=True)

            def fetch(idx_name):
                di, name = idx_name
                day = name.split("/")[-1][:-4]
                pref = f"raw/book/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
                bb = next((b.name for b in bk.client.list_blobs(bk, prefix=pref) if b.name.endswith(".parquet")), None)
                cz = np.load(io.BytesIO(bk.blob(name).download_as_bytes()))
                book = bk.blob(bb).download_as_bytes() if bb else None
                return di, cz, book
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                def _stream(items, nw):                      # bounded prefetch: <=nw days in flight (OOM-safe)
                    for s in range(0, len(items), nw):
                        for res in ex.map(fetch, items[s:s + nw]):
                            yield res
                for di, d, bookbuf in _stream(list(enumerate(te_days)), a.workers):
                    if bookbuf is None:
                        continue
                    lob = d["lob"].astype(np.float32); t0 = d["t0"].astype(np.int64)
                    feat = d["feat"].astype(np.float32); v = d["v60"].astype(bool)
                    r = d["rH60"].astype(np.float32); dtd = d["dtd"].astype(np.int64); n = len(t0)
                    bts, bmid = book_ts_mid(bookbuf)
                    room = (dtd + int(NSTEP * STEP_MS) * 1_000_000) <= bts[-1]   # book extends past path
                    keep = np.zeros(n, bool); keep[::a.score_stride] = True
                    keep &= v & room
                    dp = np.where(keep)[0]
                    if len(dp) < 50:
                        continue
                    la = predict(*modelA, lob, t0, feat, dp, None)
                    lb = predict(*modelB, lob, t0, feat, dp, sid)
                    td_dp = dtd[dp]
                    ent = bmid[np.clip(np.searchsorted(bts, td_dp, "right") - 1, 0, len(bts) - 1)]
                    np.savez(f"{bkdir}/{di}.npz", bts=bts, bmid=bmid)          # mids -> disk (no re-dl)
                    Alog.append(la); Blog.append(lb); rH.append(r[dp]); ENT.append(ent)
                    REF.append(np.stack([np.full(len(dp), di, np.int64), td_dp], axis=1))
            if not Alog:
                print(f"  {sym}: no test data"); continue
            Alog = np.concatenate(Alog); Blog = np.concatenate(Blog); rH = np.concatenate(rH)
            ENT = np.concatenate(ENT).astype(np.float64); REF = np.concatenate(REF, 0)
            nted = len(te_days); n_scored = len(Alog)
            print(f"  scored {n_scored} windows ({nted} test days)", flush=True)

            def build_paths(idx):                       # 100ms paths for given idx (from disk mids)
                mp = np.empty((len(idx), NSTEP), np.float64); ref = REF[idx]
                for di in np.unique(ref[:, 0]):
                    m = ref[:, 0] == di
                    bkz = np.load(f"{bkdir}/{di}.npz"); bts = bkz["bts"]; bmid = bkz["bmid"]
                    grid = ref[m, 1][:, None] + OFFS_NS[None, :]
                    rows = np.clip(np.searchsorted(bts, grid.ravel(), "right") - 1, 0, len(bts) - 1)
                    mp[m] = bmid[rows].reshape(int(m.sum()), NSTEP)
                return mp

            # SUPERSET cache = top CACHE_PCT% by conviction; paths built once -> GCS (never re-dl).
            thrc = np.quantile(Alog, 1 - CACHE_PCT / 100.0)
            sup = np.where(Alog >= thrc)[0]
            if len(sup) > 60000:
                sup = sup[np.argsort(-Alog[sup])[:60000]]
            MPsup = build_paths(sup)
            Asup = Alog[sup]; Bsup = Blog[sup]; ENTsup = ENT[sup]; cache_pct = CACHE_PCT
            cbuf = io.BytesIO()
            np.savez_compressed(cbuf, Alog=Asup, Blog=Bsup, ENT=ENTsup.astype(np.float64),
                                mp=MPsup.astype(np.float32), n_scored=n_scored, nted=nted, cache_pct=cache_pct,
                                Alog_full=Alog.astype(np.float32), Blog_full=Blog.astype(np.float32),
                                rH_full=rH.astype(np.float32), REF=REF)   # FULL conviction kept for history
            bk.blob(cblob).upload_from_string(cbuf.getvalue())
            print(f"  [cache] {cblob} sup={len(sup)} paths={MPsup.nbytes/1e6:.0f}MB", flush=True)
            shutil.rmtree(bkdir, ignore_errors=True)

        # ===== analysis from superset (Asup,Bsup,ENTsup,MPsup); gates are sub-quantiles of it =====
        order = np.argsort(-Asup)         # highest conviction first
        results = {"symbol": sym, "n_scored": int(n_scored), "test_days": int(nted),
                   "cache_pct": float(cache_pct), "tos_s": [t // 10 for t in tos],
                   "tiers_rt_pct": TIERS, "by_gate": {}}
        for gpct in a.gate_pct:
          try:
            if gpct > cache_pct + 1e-9:
                print(f"  [g{gpct}] SKIP (> cache_pct {cache_pct})", flush=True); continue
            k = int(round(len(Asup) * (gpct / cache_pct)))
            if k < 100:
                continue
            sel = order[:k]
            ent_g = ENTsup[sel]; long_mask = (Bsup[sel] > 0); mp = MPsup[sel]; nsel = len(sel)
            n_orig = int(round(n_scored * gpct / 100.0)); tpd = n_orig / max(nted, 1)
            np.save(f"{tmp}/el.npy", ent_g); np.save(f"{tmp}/es.npy", ent_g); np.save(f"{tmp}/mp.npy", mp)
            ce = int(max(200, min(a.chunk, 3_000_000_000 // (nsel * 16))))   # bound grid_sim output ~3GB
            print(f"  [g{gpct}] n_orig={n_orig} sim_n={nsel} ~{tpd:.0f}/d mp={mp.shape} chunk={ce}", flush=True)
            gross = np.empty(len(cfgs)); wr = np.empty(len(cfgs)); ok = True
            for ci in range(0, len(cfgs), ce):
                cc = cfgs[ci:ci + ce]
                json.dump(cc, open(f"{tmp}/cfg.json", "w"))
                r = subprocess.run([GRID, "--entry-long", f"{tmp}/el.npy", "--entry-short", f"{tmp}/es.npy",
                                    "--mid-paths", f"{tmp}/mp.npy", "--configs", f"{tmp}/cfg.json",
                                    "--out-prefix", f"{tmp}/g", "--commission-win-pct", "0",
                                    "--commission-loss-pct", "0", "--fill-latency-ms", "150"],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  GRID FAIL rc={r.returncode}: {r.stderr[-300:]}", flush=True); ok = False; break
                pl = np.load(f"{tmp}/g_pnl_long.npy"); ps = np.load(f"{tmp}/g_pnl_short.npy")
                directed = np.where(long_mask[None, :], pl, ps)      # (chunk, nsel) gross %
                gross[ci:ci + len(cc)] = directed.mean(1); wr[ci:ci + len(cc)] = (directed > 0).mean(1)
            if not ok:
                continue
            gb = gross[:NB]; wb = wr[:NB]; hg = gross[NB:]; hw = wr[NB:]   # bracket | hold slices
            # save FULL bracket surface (every config) + hold -> GCS for offline R:R slicing
            sbuf = io.BytesIO()
            np.savez_compressed(sbuf, tp=TP, sl=SL, to=TO, rr=RR, gross=gb, wr=wb,
                                hold_to=np.array(tos), hold_gross=hg, hold_wr=hw)
            bk.blob(f"{OUTP}/surface/{symk}_top{gpct}.npz").upload_from_string(sbuf.getvalue())
            # hold-to-timeout baseline (NO TP/SL)
            hold = {}
            for j, to in enumerate(tos):
                g = float(hg[j]); w = float(hw[j])
                hold[to // 10] = {"gross_bp": g * 100, "wr": w,
                                  "net_bp": {t: (g - rt) * 100 for t, rt in TIERS.items()}}
            # bracket argmax per tier
            tiers_out = {}
            for tname, rt in TIERS.items():
                net = gb - rt; bi = int(np.argmax(net))
                tiers_out[tname] = {"tp": float(TP[bi]), "sl": float(SL[bi]), "to_s": int(TO[bi]) // 10,
                                    "rr": round(float(RR[bi]), 2), "net_bp": float(net[bi] * 100),
                                    "gross_bp": float(gb[bi] * 100), "wr": float(wb[bi])}
            # R:R surface @maker_maker: best config per R:R bucket
            net_mm = gb - TIERS["maker_maker"]; rr_curve = []
            for rb in [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 18]:
                m = (RR >= rb / 1.18) & (RR < rb * 1.18)
                if not m.any():
                    continue
                idx = np.where(m)[0]; bj = int(idx[np.argmax(net_mm[idx])])
                rr_curve.append({"rr": rb, "tp": float(TP[bj]), "sl": float(SL[bj]), "to_s": int(TO[bj]) // 10,
                                 "gross_bp": float(gb[bj] * 100), "net_mm_bp": float(net_mm[bj] * 100),
                                 "wr": float(wb[bj])})
            results["by_gate"][f"top{gpct}%"] = {"n_trades": int(n_orig), "trd_per_day": float(tpd),
                                                 "sim_n": int(nsel), "hold": hold,
                                                 "bracket_best": tiers_out, "rr_curve": rr_curve}
            bb = tiers_out["maker_maker"]; h60 = hold[60]
            print(f"  g{gpct}% n={n_orig} ~{tpd:.0f}/d | HOLD60s gross={h60['gross_bp']:+.2f}bp "
                  f"net_mm={h60['net_bp']['maker_maker']:+.2f} WR={h60['wr']:.2f} || BRACKET_mm "
                  f"RR{bb['rr']} tp{bb['tp']}/sl{bb['sl']}/{bb['to_s']}s net={bb['net_bp']:+.2f} "
                  f"gross={bb['gross_bp']:+.2f}", flush=True)
          except Exception:
            print(f"  [g{gpct}] EXC {traceback.format_exc()[-700:]}", flush=True)
        bk.blob(f"{OUTP}/{symk}.json").upload_from_string(json.dumps(results, default=float))
        print(f"  [saved] {OUTP}/{symk}.json", flush=True)


if __name__ == "__main__":
    main()
