#!/usr/bin/env python3
"""Maker-fill grid_sim on the GRU/B2 ETH signal (HUSDC tooling). For each test day:
build_samples(raw L2+trades) -> maker arrays; GRU inference (A=gate, B2=side) on the
sub60 cache; match build_samples sample_ts(ms) -> nearest cache dtd(ns); keep gated
(top-q by A) samples' maker arrays; then grid_sim MAKER mode -> realistic resting-limit
fill (touch/queue/MISS, adverse from path) -> EV/trade over FILLED by predicted side.

Reuses subs60_gru_gridsim.load_gru/predict. Maker entry + taker exit (maker-maker
needs a resting exit too -> later). Run on VM.
"""
import argparse, io, json, os, shutil, subprocess, sys, tempfile, traceback
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mamba2_cascade as mc
from subs60_gru_gridsim import load_gru, predict, bk, CACHE, TOP3, NS

BS = "/tmp/husdc/rust_ingest/target/release/build_samples"
GRID = "/tmp/husdc/rust_ingest/target/release/grid_sim"      # husdc maker-mode grid_sim
OUTP = "research_runs/gru_makergrid"
H = 700; STEP = 9; MAXS = 120000                              # ~1s sampling, fwd 700 ticks (~74s)
TO_TICKS = 563                                               # 60s at ~106ms/tick
RAWB = "raw/book/exchange=BINANCE_FUTURES"; RAWT = "raw/trades/exchange=BINANCE_FUTURES"
DEV = "cpu"


def load_local(path):
    """Load a fine-tuned .best.pt from a local path -> (model,cfg,mu tensors...)."""
    st = torch.load(path, map_location=DEV)
    F = st["F"]; cfg = st["cfg"]; sd = st["model"]
    n_sym = sd["sym.weight"].shape[0] if "sym.weight" in sd else 0
    m = mc.Cascade2Stream(F, cfg["cell"], cfg["d1"], cfg["n1"], cfg["d2"], cfg["n2"],
                          n_sym=n_sym, dropout=cfg.get("dropout", 0.1)).to(DEV)
    m.load_state_dict(sd); m.eval()
    return (m, cfg, torch.tensor(st["lob_mu"]), torch.tensor(st["lob_sd"]),
            torch.tensor(st["ft_mu"]), torch.tensor(st["ft_sd"]))


def dl(blob_prefix, dst):
    name = next((b.name for b in bk.client.list_blobs(bk, prefix=blob_prefix) if b.name.endswith(".parquet")), None)
    if not name:
        return False
    bk.blob(name).download_to_filename(dst); return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETH-USDT-PERP")
    ap.add_argument("--bgate", default="/tmp/B2_ETH_hold.best.pt")   # fine-tuned side model (local)
    ap.add_argument("--gate-superset-pct", type=float, default=1.0)  # keep top-1% by A; sub-gate later
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--match-tol-ms", type=float, default=1500.0)
    ap.add_argument("--no-persist", action="store_true")   # skip uploading gated arrays
    a = ap.parse_args()
    sym = a.symbol; symk = sym.split("-")[0]; sid = TOP3.index(sym) if sym in TOP3 else 0
    tmp = tempfile.mkdtemp(prefix="mk_", dir="/tmp")
    modelA = load_gru(f"A_{symk}_gru", tmp)
    modelB = load_local(a.bgate)
    print(f"[models] A=A_{symk}_gru  B=fine-tuned {a.bgate}", flush=True)

    cblobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{CACHE}/{sym}/") if b.name.endswith(".npz"))
    nd = len(cblobs); te = cblobs[int(nd * 0.68):]
    if a.max_days:
        te = te[:a.max_days]
    # global A threshold from this run's test-day A-logits (pass 1 collects A only)
    print(f"[pass1] scoring A over {len(te)} test days for gate threshold", flush=True)
    Aall = []
    perday = []                                                # (dayblob, dtd, Alog, Blog)
    for nm in te:
        d = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
        lob = d["lob"].astype(np.float32); t0 = d["t0"].astype(np.int64)
        feat = d["feat"].astype(np.float32); v = d["v60"].astype(bool); dtd = d["dtd"].astype(np.int64)
        dp = np.where(v)[0]
        if len(dp) < 50:
            perday.append(None); continue
        la = predict(*modelA, lob, t0, feat, dp, None)
        lb = predict(*modelB, lob, t0, feat, dp, sid)
        Aall.append(la); perday.append((nm, dtd[dp], la, lb))
    Aall = np.concatenate(Aall)
    thr = np.quantile(Aall, 1 - a.gate_superset_pct / 100.0)
    print(f"[gate] top-{a.gate_superset_pct}% A-threshold={thr:.3f} over {len(Aall)} windows", flush=True)

    # pass 2: per day build_samples, match gated dtd, collect maker arrays
    keys = ["entry_long", "entry_short", "mid_paths", "book_paths", "flow_paths", "entry_q", "entry_book"]
    acc = {k: [] for k in keys}; SIDE = []; ALOG = []; BLOG = []; nten = 0
    for rec in perday:
        if rec is None:
            continue
        nm, dtd, la, lb = rec
        day = nm.split("/")[-1][:-4]; nten += 1
        gmask = la >= thr
        if gmask.sum() < 1:
            continue
        od = os.path.join(tmp, "bs"); os.makedirs(od, exist_ok=True)
        for f in os.listdir(od):
            os.remove(os.path.join(od, f))
        if not (dl(f"{RAWB}/symbol={sym}/dt={day}/", f"{tmp}/b.parquet")
                and dl(f"{RAWT}/symbol={sym}/dt={day}/", f"{tmp}/t.parquet")):
            print(f"  {day}: no raw", flush=True); continue
        r = subprocess.run([BS, "--depth", f"{tmp}/b.parquet", "--trades", f"{tmp}/t.parquet",
                            "--out-dir", od, "--window", "50", "--horizon", str(H),
                            "--step", str(STEP), "--max-samples", str(MAXS)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {day} BS FAIL: {r.stderr[-200:]}", flush=True); continue
        for u in ("X_lob", "top5_bid", "top5_ask", "sample_starts", "end_indices", "mid"):
            try:
                os.remove(f"{od}/{u}.npy")          # drop unused outputs (X_lob=118MB/day) -> disk-safe
            except OSError:
                pass
        sts = np.load(f"{od}/sample_ts.npy").astype(np.int64) * 1_000_000   # ms -> ns
        # match each gated dtd to nearest sample within tol
        gi = np.where(gmask)[0]; gdtd = dtd[gi]
        pos = np.clip(np.searchsorted(sts, gdtd), 0, len(sts) - 1)
        sel_rows, sel_lr = [], []
        for lr, gd, p in zip(gi, gdtd, pos):
            cand = [c for c in (p - 1, p) if 0 <= c < len(sts)]
            best = min(cand, key=lambda c: abs(sts[c] - gd))
            if abs(sts[best] - gd) <= a.match_tol_ms * 1_000_000:
                sel_rows.append(best); sel_lr.append(lr)
        if sel_rows:
            sr = np.array(sel_rows)
            for k in keys:
                acc[k].append(np.load(f"{od}/{k}.npy")[sr])      # full load (fd closes) -> gather matched
            for lr in sel_lr:
                SIDE.append(1 if lb[lr] > 0 else 0); ALOG.append(float(la[lr])); BLOG.append(float(lb[lr]))
        for f in os.listdir(od):                                 # free this day's arrays NOW (disk-safe)
            os.remove(os.path.join(od, f))
        for pf in (f"{tmp}/b.parquet", f"{tmp}/t.parquet"):
            try:
                os.remove(pf)
            except OSError:
                pass
        print(f"  {day}: gated={int(gmask.sum())} matched={len(SIDE)} (cum)", flush=True)

    n = len(SIDE)
    print(f"[collected] {n} gated+matched maker samples over {nten} test days", flush=True)
    if n < 50:
        print("too few; abort"); return
    ST = {k: np.concatenate(acc[k], 0) for k in keys}      # per-day matched batches -> full set
    for k in keys:
        np.save(f"{tmp}/{k}.npy", ST[k])
    SIDE = np.array(SIDE); ALOG = np.array(ALOG); BLOG = np.array(BLOG)
    np.save(f"{tmp}/side.npy", SIDE)
    if not a.no_persist:                                   # persist gated arrays -> offline entry-POLICY sweep
        buf = io.BytesIO()
        np.savez_compressed(buf, side=SIDE, alog=ALOG, blog=BLOG, **ST)
        bk.blob(f"{OUTP}/{symk}_gated_arrays.npz").upload_from_string(buf.getvalue())
        print(f"[saved arrays] {OUTP}/{symk}_gated_arrays.npz ({buf.tell()/1e6:.0f}MB) for policy sweep", flush=True)

    # configs: ETH optimum = HOLD (wide TP/SL, exit at 60s) + a few R:R for context
    cfgs = [{"tp": 50.0, "sl": 50.0, "to": TO_TICKS, "par": False, "tr": False},     # hold-60s
            {"tp": 0.30, "sl": 0.05, "to": TO_TICKS, "par": False, "tr": False},     # RR6
            {"tp": 0.20, "sl": 0.10, "to": TO_TICKS, "par": False, "tr": False}]     # RR2
    json.dump(cfgs, open(f"{tmp}/cfg.json", "w"))

    results = {"symbol": sym, "n_samples": int(n), "test_days": int(nten),
               "gate_superset_pct": a.gate_superset_pct, "by_queue": {}}
    for qm in (0.0, 1.0, 2.0):                                   # touch -> queue sweep
        g = f"{tmp}/g{qm}"
        cmd = [GRID, "--entry-long", f"{tmp}/entry_long.npy", "--entry-short", f"{tmp}/entry_short.npy",
               "--mid-paths", f"{tmp}/mid_paths.npy", "--book-paths", f"{tmp}/book_paths.npy",
               "--entry-book", f"{tmp}/entry_book.npy", "--flow-paths", f"{tmp}/flow_paths.npy",
               "--entry-q", f"{tmp}/entry_q.npy", "--configs", f"{tmp}/cfg.json", "--out-prefix", g,
               "--queue-mult", str(qm), "--entry-window-ticks", "120", "--maker-offset-frac", "0",
               "--commission-win-pct", "0", "--commission-loss-pct", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  qm{qm} GRID FAIL: {r.stderr[-300:]}", flush=True); continue
        pl = np.load(f"{g}_pnl_long.npy"); ps = np.load(f"{g}_pnl_short.npy")          # (nc, n) %, NaN=miss
        fl = np.load(f"{g}_filled_long.npy"); fs = np.load(f"{g}_filled_short.npy")    # (n,) u8
        longm = SIDE == 1
        dir_pnl = np.where(longm[None, :], pl, ps)               # (nc, n) by predicted side
        dir_fill = np.where(longm, fl, fs).astype(bool)          # (n,)
        qres = {"fill_rate": float(dir_fill.mean()), "cfgs": []}
        for ci, c in enumerate(cfgs):
            row = dir_pnl[ci]; m = dir_fill & np.isfinite(row)
            ev = float(row[m].mean() * 100) if m.any() else float("nan")              # gross bp over filled
            wr = float((row[m] > 0).mean()) if m.any() else float("nan")
            qres["cfgs"].append({"tp": c["tp"], "sl": c["sl"], "to_s": c["to"] // 10 if c["to"] else 0,
                                 "rr": round(c["tp"] / c["sl"], 2), "gross_bp": ev, "wr": wr,
                                 "n_filled": int(m.sum())})
        results["by_queue"][f"qm{qm}"] = qres
        h = qres["cfgs"][0]
        print(f"  qm{qm}: fill={qres['fill_rate']:.2f} | HOLD60s gross={h['gross_bp']:+.2f}bp "
              f"WR={h['wr']:.2f} n_filled={h['n_filled']}", flush=True)
    bk.blob(f"{OUTP}/{symk}_maker.json").upload_from_string(json.dumps(results, default=float))
    print(f"[saved] {OUTP}/{symk}_maker.json", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)             # never leak the tmpdir


if __name__ == "__main__":
    main()
