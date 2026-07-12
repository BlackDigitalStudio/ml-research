#!/usr/bin/env python3
"""HD4 rev1 amendment: extreme-tail selectivity cut of the dir10 screen — top-K windows
per day by |signal| (K=5/10 mirrors the deployed t5/t10 day-budget selectivity ~0.02%,
K=28/280 = ~0.1%/1% for the curve). Same data, same metrics (sign(x)*fwd_ret), no ML.
Reads: maker_labels_tb3s_h150 dailies (F71) + h2_dir10 dailies (R, RV). No raw book.
Signals: f62 (OBI_L1/microprice), f1 (imb_L5), f32 (top3_asym), COMP (prereg cols).
Env: SYMS (BTC,ETH,DOGE,XRP). Output: print + research_runs/h2_dir10/{SYM}_tail.npz
"""
import io, os, time
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = os.environ.get("SYMS", "BTC,ETH,DOGE,XRP").split(",")
LABSUB = "research_runs/maker_labels_tb3s_h150"
D10 = "research_runs/h2_dir10"
KS = [5, 10, 28, 280]
COMP_COLS = [0, 1, 12, 26, 27, 28, 62, 63, 64, 65, 66]
SIGS = {"f62_OBI_L1": 62, "f1_imb_L5": 1, "f32_top3asym": 32, "COMP": -1}
HORS_USE = {"5s": 0, "10s": 1, "15s": 2}

cl = storage.Client(project=PROJ); bk = cl.bucket(BUCKET)


def rank01(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x), np.float64)
    xs = x[o]; b = np.r_[True, xs[1:] != xs[:-1]]; g = np.cumsum(b) - 1
    cnt = np.bincount(g); csum = np.bincount(g, weights=np.arange(len(x)))
    r[o] = (csum / cnt)[g]
    return (r + 0.5) / len(x)


for sym in SYMS:
    days = sorted(b.name.split("/")[-1][len(sym) + 1:-4] for b in cl.list_blobs(bk, prefix=f"{D10}/daily/{sym}_")
                  if b.name.endswith(".npz"))
    # per day, per signal, per horizon, per K: (cap, hit, n)
    out = {s: {h: {k: [] for k in KS} for h in HORS_USE} for s in SIGS}
    t0 = time.time(); nd = 0
    for d in days:
        try:
            zf = np.load(io.BytesIO(bk.blob(f"{LABSUB}/daily/{sym}_{d}.npz").download_as_bytes()))
            zr = np.load(io.BytesIO(bk.blob(f"{D10}/daily/{sym}_{d}.npz").download_as_bytes()))
        except Exception:
            continue
        F = zf["F"]; R = zr["R"]; RV = zr["RV"]
        if len(F) != R.shape[1]:
            continue
        comp = np.zeros(len(F), np.float64)
        for c in COMP_COLS:
            x = F[:, c].astype(np.float64)
            if np.nanstd(x) > 0:
                comp += rank01(x) - 0.5
        nd += 1
        for sname, fc in SIGS.items():
            x = comp if fc < 0 else F[:, fc].astype(np.float64)
            two = (x < 0).any() and (x > 0).any()
            sgn = np.sign(x) if two else np.sign(rank01(x) - 0.5)
            ax = np.abs(x) if two else np.abs(rank01(x) - 0.5)
            for hname, h in HORS_USE.items():
                v = RV[h] & (sgn != 0)
                if v.sum() < 400:
                    continue
                idx = np.where(v)[0]
                order = idx[np.argsort(-ax[idx], kind="stable")]
                for K in KS:
                    sel = order[:K]
                    r = R[h][sel].astype(np.float64); s = sgn[sel]
                    out[sname][hname][K].append((float(np.mean(s * r)), float(np.mean(np.sign(r) == s)), len(sel)))
    print(f"\n================ {sym} — {nd} days (top-K per day by |signal|) ================", flush=True)
    for sname in SIGS:
        for hname in HORS_USE:
            row = []
            for K in KS:
                a = np.array(out[sname][hname][K])
                if not len(a):
                    row.append(f"K={K}: n/a"); continue
                cap = a[:, 0].mean(); hit = (a[:, 1] * a[:, 2]).sum() / a[:, 2].sum()
                dpos = (a[:, 0] > 0).mean()
                row.append(f"K={K}: cap={cap:+.3f}bp hit={hit:.3f} d+={dpos:.2f}")
            print(f"  {sname:<13} H={hname:<3} " + " | ".join(row), flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"{s}__{h}__{k}": np.array(out[s][h][k]) for s in SIGS for h in HORS_USE for k in KS})
    bk.blob(f"{D10}/{sym}_tail.npz").upload_from_string(buf.getvalue())
    print(f"  [saved] {D10}/{sym}_tail.npz ({time.time()-t0:.0f}s)", flush=True)
