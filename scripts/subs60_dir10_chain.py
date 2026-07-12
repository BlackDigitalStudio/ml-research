#!/usr/bin/env python3
"""HD4 rev1 chain amendment: the user's hypothesis-2 rule, model-free — chained 10s holds.
Episode: start at a tick with |S| >= theta_entry (per-day quantile of |S|), direction =
sign(S); each step captures the tick's forward 10s mid move (R[h=10s] from h2_dir10); at
t+10s re-evaluate: continue while sign(S)==dir AND |S| >= theta_cont, else stop (reasons:
flip / weak / data / cap30). Episodes are NON-OVERLAPPING (greedy forward scan; all ticks
inside a chain are consumed) -> also answers the clustering question. No ML, no fitting.
Sweep: S in {f62 OBI_L1, COMP} x theta_entry {q90,q99} x theta_cont {q50, =entry}.
Reads maker_labels_tb3s_h150 dailies (F71) + h2_dir10 dailies (R,RV). Env: SYMS, START, END.
Out: print + research_runs/h2_dir10/{SYM}_chain.npz (per-day per-cell tensors).
"""
import io, os, time
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = os.environ.get("SYMS", "BTC,ETH,DOGE,XRP").split(",")
START = os.environ.get("START", "0000"); END = os.environ.get("END", "9999")
LABSUB = "research_runs/maker_labels_tb3s_h150"
D10 = "research_runs/h2_dir10"
NS = 1_000_000_000
COMP_COLS = [0, 1, 12, 26, 27, 28, 62, 63, 64, 65, 66]
SIGS = ["f62", "COMP"]
CELLS = [(s, qe, qc) for s in SIGS for qe in (0.90, 0.99) for qc in (0.50, None)]  # None -> qc=qe
MAXK = 30; KTRACK = 6
NST = 14  # per-day stat vector length

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
    days = [d for d in days if START <= d <= END]
    per_day = []; keep_days = []; t0 = time.time()
    # step-position tensors accumulated pooled (per cell): sums and counts for k=1..KTRACK
    stepsum = np.zeros((len(CELLS), KTRACK)); stepcnt = np.zeros((len(CELLS), KTRACK), np.int64)
    for d in days:
        try:
            zf = np.load(io.BytesIO(bk.blob(f"{LABSUB}/daily/{sym}_{d}.npz").download_as_bytes()))
            zr = np.load(io.BytesIO(bk.blob(f"{D10}/daily/{sym}_{d}.npz").download_as_bytes()))
        except Exception:
            continue
        F = zf["F"]; R = zr["R"]; RV = zr["RV"]; dtd = zr["ts"].astype(np.int64)
        if len(F) != R.shape[1]:
            continue
        # recompute per cell with step tracking pooled
        r10 = R[1].astype(np.float64); rv = RV[1]
        comp = np.zeros(len(F), np.float64)
        for c in COMP_COLS:
            x = F[:, c].astype(np.float64)
            if np.nanstd(x) > 0:
                comp += rank01(x) - 0.5
        sigs = {"f62": F[:, 62].astype(np.float64), "COMP": comp}
        row = []
        for ci, (sname, qe, qc0) in enumerate(CELLS):
            S = sigs[sname]; sgn = np.sign(S); ax = np.abs(S)
            ok = rv & (sgn != 0)
            if ok.sum() < 500:
                row.append(np.full(NST, np.nan)); continue
            the = np.quantile(ax[ok], qe); thc = the if qc0 is None else np.quantile(ax[ok], qc0)
            n = len(S); used_until = -1
            nep = 0; slen = 0; l1 = l2 = l3p = 0; tot = 0.0; st1 = 0.0
            r_flip = r_weak = r_data = r_cap = 0
            i = 0
            while i < n:
                if i <= used_until or not (rv[i] and sgn[i] != 0 and ax[i] >= the):
                    i += 1; continue
                dd = sgn[i]; t = i; k = 0; cum = 0.0; reason = None
                while True:
                    if k >= MAXK:
                        reason = "cap"; break
                    if not rv[t]:
                        reason = "data"; break
                    cum += dd * r10[t]; k += 1
                    if k <= KTRACK:
                        stepsum[ci, k - 1] += dd * r10[t]; stepcnt[ci, k - 1] += 1
                    if k == 1:
                        st1 += dd * r10[t]
                    tgt = dtd[t] + 10 * NS
                    j = int(np.searchsorted(dtd, tgt, "left"))
                    if j >= n or abs(int(dtd[j]) - int(tgt)) > 2 * NS:
                        if j > 0 and abs(int(dtd[j - 1]) - int(tgt)) <= 2 * NS:
                            j = j - 1
                        else:
                            reason = "data"; break
                    if j <= t:
                        reason = "data"; break
                    if sgn[j] == dd and ax[j] >= thc:
                        t = j; continue
                    reason = "flip" if sgn[j] == -dd else "weak"
                    t = j
                    break
                nep += 1; slen += k; tot += cum
                l1 += k == 1; l2 += k == 2; l3p += k >= 3
                r_flip += reason == "flip"; r_weak += reason == "weak"; r_data += reason == "data"; r_cap += reason == "cap"
                used_until = t; i = t + 1
            row.append(np.array([nep, slen, l1, l2, l3p, tot, st1, r_flip, r_weak, r_data, r_cap, 0, 0, 0], np.float64))
        per_day.append(np.stack(row)); keep_days.append(d)
    P = np.stack(per_day)  # (D, ncells, NST)
    print(f"\n================ {sym} — {len(keep_days)} days, chained 10s holds ================", flush=True)
    print("  cell (sig, q_entry, q_cont): ep/day len P(>=2) P(>=3) | bp/ep total step1 added | step-k bp k=1..4 | stop flip/weak/data/cap", flush=True)
    for ci, (sname, qe, qc0) in enumerate(CELLS):
        v = P[:, ci, :]; v = v[np.isfinite(v[:, 0])]
        nep = v[:, 0].sum()
        if nep == 0:
            continue
        epd = v[:, 0].mean(); mlen = v[:, 1].sum() / nep
        p2 = v[:, 3].sum() / nep + v[:, 4].sum() / nep; p3 = v[:, 4].sum() / nep
        cap = v[:, 5].sum() / nep; s1 = v[:, 6].sum() / nep
        stops = v[:, 7:11].sum(0) / nep
        sk = stepsum[ci] / np.maximum(stepcnt[ci], 1)
        dayadd = (v[:, 5] - v[:, 6]) / np.maximum(v[:, 0], 1)
        dpos = (dayadd > 0).mean()
        qcs = "=e" if qc0 is None else f"{qc0:.2f}"
        print(f"  {sname:<5} qe={qe:.2f} qc={qcs:<4} {epd:6.1f}/d len={mlen:4.2f} P2={p2:.2f} P3={p3:.2f} | "
              f"tot={cap:+6.3f} st1={s1:+6.3f} add={cap-s1:+6.3f} (d+={dpos:.2f}) | "
              f"k:{sk[0]:+.3f}/{sk[1]:+.3f}/{sk[2]:+.3f}/{sk[3]:+.3f} | "
              f"{stops[0]:.2f}/{stops[1]:.2f}/{stops[2]:.2f}/{stops[3]:.2f}", flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, per_day=P, days=np.array(keep_days),
                        cells=np.array([f"{s}|{qe}|{qc}" for s, qe, qc in CELLS]),
                        stepsum=stepsum, stepcnt=stepcnt)
    bk.blob(f"{D10}/{sym}_chain.npz").upload_from_string(buf.getvalue())
    print(f"  [saved] {D10}/{sym}_chain.npz ({time.time()-t0:.0f}s)", flush=True)
