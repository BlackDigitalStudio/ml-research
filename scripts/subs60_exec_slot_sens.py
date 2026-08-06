#!/usr/bin/env python3
"""M-slot concurrency sensitivity + opposite-side overlap share, DOGE/XRP frozen-tau.
Slot sim: signal takes a free slot (busy = busy_fill if filled else 60s); no free slot -> skip.
Variant B: additionally skip a signal whose side opposes any currently-open trade (one-way mode)."""
import os
import numpy as np

SCR = os.getcwd()  # caches under the run dir
STEP = 3.0
CASES = [
    ("DOGE", "_recev_h150anch2_DOGE", 0.817010),
    ("XRP",  "_recev_h150anch2_XRP",  0.925442),
]
BUSY_FILL = 155.0  # empirical from live trade logs (med 155s)

def slot_sim(sel, side, fill, M, oneway):
    open_until = []  # list of (t_free, side)
    taken = []
    skipped_opp = 0
    for i in sel:
        t = i * STEP
        open_until = [(tf, sd) for tf, sd in open_until if tf > t]
        if oneway and any(sd != side[i] for _, sd in open_until):
            skipped_opp += 1
            continue
        if len(open_until) >= M:
            continue
        taken.append(i)
        open_until.append((t + (BUSY_FILL if fill[i] else 60.0), side[i]))
    return np.array(taken, dtype=int), skipped_opp

for sym, prefix, tau in CASES:
    d = os.path.join(SCR, "recev_cache", prefix)
    files = sorted(f for f in os.listdir(d) if f.startswith("D_"))
    days = [(f[2:10], np.load(os.path.join(d, f))) for f in files]
    nd = len(days)
    # reference: all signals
    Sref = 0.0; Nref = 0
    prepared = []
    opp_overlap_all = 0; tot_sig = 0
    for day, z in days:
        sc = z["score"].astype(np.float64); side = z["side"].astype(bool)
        net = np.where(side, z["netl"].astype(np.float64), z["nets"].astype(np.float64))
        fill = np.where(side, z["FL"].astype(bool), z["FS"].astype(bool))
        sel = np.where(sc >= tau)[0]
        prepared.append((sel, side, net, fill))
        ex = sel[fill[sel] & np.isfinite(net[sel])]
        Sref += net[ex].sum(); Nref += len(ex)
        # opposite-side overlap among filled signals (450s window)
        t = ex * STEP
        for k, i in enumerate(ex):
            if np.any((t < t[k]) & (t + BUSY_FILL > t[k]) & (side[ex] != side[i])):
                opp_overlap_all += 1
        tot_sig += len(ex)
    print(f"\n### {sym} tau={tau} ({nd} days) MEASURED: n={Nref} bpd={Sref/nd:+.1f} | "
          f"opposite-side overlap: {opp_overlap_all}/{tot_sig} ({100*opp_overlap_all/max(tot_sig,1):.1f}%)")
    for oneway in (False, True):
        tag = "one-way(skip-opp)" if oneway else "hedge-ok        "
        for M in (1, 2, 3, 4, 6, 8, 999):
            N = 0; S = 0.0; skop = 0
            for sel, side, net, fill in prepared:
                tk, so = slot_sim(sel, side, fill, M, oneway)
                ex = tk[fill[tk] & np.isfinite(net[tk])]
                N += len(ex); S += net[ex].sum(); skop += so
            print(f"  {tag} M={M:3d}: n={N:4d} ({N/nd:4.2f}/d) EV={S/max(N,1):+6.2f} bpd={S/nd:+6.1f} "
                  f"kept={100*S/Sref:5.1f}%" + (f" skip_opp={skop}" if oneway else ""))
