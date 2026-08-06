#!/usr/bin/env python3
"""HBV1 analysis (non-frozen): seed-diversity diagnostics ON THE TRADED TAIL.

Per user directive 2026-08-06: all diversity numbers are computed on decisions that
PASSED the causal selectivity threshold (the same causal_rolling as perseed_from_pf,
per seed, at TGT trades/day) — not on distribution-wide proxies.

Per (fold, seed): selected decision set S_s. Reported across seeds:
  - pairwise Jaccard |S_a & S_b| / |S_a | S_b|  (how different the actual trade lists are)
  - side agreement on the intersection
  - consensus buckets: decisions selected by exactly k of 4 seeds -> n, EV/tr
    (EV = mean over selecting seeds of that seed's own-side filled net; the
    consensus-selectivity axis of rev22, measured on this cell)
  - full-score corr kept as context.
Usage: seed_corr.py [SYM]; env XSYM_SUB, TGTS (default "5,10")."""
import io
import json
import os
import sys
from itertools import combinations

import numpy as np
from google.cloud import storage

SYM = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
SUB = "research_runs/" + os.environ.get("XSYM_SUB", "maker_labels_tb3s_h150anch")
TGTS = [float(x) for x in os.environ.get("TGTS", "5,10").split(",")]
SEEDS = [0, 1, 2, 3]
KDAYS = 30
bk = storage.Client(project="x").bucket("market-data-0998ac51")


def select(z, tgt):
    """causal_rolling selection — returns the selected decision indices (test-side)."""
    sc_tr = z["axb_tr"].astype(np.float64); sc_te = z["axb_te"].astype(np.float64)
    day_tr = z["day_tr"]; day_te = z["day_te"]
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    return np.array(sel, dtype=int)


def trade_net(z, idx):
    """per-seed own-side filled net for decision indices (NaN if unfilled)."""
    side = z["side"][idx]
    net = np.where(side, z["netl"].astype(np.float64)[idx], z["nets"].astype(np.float64)[idx])
    fc = np.where(side, z["fl"][idx], z["fs"][idx])
    return np.where(fc & np.isfinite(net), net, np.nan)


nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SUB}/PERFOLD_S0_{SYM}_qm0_f") if b.name.endswith(".npz"))
Z = {s: [np.load(io.BytesIO(bk.blob(f"{SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
         for f in range(nf)] for s in SEEDS}

# context: full-score corr (kept, labelled as distribution-wide)
cors = [float(np.corrcoef(Z[a][f]["axb_te"].astype(np.float64), Z[b][f]["axb_te"].astype(np.float64))[0, 1])
        for f in range(nf) for a, b in combinations(SEEDS, 2)]
print(f"{SUB.split('/')[-1]} {SYM}: folds={nf} | context full-score corr mean {np.mean(cors):.3f}", flush=True)

out = {"nf": nf, "score_corr_full": float(np.mean(cors))}
for tgt in TGTS:
    jac, sagree = [], []
    kn = {k: [] for k in (1, 2, 3, 4)}
    kev = {k: [] for k in (1, 2, 3, 4)}
    for f in range(nf):
        sels = {s: select(Z[s][f], tgt) for s in SEEDS}
        sets = {s: set(sels[s].tolist()) for s in SEEDS}
        for a, b in combinations(SEEDS, 2):
            u = sets[a] | sets[b]
            inter = sets[a] & sets[b]
            jac.append(len(inter) / max(len(u), 1))
            if inter:
                ii = np.array(sorted(inter))
                sagree.append(float((Z[a][f]["side"][ii] == Z[b][f]["side"][ii]).mean()))
        allsel = sorted(set().union(*sets.values()))
        for i in allsel:
            ks = [s for s in SEEDS if i in sets[s]]
            nets = [trade_net(Z[s][f], np.array([i]))[0] for s in ks]
            nets = [x for x in nets if np.isfinite(x)]
            kn[len(ks)].append(i)
            if nets:
                kev[len(ks)].append(float(np.mean(nets)))
    res = {"jaccard": float(np.mean(jac)), "side_agree_inter": float(np.mean(sagree)) if sagree else None,
           "consensus": {k: {"n": len(kn[k]), "ev": (float(np.mean(kev[k])) if kev[k] else None)}
                         for k in (1, 2, 3, 4)}}
    out[f"t{tgt:g}"] = res
    cons = " | ".join(f"k={k}: n={len(kn[k])} EV {np.mean(kev[k]):+.2f}bp" if kev[k] else f"k={k}: n={len(kn[k])}"
                      for k in (1, 2, 3, 4))
    print(f"  t{tgt:g}: traded-set Jaccard {np.mean(jac):.3f} | side-agree on overlap "
          f"{(np.mean(sagree)*100 if sagree else float('nan')):.1f}% | {cons}", flush=True)

bk.blob(f"{SUB}/HBV1_SEEDCORR_{SYM}.json").upload_from_string(json.dumps(out, default=float))
print("[saved HBV1_SEEDCORR]", flush=True)
