#!/usr/bin/env python3
"""HBV1 analysis (non-frozen): seed-diversity diagnostics from PERFOLD artifacts.
Per fold: pairwise Pearson corr of axb_te scores across seeds, side-agreement rate,
and the Krogh-Vedelsby-style headroom estimate for the mean-rank ensemble.
Usage: seed_corr.py [SYM]; XSYM_SUB overridable."""
import io
import json
import os
import sys
from itertools import combinations

import numpy as np
from google.cloud import storage

SYM = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
SUB = "research_runs/" + os.environ.get("XSYM_SUB", "maker_labels_tb3s_h150anch")
SEEDS = [0, 1, 2, 3]
bk = storage.Client(project="x").bucket("market-data-0998ac51")

nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SUB}/PERFOLD_S0_{SYM}_qm0_f") if b.name.endswith(".npz"))
corrs, agrees, tail_overlaps = [], [], []
for f in range(nf):
    zs = [np.load(io.BytesIO(bk.blob(f"{SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
          for s in SEEDS]
    te = [z["axb_te"].astype(np.float64) for z in zs]
    sides = [z["side"].astype(bool) for z in zs]
    n = len(te[0])
    for a, b in combinations(range(len(SEEDS)), 2):
        corrs.append(float(np.corrcoef(te[a], te[b])[0, 1]))
        agrees.append(float((sides[a] == sides[b]).mean()))
        # tail overlap: of each seed's top-1% scores, what fraction is shared
        ka = set(np.argsort(-te[a])[: max(n // 100, 1)].tolist())
        kb = set(np.argsort(-te[b])[: max(n // 100, 1)].tolist())
        tail_overlaps.append(len(ka & kb) / max(len(ka), 1))
corrs = np.array(corrs); agrees = np.array(agrees); tails = np.array(tail_overlaps)
rho = float(np.mean(corrs))
N = len(SEEDS)
var_ratio = rho + (1 - rho) / N            # ens score-noise variance vs single seed
var_ratio_8 = rho + (1 - rho) / 8
print(f"{SUB.split('/')[-1]} {SYM}: folds={nf}", flush=True)
print(f"  score corr (pairwise): mean {rho:.3f}  p10/p90 {np.quantile(corrs, .1):.3f}/{np.quantile(corrs, .9):.3f}", flush=True)
print(f"  side agreement: mean {np.mean(agrees)*100:.1f}%", flush=True)
print(f"  top-1% tail overlap: mean {np.mean(tails)*100:.1f}%", flush=True)
print(f"  ens noise-var vs single seed: N=4 -> {var_ratio:.3f}, N=8 -> {var_ratio_8:.3f} "
      f"(N=inf floor {rho:.3f})", flush=True)
out = dict(sub=SUB, nf=nf, rho_mean=rho, rho_q=[float(np.quantile(corrs, q)) for q in (.1, .5, .9)],
           side_agree=float(np.mean(agrees)), tail_overlap=float(np.mean(tails)),
           var_ratio_n4=var_ratio, var_ratio_n8=var_ratio_8)
bk.blob(f"{SUB}/HBV1_SEEDCORR_{SYM}.json").upload_from_string(json.dumps(out))
print("[saved HBV1_SEEDCORR]", flush=True)
