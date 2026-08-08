#!/usr/bin/env python3
"""HBV1 rev25 stage 3: aggregate the per-day tick-tail records into the
intra-burst tail surface per form:
  MAE210/MAE510 distribution (p50/p90/p95/p99/max, bp), MFE/END means,
  MAE by consensus-depth bucket, worst positions, and the leverage table:
  for gross G in grid — share of positions whose intra-window adverse
  excursion alone costs >=25% equity (MAE >= 2500/G bp) and the liquidation
  proxy (MAE >= 10000/G - 50 bp maintenance buffer). G_25max = 2500/max_MAE
  (largest gross with zero in-sample intra-position 25% hits), G_25p99
  analog on p99.
Env: SYM, FORMS_LIST. Output: research_runs/HBV1_TICKTAIL_AGG_{SYM}.json
"""
import json
import os

import numpy as np
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = os.environ.get("SYM", "DOGE")
FORMS_LIST = os.environ["FORMS_LIST"].split(",")
G_GRID = [1, 2, 3, 5, 7.2, 10, 15, 20, 30]

per_form = {n: [] for n in FORMS_LIST}
ndays = 0
for b in bk.client.list_blobs(bk, prefix=f"research_runs/HBV1_TICKTAIL_{SYM}/"):
    if not b.name.endswith(".json"):
        continue
    ndays += 1
    d = json.loads(b.download_as_bytes())
    for n, rows in d.items():
        if n in per_form:
            per_form[n].extend(rows)

out = {}
for name, rows in per_form.items():
    if not rows:
        continue
    # row: [ts, side, day, k, mae210, mfe210, end210, mae510, trunc, spread_bp]
    A = np.array([[r[4], r[5], r[6], r[7], r[8], r[3], r[9]] for r in rows], np.float64)
    mae210, mfe, end, mae510, trunc, kk, spr = A.T
    res = dict(n=len(rows), n_trunc=int(trunc.sum()), spread_bp_med=float(np.median(spr)))
    for tag, v in (("mae210", mae210), ("mae510", mae510)):
        res[tag] = {p: float(np.percentile(v, q)) for p, q in
                    (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))}
        res[tag]["max"] = float(v.max())
    res["mfe210_med"] = float(np.median(mfe)); res["end210_mean"] = float(end.mean())
    lev = {}
    for G in G_GRID:
        thr25 = 2500.0 / G
        thrliq = max(10000.0 / G - 50.0, 0.0)
        lev[str(G)] = dict(p_dd25=float(np.mean(mae510 >= thr25)),
                           n_dd25=int((mae510 >= thr25).sum()),
                           p_liq=float(np.mean(mae510 >= thrliq)),
                           n_liq=int((mae510 >= thrliq).sum()))
    res["leverage"] = lev
    res["G_25max"] = float(2500.0 / max(mae510.max(), 1e-9))
    res["G_25p99"] = float(2500.0 / max(np.percentile(mae510, 99), 1e-9))
    kmed = np.median(kk)
    res["mae510_p99_shallow_k"] = float(np.percentile(mae510[kk <= kmed], 99)) if (kk <= kmed).any() else None
    res["mae510_p99_deep_k"] = float(np.percentile(mae510[kk > kmed], 99)) if (kk > kmed).any() else None
    worst = np.argsort(-mae510)[:5]
    res["worst"] = [[float(rows[i][0]), int(rows[i][1]), float(mae510[i]), int(kk[i])] for i in worst]
    out[name] = res
    print(f"{name}: n={res['n']} (trunc {res['n_trunc']}) MAE510 p50 {res['mae510']['p50']:.0f} "
          f"p95 {res['mae510']['p95']:.0f} p99 {res['mae510']['p99']:.0f} max {res['mae510']['max']:.0f}bp "
          f"| G_25max {res['G_25max']:.1f} G_25p99 {res['G_25p99']:.1f} "
          f"| MFE med {res['mfe210_med']:.0f} END mean {res['end210_mean']:+.1f}", flush=True)
    for G in ("3", "7.2", "15", "30"):
        v = lev[G]
        print(f"   G={G:>4s}: P(intra-pos DD>=25%) {100*v['p_dd25']:.2f}% ({v['n_dd25']}) "
              f"P(liq-proxy) {100*v['p_liq']:.2f}% ({v['n_liq']})", flush=True)

bk.blob(f"research_runs/HBV1_TICKTAIL_AGG_{SYM}.json").upload_from_string(json.dumps(out, default=float))
print(f"\n[saved HBV1_TICKTAIL_AGG_{SYM}] from {ndays} day files", flush=True)
