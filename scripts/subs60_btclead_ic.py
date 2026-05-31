#!/usr/bin/env python3
"""Is the BTC->alt lead-lag arena alive at sub-60s? IC of BTC-lead {5,30,60}s with
future rH60 (signed), on non-flat windows. Decides whether a richer lead-lag feature
(residual/beta-adjusted) is worth a retrain BEFORE spending Modal."""
import io
import numpy as np
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
bk = storage.Client(project=PROJ).bucket(BUCKET)
COLS = {64: "btc5s", 65: "btc30s", 66: "btc60s"}

for sym in ["DOGE-USDT-PERP", "ETH-USDT-PERP", "LINK-USDT-PERP"]:
    cb = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"hd2_sub60_cache/{sym}/") if b.name.endswith(".npz"))
    samp = cb[::max(1, len(cb)//40)][:40]                     # ~40 days across full range
    F, R, V = [], [], []
    for nm in samp:
        z = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
        F.append(z["feat"][:, [64, 65, 66]].astype(np.float64)); R.append(z["rH60"].astype(np.float64))
        V.append(z["v60"].astype(bool))
    F = np.concatenate(F); R = np.concatenate(R); V = np.concatenate(V)
    nf = V & np.isfinite(R)
    Fn, Rn = F[nf], R[nf]
    print(f"\n== {sym} | {len(samp)} days, {nf.sum()} non-flat windows ==")
    for j, (c, name) in enumerate(COLS.items()):
        x = Fn[:, j]
        ic = np.corrcoef(x, Rn)[0, 1]                          # Pearson IC (feature vs future signed ret)
        # directional: does BTC's trailing direction predict alt's next-60s direction?
        hit = np.mean(np.sign(x) == np.sign(Rn))
        # rank-IC
        ric = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(Rn)))[0, 1]
        print(f"   {name:>6}: IC={ic:+.4f}  rankIC={ric:+.4f}  dir-hit={hit:.3f}  (|x|>0 n={np.mean(x!=0):.2f})")
    # combined: BTC60s sign as a standalone directional rule on non-flat
    s = np.sign(Fn[:, 2])
    cap = np.mean(s * Rn)                                       # bp captured if you trade BTC60s direction
    print(f"   BTC60s-as-signal: mean(sign(btc60)*rH60) = {cap:+.3f} bp  (vs needs >4bp maker-maker)")
