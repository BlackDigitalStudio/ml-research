#!/usr/bin/env python3
"""VERIFY (by facts, not the catalog) which feature columns are real vs placeholder-zero in the
actual training data feats_sub60, and which the deployed A/Bg models actually use. A zero/constant
column cannot be split on -> if cross-venue [50-53] (never downloaded from CL) are constant, the
models did NOT use them and my catalog-based categorization is wrong there.
"""
import io, json
import numpy as np, xgboost as xgb
from google.cloud import storage
bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")

# stack a few DOGE days of training X to measure per-column signal
Xs = []
for d in ["2026-05-08", "2026-04-15", "2026-02-01", "2025-09-01"]:
    try:
        z = np.load(io.BytesIO(bk.blob(f"feats_sub60/DOGE-USDT-PERP/{d}.npz").download_as_bytes()))
        Xs.append(z["X"].astype(np.float64))
    except Exception as e:
        print("skip", d, str(e)[:60])
X = np.concatenate(Xs); ncol = X.shape[1]
nz = (np.abs(X) > 1e-12).mean(0); sd = X.std(0)
const = np.where(sd < 1e-9)[0]
print(f"X {X.shape} | constant/zero cols ({len(const)}): {const.tolist()}", flush=True)

meta = json.loads(bk.blob("research_runs/deploy/DOGE/meta.json").download_as_bytes()); fn = meta["feat_names"]
used = {}
for nm in ("A", "Bg"):
    p = f"/tmp/{nm}.json"; bk.blob(f"research_runs/deploy/DOGE/{nm}.json").download_to_filename(p)
    m = xgb.Booster(); m.load_model(p); g = m.get_score(importance_type="gain")
    used[nm] = {int(f[1:]) if f[1:].isdigit() else fn.index(f): v for f, v in g.items()}
    tot = sum(g.values())
    used_const = sorted(set(used[nm]) & set(const))
    print(f"\n{nm}: uses {len(used[nm])}/{ncol} cols | gain from constant cols: {used_const} "
          f"(should be empty)", flush=True)

print("\n=== questionable cols: cross-venue [50-53], and check 56-63 are real ===", flush=True)
for c in [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]:
    inA = used["A"].get(c, 0); inBg = used["Bg"].get(c, 0)
    ta = 100 * inA / sum(used["A"].values()); tb = 100 * inBg / sum(used["Bg"].values())
    flag = "CONST/ZERO" if c in const else f"nz={nz[c]:.2f}"
    print(f"  col {c:2d}: {flag:12} | A gain {ta:4.1f}%  Bg gain {tb:4.1f}%", flush=True)

# the cols the models actually lean on (top), with their real/const status
print("\n=== are any TOP model features actually constant (=> mis-attributed)? ===", flush=True)
for nm in ("A", "Bg"):
    top = sorted(used[nm].items(), key=lambda kv: -kv[1])[:10]
    tot = sum(used[nm].values())
    s = ", ".join(f"x{c}({100*v/tot:.0f}%{'!CONST' if c in const else ''})" for c, v in top)
    print(f"  {nm} top10: {s}", flush=True)
