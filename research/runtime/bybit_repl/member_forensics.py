#!/usr/bin/env python3
"""HBV1 rev20 (non-frozen analysis): MEMBER FORENSICS — what do the members
that catch signal have in common?

Per member (known random feature-bag + trained boosters + realized EV):
  (a) KEEP-vs-DROP per column: mean causal-t5 EV of members that KEPT col c
      minus members that DROPPED it (the bag randomization = a feature-value
      experiment ACROSS members, orthogonal to seeds/data-bags);
  (b) Spearman(per-member gain share of col c, member EV) over members that
      kept c — do strong members WEIGHT the col differently;
  (c) gain profiles of top-quartile vs bottom-quartile members.
Env: SYM, MEMBERS "sub:seed,..." (rf members; bag recomputed as rng(1000+j)),
TGT (5), FTAG artifact suffix. EV here is per-member GROSS (selection layer),
consistent with per-seed surfaces; fees don't reorder members.
"""
import io
import json
import os

import numpy as np
import xgboost as xgb
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = os.environ.get("SYM", "DOGE")
TGT = float(os.environ.get("TGT", "5"))
KDAYS = 30
_MEMBERS = os.environ["MEMBERS"]
MEMBERS = [(("research_runs/" + p.split(":")[0]), int(p.split(":")[1])) for p in _MEMBERS.split(",")]

N56 = ("ofi imbalance_ratio imbalance_velocity spread depth_ratio_l5 large_order trade_flow_imbalance "
       "trade_intensity large_trade cvd volatility_1s vwap_deviation momentum_5s funding_rate eth_momentum_1s "
       "eth_ofi eth_leading_signal open_interest_delta long_short_ratio liquidation_proximity spoof_score "
       "volatility_ratio trade_intensity_ratio hurst sweep_intensity cancel_rate_diff ofi_1s ofi_5s ofi_30s "
       "ofi_divergence cross_exch_mom_500ms queue_pressure top3_asymmetry effective_spread_ratio momentum_30s "
       "momentum_60s momentum_120s realized_vol_60s realized_vol_120s bipower_var_120s ofi_60s ofi_120s "
       "trade_flow_imbalance_60s funding_time_to_next_min funding_basis_bps microprice_deviation "
       "ofi_top5_weighted kyle_lambda_60s vpin_60s cancel_to_trade_ratio_30s bybit_lead_lag_corr_30s "
       "okx_net_flow_30s bitget_net_flow_30s gateio_net_flow_30s eth_momentum_60s eth_btc_corr_30s").split()
NAMES = N56 + [f"ext{i}" for i in range(56, 64)] + ["btc_ret5", "btc_ret30", "btc_ret60",
                                                    "sin_h", "cos_h", "sin_f8", "cos_f8"]
DEAD = {17, 18, 19, 24, 30, 44, 50, 51, 52, 53, 56, 57, 58}


def bag(j):
    live = [i for i in range(71) if i not in DEAD and i not in (59, 60)]
    extra = np.random.default_rng(1000 + j).choice(live, size=14, replace=False)
    return sorted({59, 60} | {int(x) for x in extra})


def causal_ev(z, tgt):
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
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.nan
    side = z["side"][sel]
    net = np.where(side, z["netl"].astype(np.float64)[sel], z["nets"].astype(np.float64)[sel])
    fc = np.where(side, z["fl"][sel], z["fs"][sel])
    ex = fc & np.isfinite(net)
    return float(net[ex].mean()) if ex.any() else np.nan


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


rows = []  # (j, drop_set, ev, gain_share[71])
for sub, j in MEMBERS:
    nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{sub}/PERFOLD_S{j}_{SYM}_qm0_f")
             if b.name.endswith(".npz"))
    evs, gain = [], np.zeros(71)
    drop = set(bag(j))
    keep = [i for i in range(71) if i not in drop]
    idxmap = {mi: oi for mi, oi in enumerate(keep)}
    for f in range(nf):
        z = np.load(io.BytesIO(bk.blob(f"{sub}/PERFOLD_S{j}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
        evs.append(causal_ev(z, TGT))
        for cls in ("A", "Bg"):
            bl = bk.blob(f"{sub}/MODELS_S{j}_{SYM}_f{f}_{cls}.json")
            if not bl.exists():
                continue
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tf.write(bl.download_as_bytes()); path = tf.name
            bst = xgb.Booster(); bst.load_model(path); os.remove(path)
            for k, v in bst.get_score(importance_type="total_gain").items():
                gain[idxmap[int(k[1:])]] += v
    gshare = gain / max(gain.sum(), 1e-9)
    rows.append((j, drop, float(np.nanmean(evs)), gshare))
    print(f"member {sub.split('/')[-1]}:{j} EV(t{TGT:g}) {np.nanmean(evs):+6.2f} "
          f"dropped {sorted(c for c in drop if c not in (59, 60))}", flush=True)

evs = np.array([r[2] for r in rows])
order = np.argsort(-evs)
print(f"\n{SYM}: {len(rows)} members, EV mean {np.nanmean(evs):+.2f} sd {np.nanstd(evs):.2f} "
      f"top {evs[order[0]]:+.2f} bottom {evs[order[-1]]:+.2f}", flush=True)

# (a) keep-vs-drop per column
print("\n(a) KEEP-vs-DROP EV delta (bp; cols never/always dropped skipped):", flush=True)
tab = []
for c in range(71):
    if c in DEAD or c in (59, 60):
        continue
    kept = [r[2] for r in rows if c not in r[1]]
    dropped = [r[2] for r in rows if c in r[1]]
    if len(kept) < 2 or len(dropped) < 2:
        continue
    d = float(np.nanmean(kept) - np.nanmean(dropped))
    se = float(np.sqrt(np.nanvar(kept) / len(kept) + np.nanvar(dropped) / len(dropped)))
    tab.append((c, d, se, len(dropped)))
tab.sort(key=lambda x: -abs(x[1] / max(x[2], 1e-9)))
for c, d, se, nd in tab[:18]:
    print(f"  [{c:2d}] {NAMES[c]:26s} keep-drop {d:+7.2f} (se {se:5.2f}, z {d/max(se,1e-9):+4.1f}, n_drop {nd})", flush=True)

# (b) gain-share vs EV among keepers
print("\n(b) Spearman(gain share, member EV) among keepers (|rho| top-12, n>=6):", flush=True)
gtab = []
for c in range(71):
    if c in DEAD or c in (59, 60):
        continue
    pts = [(r[3][c], r[2]) for r in rows if c not in r[1]]
    if len(pts) < 6:
        continue
    rho = spearman([p[0] for p in pts], [p[1] for p in pts])
    if np.isfinite(rho):
        gtab.append((c, rho, len(pts)))
gtab.sort(key=lambda x: -abs(x[1]))
for c, rho, n in gtab[:12]:
    print(f"  [{c:2d}] {NAMES[c]:26s} rho {rho:+5.2f} (n {n})", flush=True)

# (c) top-vs-bottom quartile gain profiles
q = max(2, len(rows) // 4)
top = np.mean([rows[i][3] for i in order[:q]], axis=0)
bot = np.mean([rows[i][3] for i in order[-q:]], axis=0)
diff = top - bot
dord = np.argsort(-np.abs(diff))
print(f"\n(c) gain-share TOP{q} minus BOTTOM{q} (pp of total gain):", flush=True)
for c in dord[:12]:
    print(f"  [{c:2d}] {NAMES[c]:26s} {100*diff[c]:+5.2f}pp (top {100*top[c]:5.2f} bot {100*bot[c]:5.2f})", flush=True)

out = dict(sym=SYM, tgt=TGT, members=[(r[0], sorted(r[1]), r[2]) for r in rows],
           keep_drop=[(c, d, se, nd) for c, d, se, nd in tab],
           gain_ev_rho=[(c, rho, n) for c, rho, n in gtab],
           top_bot_gain=[(int(c), float(diff[c]), float(top[c]), float(bot[c])) for c in dord[:25]])
_tag = os.environ.get("FTAG", "")
bk.blob(f"research_runs/HBV1_FORENSICS_{SYM}{_tag}.json").upload_from_string(json.dumps(out, default=float))
print("\n[saved HBV1_FORENSICS]", flush=True)
