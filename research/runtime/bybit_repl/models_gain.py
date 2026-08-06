#!/usr/bin/env python3
"""HBV1 analysis (non-frozen): aggregate feature importance (total_gain) across the
dumped per-fold boosters (MODELS_S{s}_DOGE_f{k}_{A,Bg}.json). Prints top-N per model
class + the specific families of interest (funding 13/43/44, OI 17/59/60, liq 19/56-58,
btc 64-66, ToD 67-70). Usage: models_gain.py [SYM]; XSYM_SUB overridable."""
import io
import json
import os
import sys
import tempfile

import numpy as np
import xgboost as xgb
from google.cloud import storage

SYM = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
SUB = "research_runs/" + os.environ.get("XSYM_SUB", "maker_labels_tb3s_h150anch")
bk = storage.Client(project="x").bucket("market-data-0998ac51")

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

# DROP_COLS (same value the training run used): model feature index i maps to the
# original schema index AFTER re-inserting the dropped columns.
_drop = sorted(int(x) for x in os.environ.get("DROP_COLS", "").split(",") if x)
_keep = [i for i in range(len(NAMES)) if i not in _drop]
IDXMAP = {mi: oi for mi, oi in enumerate(_keep)}

blobs = [b.name for b in bk.client.list_blobs(bk, prefix=f"{SUB}/MODELS_S")
         if b.name.endswith((".json",)) and "_hp" not in b.name]
agg = {}
cnt = {}
for name in blobs:
    cls = name.rsplit("_", 1)[-1].replace(".json", "")
    if cls not in ("A", "Bg"):
        continue
    raw = bk.blob(name).download_as_bytes()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(raw); path = f.name
    b = xgb.Booster(); b.load_model(path); os.remove(path)
    sc = b.get_score(importance_type="total_gain")
    d = agg.setdefault(cls, np.zeros(len(NAMES)))
    for k, v in sc.items():
        d[IDXMAP[int(k[1:])]] += v
    cnt[cls] = cnt.get(cls, 0) + 1

FAM = {"funding": [13, 43, 44], "OI": [17, 59, 60], "liq": [19] + list(range(56, 59)),
       "btc_lead": [64, 65, 66], "ToD": [67, 68, 69, 70], "eth": [14, 15, 16, 54, 55],
       "cross_exch": [30, 50, 51, 52, 53]}
out = {}
for cls, d in agg.items():
    tot = d.sum()
    pct = 100 * d / max(tot, 1e-9)
    order = np.argsort(-d)
    print(f"\n=== {cls} (boosters: {cnt[cls]}) top-15 by total_gain% ===", flush=True)
    for i in order[:15]:
        print(f"  {pct[i]:5.2f}%  [{i:2d}] {NAMES[i]}", flush=True)
    print(f"  families: " + " | ".join(f"{k} {pct[v].sum():.2f}%" for k, v in FAM.items()), flush=True)
    out[cls] = {"top": [[int(i), NAMES[i], float(pct[i])] for i in order[:25]],
                "families_pct": {k: float(pct[v].sum()) for k, v in FAM.items()},
                "zero_gain_cols": [[int(i), NAMES[i]] for i in range(len(NAMES)) if d[i] == 0]}
    print(f"  zero-gain cols: {[NAMES[i] for i in range(len(NAMES)) if d[i]==0]}", flush=True)
bk.blob(f"{SUB}/HBV1_GAIN_{SYM}.json").upload_from_string(json.dumps(out))
print("\n[saved HBV1_GAIN]", flush=True)
