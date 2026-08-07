#!/usr/bin/env python3
"""HD4 rev1 report: aggregate {SYM}_dirstats.npz into the predictivity surface.
Per symbol: top features by |t-stat| of daily rank-IC at H=10s; horizon decay for the
leaders + COMP; strength-conditional capture (bp per horizon) at q50/q90/q99; monthly
stability of the leaders. No verdicts — the surface is the deliverable."""
import io, sys
import numpy as np
from google.cloud import storage

SYMS = sys.argv[1:] or ["BTC", "ETH", "DOGE", "XRP"]
bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")

# best-effort names (features.rs catalog + feat71 tail)
NM = {0:"ofi_W",1:"imb_L5",2:"imb_d5",3:"spread",4:"bv5/av5",5:"large_lvl",6:"tflow_5s",7:"tcnt_d1s",
      8:"tlarge_5s",9:"tsign_1s",10:"vol10",11:"vwap_dev",12:"mom_5s",13:"funding",14:"eth_r5",15:"eth_r30",
      16:"eth_r60",17:"deriv1",18:"deriv2",19:"deriv3",20:"mom_flag",21:"vol_ratio",22:"trade_int",23:"hurst",
      24:"jump",25:"cancel_d",26:"ofi_1s",27:"ofi_5s",28:"ofi_30s",29:"ofi_s-l",30:"xexch_mom",31:"qpress_ema",
      32:"top3_asym",33:"espread_ema",34:"mom_30s",35:"mom_60s",36:"mom_120s",37:"vol_30s",38:"vol_60s",39:"vol_300s",
      40:"ofi_60s",41:"ofi_300s",42:"tflow_30s",43:"liq_flag",44:"fund_basis",45:"microprice",46:"ofi_norm",
      47:"tflow_60s",48:"tflow_300s",49:"cancel_r",50:"xe_b",51:"xe_o",52:"xe_g",53:"xe_gt",54:"xe_agg",55:"oi_lvl",
      56:"liq_s5",57:"liq_s30",58:"liq_i60",59:"oi_d30",60:"oi_d300",61:"OBI_L20",62:"OBI_L1",63:"OBI_L10",
      64:"btc_r5",65:"btc_r30",66:"btc_r60",67:"sin_h",68:"cos_h",69:"sin_f8",70:"cos_f8",71:"COMP"}

for sym in SYMS:
    z = np.load(io.BytesIO(bk.blob(f"research_runs/h2_dir10/{sym}_dirstats.npz").download_as_bytes()))
    ric, hit, cap, cnt = z["ric"], z["hit"], z["cap"], z["cnt"]   # (D,72,6), (D,72,6,4)
    days = z["days"]; hors = z["hors"]; D = len(days)
    mo = np.array([d[:7] for d in days])
    h10 = int(np.argmin(np.abs(hors - 10)))
    m = np.nanmean(ric, 0); s = np.nanstd(ric, 0); nn = np.sum(np.isfinite(ric), 0)
    t = m / np.where(nn > 1, s / np.sqrt(np.maximum(nn, 1)), np.nan)
    print(f"\n================ {sym} — {D} days ({days[0]}..{days[-1]}) ================")
    print(f"-- H=10s: top-12 features by |t| of daily rank-IC --")
    order = np.argsort(-np.abs(np.nan_to_num(t[:, h10])))[:12]
    for f in order:
        c9 = np.nanmean(cap[:, f, h10, 2]); c99 = np.nanmean(cap[:, f, h10, 3])
        h9 = np.nanmean(hit[:, f, h10, 2])
        print(f"  f{f:<2} {NM.get(f,'?'):<12} ric={m[f,h10]:+.4f} t={t[f,h10]:+7.1f} "
              f"hit@q90={h9:.3f} cap@q90={c9:+.3f}bp cap@q99={c99:+.3f}bp")
    print(f"-- horizon decay (mean daily rank-IC), leaders + COMP --")
    lead = list(order[:4]) + ([71] if 71 not in order[:4] else [])
    hdr = "        " + "".join(f"{int(h):>8}s" for h in hors)
    print(hdr)
    for f in lead:
        print(f"  f{f:<3}{NM.get(f,'?'):<10}" + "".join(f"{m[f,h]:+8.4f}" for h in range(len(hors))))
    print(f"-- COMP capture bp by strength cut (rows=horizon; q0/q50/q90/q99) --")
    for h in range(len(hors)):
        row = [np.nanmean(cap[:, 71, h, qi]) for qi in range(4)]
        hh = [np.nanmean(hit[:, 71, h, qi]) for qi in range(4)]
        print(f"  H={int(hors[h]):>2}s  cap " + " ".join(f"{v:+.3f}" for v in row) +
              "   hit " + " ".join(f"{v:.3f}" for v in hh))
    print(f"-- monthly stability: COMP rank-IC @10s (mean per month) --")
    for mm in sorted(set(mo.tolist())):
        sel = mo == mm
        v = np.nanmean(ric[sel, 71, h10]); vc = np.nanmean(cap[sel, 71, h10, 2])
        print(f"  {mm}: ric={v:+.4f} cap@q90={vc:+.3f}bp  ({sel.sum()}d)")
