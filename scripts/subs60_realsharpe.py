#!/usr/bin/env python3
"""Honest Sharpe from the saved per-symbol trades: contrasts the per-trade annualization
[(EV/std)*sqrt(trd/day*365), which assumes independent trades -> inflates when many correlated
trades/day] against the DAILY-series Sharpe [mean(daily)/std(daily)*sqrt(365)] and a MONTHLY-block
Sharpe [*sqrt(12), robust to daily autocorrelation]. Reads portfolio/{SYM}_noA_t10.npz.
"""
import io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = ["DOGE", "BTC", "ETH", "SOL", "XRP", "BNB", "LINK", "LTC"]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def daily(cd, net):
    lo, hi = int(cd.min()), int(cd.max()); g = np.zeros(hi - lo + 1); np.add.at(g, cd - lo, net); return g


def ann(x, k):  # annualization factor k (365 daily, 12 monthly)
    return float(x.mean() / x.std() * np.sqrt(k)) if len(x) > 1 and x.std() > 0 else float("nan")


print(f"  {'sym':>5} {'n':>6} {'trd/d':>6} {'EV/tr':>7} {'perTrade_annS':>13} {'DAILY_annS':>11} {'MONTHLY_annS':>12}", flush=True)
rows = {}
allp = {}
for sym in SYMS:
    try:
        d = np.load(io.BytesIO(bk.blob(f"research_runs/maker_labels_h/portfolio/{sym}_noA_t10.npz").download_as_bytes()))
    except Exception as e:
        print(f"  {sym:>5} (skip: {e})"); continue
    net = d["net"].astype(float); cd = d["calday"].astype(np.int64)
    if not len(net):
        print(f"  {sym:>5} no trades"); continue
    g = daily(cd, net); span = len(g); trd_d = len(net) / max(span, 1)
    pt = (net.mean() / net.std()) * np.sqrt(trd_d * 365.0) if net.std() > 0 else float("nan")
    da = ann(g, 365.0)
    nb = max(span // 30, 1); mb = np.array([g[i * 30:(i + 1) * 30].sum() for i in range(nb)])  # 30-day blocks
    ma = ann(mb, 12.0)
    rows[sym] = dict(n=len(net), trd_d=round(trd_d, 1), ev=round(float(net.mean()), 3),
                     perTrade_annS=round(pt, 2), daily_annS=round(da, 2), monthly_annS=round(ma, 2))
    allp[sym] = (cd, net)
    print(f"  {sym:>5} {len(net):>6} {trd_d:>6.1f} {net.mean():>+7.2f} {pt:>+13.2f} {da:>+11.2f} {ma:>+12.2f}", flush=True)

# equal-weight portfolio of POSITIVE-daily-Sharpe symbols vs all
def port(symset):
    lo = min(int(allp[s][0].min()) for s in symset); hi = max(int(allp[s][0].max()) for s in symset)
    M = np.full((len(symset), hi - lo + 1), np.nan)
    for i, s in enumerate(symset):
        cd, net = allp[s]; clo, chi = int(cd.min()), int(cd.max())
        gg = np.zeros(chi - clo + 1); np.add.at(gg, cd - clo, net)
        M[i, clo - lo:chi - lo + 1] = gg
    act = ~np.isnan(M); pser = np.where(act.any(0), np.nansum(np.where(act, M, 0), 0) / np.maximum(act.sum(0), 1), 0.0)
    pser = pser[act.any(0)]; return ann(pser, 365.0), len(pser)

pos = [s for s in rows if rows[s]["daily_annS"] > 0]
pall_s, pall_n = port(list(rows.keys()))
ppos_s, ppos_n = port(pos) if pos else (float("nan"), 0)
print(f"\n  PORTFOLIO daily_annS  all-8 = {pall_s:+.2f} (n={pall_n}d)  |  positive-only {pos} = {ppos_s:+.2f} (n={ppos_n}d)", flush=True)
bk.blob("research_runs/maker_labels_h/REALSHARPE_RESULT.json").upload_from_string(
    json.dumps({"per_symbol": rows, "port_all_daily_annS": pall_s, "port_pos_daily_annS": ppos_s, "positive": pos}, default=float))
print("[saved] REALSHARPE_RESULT.json", flush=True)
