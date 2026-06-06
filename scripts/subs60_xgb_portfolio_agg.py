#!/usr/bin/env python3
"""Aggregate per-symbol noA-t10 baselines into a portfolio. Builds each symbol's DAILY P&L series
(sum of trade nets per calendar day, bp), equal-weights across symbols active that day, and reports:
per-symbol surface (annS / EV / tpd / hit / IC), portfolio daily-annualized Sharpe, uplift vs the
mean single-symbol Sharpe, and the cross-symbol daily-return correlation matrix (the diversification
driver). Reads research_runs/maker_labels_h/portfolio/{SYM}_noA_t10.{npz,json}.
"""
import io, json, sys
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = sys.argv[1:] if len(sys.argv) > 1 else ["DOGE", "BTC", "ETH", "SOL", "XRP", "BNB", "LINK", "LTC"]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def daily_series(net, cd):
    """sum of trade nets per calendar day -> (days_sorted, pnl_bp). Span = [min..max] cd, 0-fill gaps."""
    if not len(cd):
        return np.array([]), np.array([])
    cd = cd.astype(np.int64); lo, hi = int(cd.min()), int(cd.max())
    days = np.arange(lo, hi + 1); pnl = np.zeros(len(days))
    np.add.at(pnl, cd - lo, net)
    return days, pnl


def ann_sharpe_daily(pnl):
    if len(pnl) < 2 or pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(365.0))


S = {}
for sym in SYMS:
    try:
        d = np.load(io.BytesIO(bk.blob(f"research_runs/maker_labels_h/portfolio/{sym}_noA_t10.npz").download_as_bytes()))
        j = json.loads(bk.blob(f"research_runs/maker_labels_h/portfolio/{sym}_noA_t10.json").download_as_bytes())
        days, pnl = daily_series(d["net"], d["calday"])
        S[sym] = {"days": days, "pnl": pnl, "j": j}
    except Exception as e:
        print(f"  [skip {sym}] {e}", flush=True)
if not S:
    print("no symbols loaded"); sys.exit(1)

print(f"\n=== PER-SYMBOL SURFACE (noA t10, zero-fee, IC-tuned, own weights) ===", flush=True)
print(f"  {'sym':>5} {'folds':>5} {'trd/d':>6} {'EV/tr':>7} {'annS':>6} {'hit%':>6} {'dailyS':>7} {'IC(mean)':>9}  per-fold(%)", flush=True)
per_sym_annS = []
for sym in S:
    j = S[sym]["j"]; ds = ann_sharpe_daily(S[sym]["pnl"]); icm = float(np.mean(j["ic_val"])) if j["ic_val"] else float("nan")
    per_sym_annS.append(ds)
    print(f"  {sym:>5} {j['n_folds']:>5} {j['trd_per_day']:>6.1f} {j['ev_bp']:>+7.2f} {j['ann_sharpe']:>+6.2f} "
          f"{j['hit_pct']:>5.1f} {ds:>+7.2f} {icm:>+9.4f}  {j['perfold_pct']}", flush=True)

# ---- portfolio: union of spans, equal-weight across symbols active each day ----
lo = min(int(S[s]["days"][0]) for s in S); hi = max(int(S[s]["days"][-1]) for s in S)
grid = np.arange(lo, hi + 1); ns = len(grid)
M = np.full((len(S), ns), np.nan)   # symbol x day, NaN = not deployed
for i, sym in enumerate(S):
    dd = S[sym]["days"]; M[i, dd - lo] = S[sym]["pnl"]
active = ~np.isnan(M)
port_daily = np.where(active.any(0), np.nansum(np.where(active, M, 0.0), 0) / np.maximum(active.sum(0), 1), 0.0)
port_days = active.any(0)
port = port_daily[port_days]
port_annS = ann_sharpe_daily(port)
mean_single = float(np.nanmean(per_sym_annS))

# ---- cross-symbol daily-return correlation (overlap only) ----
syms = list(S.keys()); K = len(syms); C = np.full((K, K), np.nan)
for i in range(K):
    for k in range(K):
        a = M[i]; b = M[k]; ov = (~np.isnan(a)) & (~np.isnan(b))
        if ov.sum() >= 10 and a[ov].std() > 0 and b[ov].std() > 0:
            C[i, k] = np.corrcoef(a[ov], b[ov])[0, 1]
offdiag = C[~np.eye(K, dtype=bool)]; mean_corr = float(np.nanmean(offdiag))

print(f"\n=== PORTFOLIO (equal-weight, {K} symbols) ===", flush=True)
print(f"  portfolio daily-annualized Sharpe = {port_annS:+.2f}", flush=True)
print(f"  mean single-symbol daily Sharpe    = {mean_single:+.2f}", flush=True)
print(f"  diversification uplift             = {port_annS - mean_single:+.2f}  ({port_annS/mean_single:.2f}x)" if mean_single and np.isfinite(mean_single) else "", flush=True)
print(f"  mean cross-symbol daily corr       = {mean_corr:+.3f}  (lower = more diversification)", flush=True)
print(f"  portfolio trading days={int(port_days.sum())} | total EV/day(bp)={float(port.mean()):+.2f}", flush=True)
print(f"\n  corr matrix ({' '.join(syms)}):", flush=True)
for i, sym in enumerate(syms):
    print(f"  {sym:>5} " + " ".join(f"{C[i,k]:+.2f}" if np.isfinite(C[i,k]) else "  . " for k in range(K)), flush=True)

RES = {"symbols": syms, "portfolio_daily_annS": port_annS, "mean_single_daily_annS": mean_single,
       "uplift": port_annS - mean_single, "mean_cross_corr": mean_corr,
       "per_symbol": {s: S[s]["j"] for s in S}, "corr_matrix": np.nan_to_num(C, nan=0.0).tolist(),
       "port_trading_days": int(port_days.sum()), "port_ev_per_day_bp": float(port.mean())}
bk.blob("research_runs/maker_labels_h/PORTFOLIO_RESULT.json").upload_from_string(json.dumps(RES, default=float))
print("\n[saved] PORTFOLIO_RESULT.json", flush=True)
