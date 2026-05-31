#!/usr/bin/env python3
"""Phase-1b: is the hold-direction EV real edge or tail variance?
From caches (rH_full + REF day_idx). Per (symbol, signal, q): trade-mean EV,
day-clustered t-stat, day-block bootstrap 95% CI, first-half vs second-half stability.
Day-clustering is essential: ~1-17 trades/day cluster, so the unit of inference is the DAY.
"""
import io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
OUTP = "research_runs/gru_gridsim"; SYMS = ["DOGE", "ETH", "LINK"]
QS = [0.2, 0.1, 0.05, 0.02, 0.01]
MM = 4.0                                                  # maker-maker RT bp
bk = storage.Client(project=PROJ).bucket(BUCKET)
rng = np.random.default_rng(0)


def sig(x): return 1.0 / (1.0 + np.exp(-x))


def stats(realized, day):
    """trade-mean, day-clustered t-stat, day-block bootstrap CI, half-split means."""
    days = np.unique(day)
    per = [realized[day == d] for d in days]
    dmean = np.array([p.mean() for p in per])             # per-day mean PnL (equal day weight)
    n_d = len(days)
    tstat = float(dmean.mean() / (dmean.std(ddof=1) / np.sqrt(n_d))) if n_d > 1 else 0.0
    # day-block bootstrap on the TRADE-mean (resample days, pool trades)
    boot = np.empty(2000)
    for b in range(2000):
        pick = rng.integers(0, n_d, n_d)
        boot[b] = np.concatenate([per[i] for i in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # half-period stability (split days in time order)
    half = n_d // 2
    h1 = np.concatenate([per[i] for i in range(half)]).mean()
    h2 = np.concatenate([per[i] for i in range(half, n_d)]).mean()
    return {"trade_mean": float(realized.mean()), "n_trades": int(len(realized)), "n_days": n_d,
            "t_day": tstat, "ci95_lo": float(lo), "ci95_hi": float(hi),
            "half1": float(h1), "half2": float(h2)}


def main():
    out = {}
    for sym in SYMS:
        cz = np.load(io.BytesIO(bk.blob(f"{OUTP}/cache/{sym}_cache.npz").download_as_bytes()))
        A = cz["Alog_full"].astype(np.float64); B = cz["Blog_full"].astype(np.float64)
        rH = cz["rH_full"].astype(np.float64); REF = cz["REF"]
        day = REF[:, 0].astype(np.int64); n = len(A)
        side = np.sign(B); side[side == 0] = 1
        realized = side * rH                              # hold-to-60s bp
        signals = {"A": A, "EV": sig(A) * np.abs(2 * sig(B) - 1)}
        print(f"\n========= {sym} (n={n}) =========")
        out[sym] = {}
        for sname, sv in signals.items():
            for q in QS:
                k = max(int(round(n * q / 100.0)), 20)
                sel = np.argpartition(-sv, k)[:k]
                st = stats(realized[sel], day[sel])
                net = st["trade_mean"] - MM
                ci_net = (st["ci95_lo"] - MM, st["ci95_hi"] - MM)
                flag = "EDGE" if ci_net[0] > 0 else ("~" if st["trade_mean"] - MM > 0 else "")
                print(f"  {sname:>3} q{q:<5} net_mm={net:+7.2f} [CI {ci_net[0]:+6.2f},{ci_net[1]:+6.2f}] "
                      f"t_day={st['t_day']:+5.2f} WRdays={st['n_days']:>3} n={st['n_trades']:>5} "
                      f"half1={st['half1']-MM:+6.2f} half2={st['half2']-MM:+6.2f}  {flag}")
                out[sym][f"{sname}_q{q}"] = {**st, "net_mm": net, "ci_net_mm": list(ci_net)}
    bk.blob(f"{OUTP}/select_stability.json").upload_from_string(json.dumps(out, default=float))
    print(f"\n[saved] {OUTP}/select_stability.json")


if __name__ == "__main__":
    main()
