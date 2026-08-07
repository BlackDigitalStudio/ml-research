#!/usr/bin/env python3
"""Standard result battery for ANY strategy form (user directives 2026-08-06):
  - ROI and maxDD are part of EVERY reported result;
  - research capital convention FRAC=1.0 (full capital per trade; the 0.5 of
    trading_algorithm README 4.1 was an execution-side sizing choice of the small
    live account, NOT a research convention — comparisons to @0.5 rows must halve);
  - validity gate: no negative fold (per-trade EV), no single-period concentration
    (LOFO by fold stays materially positive), day-block BOOT floor > 0;
  - no jitter (standing directive HD3 rev20).

battery(trades) -> dict + one-line formatter. Sequential per-trade compounding —
fair for ~150s holds at <=10-15 tr/day (near-zero overlap); state otherwise if a
form stacks positions.
"""
import numpy as np

FRAC = 1.0


def battery(ordered_nets, ordered_days, fold_nets, days_sorted, boot_reps=1000, block=7):
    """ordered_nets/ordered_days: chronological trade nets (bp) + day index per trade.
    fold_nets: list of per-fold net arrays. days_sorted: ALL test days (incl zero-trade)."""
    a = np.asarray(ordered_nets, np.float64)
    n = len(a)
    span = max(len(days_sorted), 1)
    out = {"n": int(n), "tpd": n / span}
    if not n:
        return out
    out["ev"] = float(a.mean()); out["hit"] = float((a > 0).mean())
    # compounded curve @ FRAC=1.0
    eq = np.cumprod(1.0 + FRAC * a * 1e-4)
    curve = np.concatenate([[1.0], eq])
    out["roi_monthly"] = float(eq[-1] ** (30.0 / span) - 1.0)
    out["maxdd"] = float((1.0 - curve / np.maximum.accumulate(curve)).max())
    by_day = {}
    for net, d in zip(a, ordered_days):
        by_day.setdefault(int(d), []).append(net)
    dret = np.array([float(np.prod([1.0 + FRAC * x * 1e-4 for x in by_day.get(int(d), [])]) - 1.0)
                     for d in days_sorted])
    out["worst_day"] = float(dret.min())
    out["sharpe"] = float(dret.mean() / dret.std() * np.sqrt(365.0)) if dret.std() > 0 else 0.0
    out["roi_annual"] = float((1.0 + out["roi_monthly"]) ** 12 - 1.0)
    downside = dret[dret < 0]
    dstd = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0
    out["sortino"] = float(dret.mean() / dstd * np.sqrt(365.0)) if dstd > 0 else float("inf")
    out["calmar"] = float(out["roi_annual"] / out["maxdd"]) if out["maxdd"] > 0 else float("inf")
    out["month_roi_pct"] = [round(100 * float(np.prod(1.0 + dret[m:m + 30]) - 1.0), 1)
                            for m in range(0, span, 30)]
    # gate
    pf_ev = [float(x.mean()) if len(x) else None for x in fold_nets]
    out["perfold_ev"] = [round(v, 2) if v is not None else None for v in pf_ev]
    out["perfold_n"] = [int(len(x)) for x in fold_nets]
    lofo = []
    nf = len(fold_nets)
    for f in range(nf):
        rest = [fold_nets[g] for g in range(nf) if g != f and len(fold_nets[g])]
        lofo.append(float(np.concatenate(rest).mean()) if rest else float("nan"))
    out["lofo_min"] = float(np.nanmin(lofo)) if lofo else float("nan")
    out["neg_folds"] = int(sum(1 for v in pf_ev if v is not None and v < 0))
    # day-block bootstrap
    rng = np.random.default_rng(1)
    b_ev = []
    for _ in range(boot_reps):
        picked = []
        while len(picked) < span:
            i0 = rng.integers(0, max(span - block, 1))
            picked.extend(days_sorted[i0:i0 + block])
        tr = [x for d in picked[:span] for x in by_day.get(int(d), [])]
        if tr:
            b_ev.append(np.mean(tr))
    b_ev = np.array(b_ev)
    out["boot_p5"] = float(np.quantile(b_ev, .05)); out["boot_p95"] = float(np.quantile(b_ev, .95))
    out["boot_Ppos"] = float(100 * np.mean(b_ev > 0))
    out["gate_pass"] = bool(out["neg_folds"] == 0 and out["lofo_min"] > 0.25 * out["ev"] and out["boot_p5"] > 0)
    return out


def fmt(name, m):
    if not m.get("n"):
        return f"{name:38s} EMPTY"
    g = "PASS" if m.get("gate_pass") else "fail"
    return (f"{name:38s} EV {m['ev']:+6.2f} n={m['n']:5d} {m['tpd']:5.2f}/d hit {100*m['hit']:4.1f}% | "
            f"ROI/mo {100*m['roi_monthly']:+6.1f}% maxDD {-100*m['maxdd']:5.1f}% worst-d {100*m['worst_day']:+5.1f}% "
            f"Sh {m['sharpe']:4.1f} | LOFOmin {m['lofo_min']:+5.1f} negF {m['neg_folds']} "
            f"BOOT[{m['boot_p5']:+5.1f},{m['boot_p95']:+5.1f}] P{m['boot_Ppos']:3.0f} | {g}")
