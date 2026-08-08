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
    # day-block bootstrap (paths kept for the DD-25 robust sizing below)
    rng = np.random.default_rng(1)
    b_ev, b_paths = [], []
    for _ in range(boot_reps):
        picked = []
        while len(picked) < span:
            i0 = rng.integers(0, max(span - block, 1))
            picked.extend(days_sorted[i0:i0 + block])
        seq = np.array([x for d in picked[:span] for x in by_day.get(int(d), [])])
        b_paths.append(seq)
        if len(seq):
            b_ev.append(float(seq.mean()))
    b_ev = np.array(b_ev)
    out["boot_p5"] = float(np.quantile(b_ev, .05)); out["boot_p95"] = float(np.quantile(b_ev, .95))
    out["boot_Ppos"] = float(100 * np.mean(b_ev > 0))
    # SAMPLE-SIZE GATE (standing directive 2026-08-08): the anecdote floor is part
    # of the gate — n>=100 AND trades in >=5 folds (of 7; scaled if nf differs).
    # Below it the bootstrap/LOFO machinery itself is unreliable and floor>0
    # carries no evidence. power80_ok is a MARKER, not a gate: bootstrap
    # SE <= EV/2.49 means a TRUE edge of this size is confirmable with 80% power
    # (n_min ~ (2.49*c/EV)^2, c = SE*sqrt(n) ~ 273bp median on measured cells) —
    # thinner edges need quadratically more trades.
    out["folds_traded"] = int(sum(1 for v in pf_ev if v is not None))
    out["active_days"] = int(len(by_day))
    out["n_ok"] = bool(out["n"] >= 100 and out["folds_traded"] >= min(5, nf))
    se = (out["boot_p95"] - out["boot_p5"]) / (2 * 1.645)
    out["boot_se"] = float(se)
    out["power80_ok"] = bool(se > 0 and out["ev"] / se >= 2.49)
    out["gate_pass"] = bool(out["neg_folds"] == 0 and out["lofo_min"] > 0.25 * out["ev"]
                            and out["boot_p5"] > 0 and out["n_ok"])
    # DD-25 NORMALIZATION (standing directive 2026-08-08: variants are COMPARED
    # and recorded at a COMMON maxDD=25% capital sizing; raw FRAC=1.0 numbers
    # above remain the measurement layer, not the comparison layer).
    # F10 = robust size: P(maxDD>25%)<=10% over the bootstrap paths (rev13b method).
    def _dd(path, F):
        if not len(path):
            return 0.0
        c = np.cumprod(1.0 + F * path * 1e-4)
        c = np.concatenate([[1.0], c])
        return float((1.0 - c / np.maximum.accumulate(c)).max())
    # hi capped at 25x: beyond that leverage the number is non-operational
    # (tiny-n forms whose DD never reaches 25% would otherwise clamp at the
    # search bound and explode ROI25); f10_capped flags those cells.
    lo, hi = 0.02, 25.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if float(np.mean([_dd(p_, mid) > 0.25 for p_ in b_paths])) <= 0.10:
            lo = mid
        else:
            hi = mid
    F10 = 0.5 * (lo + hi)
    out["F10_dd25"] = F10
    out["f10_capped"] = bool(F10 > 24.5)
    eqF = np.cumprod(1.0 + F10 * a * 1e-4)
    out["roi25_monthly"] = float(eqF[-1] ** (30.0 / span) - 1.0) if len(eqF) else 0.0
    rois = [float(np.cumprod(1.0 + F10 * p_ * 1e-4)[-1] ** (30.0 / span) - 1.0) if len(p_) else 0.0
            for p_ in b_paths]
    out["roi25_p10"] = float(np.quantile(rois, .10))
    return out


def fmt(name, m):
    if not m.get("n"):
        return f"{name:38s} EMPTY"
    g = "PASS" if m.get("gate_pass") else "fail"
    if m.get("gate_pass"):
        g += "/p80" if m.get("power80_ok") else "/upow"
    return (f"{name:38s} EV {m['ev']:+6.2f} n={m['n']:5d} {m['tpd']:5.2f}/d hit {100*m['hit']:4.1f}% | "
            f"ROI25 {100*m.get('roi25_monthly',0):+6.1f}%@F{m.get('F10_dd25',0):4.1f} p10 {100*m.get('roi25_p10',0):+5.1f}% | "
            f"raw {100*m['roi_monthly']:+5.1f}% DD {-100*m['maxdd']:4.1f}% Sh {m['sharpe']:4.1f} | "
            f"LOFOmin {m['lofo_min']:+5.1f} negF {m['neg_folds']} "
            f"BOOT[{m['boot_p5']:+5.1f},{m['boot_p95']:+5.1f}] P{m['boot_Ppos']:3.0f} | {g}")
