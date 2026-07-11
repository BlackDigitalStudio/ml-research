#!/usr/bin/env python3
"""HD2 rev5 objective-screen aggregator: rank the ~11 target-forms by mean OOS
rank_IC over the 6 (sym,H) cells (seed0) + economic-grounding flag. The winner
(best seed-stable rank_IC among econ-grounded forms) is frozen as the HD2 target
for the loss/weighting round.

ECON-GROUNDED = the form's positive class requires a >= cost (0.13%) move (so a
high rank_IC is on economically-meaningful direction, not sub-cost noise).

  python scripts/hd2_obj_aggregate.py --results <dir>
"""
import argparse
import glob
import json
from collections import defaultdict

HS = ("180", "600", "1800")
SYMS = ("SOL-USDT-PERP", "LTC-USDT-PERP")
# does the positive class require a >=0.13% (cost-floor) move?
ECON_GROUNDED = {
    "fp_0.08": False, "fp_0.13": True, "fp_0.20": True, "fp_0.30": True,
    "fp_0.40": True, "fp_0.50": True, "fp_0.75": True,
    "fp_asym_up": True, "fp_asym_dn": True,        # both barriers >= 0.13
    "fp_vol_1.0": None, "fp_vol_1.5": None,        # adaptive -> inspect
    "fp_sqrt_0.13": True, "fp_sqrt_0.20": True,    # f0>=0.13 grows with sqrt(H)
    "terminal": False, "deadband_0.13": True, "volnorm": False,
}
# symmetric barrier width (% as fraction) -> breakeven hit-rate p*=(1+c/f)/2.
# Economic selection: among forms with rank_IC within seed-noise of the best,
# prefer the WIDEST barrier (lowest p*) -> more profit per correct call at fixed
# cost (user 2026-05-23). rank_IC alone is skill-only / payoff-blind.
COST_RT = 0.0011  # maker round-trip ~0.11%
BARRIER_F = {"fp_0.08": 0.0008, "fp_0.13": 0.0013, "fp_0.20": 0.0020,
             "fp_0.30": 0.0030, "fp_0.40": 0.0040, "fp_0.50": 0.0050,
             "fp_0.75": 0.0075}


def breakeven_p(t):
    f = BARRIER_F.get(t)
    return (1 + COST_RT / f) / 2 if f else None
# SELECTABLE for the winner. Exclude only the INFLATING asymmetry: fp_asym_up
# (+0.13/-0.20) puts the NEAR barrier on the predicted-positive side -> "up"
# registers on tiny moves -> rank_IC inflated by labeling skew, not real
# directional alpha. fp_asym_dn (+0.20/-0.13) is the OPPOSITE: "up" needs a
# LARGE move past a tight stop -> conservative + maps to the real deploy R:R
# ~1.5:1 -> kept (user 2026-05-23). Symmetric forms all selectable.
SELECTABLE = {
    "fp_0.08": True, "fp_0.13": True, "fp_0.20": True, "fp_0.30": True,
    "fp_0.40": True, "fp_0.50": True, "fp_0.75": True,
    "fp_asym_up": False, "fp_asym_dn": True,
    "fp_vol_1.0": True, "fp_vol_1.5": True,
    "fp_sqrt_0.13": True, "fp_sqrt_0.20": True,
    "terminal": True, "deadband_0.13": True, "volnorm": True,
}


def load(rdir):
    by = defaultdict(dict)   # target -> (sym) -> result
    for p in glob.glob(f"{rdir}/*.json"):
        r = json.load(open(p, encoding="utf8"))
        t = r.get("target_name") or r.get("flat", {}).get("target")
        sym = r.get("symbol") or r.get("flat", {}).get("sym")
        if t and sym:
            by[t][sym] = r
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    by = load(a.results)
    rows = []
    for t, per in by.items():
        ics, reached = [], []
        for sym in SYMS:
            r = per.get(sym)
            if not r:
                continue
            ntr = r.get("n_tr", 0); ndp = r.get("n_dp", 1)
            for H in HS:
                cell = r["by_H"].get(H, {}).get("all", {})
                ic = cell.get("rank_ic")
                if ic is not None:
                    ics.append(ic)
                n = cell.get("n", 0)
                reached.append(n / max(1, ndp - ntr))   # test reached-rate
        if not ics:
            continue
        mean_ic = sum(ics) / len(ics)
        rows.append({"target": t, "mean_rank_ic": mean_ic, "n_cells": len(ics),
                     "reached_rate": sum(reached) / max(1, len(reached)),
                     "econ": ECON_GROUNDED.get(t),
                     "selectable": SELECTABLE.get(t, True)})
    rows.sort(key=lambda x: -x["mean_rank_ic"])
    print(f"\n=== HD2 rev5 barrier/target-form screen (seed0, L=216000) ===")
    print(f"{'target':16s}|{'mean_rankIC':>12s}|{'reached':>9s}|{'cells':>6s}|"
          f"{'econ':>5s}|select")
    print("-" * 70)
    for r in rows:
        eg = {True: "yes", False: "no", None: "adapt"}[r["econ"]]
        sel = "" if r["selectable"] else " EXCL(inflating-asym)"
        print(f"{r['target']:16s}|{r['mean_rank_ic']:>+12.5f}|"
              f"{r['reached_rate']:>9.3f}|{r['n_cells']:>6d}|{eg:>5s}|"
              f"{r['selectable']}{sel}")
    # winner among econ-grounded AND selectable (symmetric or conservative-asym)
    grounded = [r for r in rows if r["econ"] is True and r["selectable"]]
    if grounded:
        w = grounded[0]
        ctrl = next((r for r in rows if r["target"] == "fp_0.13"), None)
        print(f"\nWINNER (econ-grounded, best mean rank_IC): {w['target']} "
              f"({w['mean_rank_ic']:+.5f})")
        if ctrl:
            print(f"vs control fp_0.13 ({ctrl['mean_rank_ic']:+.5f}): "
                  f"delta {w['mean_rank_ic']-ctrl['mean_rank_ic']:+.5f}")
        print("-> confirm this form on 3 seeds, then freeze for the loss round.")


if __name__ == "__main__":
    main()
