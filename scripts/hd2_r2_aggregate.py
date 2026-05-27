#!/usr/bin/env python3
"""HD2 R2 (loss/objective) aggregator: rank the loss variants by mean OOS rank-IC
(vs the continuous unbounded r_H, rev8) over the 6 (sym,H) cells, seed0, with the
economic annotation (top-decile |move| vs 0.13% cost floor). Winner = best
seed-stable mean rank-IC -> confirm 3 seeds -> freeze as the HD2 objective.

  python scripts/hd2_r2_aggregate.py --results <dir>
"""
import argparse
import glob
import json
from collections import defaultdict

HS = ("180", "600", "1800")
SYMS = ("SOL-USDT-PERP", "LTC-USDT-PERP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    by = defaultdict(dict)            # loss -> sym -> result
    for p in glob.glob(f"{a.results}/*.json"):
        r = json.load(open(p, encoding="utf8"))
        loss = r.get("loss_name") or r.get("flat", {}).get("loss")
        sym = r.get("symbol") or r.get("flat", {}).get("sym")
        if loss and sym:
            by[loss][sym] = r
    rows = []
    for loss, per in by.items():
        ics, caps, capg, econ, moves = [], [], [], [], []
        for sym in SYMS:
            r = per.get(sym)
            if not r:
                continue
            for H in HS:
                c = r["by_H"].get(H, {}).get("all", {})
                if c.get("rank_ic") is not None:
                    ics.append(c["rank_ic"])
                if c.get("cap_edge_net") is not None:
                    caps.append(c["cap_edge_net"])
                if c.get("cap_edge_gross") is not None:
                    capg.append(c["cap_edge_gross"])
                if c.get("econ_pass") is not None:
                    econ.append(c["econ_pass"])
                if c.get("top_decile_absmove") is not None:
                    moves.append(c["top_decile_absmove"])
        if not ics:
            continue
        rows.append({"loss": loss, "mean_ric": sum(ics) / len(ics), "n_cells": len(ics),
                     "cap_net": (sum(caps) / len(caps)) if caps else None,
                     "cap_gross": (sum(capg) / len(capg)) if capg else None,
                     "econ_frac": (sum(econ) / len(econ)) if econ else None,
                     "top_move": (sum(moves) / len(moves)) if moves else None})
    # WINNER CRITERION = mean cap_edge_net (signed captured return in the top-
    # confidence decile, net of 0.13% cost): the magnitude-aware economic edge
    # (user 2026-05-24). rank_IC is a secondary direction-skill diagnostic only.
    # Falls back to rank_IC if cap_edge absent (pre-rescore *.json -> run --rescore).
    have_cap = any(r["cap_net"] is not None for r in rows)
    rows.sort(key=(lambda x: -(x["cap_net"] if x["cap_net"] is not None else -9.0))
              if have_cap else (lambda x: -x["mean_ric"]))
    crit = "cap_edge_net" if have_cap else "rank_IC (NO cap_edge -- run --rescore!)"
    print(f"\n=== HD2 R2 loss screen (continuous r_H, seed0, L=216000) -- WINNER BY {crit} ===")
    print(f"{'loss':14s}|{'cap_net%':>9s}|{'capgrs%':>9s}|{'rank_IC':>9s}|"
          f"{'cells':>6s}|{'econ':>5s}|{'top|mv|%':>9s}")
    print("-" * 70)
    for r in rows:
        cn = f"{r['cap_net']*100:+.4f}" if r["cap_net"] is not None else "-"
        cg = f"{r['cap_gross']*100:+.4f}" if r["cap_gross"] is not None else "-"
        em = f"{r['econ_frac']:.2f}" if r["econ_frac"] is not None else "-"
        tm = f"{r['top_move']*100:.3f}" if r["top_move"] is not None else "-"
        print(f"{r['loss']:14s}|{cn:>9s}|{cg:>9s}|{r['mean_ric']:>+9.5f}|"
              f"{r['n_cells']:>6d}|{em:>5s}|{tm:>9s}")
    if rows:
        w = rows[0]
        r1 = next((x for x in rows if x["loss"] == "R1"), None)
        print(f"\nWINNER: {w['loss']} (cap_net={w['cap_net']}, rank_IC={w['mean_ric']:+.5f})")
        if r1 and have_cap and r1["cap_net"] is not None and w["cap_net"] is not None:
            print(f"vs R1 (cap_net={r1['cap_net']:+.5f}): delta {w['cap_net']-r1['cap_net']:+.5f}")
        print("-> confirm on 3 seeds; freeze if it beats R1 beyond seed-sd, else keep R1.")


if __name__ == "__main__":
    main()
