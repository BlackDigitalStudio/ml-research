#!/usr/bin/env python3
"""HD2 sweep aggregator: 18 per-unit result JSONs -> conditional alpha surface
(rank_IC vs L per (symbol,H) cell, seed mean +- sd) + ledger rows.

PRIMARY deliverable (CLAUDE.md rule 1): the surface + argmax-L + seed-stability,
reported FIRST. delta_ic vs HM6 baseline_ref is a SECONDARY, cross-window
(500d STRIDE=1 vs HM6 360d STRIDE=4) annotation only -- never the framing; the
sec5 gate is non-binding for this tier.

  python scripts/hd2_aggregate.py --results <dir> [--emit <outdir>]
"""
import argparse
import glob
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_JSONL = HERE.parent / "research" / "experiments.jsonl"
HS = (180, 600, 1800)
LS = (6000, 36000, 216000)
SYMS = ("SOL-USDT-PERP", "LTC-USDT-PERP")


def _mean_sd(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def hm6_baseline():
    """HM6 R1 rank_ic per (symbol, H) from the ledger (best abs per cell)."""
    base = {}
    for ln in open(EXP_JSONL, encoding="utf8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        r = json.loads(ln)
        if r.get("hypothesis_id") != "HM6" or r.get("kind") != "alpha":
            continue
        sym = (r.get("symbols") or [None])[0]
        H = r.get("horizon_sec")
        ic = r.get("rank_ic_oos")
        if sym and H and ic is not None:
            base.setdefault((sym, H), []).append(ic)
    # HM6 has duplicate rows per cell; take the max rank_ic as the reference
    return {k: max(v) for k, v in base.items()}


def load_results(rdir):
    res = {}
    for p in glob.glob(str(Path(rdir) / "*.json")):
        r = json.load(open(p, encoding="utf8"))
        res[(r["symbol"], r["L"], r["seed"])] = r
    return res


def build_surface(res):
    """surface[(sym,H)][L] = {all_mean, all_sd, deep_mean, deep_sd, auc_mean, seeds:[...]}"""
    surf = defaultdict(dict)
    for sym in SYMS:
        for H in HS:
            for L in LS:
                allic, deepic, aucs = [], [], []
                for seed in (0, 1, 2):
                    r = res.get((sym, L, seed))
                    if not r:
                        continue
                    byh = r["by_H"][str(H)] if str(H) in r["by_H"] else r["by_H"].get(H, {})
                    allic.append(byh.get("all", {}).get("rank_ic"))
                    deepic.append(byh.get("deep", {}).get("rank_ic"))
                    aucs.append(byh.get("all", {}).get("auc"))
                am, asd = _mean_sd(allic)
                dm, dsd = _mean_sd(deepic)
                aucm, _ = _mean_sd(aucs)
                surf[(sym, H)][L] = {"all_mean": am, "all_sd": asd,
                                     "deep_mean": dm, "deep_sd": dsd,
                                     "auc_mean": aucm, "n_seeds": len([x for x in allic if x is not None])}
    return surf


def print_surface(surf, base):
    print("\n=== HD2 Mamba-2 conditional alpha surface: rank_IC(L) per (symbol,H) ===")
    print(f"{'cell':22s}|{'L=6000':>16s}|{'L=36000':>16s}|{'L=216000':>16s}|"
          f"{'argmax-L':>10s}|{'HM6base':>9s}|{'best d_ic':>10s}")
    print("-" * 110)
    for sym in SYMS:
        for H in HS:
            cells = surf[(sym, H)]
            row = f"{sym.split('-')[0]+' H'+str(H):22s}"
            best_L, best_m = None, -9
            for L in LS:
                c = cells[L]
                m, sd = c["all_mean"], c["all_sd"]
                row += f"|{(f'{m:+.4f}±{sd:.3f}' if m is not None else '-'):>16s}"
                if m is not None and m > best_m:
                    best_m, best_L = m, L
            b = base.get((sym, H))
            dic = (best_m - b) if (b is not None and best_m > -9) else None
            row += f"|{('L='+str(best_L)):>10s}|{(f'{b:+.4f}' if b else 'n/a'):>9s}"
            row += f"|{(f'{dic:+.4f}' if dic is not None else 'n/a'):>10s}"
            print(row)
    print("\n(deep-context subset, context>=0.75L):")
    for sym in SYMS:
        for H in HS:
            cells = surf[(sym, H)]
            row = f"{sym.split('-')[0]+' H'+str(H):22s}"
            for L in LS:
                c = cells[L]
                m, sd = c["deep_mean"], c["deep_sd"]
                row += f"|{(f'{m:+.4f}±{sd:.3f}' if m is not None else '-'):>16s}"
            print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--emit", default="")
    a = ap.parse_args()
    res = load_results(a.results)
    print(f"loaded {len(res)} unit results")
    base = hm6_baseline()
    surf = build_surface(res)
    print_surface(surf, base)
    # full surface dump for the result-rev / ledger emit
    dump = {f"{sym}|H{H}": {str(L): surf[(sym, H)][L] for L in LS}
            for sym in SYMS for H in HS}
    if a.emit:
        Path(a.emit).mkdir(parents=True, exist_ok=True)
        json.dump({"surface": dump, "hm6_base": {f"{k[0]}|H{k[1]}": v
                                                  for k, v in base.items()}},
                  open(Path(a.emit) / "hd2_surface.json", "w"), indent=2, default=float)
        print(f"\nwrote {a.emit}/hd2_surface.json")
