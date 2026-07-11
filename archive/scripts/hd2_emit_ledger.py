#!/usr/bin/env python3
"""Emit HD2 ledger rows from the sweep results: 6 alpha experiment rows
(one per (symbol,H), L-surface in params) + an HD2 result-rev. Appends via
ledger.py (validated). Surface is the headline; delta_ic vs HM6 is a
SECONDARY cross-window annotation (CLAUDE.md rules 1-4)."""
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hd2_aggregate as A
import hd1_seq_core as core

EXP = HERE.parent / "research" / "experiments.jsonl"
HS = (180, 600, 1800)
LS = (6000, 36000, 216000)
SYMS = ("SOL-USDT-PERP", "LTC-USDT-PERP")
RES = sys.argv[1] if len(sys.argv) > 1 else "hd2res/hd2"
GIT = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                     capture_output=True, text=True).stdout.strip()[:12]


def hm6_ref():
    """HM6 (sym,H) -> (best rank_ic, its experiment_id)."""
    best = {}
    for ln in open(EXP, encoding="utf8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        r = json.loads(ln)
        if r.get("hypothesis_id") != "HM6" or r.get("kind") != "alpha":
            continue
        sym = (r.get("symbols") or [None])[0]; H = r.get("horizon_sec")
        ic = r.get("rank_ic_oos")
        if sym and H and ic is not None:
            if (sym, H) not in best or ic > best[(sym, H)][0]:
                best[(sym, H)] = (ic, r["experiment_id"])
    return best


def main():
    res = A.load_results(RES)
    surf = A.build_surface(res)
    hm6 = hm6_ref()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for sym in SYMS:
        for H in HS:
            cells = surf[(sym, H)]
            # argmax-L by all-points seed-mean
            bestL = max(LS, key=lambda L: (cells[L]["all_mean"]
                                           if cells[L]["all_mean"] is not None else -9))
            c = cells[bestL]
            # n + auc for the argmax-L cell (seed 0 unit; n ~ seed-invariant)
            u = res.get((sym, bestL, 0)) or res.get((sym, bestL, 1))
            byh = u["by_H"][str(H)]["all"]
            n_all = byh.get("n", 0)
            blk = core.block_size(H)
            base = hm6.get((sym, H))
            d_ic = (round(c["all_mean"] - base[0], 5)
                    if (base and c["all_mean"] is not None) else None)
            rec = {
                "experiment_id": f"hd2-20260523_{sym}_H{H}",
                "ts": ts, "git_commit": GIT or "uncommitted", "author": "claude(modal)",
                "hypothesis_id": "HD2", "kind": "alpha", "status": "exploratory",
                "setup": f"HD2 Mamba-2 streaming-stateful L-sweep (multitask-H, {sym} H{H})",
                "model_family": "mamba", "data_source": "cryptolake",
                "cache_id": f"hd2_stream_{sym}_500d_2024-12-25_2026-05-08_STRIDE1",
                "symbols": [sym], "n_samples": int(u["n_dp"]),
                "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
                "commission_loss_pct": 0.07, "split_method": "honest_val_test",
                "embargo": "64", "label_def": (
                    "first-passage F_T0=0.0013 up-first dir target; R1 |move|-weighted "
                    "BCE (HM5); streaming-stateful reset-period=L, warmup-floor, score-late"),
                "repro_cmd": ("modal run scripts/hd2_build_modal.py --full && "
                              "modal run scripts/hd2_sweep_modal.py --full && "
                              "python scripts/hd2_aggregate.py --results hd2res/hd2"),
                "alpha_target": "mamba2_stream_dir", "horizon_sec": H,
                "rank_ic_oos": round(c["all_mean"], 5),
                "auc_oos": round(c["auc_mean"], 5) if c["auc_mean"] else None,
                "baseline_ref": base[1] if base else None, "delta_ic": d_ic,
                "cost_floor_pct": 0.13, "n_eff": int(n_all // blk) if n_all else None,
                "params": {
                    "argmax_L": bestL, "seeds": [0, 1, 2], "d_model": 256,
                    "n_layers": 4, "d_state": 128, "token_budget_batch": 576000,
                    "L_surface_all": {str(L): {"mean": _r(cells[L]["all_mean"]),
                                               "seed_sd": _r(cells[L]["all_sd"])} for L in LS},
                    "L_surface_deep": {str(L): {"mean": _r(cells[L]["deep_mean"]),
                                                "seed_sd": _r(cells[L]["deep_sd"])} for L in LS},
                    "context_axis": "ticks {6000:10min,36000:1h,216000:6h} @SOL~103ms",
                },
                "note": (
                    f"HD2 Mamba-2 long-context surface, {sym} H{H}. argmax L={bestL}. "
                    f"delta_ic vs HM6 is CROSS-WINDOW (this=500d STRIDE1 vs HM6=360d "
                    f"STRIDE4) -> secondary, sec5 non-binding (HD2 rev1). "
                    f"Headline = the rank_IC(L) surface + seed-sd, not a verdict."),
            }
            rows.append(rec)
    outdir = Path(RES).parent
    for r in rows:
        p = outdir / (r["experiment_id"].replace("/", "_") + ".row.json")
        json.dump(r, open(p, "w"), ensure_ascii=False, indent=2)
        print("emit", p)
    # quick surface echo for the result-rev
    print("ARGMAX_SUMMARY " + json.dumps(
        {f"{s}|H{H}": {"argmaxL": rows[i]["params"]["argmax_L"],
                       "rank_ic": rows[i]["rank_ic_oos"], "d_ic": rows[i]["delta_ic"]}
         for i, (s, H) in enumerate([(s, H) for s in SYMS for H in HS])}, default=float))


def _r(x):
    return round(x, 5) if (x is not None and not math.isnan(x)) else None


if __name__ == "__main__":
    main()
