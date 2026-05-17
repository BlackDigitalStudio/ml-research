#!/usr/bin/env python3
"""HM6 — STANDARDIZED 4-symbol x 360-day baseline (baseline_ref).

Pre-registered & FROZEN 2026-05-17 (canon HM6; user AskUserQuestion;
HM5 objective standard; HD1 rev19 verified bucket inventory). No
post-hoc DOF.

Reuses the FROZEN ``hr1_screen.run()`` numeric logic VERBATIM (data
load, honest_val_test split, R0-R4 fit, placebo, paired-boot SE).
The ONLY differences vs HR1:
  * symbol set = {LINK, SOL, BTC, ETH}-USDT-PERP (4)
  * a COMMON ALIGNED day-window (same calendar dates for all 4),
    not hr1's per-symbol latest-N
  * record labels (hypothesis_id / experiment_id / setup / repro_cmd)

baseline_ref := R1 (HM5 standard) primary + R0 plain-logloss anchor,
per (symbol, H). DESCRIPTIVE reference, NOT a lever pass/fail.

Window rule (frozen): day-list = intersection of features_v1 dt=
partitions across ALL 4 symbols, filtered >= 2025-05-09 (BTC/ETH
availability bound), last 360. Effective per-symbol N self-reported
(run() skips days missing book/event parquet — no fabrication).

Execution: 96-vCPU VM via scripts/gcp_bootstrap.sh (NOT the ledger
sandbox). Results -> gs://.../research_runs/{run_id}/results.json
-> appended to research/experiments.jsonl -> research.db.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_cryptolake_cache import _gcs_bucket, _list_days  # noqa: E402
from scripts.hr1_screen import run  # noqa: E402  (frozen logic, reused)

SYMS = ["LINK-USDT-PERP", "SOL-USDT-PERP",
        "BTC-USDT-PERP", "ETH-USDT-PERP"]
WIN_START = "2025-05-09"   # BTC/ETH availability floor (HD1 rev19)
N_DAYS = 360
GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"


def _common_days(bk):
    """Common aligned window = dt= intersection across all 4 syms."""
    per = {s: set(_list_days(bk, s)) for s in SYMS}
    pool = sorted(d for d in set.intersection(*per.values())
                  if d >= WIN_START)
    return pool[-N_DAYS:], {s: len(per[s]) for s in SYMS}, len(pool)


def _relabel(rec, run_id):
    """HR1 -> HM6/BASELINE label rewrite; schema preserved."""
    if "experiment_id" not in rec:          # error/underpowered passthrough
        return rec
    rec["experiment_id"] = rec["experiment_id"].replace("_HR1_", "_BASELINE_")
    rec["hypothesis_id"] = "HM6"
    rec["setup"] = rec.get("setup", "").replace(
        "HR1 pre-reg reward/loss sweep R0-R4 XGB",
        "HM6 standardized 4sym x 360d baseline (R1/HM5 primary, R0 anchor)")
    rec["repro_cmd"] = f"python scripts/baseline_360.py --run-id {run_id}"
    if rec.get("cache_id"):
        rec["cache_id"] = rec["cache_id"].replace(
            "features_v1+events_", "features_v1+events_BASELINE360_")
    p = rec.get("params") or {}
    p.update(baseline=True, baseline_window="common-aligned",
             baseline_canon="HM6")
    rec["params"] = p
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--git-commit", default="unknown")
    a = ap.parse_args(argv)

    bk = _gcs_bucket()
    from google.cloud import storage
    sc = storage.Client(project=GCP_PROJECT)

    days, avail, n_pool = _common_days(bk)
    if not days:
        print("FATAL: empty common window (no dt= intersection >= "
              f"{WIN_START})", file=sys.stderr)
        return 2
    print(f"[HM6] common-aligned window {days[0]}..{days[-1]} "
          f"N={len(days)} (intersection pool={n_pool}; "
          f"per-sym features_v1 avail={avail})", flush=True)

    out = {"run_id": a.run_id, "baseline": "HM6",
           "symbols": SYMS, "window": [days[0], days[-1]],
           "n_days": len(days), "intersection_pool": n_pool,
           "per_symbol_available": avail,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "records": []}
    for sym in SYMS:
        try:
            recs = run(bk, sc, sym, days, a.run_id, a.git_commit)
            out["records"] += [_relabel(r, a.run_id) for r in recs]
        except Exception as e:                       # noqa: BLE001
            out["records"].append({"symbol": sym, "error": repr(e),
                                   "trace": traceback.format_exc()})
            print(f"[{sym}] ERROR {e}", flush=True)
        bk.blob(f"research_runs/{a.run_id}/results.json"
                ).upload_from_string(json.dumps(out, indent=2, default=str))
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bk.blob(f"research_runs/{a.run_id}/results.json"
            ).upload_from_string(json.dumps(out, indent=2, default=str))
    print("PHASE_B_DONE", a.run_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
