#!/usr/bin/env python3
"""HM6 — STANDARDIZED 4-symbol x 360-day baseline (baseline_ref).

Pre-registered & FROZEN 2026-05-17 (canon HM6; user AskUserQuestion;
HM5 objective standard; HD1 rev19 verified bucket inventory). No
post-hoc DOF.

Reuses the FROZEN ``hr1_screen.run()`` numeric logic VERBATIM (data
load, honest_val_test split, R0-R4 fit, placebo, paired-boot SE).
The ONLY differences vs HR1:
  * symbol set = {LINK, SOL, BTC, ETH}-USDT-PERP (4)
  * a COMMON ALIGNED calendar range (same [start,end] for all 4),
    not hr1's per-symbol latest-N
  * record labels (hypothesis_id / experiment_id / setup / repro_cmd)

baseline_ref := R1 (HM5 standard) primary + R0 plain-logloss anchor,
per (symbol, H). DESCRIPTIVE reference, NOT a lever pass/fail.

Symbol set (canon HM6 rev3 — user decision): LINK-USDT-PERP DROPPED
and replaced by LTC-USDT-PERP. LINK has a genuine, non-recoverable
~119-day source outage (2025-12-11..2026-04-10, absent in raw AND
features_v1) — unfit for a canonical baseline_ref. LTC chosen:
verified ZERO-gap full coverage in-window across features_v1 / book /
trades, AND it is the deepest-history symbol in the bucket
(features 1243d, raw/trades ~4.4yr) — a built-in long-window
extension path for the sequence build (HD1 rev19 "depth lives in
LTC/LINK"). All 4 of {SOL,BTC,ETH,LTC} are complete in-window.

Window rule (canon HM6 rev2): common ALIGNED CALENDAR range, NOT a
strict dt= intersection. start = max(symbol mins, floor 2025-05-09);
end = min(symbol maxes); take the last N_DAYS=360 CALENDAR days; give
EACH symbol its OWN available partitions in that range (run() skips
days a symbol lacks — defensive; with this set the per-symbol N is
~uniform ~360, no fabrication).

Execution: 96-vCPU VM via scripts/gcp_bootstrap.sh (NOT the ledger
sandbox). Results -> gs://.../research_runs/{run_id}/results.json
-> appended to research/experiments.jsonl -> research.db.
"""
import argparse
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_cryptolake_cache import _gcs_bucket, _list_days  # noqa: E402
from scripts.hr1_screen import run  # noqa: E402  (frozen logic, reused)

SYMS = ["SOL-USDT-PERP", "BTC-USDT-PERP",
        "ETH-USDT-PERP", "LTC-USDT-PERP"]   # LINK dropped (HM6 rev3)
FLOOR = "2025-05-09"       # BTC/ETH availability floor (HD1 rev19)
N_DAYS = 360
GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"


def _window(bk):
    """Common ALIGNED CALENDAR range; per-symbol available days.

    NOT a strict dt= intersection (canon HM6 rev2): start =
    max(symbol mins, FLOOR); end = min(symbol maxes); last N_DAYS
    calendar days; each symbol uses its OWN partitions in [winlo,end]
    (run() skips days it lacks). LINK's genuine 119-day outage stays
    LINK-only instead of truncating SOL/BTC/ETH.
    """
    per = {s: _list_days(bk, s) for s in SYMS}          # sorted lists
    if not all(per.values()):
        return None, None, {}, {s: len(per[s]) for s in SYMS}
    start = max([FLOOR] + [d[0] for d in per.values()])
    end = min(d[-1] for d in per.values())
    lo_cal = (dt.date.fromisoformat(end)
              - dt.timedelta(days=N_DAYS - 1)).isoformat()
    winlo = max(start, lo_cal)
    psd = {s: [d for d in per[s] if winlo <= d <= end] for s in SYMS}
    return winlo, end, psd, {s: len(per[s]) for s in SYMS}


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
    p.update(baseline=True,
             baseline_window="common-calendar-range",
             baseline_canon="HM6-rev2")
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

    winlo, winhi, psd, avail = _window(bk)
    if not winlo or not any(psd.values()):
        print("FATAL: empty common calendar window", file=sys.stderr)
        return 2
    n_by = {s: len(psd[s]) for s in SYMS}
    print(f"[HM6] aligned calendar window {winlo}..{winhi} "
          f"(<= {N_DAYS} cal days); per-symbol available days={n_by}; "
          f"full-history counts={avail}", flush=True)

    out = {"run_id": a.run_id, "baseline": "HM6",
           "symbol_set": SYMS, "window": [winlo, winhi],
           "n_days_per_symbol": n_by,
           "per_symbol_full_available": avail,
           "window_rule": ("common-calendar-range; per-symbol "
                           "available days; canon HM6 rev2 "
                           "(NOT a strict dt= intersection)"),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "records": []}
    for sym in SYMS:
        try:
            recs = run(bk, sc, sym, psd[sym], a.run_id, a.git_commit)
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
