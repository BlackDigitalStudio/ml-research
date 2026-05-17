#!/usr/bin/env python3
"""scalper-bot research ledger — stdlib-only, zero-dependency.

The append-only JSONL files are the source of truth; research.db is a
derived SQLite index. This module is the *gate*: it refuses to record a
result that lacks the provenance which made the last three false-positive
findings expensive to unwind (TAKER<->MAKER fee confusion, lost caches,
mixed split methods).

Subcommands
-----------
  validate                 lint every JSONL record against the contract
  append <jsonfile|->      validate one record + append to the right ledger
  build-db [--db PATH]     materialise research.db from the JSONL ledgers
  frontier                 regenerate the RESEARCH_LOG section-3 table (stdout)
  check                    CI gate: validate + invariants, non-zero on fail

Files (all relative to this directory):
  experiments.jsonl   append-only, one experiment per line
  hypotheses.jsonl    append-only event log, one hypothesis revision per line
  schema.sql          canonical DDL (research.db is rebuilt from it)
  research.db         derived, gitignored, never hand-edited
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_JSONL = HERE / "experiments.jsonl"
HYP_JSONL = HERE / "hypotheses.jsonl"
SCHEMA_SQL = HERE / "schema.sql"
DEFAULT_DB = HERE / "research.db"

# ---------------------------------------------------------------------------
# Contract — kept in lockstep with schema.sql and the live code constants.
# ---------------------------------------------------------------------------

FEE_REGIMES = ("TAKER", "MAKER_FIRST", "MAKER")
DATA_SOURCES = ("v3_btc", "cryptolake", "recorder", "volaware", "other")
SPLIT_METHODS = ("CPCV_6_2", "walkforward_7525", "honest_val_test", "other")
EXP_STATUS = ("confirmed", "suspect", "refuted", "artifact", "exploratory")
HYP_STATUS = ("active", "testing", "confirmed", "refuted", "blocked",
              "superseded")

# live_sim.TradeOutcome.REASONS — the 12 canonical exit reasons.
EXIT_REASONS = (
    "tp_hit", "sl_hit", "trailing_sl_1", "trailing_sl_2",
    "partial_plus_tp", "partial_plus_trailing_sl_1",
    "partial_plus_trailing_sl_2", "timeout_limit", "timeout_market",
    "fast_fill_adverse", "fast_fill_sl", "no_forward_data",
)

# Mandatory experiment fields. Each maps to a documented, expensive lesson.
# Provenance is mandatory for BOTH kinds (a number without its data
# origin / fee regime / split is noise — the chaos this ledger closes).
COMMON_REQUIRED = (
    "experiment_id", "ts", "git_commit", "author", "status",
    "setup", "model_family", "params", "data_source", "cache_id",
    "symbols", "n_samples", "fee_regime", "commission_win_pct",
    "commission_loss_pct", "split_method", "label_def", "repro_cmd",
)
STRATEGY_REQUIRED = ("n_trades",)              # EV/owner-metric runs
ALPHA_REQUIRED = ("alpha_target", "horizon_sec", "rank_ic_oos",
                  "cost_floor_pct", "n_eff")   # prediction-only screens
HYP_REQUIRED = ("hypothesis_id", "rev", "ts", "statement", "status")

OWNER_SHARE_KEYS = ("pct_full_tp", "pct_full_sl", "pct_timeout",
                    "pct_trailing", "pct_partial_only")

# Columns materialised into SQLite (JSON sub-objects flattened to *_json).
EXP_COLUMNS = (
    "experiment_id", "ts", "git_commit", "author", "hypothesis_id",
    "status", "supersedes", "note", "setup", "model_family", "params_json",
    "data_source", "cache_id", "symbols_json", "date_range_start",
    "date_range_end", "n_samples", "label_horizon_ticks", "fee_regime",
    "commission_win_pct", "commission_loss_pct", "split_method", "embargo",
    "purge", "gap_ticks", "label_def", "pct_full_tp", "pct_full_sl",
    "pct_timeout", "pct_trailing", "pct_partial_only", "pnl_gross_pct",
    "pnl_gross_usd", "pnl_net_pct", "pnl_net_usd", "ev_per_trade_pct",
    "trades_per_day", "net_return_pct", "kelly_frac", "win_rate_pct",
    "base_rate_pct", "n_trades", "sharpe", "max_dd_pct", "exit_hist_json",
    "artifact_path", "artifact_sha256", "repro_cmd",
    "kind", "alpha_target", "horizon_sec", "rank_ic_oos", "r2_oos",
    "auc_oos", "top_decile_absmove_pct", "bot_decile_absmove_pct",
    "cost_floor_pct", "decile_monotonic", "economic_pass", "n_eff",
)
HYP_COLUMNS = (
    "hypothesis_id", "rev", "ts", "statement", "expected_lift",
    "compute_cost_usd", "compute_time", "prerequisites", "priority_rank",
    "status", "result_experiment_id", "note",
)


class LedgerError(ValueError):
    """A record violates the contract. The message is the audit reason."""


# ---------------------------------------------------------------------------
# Validation — the gate
# ---------------------------------------------------------------------------

def validate_experiment(r: dict) -> None:
    eid = r.get("experiment_id", "<no-id>")
    kind = r.get("kind") or "strategy"
    if kind not in ("alpha", "strategy"):
        raise LedgerError(f"{eid}: kind must be 'alpha' or 'strategy'")
    req = COMMON_REQUIRED + (ALPHA_REQUIRED if kind == "alpha"
                             else STRATEGY_REQUIRED)
    miss = [k for k in req if r.get(k) in (None, "", [], {})]
    if miss:
        raise LedgerError(f"{eid}: missing mandatory fields {miss} "
                          f"(provenance is non-optional; kind={kind})")
    if r["fee_regime"] not in FEE_REGIMES:
        raise LedgerError(f"{eid}: fee_regime={r['fee_regime']!r} not in "
                          f"{FEE_REGIMES} — TAKER/MAKER must be explicit")
    if r["data_source"] not in DATA_SOURCES:
        raise LedgerError(f"{eid}: data_source not in {DATA_SOURCES}")
    if r["split_method"] not in SPLIT_METHODS:
        raise LedgerError(f"{eid}: split_method not in {SPLIT_METHODS}")
    if r["status"] not in EXP_STATUS:
        raise LedgerError(f"{eid}: status not in {EXP_STATUS}")
    if not isinstance(r["symbols"], list) or not r["symbols"]:
        raise LedgerError(f"{eid}: symbols must be a non-empty list")
    if not isinstance(r["params"], dict):
        raise LedgerError(f"{eid}: params must be an object")

    if kind == "alpha":
        # The alpha killer rule: an RL agent converts existing alpha, it
        # cannot create it and cannot beat the fee/spread floor. A
        # 'confirmed' alpha that does not clear the cost floor is the
        # "significant but economically worthless" trap.
        if r["status"] == "confirmed" and r.get("economic_pass") != 1:
            raise LedgerError(
                f"{eid}: refusing to confirm an alpha with "
                f"economic_pass!=1 — statistically-significant but "
                f"sub-cost signal is not harvestable. Use 'exploratory' "
                f"or show top-decile |move| > cost_floor.")
        return  # alpha rows skip the strategy-only EV/owner/exit checks

    # The killer rule: a positive EV under TAKER labels was a false
    # positive all 3 prior times. The ledger will not store such a row as
    # 'confirmed' — it must be 'suspect' until revalidated MAKER-first.
    ev = r.get("ev_per_trade_pct")
    if (r["status"] == "confirmed" and ev is not None and ev > 0
            and r["fee_regime"] == "TAKER"):
        raise LedgerError(
            f"{eid}: refusing to store a positive-EV TAKER result as "
            f"'confirmed'. This exact shape was wrong 3x (DOGE/ETH/phase56). "
            f"Set status='suspect' and add a MAKER_FIRST revalidation, or "
            f"correct fee_regime.")

    # Owner shares, when present, are mutually exclusive and sum to ~1.
    shares = [r.get(k) for k in OWNER_SHARE_KEYS]
    if all(s is not None for s in shares):
        tot = sum(shares)
        if not (0.98 <= tot <= 1.02):
            raise LedgerError(f"{eid}: owner exit-shares sum to {tot:.4f}, "
                              f"must be ~1.0 (categories are exclusive)")

    # Fees cannot add PnL.
    g, n = r.get("pnl_gross_pct"), r.get("pnl_net_pct")
    if g is not None and n is not None and abs(g) + 1e-9 < abs(n):
        raise LedgerError(f"{eid}: |gross|<|net| ({g} vs {n}) — fees cannot "
                          f"add PnL")

    if r.get("exit_hist"):
        bad = set(r["exit_hist"]) - set(EXIT_REASONS)
        if bad:
            raise LedgerError(f"{eid}: unknown exit reasons {bad} "
                              f"(must be subset of live_sim REASONS)")


def validate_hypothesis(r: dict) -> None:
    hid = r.get("hypothesis_id", "<no-id>")
    miss = [k for k in HYP_REQUIRED if r.get(k) in (None, "")]
    if miss:
        raise LedgerError(f"{hid}: missing mandatory fields {miss}")
    if r["status"] not in HYP_STATUS:
        raise LedgerError(f"{hid}: status not in {HYP_STATUS}")
    if not isinstance(r["rev"], int) or r["rev"] < 1:
        raise LedgerError(f"{hid}: rev must be a positive int")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    for ln, raw in enumerate(path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            yield ln, json.loads(raw)
        except json.JSONDecodeError as e:
            raise LedgerError(f"{path.name}:{ln}: invalid JSON — {e}")


def cmd_validate(_a) -> int:
    n_e = n_h = 0
    for _, r in _iter_jsonl(EXP_JSONL):
        validate_experiment(r)
        n_e += 1
    for _, r in _iter_jsonl(HYP_JSONL):
        validate_hypothesis(r)
        n_h += 1
    # Referential + lineage sanity across the whole ledger.
    exp_ids = {r["experiment_id"] for _, r in _iter_jsonl(EXP_JSONL)}
    hyp_ids = {r["hypothesis_id"] for _, r in _iter_jsonl(HYP_JSONL)}
    for _, r in _iter_jsonl(EXP_JSONL):
        sup = r.get("supersedes")
        if sup and sup not in exp_ids:
            raise LedgerError(f"{r['experiment_id']}: supersedes unknown "
                              f"experiment {sup!r}")
        hid = r.get("hypothesis_id")
        if hid and hid not in hyp_ids:
            raise LedgerError(f"{r['experiment_id']}: hypothesis_id {hid!r} "
                              f"not in hypotheses ledger")
    print(f"OK: {n_e} experiments, {n_h} hypothesis revisions valid.")
    return 0


# ---------------------------------------------------------------------------
# Append — the only sanctioned writer
# ---------------------------------------------------------------------------

def cmd_append(a) -> int:
    blob = sys.stdin.read() if a.src == "-" else Path(a.src).read_text()
    rec = json.loads(blob)
    if a.kind == "experiment":
        validate_experiment(rec)
        path = EXP_JSONL
    else:
        validate_hypothesis(rec)
        path = HYP_JSONL
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"appended {a.kind} -> {path.name}")
    return 0


# ---------------------------------------------------------------------------
# build-db — JSONL -> SQLite (derived, rebuildable)
# ---------------------------------------------------------------------------

def _flatten_exp(r: dict) -> dict:
    o = dict(r)
    o["params_json"] = json.dumps(r.get("params", {}), sort_keys=True)
    o["symbols_json"] = json.dumps(r.get("symbols", []))
    o["exit_hist_json"] = json.dumps(r.get("exit_hist", {}), sort_keys=True)
    return {c: o.get(c) for c in EXP_COLUMNS}


def cmd_build_db(a) -> int:
    db = Path(a.db)
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.executescript(SCHEMA_SQL.read_text())
    for _, r in _iter_jsonl(HYP_JSONL):
        validate_hypothesis(r)
        con.execute(
            f"INSERT INTO hypotheses ({','.join(HYP_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(HYP_COLUMNS))})",
            [r.get(c) for c in HYP_COLUMNS])
    for _, r in _iter_jsonl(EXP_JSONL):
        validate_experiment(r)
        o = _flatten_exp(r)
        con.execute(
            f"INSERT INTO experiments ({','.join(EXP_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(EXP_COLUMNS))})",
            [o.get(c) for c in EXP_COLUMNS])
    con.commit()
    audit = con.execute("SELECT COUNT(*) FROM v_methodology_audit").fetchone()
    if audit[0]:
        con.close()
        raise LedgerError(f"{audit[0]} confirmed positive-EV TAKER rows "
                          f"slipped past the gate — ledger is inconsistent")
    aa = con.execute("SELECT COUNT(*) FROM v_alpha_audit").fetchone()
    if aa[0]:
        con.close()
        raise LedgerError(f"{aa[0]} confirmed sub-cost alpha rows slipped "
                          f"past the gate — ledger is inconsistent")
    ne = con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    nh = con.execute("SELECT COUNT(*) FROM v_current_hypotheses").fetchone()[0]
    con.close()
    print(f"built {db} — {ne} experiments, {nh} current hypotheses")
    return 0


# ---------------------------------------------------------------------------
# frontier — regenerate RESEARCH_LOG section 3 from data (never hand-typed)
# ---------------------------------------------------------------------------

def cmd_frontier(a) -> int:
    db = Path(a.db)
    if not db.exists():
        cmd_build_db(argparse.Namespace(db=str(db)))
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM v_frontier").fetchall()
    print("| Date | Setup | Symbols | Fee | EV/tr% | tr/day | "
          "net%@k | n | status |")
    print("|---|---|---|---|--:|--:|--:|--:|---|")
    for r in rows:
        ev = "" if r["ev_per_trade_pct"] is None else f"{r['ev_per_trade_pct']:+.3f}"
        td = "" if r["trades_per_day"] is None else f"{r['trades_per_day']:.1f}"
        nr = "" if r["net_return_pct"] is None else f"{r['net_return_pct']:+.2f}"
        print(f"| {r['date']} | {r['setup']} | {r['symbols']} | "
              f"{r['fee_regime']} | {ev} | {td} | {nr} | "
              f"{r['n_trades']} | {r['status']} |")
    con.close()
    return 0


def cmd_alpha(a) -> int:
    db = Path(a.db)
    if not db.exists():
        cmd_build_db(argparse.Namespace(db=str(db)))
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM v_alpha").fetchall()
    print("| Date | Setup | Symbols | target | h(s) | rankIC | AUC | "
          "top|mv|% | cost% | econ | mono | n_eff | status |")
    print("|---|---|---|---|--:|--:|--:|--:|--:|:--:|:--:|--:|---|")
    for r in rows:
        f = lambda v, p=4: "" if v is None else f"{v:.{p}f}"
        print(f"| {r['date']} | {r['setup']} | {r['symbols']} | "
              f"{r['alpha_target']} | {r['horizon_sec']} | "
              f"{f(r['rank_ic_oos'])} | {f(r['auc_oos'],3)} | "
              f"{f(r['top_decile_absmove_pct'],3)} | "
              f"{f(r['cost_floor_pct'],3)} | {r['economic_pass']} | "
              f"{r['decile_monotonic']} | {r['n_eff']} | {r['status']} |")
    con.close()
    return 0


# ---------------------------------------------------------------------------
# check — CI gate
# ---------------------------------------------------------------------------

def cmd_check(a) -> int:
    try:
        cmd_validate(a)
        cmd_build_db(argparse.Namespace(db=str(DEFAULT_DB)))
    except LedgerError as e:
        print(f"LEDGER CHECK FAILED: {e}", file=sys.stderr)
        return 1
    print("LEDGER CHECK PASSED")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ledger")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    ap = sub.add_parser("append")
    ap.add_argument("kind", choices=("experiment", "hypothesis"))
    ap.add_argument("src", help="path to a JSON record, or - for stdin")
    bp = sub.add_parser("build-db")
    bp.add_argument("--db", default=str(DEFAULT_DB))
    fp = sub.add_parser("frontier")
    fp.add_argument("--db", default=str(DEFAULT_DB))
    alp = sub.add_parser("alpha")
    alp.add_argument("--db", default=str(DEFAULT_DB))
    sub.add_parser("check")
    a = p.parse_args(argv)
    fn = {
        "validate": cmd_validate, "append": cmd_append,
        "build-db": cmd_build_db, "frontier": cmd_frontier,
        "alpha": cmd_alpha, "check": cmd_check,
    }[a.cmd]
    try:
        return fn(a)
    except LedgerError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
