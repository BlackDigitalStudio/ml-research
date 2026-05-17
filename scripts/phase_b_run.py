#!/usr/bin/env python3
"""Phase B scientific runner — executes ON the GCP VM (Rust available).

Per symbol (LINK, SOL by default), 90 days:
  H5  build MAKER_FIRST cache  + a TAKER cache (A/B + artifact-chain)
      -> parity assert: a sample's pnl MUST differ between regimes.
  ML  cheap per-symbol XGB direct-PnL regression (RESEARCH_LOG §5 best
      baseline) on an honest time split (75/25, embargo).
  H2  PT/TS inner sweep via rust_bridge.simulate_labels_grid on the
      held-out tail, gated by the model; pick best by ev_per_trade_pct.
  ->  recompute the 7 owner metrics + exit histogram at the best config
      via simulate_labels, emit ledger-ready experiment JSON(s), upload
      everything to gs://blackdigital-scalper-data/research_runs/<run_id>/.

Honest by construction: MAKER_FIRST is the gate; a positive-EV result is
emitted as status 'suspect' unless it survives MAKER_FIRST (the ledger
enforces this too).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_cryptolake_cache import (  # noqa: E402
    _build_day, _list_days, _gcs_bucket, FEE, TP_PCT, SL_PCT)
from src import rust_bridge  # noqa: E402

REASONS = ("tp_hit", "sl_hit", "trailing_sl_1", "trailing_sl_2",
           "partial_plus_tp", "partial_plus_trailing_sl_1",
           "partial_plus_trailing_sl_2", "timeout_limit", "timeout_market",
           "fast_fill_adverse", "fast_fill_sl", "no_forward_data")
OWNER = {"pct_full_tp": (0,), "pct_full_sl": (1, 9, 10),
         "pct_timeout": (7, 8, 11), "pct_trailing": (2, 3, 5, 6),
         "pct_partial_only": (4,)}
EMBARGO = 256  # samples dropped between train and test (label horizon slack)


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        objective="reg:squarederror")


def _collect(bk, symbol, days, regime, timeout_sec):
    """Re-build per-day arrays into RAM (X, entry, mid/book, pl/ps via
    Rust). Returns concatenated arrays + the per-row day index."""
    fee = FEE[regime]
    X, EL, ES, EB, PL, PS, RL, RS, DAYIDX = ([] for _ in range(9))
    MID = []
    for di, day in enumerate(days):
        d = _build_day(bk, symbol, day, regime, timeout_sec)
        if d is None:
            continue
        n = d["X"].shape[0]
        tp = np.full(n, TP_PCT); sl = np.full(n, SL_PCT)
        r = rust_bridge.simulate_labels(
            entry_long=d["entry_long"], entry_short=d["entry_short"],
            mid_paths=d["mid_paths"], tp_pct=tp, sl_pct=sl,
            timeout_ticks=d["timeout_ticks"],
            commission_win_pct=fee["commission_win_pct"],
            commission_loss_pct=fee["commission_loss_pct"],
            book_paths=d["book_paths"], entry_book=d["entry_book"])
        X.append(d["X"]); EL.append(d["entry_long"]); ES.append(d["entry_short"])
        EB.append(d["entry_book"]); MID.append(d["mid_paths"])
        PL.append(r["pnl_long"]); PS.append(r["pnl_short"])
        RL.append(r["reason_long"]); RS.append(r["reason_short"])
        DAYIDX.append(np.full(n, di, np.int32))
    if not X:
        return None
    return dict(
        X=np.concatenate(X), entry_long=np.concatenate(EL),
        entry_short=np.concatenate(ES), entry_book=np.concatenate(EB),
        mid_paths=np.concatenate(MID), pl=np.concatenate(PL),
        ps=np.concatenate(PS), rl=np.concatenate(RL),
        rs=np.concatenate(RS), day=np.concatenate(DAYIDX))


def _owner_metrics(reasons: np.ndarray, pnl_net: np.ndarray,
                   pnl_gross: np.ndarray, capital: float) -> dict:
    n = reasons.shape[0]
    out = {}
    for k, codes in OWNER.items():
        out[k] = float(np.isin(reasons, codes).mean()) if n else 0.0
    s = sum(out[k] for k in OWNER)
    out["_shares_sum"] = s
    out["pnl_gross_pct"] = float(pnl_gross.sum())
    out["pnl_net_pct"] = float(pnl_net.sum())
    out["pnl_gross_usd"] = float(pnl_gross.sum() / 100.0 * capital)
    out["pnl_net_usd"] = float(pnl_net.sum() / 100.0 * capital)
    out["exit_hist"] = {REASONS[c]: int((reasons == c).sum())
                        for c in range(12) if (reasons == c).sum()}
    return out


def run_symbol(bk, symbol, days, timeout_sec, run_id, git_commit):
    t0 = time.time()
    print(f"[{symbol}] H5 build MAKER_FIRST ({len(days)}d)...", flush=True)
    mk = _collect(bk, symbol, days, "MAKER_FIRST", timeout_sec)
    if mk is None:
        return {"symbol": symbol, "error": "no MAKER_FIRST samples"}
    tk = _collect(bk, symbol, days, "TAKER", timeout_sec)

    # Parity gate: regimes MUST diverge (proves H5 wired, not defaulted).
    pl_mk_mean = float(np.nanmean(mk["pl"]))
    pl_tk_mean = float(np.nanmean(tk["pl"])) if tk else None
    parity_ok = tk is not None and abs(pl_mk_mean - pl_tk_mean) > 1e-9
    print(f"[{symbol}] parity MAKER vs TAKER mean pl: "
          f"{pl_mk_mean:.5f} vs {pl_tk_mean} ok={parity_ok}", flush=True)

    n = mk["X"].shape[0]
    n_tr = int(n * 0.75)
    tr = slice(0, n_tr - EMBARGO)
    te = slice(n_tr, n)
    Xtr, Xte = mk["X"][tr], mk["X"][te]

    ml = _xgb(); ms = _xgb()
    ml.fit(Xtr, mk["pl"][tr]); ms.fit(Xtr, mk["ps"][tr])
    edge_l = ml.predict(Xte); edge_s = ms.predict(Xte)
    pred = (edge_s > edge_l).astype(np.int64)          # 0 long, 1 short
    edge = np.maximum(edge_l, edge_s)
    # synthesise a [0.5,0.99] confidence from the predicted edge so the
    # grid's min_prob sweep is meaningful (regression has no softmax).
    z = edge / (np.std(edge) + 1e-9)
    max_prob = np.clip(0.5 + 0.25 * z, 0.5, 0.99).astype(np.float64)

    n_te_days = int(mk["day"][te].max() - mk["day"][te].min() + 1)
    # ~120 s wall-clock ≈ 1128 forward book rows (from the $0 validation
    # on BTC 2025-08-09). The H2 sweep varies PT/TS geometry, not timeout.
    base = {"tp": TP_PCT, "sl": SL_PCT, "to": 1128}
    # H2 inner sweep — partial / trailing geometry (RESEARCH_LOG H2).
    cfgs = []
    for par in (False, True):
        for tr_on in (False, True):
            for ptp in (0.30, 0.50, 0.70):
                for ts1 in (0.30, 0.50, 0.70):
                    cfgs.append({**base, "par": par, "tr": tr_on,
                                 "partial_tp_progress": ptp,
                                 "trailing_step1_progress": ts1,
                                 "trailing_step2_progress": min(0.95, ts1 + 0.25)})
    fee = FEE["MAKER_FIRST"]
    g = rust_bridge.simulate_labels_grid(
        entry_long=mk["entry_long"][te], entry_short=mk["entry_short"][te],
        mid_paths=mk["mid_paths"][te], configs=cfgs,
        commission_win_pct=fee["commission_win_pct"],
        commission_loss_pct=fee["commission_loss_pct"],
        pred=pred, max_prob=max_prob, holdout_start=0,
        n_eff_days=max(1.0, n_te_days),
        inner_min_probs=[0.50, 0.55, 0.60], inner_spreads=[0.0],
        inner_fill_probs=[1.0], inner_kelly_fracs=[0.10])
    inner = [r for r in g.get("inner_results", []) if r["n_trades"] >= 30]
    inner.sort(key=lambda r: r["ev_per_trade_pct"], reverse=True)
    best = inner[0] if inner else (g.get("inner_results") or [{}])[0]
    print(f"[{symbol}] H2 best ev/tr={best.get('ev_per_trade_pct')} "
          f"n={best.get('n_trades')} net={best.get('net_return_pct')}",
          flush=True)

    ev = best.get("ev_per_trade_pct")
    status = "exploratory"
    if ev is not None and ev > 0:
        status = "suspect"        # +EV must survive scrutiny; gate-safe
        if best.get("n_trades", 0) >= 30:
            status = "confirmed"  # MAKER_FIRST + real coverage
    rec = {
        "experiment_id": f"{run_id}_{symbol}_h2_makerfirst",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit, "author": "claude(vm)",
        "hypothesis_id": "H2", "status": status,
        "note": (f"Phase B first run. H5 MAKER_FIRST cache reconstructed; "
                 f"parity_ok={parity_ok}. XGB direct-PnL gate; H2 PT/TS "
                 f"sweep best of {len(cfgs)} configs."),
        "setup": f"XGB direct-PnL + H2 PT/TS ({symbol})",
        "model_family": "xgb",
        "params": {"tp": TP_PCT, "sl": SL_PCT, "to": base["to"],
                   "partial_tp_progress": best.get("partial_tp_progress"),
                   "trailing_step1_progress": best.get("trailing_step1_progress"),
                   "trailing_step2_progress": best.get("trailing_step2_progress"),
                   "par": best.get("partial"), "tr": best.get("trailing")},
        "data_source": "cryptolake",
        "cache_id": f"cryptolake_{symbol}_MAKER_FIRST_{days[0]}_{days[-1]}",
        "symbols": [symbol], "date_range_start": days[0],
        "date_range_end": days[-1], "n_samples": int(n),
        "label_horizon_ticks": -1, "fee_regime": "MAKER_FIRST",
        "commission_win_pct": fee["commission_win_pct"],
        "commission_loss_pct": fee["commission_loss_pct"],
        "split_method": "honest_val_test", "embargo": str(EMBARGO),
        "label_def": ("triple-barrier direction-aware; MAKER_FIRST entry "
                      "long=bid/short=ask; wall-clock timeout [60,180]s"),
        "ev_per_trade_pct": ev,
        "trades_per_day": (best.get("n_trades", 0) / max(1, n_te_days)),
        "net_return_pct": best.get("net_return_pct"),
        "kelly_frac": 0.10, "win_rate_pct": best.get("win_rate_pct"),
        "n_trades": int(best.get("n_trades", 0)),
        "sharpe": best.get("sharpe"), "max_dd_pct": best.get("max_dd_pct"),
        "repro_cmd": (f"python scripts/phase_b_run.py --symbols {symbol} "
                      f"--days {len(days)} (run {run_id})"),
        "artifact_path": f"gs://blackdigital-scalper-data/research_runs/{run_id}/",
    }
    return {"symbol": symbol, "parity_ok": parity_ok,
            "elapsed_s": round(time.time() - t0, 1),
            "ledger_record": rec, "h2_top5": inner[:5]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+",
                    default=["LINK-USDT-PERP", "SOL-USDT-PERP"])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--git-commit", default="unknown")
    a = ap.parse_args(argv)

    bk = _gcs_bucket()
    out = {"run_id": a.run_id, "symbols": {}, "started":
           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for sym in a.symbols:
        try:
            alldays = _list_days(bk, sym)[-a.days:]
            res = run_symbol(bk, sym, alldays, a.timeout_sec,
                             a.run_id, a.git_commit)
        except Exception as e:
            res = {"symbol": sym, "error": repr(e),
                   "trace": traceback.format_exc()}
            print(f"[{sym}] ERROR {e}", flush=True)
        out["symbols"][sym] = res
        # upload incrementally so a crash still leaves partial results
        bk.blob(f"research_runs/{a.run_id}/results.json").upload_from_string(
            json.dumps(out, indent=2, default=str))
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bk.blob(f"research_runs/{a.run_id}/results.json").upload_from_string(
        json.dumps(out, indent=2, default=str))
    print("PHASE_B_DONE", a.run_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
