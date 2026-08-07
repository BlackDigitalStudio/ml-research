"""Contract tests for the research ledger (research/ledger.py).

Stdlib-only on purpose — the ledger must validate in any environment,
including the planning container that has no numpy/pandas. Guards:

  1. The committed ledger passes `check` (validate + build-db + audit).
  2. The gate REFUSES a positive-EV TAKER 'confirmed' row (the exact
     shape that was a false positive 3x: DOGE / ETH / phase56).
  3. The gate REFUSES a row missing provenance (fee_regime).
  4. Owner exit-shares must sum to ~1.0 (categories are exclusive).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

LEDGER = Path(__file__).resolve().parent.parent / "research" / "ledger.py"
_spec = importlib.util.spec_from_file_location("ledger", LEDGER)
ledger = importlib.util.module_from_spec(_spec)
sys.modules["ledger"] = ledger
_spec.loader.exec_module(ledger)


def _valid_record(**over) -> dict:
    r = {
        "experiment_id": "t_x", "ts": "2026-05-16T00:00:00Z",
        "git_commit": "deadbeef", "author": "test", "status": "exploratory",
        "setup": "unit", "model_family": "xgb", "params": {"tp_pct": 0.2},
        "data_source": "v3_btc", "cache_id": "c", "symbols": ["BTCUSDT"],
        "n_samples": 100, "fee_regime": "MAKER_FIRST",
        "commission_win_pct": 0.04, "commission_loss_pct": 0.07,
        "split_method": "walkforward_7525", "label_def": "canonical TB",
        "n_trades": 50, "repro_cmd": "pytest",
    }
    r.update(over)
    return r


def test_committed_ledger_passes_check() -> None:
    rc = ledger.main(["check"])
    assert rc == 0, "committed research ledger must pass the CI gate"


def test_valid_record_accepted() -> None:
    ledger.validate_experiment(_valid_record())


def test_positive_ev_taker_confirmed_is_refused() -> None:
    """The killer rule. A +EV TAKER result must not be stored confirmed."""
    with pytest.raises(ledger.LedgerError, match="positive-EV TAKER"):
        ledger.validate_experiment(
            _valid_record(status="confirmed", fee_regime="TAKER",
                          commission_win_pct=0.07, commission_loss_pct=0.10,
                          ev_per_trade_pct=0.05))


def test_missing_provenance_is_refused() -> None:
    bad = _valid_record()
    del bad["fee_regime"]
    with pytest.raises(ledger.LedgerError, match="missing mandatory"):
        ledger.validate_experiment(bad)


def test_owner_shares_must_sum_to_one() -> None:
    with pytest.raises(ledger.LedgerError, match="exit-shares sum"):
        ledger.validate_experiment(_valid_record(
            pct_full_tp=0.1, pct_full_sl=0.1, pct_timeout=0.1,
            pct_trailing=0.1, pct_partial_only=0.1))


def test_fees_cannot_add_pnl() -> None:
    with pytest.raises(ledger.LedgerError, match="fees cannot"):
        ledger.validate_experiment(
            _valid_record(pnl_gross_pct=0.1, pnl_net_pct=0.5))


def test_unknown_exit_reason_is_refused() -> None:
    with pytest.raises(ledger.LedgerError, match="unknown exit reasons"):
        ledger.validate_experiment(
            _valid_record(exit_hist={"tp_hit": 5, "bogus_reason": 1}))
