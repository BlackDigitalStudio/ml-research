"""Contract tests for the 7 canonical business metrics.

Source of truth: memory/business_metrics_canonical.md — the business owner
reads these first on every backtest report. This test guards the contract:

    1. All 7 metrics are exposed on BacktestResult and produce correct numbers.
    2. Categories 1-5 (exit-reason shares) are mutually exclusive and sum
       to 1.0 across every valid live_sim exit reason — adding a new
       reason in live_sim WITHOUT updating the mapping breaks this test
       on purpose.
    3. gross >= net in absolute terms (fees cannot add PnL).
    4. Legacy fields (total_pnl, tp_hit_rate, sl_hit_rate) retain meaning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest import BacktestResult, Trade
from src.live_sim import TradeOutcome


def _make_trade(reason: str, net: float = 1.0, gross: float = 1.0) -> Trade:
    """Build a Trade fixture with the given exit reason and PnL figures."""
    return Trade(
        direction="LONG",
        entry_price=100000.0,
        exit_price=100200.0,
        pnl=gross,          # gross_pnl_usd
        fees=gross - net,   # implied
        net_pnl=net,
        reason=reason,
        duration_ticks=100,
        gross_pnl_pct=(gross / 950.0) * 100,   # notional $950 stub
    )


@pytest.fixture
def one_of_each_result() -> BacktestResult:
    """Result holding exactly one trade per live_sim exit reason.

    Sizes `N` to the current `TradeOutcome.REASONS` (12 as of 2026-04-12).
    Equity curve starts at $100 for trivial percentage math.
    PnL per trade: +$1 net / +$2 gross — aggregates are multiples of N.
    """
    result = BacktestResult()
    result.equity_curve.append(100.0)
    for reason in TradeOutcome.REASONS:
        result.trades.append(_make_trade(reason, net=1.0, gross=2.0))
        result.equity_curve.append(result.equity_curve[-1] + 1.0)
    return result


# Total number of live_sim reasons — cached so bucket expectations stay
# in sync if live_sim adds a new reason AND BacktestResult is updated.
N_REASONS = len(TradeOutcome.REASONS)


# ---------------------------------------------------------------------------
# Metric 1-5: exit-reason shares, mutually exclusive, sum to 100%
# ---------------------------------------------------------------------------


def test_full_tp_rate(one_of_each_result: BacktestResult) -> None:
    # Only `tp_hit` counts as a full TP win.
    assert one_of_each_result.full_tp_rate == pytest.approx(1 / N_REASONS)


def test_full_sl_rate(one_of_each_result: BacktestResult) -> None:
    # sl_hit + fast_fill_adverse + fast_fill_sl
    assert one_of_each_result.full_sl_rate == pytest.approx(3 / N_REASONS)


def test_timeout_rate_covers_all_timeout_variants(
    one_of_each_result: BacktestResult,
) -> None:
    # timeout_limit + timeout_market + no_forward_data
    assert one_of_each_result.timeout_rate == pytest.approx(3 / N_REASONS)


def test_trailing_stop_rate_covers_pure_and_partial(
    one_of_each_result: BacktestResult,
) -> None:
    # trailing_sl_1 + trailing_sl_2 + partial_plus_trailing_sl_1/2
    assert one_of_each_result.trailing_stop_rate == pytest.approx(4 / N_REASONS)


def test_partial_tp_only_rate(one_of_each_result: BacktestResult) -> None:
    # Only `partial_plus_tp` — remainder of partials live in trailing bucket
    assert one_of_each_result.partial_tp_only_rate == pytest.approx(1 / N_REASONS)


def test_exit_rate_categories_sum_to_one(
    one_of_each_result: BacktestResult,
) -> None:
    """Contract: the 5 business categories cover every live_sim reason exactly
    once. Adding a new reason to live_sim.TradeOutcome.REASONS without
    updating BacktestResult mappings must fail this test."""
    total = (
        one_of_each_result.full_tp_rate
        + one_of_each_result.full_sl_rate
        + one_of_each_result.timeout_rate
        + one_of_each_result.trailing_stop_rate
        + one_of_each_result.partial_tp_only_rate
    )
    assert total == pytest.approx(1.0), (
        f"Exit-reason categories do not cover all live_sim reasons "
        f"({total:.4f} ≠ 1.0). If you added a new reason to "
        f"live_sim.TradeOutcome.REASONS, update the mapping in "
        f"BacktestResult (full_tp_rate / full_sl_rate / timeout_rate / "
        f"trailing_stop_rate / partial_tp_only_rate) and "
        f"memory/business_metrics_canonical.md."
    )


def test_live_sim_reason_set_fully_mapped() -> None:
    """Stronger structural check: every string in REASONS belongs to exactly
    one business bucket. Protects against typos in the mapping constants."""
    buckets = {
        "full_tp": set(BacktestResult._FULL_TP_REASONS),
        "full_sl": set(BacktestResult._FULL_SL_REASONS),
        "timeout": set(BacktestResult._TIMEOUT_REASONS),
        "trailing": set(BacktestResult._TRAILING_REASONS),
        "partial_tp": set(BacktestResult._PARTIAL_TP_REASONS),
    }
    covered: set[str] = set()
    for bucket_name, reasons in buckets.items():
        overlap = covered & reasons
        assert not overlap, f"{bucket_name} overlaps existing buckets: {overlap}"
        covered |= reasons
    missing = set(TradeOutcome.REASONS) - covered
    extra = covered - set(TradeOutcome.REASONS)
    assert not missing, f"Reasons missing from business mapping: {missing}"
    assert not extra, f"Business mapping references unknown reasons: {extra}"


# ---------------------------------------------------------------------------
# Metric 6-7: PnL in $ and %, gross and net
# ---------------------------------------------------------------------------


def test_gross_pnl_usd(one_of_each_result: BacktestResult) -> None:
    # N trades × $2 gross
    assert one_of_each_result.gross_pnl_usd == pytest.approx(2.0 * N_REASONS)


def test_net_pnl_usd_matches_legacy_total_pnl(
    one_of_each_result: BacktestResult,
) -> None:
    # N trades × $1 net. total_pnl is the legacy name, must still work.
    assert one_of_each_result.total_pnl == pytest.approx(1.0 * N_REASONS)


def test_gross_pnl_pct(one_of_each_result: BacktestResult) -> None:
    # gross_usd / $100 initial × 100
    assert one_of_each_result.gross_pnl_pct == pytest.approx(2.0 * N_REASONS)


def test_net_pnl_pct(one_of_each_result: BacktestResult) -> None:
    assert one_of_each_result.net_pnl_pct == pytest.approx(1.0 * N_REASONS)


def test_gross_absolute_ge_net_absolute(
    one_of_each_result: BacktestResult,
) -> None:
    """Fees cannot add PnL: gross >= net in absolute value on any consistent
    fixture. Stronger than equality because both can be negative."""
    assert abs(one_of_each_result.gross_pnl_usd) >= abs(
        one_of_each_result.total_pnl
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_result_no_division_by_zero() -> None:
    result = BacktestResult()
    assert result.full_tp_rate == 0.0
    assert result.full_sl_rate == 0.0
    assert result.timeout_rate == 0.0
    assert result.trailing_stop_rate == 0.0
    assert result.partial_tp_only_rate == 0.0
    assert result.gross_pnl_usd == 0.0
    assert result.gross_pnl_pct == 0.0
    assert result.net_pnl_pct == 0.0


def test_metrics_are_fractions_not_percentages() -> None:
    """Document the unit: rate properties return a fraction in [0, 1].
    Display code multiplies by 100 via `:.1%` format. Tests rely on this."""
    result = BacktestResult()
    result.equity_curve.append(50.0)
    result.trades.append(_make_trade("tp_hit", net=0.1, gross=0.2))
    assert result.full_tp_rate == 1.0
    assert 0.0 <= result.full_tp_rate <= 1.0
