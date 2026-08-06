"""Unit tests for src/filters.py.

Each stateless filter gets one pass + one fail case. Adaptive TP/SL and
dynamic timeout get boundary checks so accidental drift from the
constants in signal.py / executor.py surfaces immediately.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BINANCE_API_KEY", "dummy")
os.environ.setdefault("BINANCE_API_SECRET", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import filters  # noqa: E402


def _ts_ms(year: int, month: int, day: int, hour: int, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc).timestamp() * 1000.0


# ---- Time-of-day --------------------------------------------------------


def test_time_of_day_inside_asia_night_rejects() -> None:
    # 04:30 UTC is inside [04, 07).
    assert not filters.passes_time_of_day(_ts_ms(2026, 4, 10, 4, 30))


def test_time_of_day_outside_asia_night_passes() -> None:
    assert filters.passes_time_of_day(_ts_ms(2026, 4, 10, 12, 0))


def test_time_of_day_boundary_end_is_exclusive() -> None:
    # 07:00 UTC is the end of the half-open window — passes.
    assert filters.passes_time_of_day(_ts_ms(2026, 4, 10, 7, 0))


# ---- Funding blackout ---------------------------------------------------


def test_funding_blackout_exact_hour_rejects() -> None:
    for h in (0, 8, 16):
        assert not filters.passes_funding_blackout(_ts_ms(2026, 4, 10, h, 0))


def test_funding_blackout_within_3min_rejects() -> None:
    assert not filters.passes_funding_blackout(_ts_ms(2026, 4, 10, 7, 58))
    assert not filters.passes_funding_blackout(_ts_ms(2026, 4, 10, 8, 3))


def test_funding_blackout_outside_guard_passes() -> None:
    assert filters.passes_funding_blackout(_ts_ms(2026, 4, 10, 7, 50))
    assert filters.passes_funding_blackout(_ts_ms(2026, 4, 10, 12, 30))


def test_funding_blackout_wraps_midnight() -> None:
    assert not filters.passes_funding_blackout(_ts_ms(2026, 4, 10, 23, 58))


# ---- Spread / vol / liquidity / spoof ---------------------------------


def test_spread_pass_fail() -> None:
    # MAX_SPREAD_USD = 0.20 (= 2 ticks on BTCUSDT tick 0.10). The legacy
    # 0.02 value was a 10× typo; see the filters.py comment for the fix.
    assert filters.passes_spread(0.10)
    assert filters.passes_spread(0.20)  # boundary: inclusive pass
    assert not filters.passes_spread(0.21)


def test_vol_spike_pass_fail() -> None:
    assert filters.passes_vol_spike(0.001, 0.003)
    assert not filters.passes_vol_spike(0.004, 0.003)


def test_vol_band_pass_fail() -> None:
    assert filters.passes_vol_band(1.0)
    assert not filters.passes_vol_band(0.5)
    assert not filters.passes_vol_band(3.0)
    # Strict boundaries — 0.7 and 2.5 both reject.
    assert not filters.passes_vol_band(0.7)
    assert not filters.passes_vol_band(2.5)


def test_liquidity_pass_fail() -> None:
    # MIN_TOP5_BTC_LIQUIDITY = 2.0 (= ~BTC median on real data). See
    # filters.py comment for the ETH→BTC migration fix that landed with
    # the label/live-sync rewrite.
    assert filters.passes_liquidity(5.0, 10.0)
    assert filters.passes_liquidity(2.0, 2.0)  # boundary
    assert not filters.passes_liquidity(1.9, 10.0)
    assert not filters.passes_liquidity(10.0, 1.9)


def test_spoof_pass_fail() -> None:
    assert filters.passes_spoof(0.4)
    assert filters.passes_spoof(0.5)  # boundary
    assert not filters.passes_spoof(0.51)


# ---- Imbalance / hurst -------------------------------------------------


def test_imbalance_long_requires_positive_threshold() -> None:
    assert filters.passes_imbalance_long(0.20)
    assert not filters.passes_imbalance_long(0.10)
    assert not filters.passes_imbalance_long(-0.20)


def test_imbalance_short_requires_negative_threshold() -> None:
    assert filters.passes_imbalance_short(-0.20)
    assert not filters.passes_imbalance_short(-0.10)
    assert not filters.passes_imbalance_short(0.20)


def test_hurst_regime_trending_noop() -> None:
    # H >= 0.45 always passes regardless of confidence.
    assert filters.passes_hurst_regime(0.55, 0.58, 0.58)
    assert filters.passes_hurst_regime(0.45, 0.30, 0.58)


def test_hurst_regime_mean_reverting_requires_bonus() -> None:
    # H = 0.40, base threshold = 0.58 → need confidence >= 0.63.
    assert not filters.passes_hurst_regime(0.40, 0.60, 0.58)
    assert filters.passes_hurst_regime(0.40, 0.63, 0.58)
    assert filters.passes_hurst_regime(0.40, 0.80, 0.58)


# ---- adaptive_tp_sl -----------------------------------------------------


def test_adaptive_tp_sl_identity_at_vol_ratio_1() -> None:
    tp, sl = filters.adaptive_tp_sl(vol_ratio=1.0, base_tp_pct=0.20, base_sl_pct=0.10)
    assert abs(tp - 0.20) < 1e-12
    assert abs(sl - 0.10) < 1e-12


def test_adaptive_tp_sl_clamps_at_low_vol() -> None:
    # vol_ratio clamped to 0.5 → base is halved, but then SL hits the 0.05 floor.
    tp, sl = filters.adaptive_tp_sl(vol_ratio=0.1, base_tp_pct=0.20, base_sl_pct=0.10)
    # 0.20 * 0.5 = 0.10 — exactly the MIN_TP_PCT floor (still inside bounds).
    assert abs(tp - 0.10) < 1e-12
    # 0.10 * 0.5 = 0.05 — exactly the MIN_SL_PCT floor.
    assert abs(sl - 0.05) < 1e-12


def test_adaptive_tp_sl_clamps_at_high_vol() -> None:
    tp, sl = filters.adaptive_tp_sl(vol_ratio=5.0, base_tp_pct=0.20, base_sl_pct=0.10)
    # vol_ratio clamps to 3.0 → tp=0.60 (=MAX), sl=0.30 (=MAX).
    assert abs(tp - 0.60) < 1e-12
    assert abs(sl - 0.30) < 1e-12


def test_adaptive_tp_sl_matches_signal_generator() -> None:
    """Parity check: the new filters.adaptive_tp_sl must produce identical
    output to the live SignalGenerator._adaptive_tp_sl for the same input.

    Uses __new__ to bypass the executor/feature dependencies — we only need
    the instance method bound to a fake config.
    """
    from src.signal import SignalGenerator

    class _Cfg:
        take_profit_pct = 0.20
        stop_loss_pct = 0.10

    g = SignalGenerator.__new__(SignalGenerator)
    g._cfg = _Cfg()
    for vr in (0.3, 0.6, 1.0, 1.5, 2.2, 4.0):
        legacy_tp, legacy_sl = g._adaptive_tp_sl({"volatility_ratio": vr})
        new_tp, new_sl = filters.adaptive_tp_sl(vr, 0.20, 0.10)
        assert abs(legacy_tp - new_tp) < 1e-12, vr
        assert abs(legacy_sl - new_sl) < 1e-12, vr


# ---- dynamic_timeout ----------------------------------------------------


def test_dynamic_timeout_returns_base_when_vol_missing() -> None:
    assert filters.dynamic_timeout_sec(0.0, 0.0, 60.0) == 60.0
    assert filters.dynamic_timeout_sec(0.001, 0.0, 60.0) == 60.0


def test_dynamic_timeout_scales_inverse_to_current_vol() -> None:
    # current = 2 × avg → timeout shrinks to half.
    t = filters.dynamic_timeout_sec(0.001, 0.002, 60.0)
    assert abs(t - 30.0) < 1e-9


def test_dynamic_timeout_clamps() -> None:
    # Extreme ratios get clamped to [15, 120] seconds.
    assert filters.dynamic_timeout_sec(0.001, 10.0, 60.0) == 15.0
    assert filters.dynamic_timeout_sec(10.0, 0.001, 60.0) == 120.0


def test_dynamic_timeout_ticks_conversion() -> None:
    assert filters.dynamic_timeout_ticks(0.001, 0.001, 60.0) == 600
    assert filters.dynamic_timeout_ticks(0.001, 0.002, 60.0) == 300  # half


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    n = 0
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            fn()
            print(f"  ok: {name}")
            n += 1
    print(f"filter tests OK ({n} tests)")
