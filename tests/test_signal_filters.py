"""Unit tests for the Lever 3 entry filters in src/signal.py.

These tests cover the pure functions/predicates only — the rest of
`SignalGenerator.generate` requires a live order book and a loaded model.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BINANCE_API_KEY", "dummy")
os.environ.setdefault("BINANCE_API_SECRET", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signal import (
    ASIA_NIGHT_END_UTC,
    ASIA_NIGHT_START_UTC,
    FUNDING_GUARD_MIN,
    SignalGenerator,
)


class _Stub:
    pass


def _gen() -> SignalGenerator:
    # Construct without invoking __init__ — we only need _is_funding_window.
    g = SignalGenerator.__new__(SignalGenerator)
    g._executor = None
    g._recent_wr_pause_until = 0.0
    return g


def test_funding_window_exact_hour() -> None:
    g = _gen()
    for h in (0, 8, 16):
        t = datetime(2026, 4, 10, h, 0, 0, tzinfo=timezone.utc)
        assert g._is_funding_window(t), f"expected guard at {h:02d}:00 UTC"


def test_funding_window_within_guard() -> None:
    g = _gen()
    # 07:58 is 2 minutes before 08:00 — must be inside the ±3 min guard.
    t = datetime(2026, 4, 10, 7, 58, 0, tzinfo=timezone.utc)
    assert g._is_funding_window(t)
    # 08:03 is 3 minutes after 08:00 — boundary, still inside.
    t = datetime(2026, 4, 10, 8, FUNDING_GUARD_MIN, 0, tzinfo=timezone.utc)
    assert g._is_funding_window(t)


def test_funding_window_outside_guard() -> None:
    g = _gen()
    # 07:50 is 10 minutes before 08:00 — outside.
    t = datetime(2026, 4, 10, 7, 50, 0, tzinfo=timezone.utc)
    assert not g._is_funding_window(t)
    # 12:30 is mid-day, far from any funding hour.
    t = datetime(2026, 4, 10, 12, 30, 0, tzinfo=timezone.utc)
    assert not g._is_funding_window(t)


def test_funding_window_wraps_midnight() -> None:
    g = _gen()
    # 23:58 is 2 minutes before 00:00 of the next day — must wrap correctly.
    t = datetime(2026, 4, 10, 23, 58, 0, tzinfo=timezone.utc)
    assert g._is_funding_window(t)


def test_asia_night_constants_are_consistent() -> None:
    # Half-open interval [start, end) — sanity check on the constants.
    assert ASIA_NIGHT_START_UTC < ASIA_NIGHT_END_UTC
    assert 0 <= ASIA_NIGHT_START_UTC < 24
    assert 0 < ASIA_NIGHT_END_UTC <= 24


if __name__ == "__main__":
    test_funding_window_exact_hour()
    test_funding_window_within_guard()
    test_funding_window_outside_guard()
    test_funding_window_wraps_midnight()
    test_asia_night_constants_are_consistent()
    print("signal-filter tests OK")
