"""Contract tests for Kelly position sizing.

Locks in the formulas + cap behaviour against well-known reference values.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.sizing import (
    KellyConfig, kelly_fraction, asymmetric_kelly_fraction,
    size_trade, size_trades_batch,
)


# ─── kelly_fraction (pure formula, decimal payouts) ────────────────────────


def test_kelly_symmetric_fair_coin():
    """50/50 with 1:1 payoffs → f* = 0 (no edge)."""
    assert kelly_fraction(0.5, 1.0, 1.0) == pytest.approx(0.0)


def test_kelly_symmetric_with_edge():
    """60/40 with 1:1 payoffs → f* = 0.20 (classic textbook)."""
    assert kelly_fraction(0.6, 1.0, 1.0) == pytest.approx(0.20)


def test_kelly_negative_edge_returns_negative():
    """40/60 with 1:1 → f* = -0.20 (caller should treat as 'skip')."""
    assert kelly_fraction(0.4, 1.0, 1.0) == pytest.approx(-0.20)


def test_kelly_invalid_inputs_zero():
    assert kelly_fraction(0.6, 0.0, 1.0) == 0.0
    assert kelly_fraction(0.6, 1.0, 0.0) == 0.0


# ─── asymmetric_kelly_fraction (% inputs) ──────────────────────────────────


def test_asymmetric_no_commissions_2to1():
    """50/50 with 2:1 payoffs (TP=2%, SL=1%) → in DECIMAL: w=0.02, l=0.01.
    f* = (0.5*0.02 - 0.5*0.01) / (0.02*0.01) = 0.005 / 0.0002 = 25.

    Yes, full Kelly says 25× capital — which is why scalping NEEDS the cap.
    """
    f = asymmetric_kelly_fraction(0.5, win_pct=2.0, loss_pct=1.0)
    assert f == pytest.approx(25.0, rel=1e-6)


def test_asymmetric_with_commissions_eats_edge():
    """Commissions strictly bigger than win → invalid → 0."""
    f = asymmetric_kelly_fraction(0.6, win_pct=0.05, loss_pct=0.05,
                                   commission_win_pct=0.10,
                                   commission_loss_pct=0.10)
    # net_win = -0.05 → invalid → 0
    assert f == 0.0


# ─── size_trade ────────────────────────────────────────────────────────────


def test_size_trade_skip_below_min_probability():
    cfg = KellyConfig(min_probability=0.55)
    d = size_trade(0.50, win_pct=0.20, loss_pct=0.20, cfg=cfg)
    assert d.take is False
    assert d.fraction == 0.0


def test_size_trade_caps_at_max_position_fraction():
    """Strong edge → raw Kelly is huge, but cap pulls down to 19× max."""
    cfg = KellyConfig(fraction=1.0, max_position_fraction=19.0,
                      min_probability=0.5, min_position_fraction=0.1)
    d = size_trade(0.99, win_pct=1.0, loss_pct=1.0,
                   commission_win_pct=0.0, commission_loss_pct=0.0, cfg=cfg)
    # Net win=0.01, net loss=0.01 → raw Kelly = (0.99*0.01 - 0.01*0.01)/(0.01*0.01) = 98
    assert d.take is True
    assert d.fraction == pytest.approx(19.0)
    assert d.raw_kelly == pytest.approx(98.0, rel=1e-3)


def test_size_trade_dust_skip():
    """Tiny scaled fraction (< min_position_fraction) → skip."""
    cfg = KellyConfig(fraction=0.001, max_position_fraction=10.0,
                      min_position_fraction=0.5, min_probability=0.5)
    # raw≈25 → scaled=0.025 < 0.5 → skip
    d = size_trade(0.5001, win_pct=2.0, loss_pct=1.0,
                   commission_win_pct=0.0, commission_loss_pct=0.0, cfg=cfg)
    # Even with edge, scaled fraction too small.
    assert d.take is False


def test_size_trade_realistic_scalping():
    """Realistic case: TP=0.20%, SL=0.20%, p=0.55, x20 leverage cap.
    Expect to trade with fraction near max (19) due to tiny payouts.
    """
    cfg = KellyConfig(fraction=0.25, max_position_fraction=19.0,
                      min_probability=0.51)
    d = size_trade(0.55, win_pct=0.20, loss_pct=0.20,
                   commission_win_pct=0.04, commission_loss_pct=0.07, cfg=cfg)
    # Net win=0.16, net loss=0.27 → edge per unit:
    #   0.55*0.16 - 0.45*0.27 = 0.088 - 0.1215 = -0.0335 → NEGATIVE edge
    # Should NOT take.
    assert d.take is False


def test_size_trade_realistic_scalping_with_real_edge():
    """Same setup but p=0.65 (real edge after commissions)."""
    cfg = KellyConfig(fraction=0.25, max_position_fraction=19.0,
                      min_probability=0.51)
    d = size_trade(0.65, win_pct=0.20, loss_pct=0.20,
                   commission_win_pct=0.04, commission_loss_pct=0.07, cfg=cfg)
    # Net win=0.16, net loss=0.27 → edge: 0.65*0.16 - 0.35*0.27 = 0.0095 → small +
    # raw Kelly = (0.65*0.0016 - 0.35*0.0027)/(0.0016*0.0027) = 0.000095/4.32e-6 ≈ 22
    # Scaled by 0.25 → 5.5 → between min(0.1) and max(19) → take.
    assert d.take is True
    assert 0.5 < d.fraction < 19.0


# ─── size_trades_batch ─────────────────────────────────────────────────────


def test_batch_matches_per_trade():
    """Vectorised result must equal per-trade for each entry."""
    cfg = KellyConfig(fraction=0.25, max_position_fraction=19.0,
                      min_probability=0.51, min_position_fraction=0.1)
    p = np.array([0.40, 0.55, 0.70, 0.99])
    w = np.full(4, 0.20)
    l = np.full(4, 0.10)
    out = size_trades_batch(p, w, l, cfg=cfg)
    for i in range(4):
        d = size_trade(float(p[i]), 0.20, 0.10, cfg=cfg)
        assert bool(out["take"][i]) == d.take, f"row {i}"
        assert out["fraction"][i] == pytest.approx(d.fraction), f"row {i}"
        assert out["raw_kelly"][i] == pytest.approx(d.raw_kelly), f"row {i}"


def test_batch_negative_edge_skipped():
    """Trades with p < min_probability or negative Kelly → take=False."""
    cfg = KellyConfig(min_probability=0.5)
    p = np.array([0.45, 0.50, 0.49])
    out = size_trades_batch(p, win_pct=0.20, loss_pct=0.20, cfg=cfg)
    assert (out["take"] == False).all()
    assert (out["fraction"] == 0.0).all()


def test_kelly_growth_intuition():
    """Sanity: full-Kelly growth on 60/40 1:1 ≈ 0.0201 per bet (log).
    Half-Kelly grows slightly less but with much lower variance."""
    p = 0.6
    f = kelly_fraction(p, 1.0, 1.0)  # 0.20
    g_full = p * np.log(1 + f) + (1 - p) * np.log(1 - f)
    g_half = p * np.log(1 + f / 2) + (1 - p) * np.log(1 - f / 2)
    assert g_full > 0
    assert g_half > 0
    assert g_full > g_half
    assert g_full == pytest.approx(0.02014, abs=1e-4)
