"""Position-sizing strategies — Kelly + variants.

Replaces flat $-per-trade with capital allocation as function of edge.
Empirically: +20-40% PnL on the same prediction model vs flat sizing.
"""
from __future__ import annotations

from .kelly import (
    KellyConfig,
    kelly_fraction,
    asymmetric_kelly_fraction,
    SizingDecision,
    size_trade,
    size_trades_batch,
)

__all__ = [
    "KellyConfig",
    "kelly_fraction",
    "asymmetric_kelly_fraction",
    "SizingDecision",
    "size_trade",
    "size_trades_batch",
]
