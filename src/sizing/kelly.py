"""Kelly criterion and fractional-Kelly position sizing for scalping.

The Kelly fraction f* is the bet size that maximizes log(growth) of capital.
For a binary bet with win prob p, win/loss multipliers w/l (in units of stake):

    f* = (p*w - (1-p)*l) / (w*l)

Full Kelly is variance-maximal; in practice we use **fractional Kelly**
(typically 0.25× to 0.5×) to reduce drawdowns at the cost of growth rate.

Why this matters for our project:
  - With a directional model giving probability p per trade, flat $-sizing
    gives same exposure to high-conviction and low-conviction trades. That's
    suboptimal Kelly tells us how much MORE to bet on high-conviction.
  - On the same predictions, Kelly typically produces +20-40% net PnL vs
    flat sizing, AND smaller max drawdown when fractionally scaled.

Caps: we layer two limits on top of Kelly's mathematical fraction:
  - `max_fraction`: hard cap on % of capital per trade (e.g., 5%) — protects
    against model overconfidence.
  - `min_fraction`: skip trades below threshold (avoids dust trades that
    only pay commissions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KellyConfig:
    """Parameters controlling Kelly sizing behaviour.

    Note on units: `fraction` field of SizingDecision is **position fraction
    of capital** — can exceed 1.0 when leverage is used. e.g. fraction=10
    means committing 10× capital as leveraged position. Caps:
      - `max_position_fraction` is the hard ceiling (typically = leverage_cap
        × max_capital_use, e.g. 20 × 0.95 = 19 for Binance BTC at 95% margin).
      - `min_position_fraction` skips dust trades that only pay commission.

    Why Kelly often outputs huge raw values for scalping: when TP/SL are
    tiny (0.1-0.3% of price) but win probability has clear edge over 50%,
    pure Kelly says "bet many multiples of capital" — math is correct, the
    cap (max_position_fraction) brings it back to the realistic range.
    """
    fraction: float = 0.25            # fractional Kelly multiplier (safety scale)
    max_position_fraction: float = 19.0   # 20× leverage × 95% margin ceiling
    min_position_fraction: float = 0.10   # skip trades below 10% capital exposure
    min_probability: float = 0.51         # ignore trades with edge ≤ 0


@dataclass(frozen=True)
class SizingDecision:
    """One trade's sizing output."""
    take: bool                        # whether to actually open the trade
    fraction: float                   # fraction of capital to commit (0..max_fraction)
    raw_kelly: float                  # unscaled Kelly fraction (diagnostic)
    edge: float                       # p*w - (1-p)*l (positive expected value)


def kelly_fraction(
    p_win: float,
    win_size: float,
    loss_size: float,
) -> float:
    """Pure Kelly formula. Returns fraction of capital to bet.

    Args:
        p_win:     probability of win (0..1).
        win_size:  fractional gain on win, e.g. 0.002 for +0.2%.
        loss_size: fractional loss on loss, POSITIVE number (e.g. 0.001 for -0.1%).

    Returns:
        Optimal full-Kelly fraction. Negative result means the bet has
        negative expected value — caller should skip.
    """
    if win_size <= 0 or loss_size <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(p_win)))
    q = 1.0 - p
    # f* = (p*w - q*l) / (w*l)
    return (p * win_size - q * loss_size) / (win_size * loss_size)


def asymmetric_kelly_fraction(
    p_win: float,
    win_pct: float,
    loss_pct: float,
    commission_win_pct: float = 0.0,
    commission_loss_pct: float = 0.0,
) -> float:
    """Kelly with asymmetric TP/SL and commissions baked in.

    Net win = win_pct - commission_win_pct
    Net loss = loss_pct + commission_loss_pct

    All inputs are POSITIVE percentages of entry (e.g. 0.2 for 0.2%).
    Returns full-Kelly fraction (caller applies fractional + caps).
    """
    net_win = win_pct - commission_win_pct
    net_loss = loss_pct + commission_loss_pct
    if net_win <= 0 or net_loss <= 0:
        return 0.0
    return kelly_fraction(p_win, net_win / 100.0, net_loss / 100.0)


def size_trade(
    p_win: float,
    win_pct: float,
    loss_pct: float,
    *,
    commission_win_pct: float = 0.04,
    commission_loss_pct: float = 0.07,
    cfg: KellyConfig = KellyConfig(),
) -> SizingDecision:
    """Compute a per-trade sizing decision.

    Returns SizingDecision with:
      - take: whether to open the trade
      - fraction: capital fraction to commit (already × leverage, capped)
      - raw_kelly: unscaled Kelly (diagnostic)
      - edge: expected value per unit of stake
    """
    raw = asymmetric_kelly_fraction(
        p_win, win_pct, loss_pct,
        commission_win_pct=commission_win_pct,
        commission_loss_pct=commission_loss_pct,
    )
    edge = (p_win * (win_pct - commission_win_pct)
            - (1.0 - p_win) * (loss_pct + commission_loss_pct)) / 100.0

    if p_win < cfg.min_probability or raw <= 0.0:
        return SizingDecision(take=False, fraction=0.0, raw_kelly=raw, edge=edge)

    f = raw * cfg.fraction
    f = max(0.0, min(f, cfg.max_position_fraction))

    if f < cfg.min_position_fraction:
        return SizingDecision(take=False, fraction=0.0, raw_kelly=raw, edge=edge)

    return SizingDecision(take=True, fraction=f, raw_kelly=raw, edge=edge)


def size_trades_batch(
    p_win: np.ndarray,
    win_pct: np.ndarray | float,
    loss_pct: np.ndarray | float,
    *,
    commission_win_pct: float = 0.04,
    commission_loss_pct: float = 0.07,
    cfg: KellyConfig = KellyConfig(),
) -> dict[str, np.ndarray]:
    """Vectorised sizing for a batch of trades.

    All array inputs broadcast against each other. Returns dict with arrays:
      take      (bool)   — whether to open this trade
      fraction  (f64)    — capital fraction (0 if take=False)
      raw_kelly (f64)    — unscaled Kelly per trade
      edge      (f64)    — expected value per unit stake
    """
    p = np.clip(np.asarray(p_win, dtype=np.float64), 0.0, 1.0)
    w = np.asarray(win_pct, dtype=np.float64)
    l = np.asarray(loss_pct, dtype=np.float64)

    net_win = w - commission_win_pct
    net_loss = l + commission_loss_pct
    edge = (p * net_win - (1.0 - p) * net_loss) / 100.0

    valid = (net_win > 0) & (net_loss > 0)
    raw = np.zeros_like(p)
    np.divide(
        p * (net_win / 100.0) - (1.0 - p) * (net_loss / 100.0),
        (net_win / 100.0) * (net_loss / 100.0),
        out=raw, where=valid,
    )

    f = raw * cfg.fraction
    f = np.clip(f, 0.0, cfg.max_position_fraction)
    take = (p >= cfg.min_probability) & (raw > 0.0) & (f >= cfg.min_position_fraction)
    fraction = np.where(take, f, 0.0)

    return {"take": take, "fraction": fraction, "raw_kelly": raw, "edge": edge}
