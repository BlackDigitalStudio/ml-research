"""Phase 1: Walk-forward backtest with realistic simulation.

Simulates:
- 10ms execution delay
- 10% Post-Only rejection rate
- Maker commission 0.036% round-trip

Usage:
    python scripts/backtest.py --data-hours 72 --mode walk-forward --n-jobs 2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- jemalloc bootstrap ---------------------------------------------------
# Re-exec with LD_PRELOAD set so pandas-heavy build_samples uses jemalloc
# instead of glibc malloc. See scripts/train_initial.py for rationale.
# Guarded to only run in CLI mode — importing this module (e.g. from pytest)
# must not re-exec the interpreter or parse CLI args.
_JEMALLOC = Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")
if (
    __name__ == "__main__"
    and _JEMALLOC.exists()
    and "SCALPER_JEMALLOC_ACTIVE" not in os.environ
):
    os.environ["LD_PRELOAD"] = str(_JEMALLOC)
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ["SCALPER_JEMALLOC_ACTIVE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
# --- end jemalloc bootstrap -----------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--data-hours", type=int, default=24)
    parser.add_argument("--confidence", type=float, default=0.58)
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of data for training in walk-forward mode (default 0.8)")
    parser.add_argument("--mode", choices=["walk-forward", "model"], default="walk-forward",
                        help="walk-forward: train+test in one run; model: use pre-trained model from disk")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Threads per library. 1 (default) isolates training to 1 core. "
             "On the 3vCPU/8GB VPS, --n-jobs 2 is the recommended balance "
             "(leaves 1 core for recorder). -1 uses all cores (dev only).",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Ignore the sample cache and rebuild from raw parquet data.",
    )
    return parser.parse_args()


# Parse and apply thread env BEFORE importing numpy/torch/xgboost. NumPy
# MKL reads OMP_NUM_THREADS at import time and caches the value.
# Guarded to CLI mode so test imports don't call argparse on pytest's argv.
_args: argparse.Namespace | None = None
if __name__ == "__main__":
    _args = _parse_args()
    if _args.n_jobs > 0:
        os.environ["OMP_NUM_THREADS"] = str(_args.n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(_args.n_jobs)
        os.environ["OPENBLAS_NUM_THREADS"] = str(_args.n_jobs)
        os.environ["NUMEXPR_NUM_THREADS"] = str(_args.n_jobs)
    # Pin thread pools — see train_initial.py for rationale.
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import numpy as np    # noqa: E402 — must follow env setup
import pandas as pd   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import filters                                # noqa: E402
from src.config import load_config                     # noqa: E402
from src.live_sim import (                              # noqa: E402
    LiveSimConfig, SimDirection, simulate_trade,
)
from src.model import LOBEncoder, UP, DOWN, FLAT        # noqa: E402
from src.trainer import (                               # noqa: E402
    Trainer, WINDOW_SIZE, HORIZON, SIM_HORIZON,
)

logger = logging.getLogger("backtest")

# Post-only rejection rate is the execution artefact backtest still
# simulates — it gets dropped into live_sim as a pre-entry filter so the
# downstream metrics include 10% rejected entries. This is the one place
# we deliberately diverge from live_sim's deterministic outputs (live_sim
# itself does NOT model post-only rejection so labels stay stable).
POST_ONLY_REJECT_RATE = 0.10
EXECUTION_DELAY_TICKS = 1       # 100 ms = 1 tick
BASE_TP_PCT = 0.20              # base TP/SL used for per-sample adaptive scaling
BASE_SL_PCT = 0.10
BASE_TIMEOUT_SEC = 60.0
COMMISSION_WIN_PCT = 0.04       # maker + maker (all-maker exit)
COMMISSION_LOSS_PCT = 0.07      # maker + taker (SL / market exit)
# Deploy-gate thresholds — pulled from `handoff_current.md` "Trader
# priorities". Change here drives the `DEPLOY` block at the end.
DEPLOY_GATES = {
    "tp_wr_min": 0.52,
    "profit_factor_min": 1.10,
    "max_consec_losses_max": 10,
    "trades_per_day_min": 2.0,
    "max_drawdown_pct_max": 20.0,
}


@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    net_pnl: float
    reason: str
    duration_ticks: int
    gross_pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def max_consecutive_losses(self) -> int:
        max_cl = 0
        cl = 0
        for t in self.trades:
            if t.net_pnl <= 0:
                cl += 1
                max_cl = max(max_cl, cl)
            else:
                cl = 0
        return max_cl

    @property
    def sharpe_daily(self) -> float:
        if len(self.trades) < 2:
            return 0
        pnls = [t.net_pnl for t in self.trades]
        mean = np.mean(pnls)
        std = np.std(pnls)
        if std == 0:
            return 0
        # Annualize: assume ~30 trades/day
        return (mean / std) * np.sqrt(30)

    @property
    def avg_win(self) -> float:
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        return np.mean(losses) if losses else 0

    # ---- Trading-metric block (for DEPLOY verdict) ---------------------

    @property
    def tp_hit_rate(self) -> float:
        """% of trades closed at full TP (maps to trader-facing `TP-WR`)."""
        if not self.trades:
            return 0.0
        tps = sum(1 for t in self.trades if t.reason == "tp_hit")
        return tps / len(self.trades)

    @property
    def sl_hit_rate(self) -> float:
        if not self.trades:
            return 0.0
        sls = sum(1 for t in self.trades if "sl" in t.reason)
        return sls / len(self.trades)

    # ---- Canonical business metrics (owner-defined, see memory
    # ---- business_metrics_canonical.md). Categories 1-5 are mutually
    # ---- exclusive and MUST sum to 1.0 across the 9 live_sim reasons.

    # Business-metric categories (see memory/business_metrics_canonical.md).
    # Taxonomy mirrors live_sim.TradeOutcome.REASONS — every reason must
    # belong to exactly one bucket; the test enforces full coverage.
    _FULL_TP_REASONS = ("tp_hit",)
    _FULL_SL_REASONS = ("sl_hit", "fast_fill_adverse", "fast_fill_sl")
    _TIMEOUT_REASONS = ("timeout_limit", "timeout_market", "no_forward_data")
    _TRAILING_REASONS = (
        "trailing_sl_1", "trailing_sl_2",
        "partial_plus_trailing_sl_1", "partial_plus_trailing_sl_2",
    )
    _PARTIAL_TP_REASONS = ("partial_plus_tp",)

    def _share_by_reasons(self, reasons: tuple[str, ...]) -> float:
        if not self.trades:
            return 0.0
        n = sum(1 for t in self.trades if t.reason in reasons)
        return n / len(self.trades)

    @property
    def full_tp_rate(self) -> float:
        """% сделок закрытых на полный TP."""
        return self._share_by_reasons(self._FULL_TP_REASONS)

    @property
    def full_sl_rate(self) -> float:
        """% сделок закрытых на полный SL (включая fast-fill adverse и SL)."""
        return self._share_by_reasons(self._FULL_SL_REASONS)

    @property
    def timeout_rate(self) -> float:
        """% сделок закрытых по таймауту (limit, market, no_forward_data)."""
        return self._share_by_reasons(self._TIMEOUT_REASONS)

    @property
    def trailing_stop_rate(self) -> float:
        """% сделок закрытых по трейлинг-стопу (pure trailing + partial+trailing)."""
        return self._share_by_reasons(self._TRAILING_REASONS)

    @property
    def partial_tp_only_rate(self) -> float:
        """% сделок с частичным TP и полным закрытием остатка по TP."""
        return self._share_by_reasons(self._PARTIAL_TP_REASONS)

    @property
    def initial_equity(self) -> float:
        return self.equity_curve[0] if self.equity_curve else 0.0

    @property
    def gross_pnl_usd(self) -> float:
        """P&L до вычета комиссий/спрэда, в $."""
        return sum(t.pnl for t in self.trades)

    @property
    def gross_pnl_pct(self) -> float:
        """P&L до вычета комиссий/спрэда, в % к стартовому балансу."""
        eq0 = self.initial_equity
        return self.gross_pnl_usd / eq0 * 100 if eq0 > 0 else 0.0

    @property
    def net_pnl_pct(self) -> float:
        """P&L после комиссий и спрэда, в % к стартовому балансу."""
        eq0 = self.initial_equity
        return self.total_pnl / eq0 * 100 if eq0 > 0 else 0.0

    @property
    def trades_per_day(self) -> float:
        """Trades / (total backtest wall-time in days).

        Uses the bar-count approximation: each mid-price tick is 100 ms, so
        the session duration in days = `n_ticks / 864000`. The backtest
        caller sets `result.session_ticks` before printing metrics.
        """
        ticks = getattr(self, "session_ticks", 0)
        if ticks <= 0:
            return 0.0
        days = ticks / 864_000.0  # 10 ticks/sec × 86 400 sec/day
        return len(self.trades) / max(days, 1e-9)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd


def _trading_signal_mask(
    predictions: np.ndarray,
    confidences: np.ndarray,
    X_feat: np.ndarray,
    sample_ts_ms: np.ndarray | None,
    confidence_threshold: float,
) -> np.ndarray:
    """Vectorised Tier-2 filter check matching `src/filters.py`.

    Returns a boolean mask of samples where a non-FLAT trade would be taken
    by the live signal generator (minus the stateful filters that the
    backtest cannot replay). Used by `run_backtest` to decide which samples
    become trades.
    """
    mask = predictions != FLAT
    mask &= confidences >= confidence_threshold
    mask &= X_feat[:, 3] <= filters.MAX_SPREAD_USD           # spread
    vol_ratio = X_feat[:, 21]
    mask &= (vol_ratio > filters.VOL_BAND_LOW) & (vol_ratio < filters.VOL_BAND_HIGH)
    mask &= X_feat[:, 20] <= filters.SPOOF_SCORE_MAX         # spoof
    mask &= X_feat[:, 23] >= filters.HURST_MEAN_REVERTING_MAX  # hurst (static conf)

    imb = X_feat[:, 1]
    long_ok = (predictions == UP) & (imb > filters.IMBALANCE_LONG_MIN)
    short_ok = (predictions == DOWN) & (imb < filters.IMBALANCE_SHORT_MAX)
    mask &= long_ok | short_ok

    if sample_ts_ms is not None and len(sample_ts_ms) == len(predictions):
        sec_of_day = (sample_ts_ms // 1000) % (24 * 3600)
        hour_utc = sec_of_day // 3600
        mask &= ~((hour_utc >= filters.ASIA_NIGHT_START_UTC) & (hour_utc < filters.ASIA_NIGHT_END_UTC))
        min_of_day = (sample_ts_ms // 60_000) % (24 * 60)
        funding_ok = np.ones(len(predictions), dtype=bool)
        for h in filters.FUNDING_HOURS_UTC:
            center = h * 60
            delta = np.abs(min_of_day - center)
            delta = np.minimum(delta, 24 * 60 - delta)
            funding_ok &= delta > filters.FUNDING_GUARD_MIN
        mask &= funding_ok
    return mask


def run_backtest(
    mid_prices: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    imbalances: np.ndarray,
    spreads: np.ndarray,
    X_feat: np.ndarray | None = None,
    sample_ts_ms: np.ndarray | None = None,
    confidence_threshold: float = 0.58,
    initial_equity: float = 50.0,
    leverage: int = 20,
    position_size_pct: int = 95,
    seed: int = 42,
) -> BacktestResult:
    """Walk the test window, delegate trade realization to `live_sim`.

    Every Tier-1 divergence (partial / trailing / timeout limit→market /
    fast-fill / adaptive TP/SL / dynamic timeout / bid-ask entry) is handled
    by `live_sim.simulate_trade`. The backtest layer only decides WHEN to
    enter (matching the live `SignalGenerator` filter stack) and tracks
    equity across trades. Post-only rejection is still simulated here as
    an execution artefact because `live_sim` deliberately omits it.
    """
    result = BacktestResult()
    equity = initial_equity
    result.equity_curve.append(equity)

    rng = np.random.default_rng(seed)
    n = len(predictions)
    result.session_ticks = n

    if X_feat is None:
        # Backwards compatibility: reconstruct a minimal X_feat from the
        # legacy (imbalances, spreads) arguments. Vol_ratio / spoof / hurst
        # default to permissive values so the filter mask reduces to the
        # original 0.03 spread + imbalance gate.
        stub = np.zeros((n, 34), dtype=np.float32)
        stub[:, 1] = imbalances
        stub[:, 3] = spreads
        stub[:, 20] = 0.0           # spoof score
        stub[:, 21] = 1.0           # vol_ratio → inside band
        stub[:, 23] = 0.5           # hurst → pass regime gate
        X_feat = stub

    entry_mask = _trading_signal_mask(
        predictions, confidences, X_feat, sample_ts_ms, confidence_threshold,
    )

    # Iterate forward. When a trade opens we skip ahead by the trade's
    # duration before evaluating the next entry so trades don't overlap.
    i = 0
    while i < n:
        if not entry_mask[i]:
            i += 1
            continue

        pred = int(predictions[i])
        vol_ratio = float(X_feat[i, 21])
        tp_pct, sl_pct = filters.adaptive_tp_sl(vol_ratio, BASE_TP_PCT, BASE_SL_PCT)
        timeout_sec = filters.dynamic_timeout_sec(
            avg_volatility=1.0,
            current_volatility=max(vol_ratio, 1e-9),
            base_timeout_sec=BASE_TIMEOUT_SEC,
        )
        timeout_ticks = int(round(timeout_sec * 10.0))

        cfg = LiveSimConfig(
            tp_pct=tp_pct, sl_pct=sl_pct, timeout_ticks=timeout_ticks,
            commission_win_pct=COMMISSION_WIN_PCT,
            commission_loss_pct=COMMISSION_LOSS_PCT,
        )

        # Post-only rejection — if the entry order is rejected we lose
        # the tick and look for the next signal. Keeps the 10% failure
        # rate the old backtest modelled without bending live_sim.
        if rng.random() < POST_ONLY_REJECT_RATE:
            i += 1
            continue

        # Execution delay — enter 1 tick after the signal.
        entry_idx = min(i + EXECUTION_DELAY_TICKS, n - 1)
        entry_px = float(mid_prices[entry_idx])
        if entry_px <= 0:
            i += 1
            continue

        fwd_start = entry_idx + 1
        fwd_end = min(fwd_start + SIM_HORIZON, n)
        mid_path = mid_prices[fwd_start:fwd_end]
        if len(mid_path) < 2:
            break

        direction = SimDirection.LONG if pred == UP else SimDirection.SHORT
        outcome = simulate_trade(
            direction, entry_px, mid_path, cfg,
            fill_latency_ms=150.0,
        )

        # Sizing + equity accounting ----------------------------------
        notional = equity * leverage * position_size_pct / 100
        size_btc = round(notional / entry_px, 3)
        if size_btc * entry_px < 100.0:
            # Below Binance min notional — skip.
            i = entry_idx + 1
            continue

        # Net PnL in USD from the per-sample net_pnl_pct. gross & fees are
        # recomputed so the Trade record can still feed the existing
        # `avg_win`/`avg_loss`/`profit_factor` aggregates.
        net_pnl_usd = size_btc * entry_px * outcome.net_pnl_pct / 100
        gross_pnl_usd = size_btc * entry_px * outcome.gross_pnl_pct / 100
        fees_usd = gross_pnl_usd - net_pnl_usd

        equity += net_pnl_usd
        result.equity_curve.append(equity)
        result.trades.append(
            Trade(
                direction="LONG" if direction == SimDirection.LONG else "SHORT",
                entry_price=entry_px,
                exit_price=entry_px * (1 + outcome.gross_pnl_pct / 100 * (1 if direction == SimDirection.LONG else -1)),
                pnl=gross_pnl_usd,
                fees=fees_usd,
                net_pnl=net_pnl_usd,
                reason=outcome.exit_reason,
                duration_ticks=outcome.duration_ticks,
                gross_pnl_pct=outcome.gross_pnl_pct,
            )
        )

        # Skip past the trade so we don't re-enter mid-position.
        i = entry_idx + 1 + max(outcome.duration_ticks, 1)

    return result


def print_results(result: BacktestResult, label: str = "BACKTEST RESULTS") -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total trades:          {result.total_trades}")
    print(f"  Wins (net > 0):        {result.wins}")
    print(f"  Losses (net <= 0):     {result.losses}")
    print(f"  Net-WR (net > 0):      {result.win_rate:.1%}")
    print(f"  TP-WR (full TP hit):   {result.tp_hit_rate:.1%}")
    print(f"  SL-WR (any SL hit):    {result.sl_hit_rate:.1%}")
    # --- Canonical business metrics (business_metrics_canonical.md) ----
    print(f"  — Canonical metrics —")
    print(f"  Full TP %:             {result.full_tp_rate:.1%}")
    print(f"  Full SL %:             {result.full_sl_rate:.1%}")
    print(f"  Timeout %:             {result.timeout_rate:.1%}")
    print(f"  Trailing-stop %:       {result.trailing_stop_rate:.1%}")
    print(f"  Partial-TP-only %:     {result.partial_tp_only_rate:.1%}")
    print(f"  Gross P&L (pre-fees):  ${result.gross_pnl_usd:.2f}  ({result.gross_pnl_pct:+.2f}%)")
    print(f"  Net P&L (post-fees):   ${result.total_pnl:.2f}  ({result.net_pnl_pct:+.2f}%)")
    print(f"  — Other metrics —")
    print(f"  Profit factor:         {result.profit_factor:.2f}")
    print(f"  Total P&L:             ${result.total_pnl:.2f}")
    print(f"  Avg win:               ${result.avg_win:.4f}")
    print(f"  Avg loss:              ${result.avg_loss:.4f}")
    print(f"  Max drawdown ($):      ${result.max_drawdown:.2f}")
    print(f"  Max drawdown (%):      {result.max_drawdown_pct:.2f}%")
    print(f"  Max consec losses:     {result.max_consecutive_losses}")
    print(f"  Trades / day:          {result.trades_per_day:.2f}")
    print(f"  Sharpe (daily):        {result.sharpe_daily:.2f}")
    print(f"  Final equity:          ${result.equity_curve[-1]:.2f}")

    # --- Exit reason distribution --------------------------------------
    # Quick sanity check that the bot isn't permanently stuck in one
    # branch. A well-balanced run should see a mix of tp_hit / sl_hit /
    # trailing_* / timeout_*.
    reason_counts: dict[str, int] = {}
    for t in result.trades:
        reason_counts[t.reason] = reason_counts.get(t.reason, 0) + 1
    if reason_counts:
        print(f"  Exit reason distribution:")
        total = len(result.trades)
        for r, c in sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {r:<30s} {c:5d}  ({c/total*100:5.1f}%)")

    # --- Deploy verdict ------------------------------------------------
    # Gate criteria come from handoff_current.md → "Trader priorities".
    # The verdict is advisory only — a trader still has to eyeball the
    # equity curve and reason distribution before going live.
    print(f"  Deploy gates ({list(DEPLOY_GATES.keys())}):")
    checks = {
        "tp_wr_min": (result.tp_hit_rate, DEPLOY_GATES["tp_wr_min"], ">="),
        "profit_factor_min": (result.profit_factor, DEPLOY_GATES["profit_factor_min"], ">="),
        "max_consec_losses_max": (result.max_consecutive_losses, DEPLOY_GATES["max_consec_losses_max"], "<="),
        "trades_per_day_min": (result.trades_per_day, DEPLOY_GATES["trades_per_day_min"], ">="),
        "max_drawdown_pct_max": (result.max_drawdown_pct, DEPLOY_GATES["max_drawdown_pct_max"], "<="),
    }
    all_pass = True
    for name, (actual, target, op) in checks.items():
        if op == ">=":
            ok = actual >= target
        else:
            ok = actual <= target
        marker = "PASS" if ok else "FAIL"
        print(f"    [{marker}] {name:<25s} actual={actual:.3f} {op} target={target}")
        all_pass = all_pass and ok
    verdict = "DEPLOY" if all_pass else "DO NOT DEPLOY"
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}\n")


def save_results(result: BacktestResult, cfg, suffix: str = "") -> None:
    tag = f"_{suffix}" if suffix else ""

    eq_path = cfg.data_dir / f"backtest_equity{tag}.csv"
    pd.DataFrame({"equity": result.equity_curve}).to_csv(eq_path, index=False)
    print(f"Equity curve saved to {eq_path}")

    trades_path = cfg.data_dir / f"backtest_trades{tag}.csv"
    trades_data = [{
        "direction": t.direction,
        "entry": t.entry_price,
        "exit": t.exit_price,
        "pnl": t.net_pnl,
        "fees": t.fees,
        "reason": t.reason,
        "duration_ticks": t.duration_ticks,
    } for t in result.trades]
    pd.DataFrame(trades_data).to_csv(trades_path, index=False)
    print(f"Trades saved to {trades_path}")


def run_with_model(
    trainer: Trainer,
    X_lob: np.ndarray,
    X_feat: np.ndarray,
    mid_prices: np.ndarray,
    confidence: float,
    sample_ts_ms: np.ndarray | None = None,
) -> BacktestResult:
    """Run backtest using a pre-trained model from disk."""
    from src.model import HybridModel
    cfg = trainer._cfg
    model = HybridModel(cfg)
    if not model.load():
        print("Error: No trained model found. Run train_initial.py first.")
        sys.exit(1)

    logger.info("Generating predictions on %d samples...", len(X_lob))
    predictions = []
    confidences = []
    for i in range(len(X_lob)):
        pred, conf = model.predict(X_lob[i], X_feat[i])
        predictions.append(pred)
        confidences.append(conf)

    return _run_bt(
        mid_prices, np.array(predictions), np.array(confidences),
        X_feat, confidence, sample_ts_ms,
    )


def run_walk_forward(
    trainer: Trainer,
    X_lob: np.ndarray,
    X_feat: np.ndarray,
    y: np.ndarray,
    mid_prices: np.ndarray,
    target_pnl: np.ndarray,
    confidence: float,
    train_ratio: float = 0.8,
    n_jobs: int = 1,
    sample_ts_ms: np.ndarray | None = None,
) -> BacktestResult:
    """Walk-forward: train on first train_ratio, backtest on the rest."""
    n = len(X_lob)
    split = int(n * train_ratio)

    X_lob_train, X_lob_test = X_lob[:split], X_lob[split:]
    X_feat_train, X_feat_test = X_feat[:split], X_feat[split:]
    y_train, y_test = y[:split], y[split:]
    pnl_train = target_pnl[:split]
    mid_test = mid_prices[split:]
    ts_test = sample_ts_ms[split:] if sample_ts_ms is not None else None

    print(f"\n  Walk-forward split: train={split}, test={n - split}")
    print(f"  Test labels: UP={int((y_test==UP).sum())} DOWN={int((y_test==DOWN).sum())} FLAT={int((y_test==FLAT).sum())}")

    # Free train LOB early — CNN training copies to torch tensors
    encoder = trainer.train_cnn(X_lob_train, y_train, n_jobs=n_jobs)

    logger.info("Extracting embeddings...")
    emb_train = trainer.extract_embeddings(encoder, X_lob_train)
    emb_test = trainer.extract_embeddings(encoder, X_lob_test)
    del X_lob_train, X_lob_test

    logger.info("Training ensemble on %d samples...", split)
    (
        xgb_models, lgb_model, logreg, top5_features, calibrators, _regressor,
    ) = trainer.train_ensemble(
        emb_train, X_feat_train, y_train, n_jobs=n_jobs, target_pnl=pnl_train,
    )

    # Predict on test set using ensemble majority vote + calibrated confidence
    import xgboost as xgb_lib
    X_test_combined = np.hstack([emb_test, X_feat_test])
    n_test = len(X_test_combined)

    # Collect proba from all 5 models for majority vote
    all_proba = []
    for xgb_m in xgb_models:
        p = xgb_m.predict(xgb_lib.DMatrix(X_test_combined))
        if p.ndim == 1:
            p = p.reshape(-1, 3)
        all_proba.append(p)
    lgb_p = lgb_model.predict(X_test_combined)
    if lgb_p.ndim == 1:
        lgb_p = lgb_p.reshape(-1, 3)
    all_proba.append(lgb_p)
    lr_p = logreg.predict_proba(X_test_combined[:, top5_features])
    all_proba.append(lr_p)

    # Majority vote
    ensemble_votes = np.zeros((n_test, 3), dtype=np.int32)
    for p in all_proba:
        classes = p.argmax(axis=1)
        ensemble_votes[np.arange(n_test), classes] += 1

    predictions = ensemble_votes.argmax(axis=1)

    # Calibrated mean-proba confidence (matches HybridModel.predict at runtime)
    mean_proba = np.mean(all_proba, axis=0)  # (n_test, 3)
    cal_proba = trainer._apply_calibrators(calibrators, mean_proba)
    confidences = cal_proba[np.arange(n_test), predictions]

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, predictions)
    print(f"\n  Test accuracy: {acc:.4f}")
    # `labels=` + `zero_division=0` — matches the trainer.train_ensemble fix
    # so degenerate val sets no longer crash the walk-forward run.
    print(classification_report(
        y_test, predictions,
        labels=[UP, DOWN, FLAT],
        target_names=["UP", "DOWN", "FLAT"],
        zero_division=0,
    ))

    return _run_bt(
        mid_test, predictions, confidences, X_feat_test, confidence, ts_test,
    )


def _run_bt(
    mid_prices: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    X_feat: np.ndarray,
    confidence: float,
    sample_ts_ms: np.ndarray | None = None,
) -> BacktestResult:
    """Common backtest runner. mid_prices already aligned with predictions."""
    n = min(len(mid_prices), len(predictions))
    return run_backtest(
        mid_prices=mid_prices[:n],
        predictions=predictions[:n],
        confidences=confidences[:n],
        imbalances=X_feat[:n, 1],
        spreads=X_feat[:n, 3],
        X_feat=X_feat[:n],
        sample_ts_ms=sample_ts_ms[:n] if sample_ts_ms is not None else None,
        confidence_threshold=confidence,
    )


def main() -> None:
    args = _args  # parsed at import time so env vars could be set before numpy

    cfg = load_config("config.env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*50}")
    print(f"  Walk-Forward Backtest — {args.mode} mode")
    print(f"  Data:       last {args.data_hours} hours")
    print(f"  Threads:    n_jobs={args.n_jobs}  "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    print(f"  Commission: win={COMMISSION_WIN_PCT}% loss={COMMISSION_LOSS_PCT}%")
    print(f"  Post-Only rejection: {POST_ONLY_REJECT_RATE * 100:.0f}%")
    if args.mode == "walk-forward":
        print(f"  Train/test split: {args.train_ratio:.0%}/{1 - args.train_ratio:.0%}")
    print(f"{'='*50}")

    trainer = Trainer(cfg)

    # build_samples_cached loads data internally and frees DataFrames to save RAM.
    # Cache key is (schema_version, hours, newest_depth_mtime) — any new
    # data, bumped schema, or different window invalidates it.
    # --force-rebuild bypasses the cache.
    X_lob, X_feat, y, mid_prices, target_pnl = trainer.build_samples_cached(
        hours=args.data_hours, force_rebuild=args.force_rebuild,
    )

    # Cached samples already passed the global filter mask (including
    # time-of-day and funding blackout) during build_samples, so the
    # backtest layer skips those re-checks and passes `sample_ts_ms=None`.
    # Any non-FLAT prediction on a kept sample is time-valid by construction.
    result = None
    if args.mode == "model":
        result = run_with_model(
            trainer, X_lob, X_feat, mid_prices, args.confidence,
        )
    else:
        result = run_walk_forward(
            trainer, X_lob, X_feat, y, mid_prices, target_pnl,
            args.confidence, args.train_ratio, n_jobs=args.n_jobs,
        )

    print_results(result, f"BACKTEST RESULTS ({args.mode})")
    save_results(result, cfg, args.mode)


if __name__ == "__main__":
    main()
