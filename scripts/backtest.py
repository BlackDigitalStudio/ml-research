"""Phase 1: Walk-forward backtest with realistic simulation.

Simulates:
- 10ms execution delay
- 10% Post-Only rejection rate
- Maker commission 0.036% round-trip

Usage:
    python scripts/backtest.py --data-hours 72 --mode walk-forward --n-jobs 1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


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
        help="Threads per library. 1 (default) isolates training to 1 core "
             "(safe on the 2-vCPU prod server). -1 uses all cores for "
             "faster dev iteration — only when the live bot is off.",
    )
    return parser.parse_args()


# Parse and apply thread env BEFORE importing numpy/torch/xgboost. NumPy
# MKL reads OMP_NUM_THREADS at import time and caches the value.
_args = _parse_args()
if _args.n_jobs > 0:
    os.environ["OMP_NUM_THREADS"] = str(_args.n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(_args.n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(_args.n_jobs)
    os.environ["NUMEXPR_NUM_THREADS"] = str(_args.n_jobs)

import numpy as np    # noqa: E402 — must follow env setup
import pandas as pd   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config                     # noqa: E402
from src.model import LOBEncoder, UP, DOWN, FLAT        # noqa: E402
from src.trainer import Trainer, WINDOW_SIZE, HORIZON   # noqa: E402

logger = logging.getLogger("backtest")

POST_ONLY_REJECT_RATE = 0.10
EXECUTION_DELAY_TICKS = 1  # 100ms = 1 tick
STOP_LOSS_PCT = 0.10       # 0.1% of price
TAKE_PROFIT_PCT = 0.20     # 0.2% of price (2:1 ratio)
COMMISSION_WIN_PCT = 0.04  # maker + maker
COMMISSION_LOSS_PCT = 0.07 # maker + taker (stop-market)
POSITION_TIMEOUT_TICKS = 600  # 60 seconds


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


def run_backtest(
    mid_prices: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    imbalances: np.ndarray,
    spreads: np.ndarray,
    confidence_threshold: float = 0.58,
    initial_equity: float = 50.0,
    leverage: int = 20,
    position_size_pct: int = 95,
) -> BacktestResult:
    result = BacktestResult()
    equity = initial_equity
    result.equity_curve.append(equity)

    n = len(predictions)
    i = 0
    in_position = False
    direction = ""
    entry_price = 0.0
    entry_tick = 0
    position_size = 0.0

    while i < n:
        if not in_position:
            pred = predictions[i]
            conf = confidences[i]
            imb = imbalances[i]
            spread = spreads[i]

            # Entry filters
            if pred == FLAT or conf < confidence_threshold:
                i += 1
                continue
            if spread > 0.03:
                i += 1
                continue

            if pred == UP and imb > 0.15:
                direction = "LONG"
            elif pred == DOWN and imb < -0.15:
                direction = "SHORT"
            else:
                i += 1
                continue

            # Post-Only rejection simulation
            if np.random.random() < POST_ONLY_REJECT_RATE:
                i += 1
                continue

            # Execution delay
            entry_idx = min(i + EXECUTION_DELAY_TICKS, n - 1)
            entry_price = mid_prices[entry_idx]
            entry_tick = entry_idx
            # Dynamic position size from current equity
            notional = equity * leverage * position_size_pct / 100
            position_size = notional / entry_price
            position_size = round(position_size, 3)
            in_position = True
            i = entry_idx + 1
            continue

        # In position — check exit (%-based TP/SL)
        current_price = mid_prices[i]
        ticks_held = i - entry_tick

        tp_dist = entry_price * TAKE_PROFIT_PCT / 100
        sl_dist = entry_price * STOP_LOSS_PCT / 100

        if direction == "LONG":
            pnl_raw = (current_price - entry_price) * position_size
            hit_tp = current_price >= entry_price + tp_dist
            hit_sl = current_price <= entry_price - sl_dist
        else:
            pnl_raw = (entry_price - current_price) * position_size
            hit_tp = current_price <= entry_price - tp_dist
            hit_sl = current_price >= entry_price + sl_dist

        reason = ""
        if hit_tp:
            reason = "take_profit"
        elif hit_sl:
            reason = "stop_loss"
        elif ticks_held >= POSITION_TIMEOUT_TICKS:
            reason = "timeout"

        if reason:
            # Different commission for win vs loss
            notional_val = entry_price * position_size
            if reason == "stop_loss":
                fees = notional_val * COMMISSION_LOSS_PCT / 100
            else:
                fees = notional_val * COMMISSION_WIN_PCT / 100
            net = pnl_raw - fees
            equity += net
            result.equity_curve.append(equity)

            result.trades.append(Trade(
                direction=direction,
                entry_price=entry_price,
                exit_price=current_price,
                pnl=pnl_raw,
                fees=fees,
                net_pnl=net,
                reason=reason,
                duration_ticks=ticks_held,
            ))

            in_position = False
            i += 1
            continue

        i += 1

    return result


def print_results(result: BacktestResult, label: str = "BACKTEST RESULTS") -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Total trades:          {result.total_trades}")
    print(f"  Wins:                  {result.wins}")
    print(f"  Losses:                {result.losses}")
    print(f"  Win rate:              {result.win_rate:.1%}")
    print(f"  Profit factor:         {result.profit_factor:.2f}")
    print(f"  Total P&L:             ${result.total_pnl:.2f}")
    print(f"  Avg win:               ${result.avg_win:.4f}")
    print(f"  Avg loss:              ${result.avg_loss:.4f}")
    print(f"  Max drawdown:          ${result.max_drawdown:.2f}")
    print(f"  Max consec losses:     {result.max_consecutive_losses}")
    print(f"  Sharpe (daily):        {result.sharpe_daily:.2f}")
    print(f"  Final equity:          ${result.equity_curve[-1]:.2f}")
    print(f"{'='*50}\n")


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
        X_feat, confidence,
    )


def run_walk_forward(
    trainer: Trainer,
    X_lob: np.ndarray,
    X_feat: np.ndarray,
    y: np.ndarray,
    mid_prices: np.ndarray,
    confidence: float,
    train_ratio: float = 0.8,
    n_jobs: int = 1,
) -> BacktestResult:
    """Walk-forward: train on first train_ratio, backtest on the rest."""
    n = len(X_lob)
    split = int(n * train_ratio)

    X_lob_train, X_lob_test = X_lob[:split], X_lob[split:]
    X_feat_train, X_feat_test = X_feat[:split], X_feat[split:]
    y_train, y_test = y[:split], y[split:]
    mid_test = mid_prices[split:]

    print(f"\n  Walk-forward split: train={split}, test={n - split}")
    print(f"  Test labels: UP={int((y_test==UP).sum())} DOWN={int((y_test==DOWN).sum())} FLAT={int((y_test==FLAT).sum())}")

    # Free train LOB early — CNN training copies to torch tensors
    encoder = trainer.train_cnn(X_lob_train, y_train, n_jobs=n_jobs)

    logger.info("Extracting embeddings...")
    emb_train = trainer.extract_embeddings(encoder, X_lob_train)
    emb_test = trainer.extract_embeddings(encoder, X_lob_test)
    del X_lob_train, X_lob_test

    logger.info("Training ensemble on %d samples...", split)
    xgb_models, lgb_model, logreg, top5_features, calibrators = trainer.train_ensemble(
        emb_train, X_feat_train, y_train, n_jobs=n_jobs,
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
    print(classification_report(
        y_test, predictions, target_names=["UP", "DOWN", "FLAT"],
    ))

    return _run_bt(mid_test, predictions, confidences, X_feat_test, confidence)


def _run_bt(
    mid_prices: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    X_feat: np.ndarray,
    confidence: float,
) -> BacktestResult:
    """Common backtest runner. mid_prices already aligned with predictions."""
    n = min(len(mid_prices), len(predictions))
    return run_backtest(
        mid_prices=mid_prices[:n],
        predictions=predictions[:n],
        confidences=confidences[:n],
        imbalances=X_feat[:n, 1],
        spreads=X_feat[:n, 3],
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

    # build_samples loads data internally and frees DataFrames to save RAM
    X_lob, X_feat, y, mid_prices = trainer.build_samples(hours=args.data_hours)

    if args.mode == "model":
        result = run_with_model(trainer, X_lob, X_feat, mid_prices, args.confidence)
    else:
        result = run_walk_forward(
            trainer, X_lob, X_feat, y, mid_prices,
            args.confidence, args.train_ratio, n_jobs=args.n_jobs,
        )

    print_results(result, f"BACKTEST RESULTS ({args.mode})")
    save_results(result, cfg, args.mode)


if __name__ == "__main__":
    main()
