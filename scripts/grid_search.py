"""Grid search Stage 1 — execution params only, teacher fixed.

Single teacher is trained once on 80% train split. Predictions cached for
the test window. Then we enumerate (TP, SL, conf, leverage, timeout,
partial, trailing) combinations and call `run_backtest` for each with
per-config overrides. Canonical 7 metrics + a few others recorded per
config to `grid_results.csv`.

Stage 1 does NOT re-train the teacher per TP/SL — labels were computed
under the baseline TP=0.20/SL=0.10 in `build_samples`, so wider-barrier
configs ride on the baseline-labeled signal. Stage 2 is re-label + re-train
for the winners (out of scope here).

Usage:
    python scripts/grid_search.py --data-hours 64

Output:
    data/grid_results.csv         — all configs, one row each
    data/grid_top5_<ranking>.csv  — top-5 per ranking criterion
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest import BacktestResult, run_backtest       # noqa: E402
from src.config import load_config                               # noqa: E402
from src.model import UP, DOWN, FLAT                             # noqa: E402
from src.teacher import TeacherConfig, train_teacher, predict_teacher  # noqa: E402
from src.trainer import Trainer, SIM_HORIZON                      # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grid_search")


# ---------------------------------------------------------------------------
# Grid definition — 2700 configs. Update here to re-size the experiment.
# ---------------------------------------------------------------------------


TP_SL_PAIRS: tuple[tuple[float, float], ...] = (
    # Ratio 2:1 — magnitude ladder
    (0.10, 0.05),
    (0.15, 0.075),
    (0.20, 0.10),    # baseline
    (0.30, 0.15),
    (0.40, 0.20),
    (0.60, 0.30),
    # Ratio 3:1 and 4:1 — asymmetric in favor of profit
    (0.30, 0.10),
    (0.40, 0.10),
    (0.60, 0.20),
    # Ratio 1.33:1 and 1.5:1 — more breathing room
    (0.20, 0.15),
    (0.30, 0.20),
    (0.40, 0.30),
    # Ratio 1:1 — symmetric
    (0.10, 0.10),
    (0.20, 0.20),
    (0.30, 0.30),
)
CONFIDENCES = (0.55, 0.60, 0.65, 0.70, 0.75)
LEVERAGES = (1, 10, 20)
TIMEOUTS_SEC = (30, 60, 120)
PARTIALS = (True, False)
TRAILINGS = (True, False)


@dataclass
class ConfigRow:
    config_id: int
    tp_pct: float
    sl_pct: float
    confidence: float
    leverage: int
    timeout_sec: int
    partial: bool
    trailing: bool

    # Metrics filled after run_backtest
    trades: int = 0
    full_tp_pct: float = 0.0
    full_sl_pct: float = 0.0
    timeout_pct: float = 0.0
    trailing_pct: float = 0.0
    partial_tp_only_pct: float = 0.0
    gross_usd: float = 0.0
    gross_pct: float = 0.0
    net_usd: float = 0.0
    net_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_daily: float = 0.0
    max_dd_pct: float = 0.0
    max_consec_losses: int = 0
    trades_per_day: float = 0.0
    # Derived ranking metric
    tp_weighted_net_pct: float = 0.0     # R4
    min_net_dd_pct: float = 0.0          # R5


def _fill_metrics_from_result(row: ConfigRow, result: BacktestResult) -> None:
    row.trades = result.total_trades
    row.full_tp_pct = result.full_tp_rate * 100
    row.full_sl_pct = result.full_sl_rate * 100
    row.timeout_pct = result.timeout_rate * 100
    row.trailing_pct = result.trailing_stop_rate * 100
    row.partial_tp_only_pct = result.partial_tp_only_rate * 100
    row.gross_usd = result.gross_pnl_usd
    row.gross_pct = result.gross_pnl_pct
    row.net_usd = result.total_pnl
    row.net_pct = result.net_pnl_pct
    # profit_factor returns inf when there are no losses — clamp for CSV
    pf = result.profit_factor
    row.profit_factor = float(pf) if np.isfinite(pf) else 999.0
    row.sharpe_daily = result.sharpe_daily
    row.max_dd_pct = result.max_drawdown_pct
    row.max_consec_losses = result.max_consecutive_losses
    row.trades_per_day = result.trades_per_day
    row.tp_weighted_net_pct = row.full_tp_pct * row.net_pct
    row.min_net_dd_pct = min(row.net_pct, -row.max_dd_pct)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search Stage 1")
    p.add_argument("--data-hours", type=int, default=64)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--teacher-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--min-notional-usd", type=float, default=100.0,
                   help="Binance min notional gate. Set to 1.0 for x1-leverage "
                        "paper tests where small equity won't meet the $100 floor.")
    p.add_argument("--teacher-checkpoint", type=str, default="",
                   help="Path to pre-trained teacher .pt. Empty → train fresh.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("  Grid Search Stage 1 — execution-param sweep")
    logger.info("  TP/SL pairs: %d | Confidences: %d | Leverages: %d",
                len(TP_SL_PAIRS), len(CONFIDENCES), len(LEVERAGES))
    logger.info("  Timeouts: %d | Partials: %d | Trailings: %d",
                len(TIMEOUTS_SEC), len(PARTIALS), len(TRAILINGS))
    total = (len(TP_SL_PAIRS) * len(CONFIDENCES) * len(LEVERAGES) *
             len(TIMEOUTS_SEC) * len(PARTIALS) * len(TRAILINGS))
    logger.info("  Total configs: %d", total)
    logger.info("=" * 60)

    cfg = load_config()
    trainer = Trainer(cfg)

    # ---- Load sample cache ----------------------------------------------
    X_lob, X_feat, y, mid_prices, target_pnl = trainer.build_samples_cached(
        hours=args.data_hours, force_rebuild=args.force_rebuild,
    )
    n = len(y)
    logger.info("Samples: %d | features: %d", n, X_feat.shape[1])

    # ---- Walk-forward split ---------------------------------------------
    train_end = int(n * args.train_ratio)
    gap = 650
    test_start = train_end + gap
    test_end = n - SIM_HORIZON
    if test_start >= test_end:
        raise ValueError(f"Not enough test samples: start={test_start} end={test_end}")
    logger.info("Walk-forward: train 0..%d gap %d test %d..%d (%d test samples)",
                train_end, gap, test_start, test_end, test_end - test_start)

    # ---- Train teacher once --------------------------------------------
    if args.teacher_checkpoint:
        from src.teacher import load_teacher
        logger.info("Loading teacher from %s", args.teacher_checkpoint)
        teacher = load_teacher(Path(args.teacher_checkpoint),
                                num_feat=X_feat.shape[1])
    else:
        tcfg = TeacherConfig(epochs=args.teacher_epochs,
                              batch_size=args.teacher_batch)
        teacher, metrics = train_teacher(
            X_lob=X_lob[:train_end],
            X_feat=X_feat[:train_end],
            y=y[:train_end],
            target_pnl=target_pnl[:train_end],
            cfg=tcfg, val_frac=0.15, gap=gap, seed=args.seed,
        )
        logger.info("Teacher trained. bal_acc=%.4f params=%.2fM",
                    metrics["best_bal_acc"], metrics["params_M"])
        from src.teacher import save_teacher
        save_teacher(teacher, cfg.model_dir / "teacher_grid.pt")

    # ---- Predict once on test window -----------------------------------
    preds, confs, reg = predict_teacher(
        teacher,
        X_lob=X_lob[test_start:test_end],
        X_feat=X_feat[test_start:test_end],
        batch_size=512,
    )
    dist = {int(k): int((preds == k).sum()) for k in (UP, DOWN, FLAT)}
    logger.info("Predictions: UP=%d DOWN=%d FLAT=%d", dist[UP], dist[DOWN], dist[FLAT])

    # Pre-slice arrays for run_backtest — avoid repeated slicing in hot loop
    mid_test = mid_prices[test_start:test_end]
    feat_test = X_feat[test_start:test_end]
    imb_test = feat_test[:, 1]
    spr_test = feat_test[:, 3]

    # ---- Grid search main loop ------------------------------------------
    rows: list[ConfigRow] = []
    config_id = 0
    t_loop_start = time.time()

    for (tp_pct, sl_pct), conf_thr, lev, timeout_s, partial, trailing in itertools.product(
        TP_SL_PAIRS, CONFIDENCES, LEVERAGES, TIMEOUTS_SEC, PARTIALS, TRAILINGS,
    ):
        config_id += 1
        row = ConfigRow(
            config_id=config_id,
            tp_pct=tp_pct, sl_pct=sl_pct, confidence=conf_thr,
            leverage=lev, timeout_sec=timeout_s,
            partial=partial, trailing=trailing,
        )
        result = run_backtest(
            mid_prices=mid_test,
            predictions=preds,
            confidences=confs,
            imbalances=imb_test,
            spreads=spr_test,
            X_feat=feat_test,
            sample_ts_ms=None,
            confidence_threshold=conf_thr,
            leverage=lev,
            tp_pct_override=tp_pct,
            sl_pct_override=sl_pct,
            timeout_sec_override=float(timeout_s),
            partial_enabled=partial,
            trailing_enabled=trailing,
            min_notional_usd=args.min_notional_usd,
        )
        _fill_metrics_from_result(row, result)
        rows.append(row)

        if config_id % 100 == 0:
            elapsed = time.time() - t_loop_start
            rate = config_id / max(elapsed, 1e-9)
            eta = (total - config_id) / max(rate, 1e-9)
            logger.info("  %d/%d configs (%.1f/s, ETA %.1fm)",
                         config_id, total, rate, eta / 60)

    logger.info("Grid complete: %d configs in %.1fs", len(rows), time.time() - t_loop_start)

    # ---- Save CSV --------------------------------------------------------
    out_dir = cfg.data_dir
    out_csv = out_dir / "grid_results.csv"
    fieldnames = list(asdict(rows[0]).keys())
    with open(out_csv, "w") as fh:
        fh.write(",".join(fieldnames) + "\n")
        for r in rows:
            d = asdict(r)
            fh.write(",".join(str(d[k]) for k in fieldnames) + "\n")
    logger.info("Saved %s", out_csv)

    # ---- Top-5 per ranking criterion ------------------------------------
    rankings = {
        "R1_net_pct": lambda r: r.net_pct,
        "R2_profit_factor": lambda r: r.profit_factor if r.trades > 0 else -1e9,
        "R3_sharpe": lambda r: r.sharpe_daily,
        "R4_tp_weighted_net": lambda r: r.tp_weighted_net_pct,
        "R5_min_net_dd": lambda r: r.min_net_dd_pct,
    }
    # Additional filter: only rank configs with >= 5 trades (otherwise PF is
    # meaningless). R2 applies its own stricter filter above.
    meaningful = [r for r in rows if r.trades >= 5]
    logger.info("Configs with ≥5 trades: %d / %d", len(meaningful), len(rows))

    summary = {"generated_at": time.time(), "total_configs": len(rows),
               "meaningful_configs": len(meaningful), "top_by_ranking": {}}

    for name, keyfn in rankings.items():
        pool = meaningful if name != "R1_net_pct" else rows
        top = sorted(pool, key=keyfn, reverse=True)[:5]
        summary["top_by_ranking"][name] = [asdict(r) for r in top]
        out_top = out_dir / f"grid_top5_{name}.csv"
        with open(out_top, "w") as fh:
            fh.write(",".join(fieldnames) + "\n")
            for r in top:
                d = asdict(r)
                fh.write(",".join(str(d[k]) for k in fieldnames) + "\n")
        logger.info("Saved %s", out_top)

    # ---- Configs making top-5 in ≥ 3 of 5 rankings — strong candidates --
    ranked_in: dict[int, int] = {}
    for keyfn in rankings.values():
        pool = meaningful
        top = sorted(pool, key=keyfn, reverse=True)[:5]
        for r in top:
            ranked_in[r.config_id] = ranked_in.get(r.config_id, 0) + 1
    strong = sorted(
        [r for r in rows if ranked_in.get(r.config_id, 0) >= 3],
        key=lambda r: ranked_in[r.config_id], reverse=True,
    )
    summary["strong_candidates"] = [
        {"config_id": r.config_id, "ranked_in": ranked_in[r.config_id],
         "tp_pct": r.tp_pct, "sl_pct": r.sl_pct, "confidence": r.confidence,
         "leverage": r.leverage, "timeout_sec": r.timeout_sec,
         "partial": r.partial, "trailing": r.trailing,
         "net_pct": r.net_pct, "profit_factor": r.profit_factor,
         "trades": r.trades}
        for r in strong
    ]
    logger.info("Strong candidates (top-5 in ≥3 rankings): %d", len(strong))
    if strong:
        for r in strong[:10]:
            logger.info("  cfg#%d (top-5×%d): TP=%.2f SL=%.2f conf=%.2f "
                        "lev=x%d timeout=%ds partial=%s trailing=%s → "
                        "net=%+.2f%% PF=%.2f trades=%d",
                        r.config_id, ranked_in[r.config_id],
                        r.tp_pct, r.sl_pct, r.confidence, r.leverage,
                        r.timeout_sec, r.partial, r.trailing,
                        r.net_pct, r.profit_factor, r.trades)

    with open(out_dir / "grid_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Total wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
