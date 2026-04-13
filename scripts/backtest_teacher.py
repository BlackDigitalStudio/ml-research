"""Walk-forward backtest driver for the Transformer teacher.

Parallel to scripts/backtest.py (CNN+ensemble), kept separate so the
validated CNN baseline stays intact. Consumes the same cached
build_samples output, trains the Transformer teacher, runs the existing
`run_backtest` simulation layer on teacher predictions.

Usage:
    python scripts/backtest_teacher.py --data-hours 64 --n-jobs 16

No jemalloc re-exec (on a fat pod we don't need it); no env-var thread
pinning beyond OMP (teacher training is GPU-bound — CPU threads only
help the build_samples pre-work).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Basic thread env pre-numpy import.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import numpy as np   # noqa: E402
import torch          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest import (  # noqa: E402
    BacktestResult, print_results, run_backtest,
)
from src.config import load_config      # noqa: E402
from src.model import UP, DOWN, FLAT     # noqa: E402
from src.teacher import (                # noqa: E402
    MultiStreamTransformer, TeacherConfig,
    train_teacher, predict_teacher, save_teacher,
)
from src.trainer import Trainer, SIM_HORIZON, WINDOW_SIZE  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_teacher")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher walk-forward backtest")
    p.add_argument("--data-hours", type=int, default=64)
    p.add_argument("--confidence", type=float, default=0.58)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("=" * 60)
    logger.info("  Transformer Teacher Backtest")
    logger.info("  Data hours: %d | epochs: %d | batch: %d | n_jobs: %d",
                args.data_hours, args.epochs, args.batch_size, args.n_jobs)
    logger.info("=" * 60)

    cfg = load_config()
    trainer = Trainer(cfg)

    # ---- Build / reuse sample cache --------------------------------------
    X_lob, X_feat, y, mid_prices, target_pnl = trainer.build_samples_cached(
        hours=args.data_hours, force_rebuild=args.force_rebuild,
    )
    n = len(y)
    logger.info("Samples: %d | features: %d | mid[0]=%.2f", n, X_feat.shape[1], mid_prices[0])

    # ---- Walk-forward split on time-ordered data ------------------------
    train_end = int(n * args.train_ratio)
    gap = 650
    val_end = train_end
    test_start = train_end + gap
    test_end = n - SIM_HORIZON  # leave forward slice for live_sim in backtest

    if test_start >= test_end:
        raise ValueError(f"Not enough test samples: start={test_start} end={test_end}")

    logger.info("Walk-forward: train 0..%d, gap %d, test %d..%d",
                train_end, gap, test_start, test_end)

    # ---- Train teacher ---------------------------------------------------
    tcfg = TeacherConfig(epochs=args.epochs, batch_size=args.batch_size)
    teacher, metrics = train_teacher(
        X_lob=X_lob[:train_end],
        X_feat=X_feat[:train_end],
        y=y[:train_end],
        target_pnl=target_pnl[:train_end],
        cfg=tcfg,
        val_frac=0.15,
        gap=gap,
        seed=args.seed,
    )
    logger.info("Teacher trained. Best bal_acc=%.4f params=%.2fM",
                metrics["best_bal_acc"], metrics["params_M"])

    save_teacher(teacher, cfg.model_dir / "teacher_latest.pt")
    logger.info("Teacher saved to %s", cfg.model_dir / "teacher_latest.pt")

    # ---- Predict on test window -----------------------------------------
    preds, confs, reg = predict_teacher(
        teacher,
        X_lob=X_lob[test_start:test_end],
        X_feat=X_feat[test_start:test_end],
        batch_size=512,
    )

    # Distribution of predictions
    dist = {int(k): int((preds == k).sum()) for k in (UP, DOWN, FLAT)}
    logger.info("Test preds: UP=%d DOWN=%d FLAT=%d",
                dist[UP], dist[DOWN], dist[FLAT])

    test_acc = (preds == y[test_start:test_end]).mean()
    logger.info("Test accuracy: %.4f", test_acc)

    # ---- Walk-forward simulation via live_sim ----------------------------
    # run_backtest expects mid_prices over the test slice. We pass the SAME
    # slice predictions correspond to, so sample i aligns with mid[i].
    result = run_backtest(
        mid_prices=mid_prices[test_start:test_end],
        predictions=preds,
        confidences=confs,
        imbalances=X_feat[test_start:test_end, 1],
        spreads=X_feat[test_start:test_end, 3],
        X_feat=X_feat[test_start:test_end],
        sample_ts_ms=None,
        confidence_threshold=args.confidence,
    )
    print_results(result, label="TEACHER BACKTEST RESULTS (walk-forward)")

    # ---- Persist artefacts ----------------------------------------------
    out_dir = cfg.data_dir
    equity = np.asarray(result.equity_curve, dtype=np.float64)
    np.savetxt(out_dir / "backtest_equity_teacher.csv", equity,
               header="equity", comments="", fmt="%.6f")
    with open(out_dir / "backtest_trades_teacher.csv", "w") as fh:
        fh.write("direction,entry,exit,pnl,fees,net_pnl,reason,gross_pct\n")
        for t in result.trades:
            fh.write(f"{t.direction},{t.entry_price:.2f},{t.exit_price:.2f},"
                     f"{t.pnl:.4f},{t.fees:.4f},{t.net_pnl:.4f},{t.reason},"
                     f"{t.gross_pnl_pct:.4f}\n")
    logger.info("Artefacts: %s, %s",
                out_dir / "backtest_equity_teacher.csv",
                out_dir / "backtest_trades_teacher.csv")


if __name__ == "__main__":
    main()
