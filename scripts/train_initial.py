"""Phase 1: Initial model training from recorded data.

Usage:
    python scripts/train_initial.py --data-hours 72 --n-jobs 1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _parse_early() -> argparse.Namespace:
    """Parse args before importing any numpy/torch/sklearn — needed so the
    OMP/MKL thread caps are in place before those libraries cache them.
    """
    parser = argparse.ArgumentParser(description="Train initial CNN + XGBoost model")
    parser.add_argument("--data-hours", type=int, default=24, help="Hours of data to use")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Threads per library. 1 (default) isolates training to 1 core "
             "(safe on the 2-vCPU prod server). -1 uses all cores (dev only).",
    )
    return parser.parse_args()


# Parse + apply thread env BEFORE any heavy imports. NumPy MKL/OpenBLAS
# read OMP_NUM_THREADS/MKL_NUM_THREADS at import time and bake the value
# into their thread pool — setting the env afterwards is a no-op.
_args = _parse_early()
if _args.n_jobs > 0:
    os.environ["OMP_NUM_THREADS"] = str(_args.n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(_args.n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(_args.n_jobs)
    os.environ["NUMEXPR_NUM_THREADS"] = str(_args.n_jobs)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402 — must follow env setup
from src.trainer import Trainer       # noqa: E402


def main() -> None:
    args = _args  # use the args parsed above

    cfg = load_config("config.env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.log_dir / "training.log"),
        ],
    )

    print(f"\n{'='*50}")
    print(f"  Initial Model Training")
    print(f"  Data:     last {args.data_hours} hours")
    print(f"  Threads:  n_jobs={args.n_jobs}  "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")
    print(f"  Output:   {cfg.model_dir}")
    print(f"{'='*50}\n")

    trainer = Trainer(cfg)

    try:  # noqa: E501 — train_full handles build_samples 4-tuple internally
        result = trainer.train_full(hours=args.data_hours, n_jobs=args.n_jobs)
        print(f"\n{'='*50}")
        print(f"  Training Complete!")
        print(f"  Samples:  {result['samples']}")
        print(f"  Time:     {result['elapsed_min']:.1f} minutes")
        print(f"  Encoder:  {result['encoder_path']}")
        print(f"  XGBoost:  {result['xgb_path']}")
        print(f"{'='*50}\n")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Run the data recorder first: systemctl start scalper-recorder")
        print("Wait at least 1 hour for sufficient data.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
