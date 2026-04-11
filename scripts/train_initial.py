"""Phase 1: Initial model training from recorded data.

Usage:
    python scripts/train_initial.py --data-hours 72 --n-jobs 2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# --- jemalloc bootstrap ---------------------------------------------------
# Re-exec ourselves with LD_PRELOAD set so pandas-heavy build_samples uses
# jemalloc instead of glibc malloc (20-40% RAM savings, no list-column
# fragmentation). Guarded by an env var so the re-exec happens at most once.
_JEMALLOC = Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")
if _JEMALLOC.exists() and "SCALPER_JEMALLOC_ACTIVE" not in os.environ:
    os.environ["LD_PRELOAD"] = str(_JEMALLOC)
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ["SCALPER_JEMALLOC_ACTIVE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
# --- end jemalloc bootstrap -----------------------------------------------


def _parse_early() -> argparse.Namespace:
    """Parse args before importing any numpy/torch/sklearn — needed so the
    OMP/MKL thread caps are in place before those libraries cache them.
    """
    parser = argparse.ArgumentParser(description="Train initial CNN + XGBoost model")
    parser.add_argument("--data-hours", type=int, default=24, help="Hours of data to use")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Threads per library. 1 (default) isolates training to 1 core. "
             "On the 3vCPU/8GB VPS, --n-jobs 2 is the recommended balance "
             "(leaves 1 core for recorder). -1 uses all cores (dev only).",
    )
    parser.add_argument(
        "--warm-start-encoder", type=Path, default=None,
        help="Path to a previously saved encoder .pt file. CNN loads these "
             "weights and runs only 5 fine-tune epochs instead of training "
             "from scratch. Used for daily prod retrain cycles.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Ignore the sample cache and rebuild X_lob/X_feat/y from raw "
             "parquet data. Use after code changes to features/labels.",
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
# Pin MKL/OMP thread pool sizes — dynamic scaling causes context-switching
# overhead on a small core count and measurably slows matrix kernels.
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
# CPU affinity for Intel OMP: tightly pack threads onto physical cores.
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

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
        result = trainer.train_full(
            hours=args.data_hours,
            n_jobs=args.n_jobs,
            warm_start_encoder=args.warm_start_encoder,
            force_rebuild=args.force_rebuild,
        )
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
