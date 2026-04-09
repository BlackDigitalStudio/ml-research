"""Phase 1: Initial model training from recorded data.

Usage:
    python scripts/train_initial.py --data-hours 24
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train initial CNN + XGBoost model")
    parser.add_argument("--data-hours", type=int, default=24, help="Hours of data to use")
    args = parser.parse_args()

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
    print(f"  Data: last {args.data_hours} hours")
    print(f"  Output: {cfg.model_dir}")
    print(f"{'='*50}\n")

    trainer = Trainer(cfg)

    try:  # noqa: E501 — train_full handles build_samples 4-tuple internally
        result = trainer.train_full(hours=args.data_hours)
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
