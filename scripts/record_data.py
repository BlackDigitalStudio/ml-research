"""Phase 0: Standalone data recorder for ETHUSDT depth + trades.

Connects to Binance WebSocket and writes raw data to Parquet files.
Runs independently — no model, no trading.

Usage:
    python scripts/record_data.py
    # Ctrl+C to stop gracefully
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.ws_client import BinanceWSClient
from src.order_book import OrderBook
from src.recorder import Recorder

logger = logging.getLogger("recorder")


async def main() -> None:
    cfg = load_config("config.env")

    log_fmt = "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.log_dir / "recorder.log"),
        ],
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    logger.info("=" * 50)
    logger.info("Data recorder starting — %s", cfg.symbol)
    logger.info("=" * 50)

    ws = BinanceWSClient(cfg)
    ob = OrderBook(ws, cfg.symbol)
    rec = Recorder(cfg)

    depth_count = 0
    trade_count = 0
    t_start = time.monotonic()

    def on_snapshot(snap) -> None:
        nonlocal depth_count
        rec.record_depth(snap)
        depth_count += 1

    async def on_trade(data: dict) -> None:
        nonlocal trade_count
        rec.record_trade(data)
        trade_count += 1

    ob.on_snapshot(on_snapshot)

    await ws.start()
    ws.on_aggtrade(on_trade)
    await asyncio.sleep(2)
    await ob.start()
    await rec.start()

    logger.info("Recording to %s", cfg.data_dir)

    # Graceful shutdown
    shutdown = asyncio.Event()

    def _stop() -> None:
        logger.info("Shutdown signal received")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    # Stats loop
    while not shutdown.is_set():
        await asyncio.sleep(60)
        elapsed = time.monotonic() - t_start
        hours = elapsed / 3600
        depth_dir = cfg.data_dir / "depth"
        trades_dir = cfg.data_dir / "trades"
        depth_files = len(list(depth_dir.glob("*.parquet")))
        trade_files = len(list(trades_dir.glob("*.parquet")))
        depth_size = sum(f.stat().st_size for f in depth_dir.glob("*.parquet"))
        trade_size = sum(f.stat().st_size for f in trades_dir.glob("*.parquet"))
        total_mb = (depth_size + trade_size) / 1024 / 1024

        logger.info(
            "Stats: %.1fh | depth=%d msgs (%d files, %.1f MB) | trades=%d msgs (%d files, %.1f MB) | total=%.1f MB",
            hours, depth_count, depth_files, depth_size / 1024 / 1024,
            trade_count, trade_files, trade_size / 1024 / 1024, total_mb,
        )

    logger.info("Flushing final data...")
    await rec.stop()
    await ws.stop()

    elapsed = time.monotonic() - t_start
    logger.info(
        "Recording complete: %.1f hours, %d depth, %d trades",
        elapsed / 3600, depth_count, trade_count,
    )


if __name__ == "__main__":
    asyncio.run(main())
