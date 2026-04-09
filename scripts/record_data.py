"""Phase 0: Full data recorder — BTC + ETH + Bybit + funding + derivatives.

Connects to Binance & Bybit WebSocket, polls derivatives data,
and writes everything to Parquet files for training.

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
    logger.info("Data recorder starting — %s + %s + Bybit", cfg.symbol, cfg.secondary_symbol)
    logger.info("=" * 50)

    ws = BinanceWSClient(cfg)
    ob = OrderBook(ws, cfg.symbol)
    eth_ob = OrderBook(ws, cfg.secondary_symbol, secondary=True)
    rec = Recorder(cfg)

    # Counters
    counts = {"depth": 0, "trade": 0, "bybit": 0, "eth_depth": 0, "eth_trade": 0, "funding": 0, "deriv": 0}
    t_start = time.monotonic()

    # --- BTC depth ---
    def on_snapshot(snap) -> None:
        rec.record_depth(snap)
        counts["depth"] += 1

    # --- BTC trades ---
    async def on_trade(data: dict) -> None:
        rec.record_trade(data)
        counts["trade"] += 1

    # --- Bybit trades ---
    async def on_bybit_trade(data: dict) -> None:
        rec.record_bybit_trade(data)
        counts["bybit"] += 1

    # --- ETH depth ---
    def on_eth_snapshot(snap) -> None:
        rec.record_eth_depth(snap)
        counts["eth_depth"] += 1

    # --- ETH trades ---
    async def on_eth_trade(data: dict) -> None:
        rec.record_eth_trade(data)
        counts["eth_trade"] += 1

    # --- Cross-exchange trades (5 exchanges) ---
    async def on_exchange_trade(data: dict) -> None:
        rec.record_exchange_trade(data)
        ex = data.get("exchange", "?")
        counts[f"ex_{ex}"] = counts.get(f"ex_{ex}", 0) + 1

    # --- Funding rate (from markPrice stream, every 1s) ---
    async def on_markprice(data: dict) -> None:
        rec.record_funding(data)
        counts["funding"] += 1

    # --- Derivatives (OI + L/S ratio, polled every 15s by ws_client) ---
    # We hook into ws_client's polling by periodically reading its values
    async def derivatives_recorder() -> None:
        """Piggyback on ws_client's _poll_derivatives_data to save OI + L/S."""
        await asyncio.sleep(20)  # let polling start
        while not shutdown.is_set():
            if ws.open_interest > 0:
                rec.record_derivatives(ws.open_interest, ws.long_short_ratio)
                counts["deriv"] += 1
            await asyncio.sleep(15)

    # Wire up callbacks
    ob.on_snapshot(on_snapshot)
    eth_ob.on_snapshot(on_eth_snapshot)

    await ws.start()
    ws.on_aggtrade(on_trade)
    ws.on_bybit_aggtrade(on_bybit_trade)
    ws.on_secondary_aggtrade(on_eth_trade)
    for ex in ("okx", "bitget", "gateio", "htx", "deribit"):
        ws.on_exchange_trade(ex, on_exchange_trade)
    ws.on_markprice(on_markprice)

    await asyncio.sleep(2)
    await ob.start()
    await eth_ob.start()
    await rec.start()

    logger.info("Recording to %s (BTC + ETH + Bybit + funding + derivatives)", cfg.data_dir)

    # Graceful shutdown
    shutdown = asyncio.Event()

    def _stop() -> None:
        logger.info("Shutdown signal received")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    # Start derivatives recorder
    deriv_task = asyncio.create_task(derivatives_recorder())

    # Stats loop
    while not shutdown.is_set():
        await asyncio.sleep(60)
        elapsed = time.monotonic() - t_start
        hours = elapsed / 3600

        # Calculate total size across all data dirs
        total_size = 0
        for subdir in cfg.data_dir.iterdir():
            if subdir.is_dir():
                total_size += sum(f.stat().st_size for f in subdir.glob("*.parquet"))

        total_mb = total_size / 1024 / 1024

        # Exchange counts
        ex_str = " ".join(f"{k.replace('ex_','')}={v}" for k, v in sorted(counts.items()) if k.startswith("ex_"))

        logger.info(
            "Stats: %.1fh | depth=%d trade=%d eth_d=%d eth_t=%d bybit=%d fund=%d deriv=%d | %s | %.1f MB",
            hours,
            counts["depth"], counts["trade"],
            counts["eth_depth"], counts["eth_trade"],
            counts["bybit"], counts["funding"], counts["deriv"],
            ex_str or "exchanges=connecting",
            total_mb,
        )

    deriv_task.cancel()
    logger.info("Flushing final data...")
    await rec.stop()
    await ws.stop()

    elapsed = time.monotonic() - t_start
    logger.info("Recording complete: %.1f hours, %s", elapsed / 3600, counts)


if __name__ == "__main__":
    asyncio.run(main())
