"""Scalper bot entry point — BTCUSDT Perpetual on Binance Futures."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.ws_client import BinanceWSClient
from src.order_book import OrderBook
from src.features import FeatureEngine
from src.model import HybridModel
from src.signal import SignalGenerator, Direction
from src.executor import Executor, State
from src.risk import RiskManager
from src.recorder import Recorder
from src.notifier import Notifier

logger = logging.getLogger("scalper")

TICK_INTERVAL = 0.1  # 100ms main loop
MODEL_CHECK_INTERVAL = 60  # check for model updates every 60s


async def main() -> None:
    # --- Setup logging ---
    cfg = load_config("config.env")

    log_fmt = "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s"
    log_datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        datefmt=log_datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.log_dir / "bot.log"),
        ],
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("Scalper bot starting — %s x%d", cfg.symbol, cfg.leverage)
    logger.info("=" * 60)

    # --- Initialize components ---
    ws = BinanceWSClient(cfg)
    ob = OrderBook(ws, cfg.symbol)
    features = FeatureEngine(ob)
    model = HybridModel(cfg)
    recorder = Recorder(cfg)
    notifier = Notifier(cfg)
    risk = RiskManager(cfg, features, ob, ws)
    features.set_ws_client(ws)  # for OI / long-short ratio access
    signal_gen = SignalGenerator(cfg, model, features)
    executor = Executor(cfg, ws, risk, recorder, notifier)
    signal_gen.set_executor(executor)  # for dynamic threshold + fill rate

    # --- Load model (if available) ---
    if model.load():
        logger.info("Model loaded successfully")
    else:
        logger.warning("No model found — running in data-collection mode (no trading)")

    # --- Connect ---
    await ws.start()
    await asyncio.sleep(1)  # let WS connect
    await ob.start()
    await recorder.start()
    await notifier.start()
    await executor.start()

    # Wire recorder to OrderBook snapshots (not raw WS diffs)
    def on_ob_snapshot(snap) -> None:
        recorder.record_depth(snap)
        # Sim-parallel diagnostic (Point 4): feed the latest mid into
        # the executor's in-memory ring buffer so every live trade can
        # be replayed through `src.live_sim.simulate_trade` at close
        # time. Cheap — one deque append.
        executor.record_mid_tick(snap.mid_price)

    ob.on_snapshot(on_ob_snapshot)

    async def on_aggtrade(data: dict) -> None:
        features.on_aggtrade(data)
        recorder.record_trade(data)

    async def on_markprice(data: dict) -> None:
        features.on_markprice(data)

    async def on_disconnect(data: dict) -> None:
        stream = data.get("stream", "unknown")
        logger.warning("Stream disconnected: %s", stream)
        notifier.alert("WS Disconnected", f"Stream: {stream}")
        # Emergency close if in position
        close_needed, reason = risk.should_emergency_close()
        if close_needed:
            await executor.emergency_close(reason)

    # Secondary instrument callbacks (ETH leading signal)
    async def on_secondary_depth(data: dict) -> None:
        features.on_secondary_depth(data)

    async def on_secondary_aggtrade(data: dict) -> None:
        features.on_secondary_aggtrade(data)

    # Bybit + cross-exchange callbacks (feed feature 30 cross_exchange_momentum)
    async def on_bybit_aggtrade(data: dict) -> None:
        features.on_bybit_aggtrade(data)

    async def on_exchange_trade(data: dict) -> None:
        features.on_exchange_trade(data)

    ws.on_aggtrade(on_aggtrade)
    ws.on_markprice(on_markprice)
    ws.on_secondary_depth(on_secondary_depth)
    ws.on_secondary_aggtrade(on_secondary_aggtrade)
    ws.on_bybit_aggtrade(on_bybit_aggtrade)
    for ex in ("okx", "bitget", "gateio"):
        ws.on_exchange_trade(ex, on_exchange_trade)
    ws.on_disconnect(on_disconnect)

    notifier.info("Bot Started", f"{cfg.symbol} x{cfg.leverage} | Balance: ${executor.balance:.2f}")

    # --- Graceful shutdown ---
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, _signal_handler)

    # --- Main loop ---
    last_model_check = time.monotonic()
    last_threshold_tune = time.monotonic()
    tick_count = 0

    logger.info("Entering main loop (tick=%.0fms)", TICK_INTERVAL * 1000)

    try:
        while not shutdown_event.is_set():
            t0 = time.monotonic()

            if not ob.is_synced:
                await asyncio.sleep(TICK_INTERVAL)
                continue

            # Update features
            features.update()
            tick_count += 1

            # Self-tune threshold every hour
            if t0 - last_threshold_tune > 3600:
                executor.tune_threshold()
                last_threshold_tune = t0

            # Check for model hot-swap
            if t0 - last_model_check > MODEL_CHECK_INTERVAL:
                model.check_reload()
                last_model_check = t0

            # Emergency close check
            close_needed, reason = risk.should_emergency_close()
            if close_needed and executor.state == State.IN_POSITION:
                await executor.emergency_close(reason)
                await asyncio.sleep(5)
                continue

            # Generate signal and trade
            if model.is_loaded and executor.state == State.IDLE:
                sig = signal_gen.generate(executor.balance)
                if sig.direction != Direction.NONE:
                    await executor.process_signal(sig)

            elif model.is_loaded and executor.state == State.IN_POSITION:
                # Check for reversal
                sig = signal_gen.generate(executor.balance)
                if sig.direction != Direction.NONE:
                    await executor.handle_reversal_signal(sig)

            # Maintain tick rate
            elapsed = time.monotonic() - t0
            sleep_time = max(0, TICK_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

            # Periodic logging
            if tick_count % 600 == 0:  # every 60 seconds
                logger.info(
                    "Tick #%d | mid=$%.2f | spread=$%.2f | state=%s | balance=$%.2f | daily_pnl=$%.2f",
                    tick_count,
                    ob.current.mid_price if ob.current else 0,
                    ob.current.spread if ob.current else 0,
                    executor.state.value,
                    executor.balance,
                    risk.daily_pnl,
                )

    except Exception as e:
        logger.exception("Fatal error in main loop: %s", e)
        notifier.alert("Fatal Error", str(e))
        if executor.state == State.IN_POSITION:
            await executor.emergency_close("fatal_error")

    finally:
        logger.info("Shutting down...")
        notifier.info("Bot Stopping", "Graceful shutdown")

        if executor.state == State.IN_POSITION:
            await executor.emergency_close("shutdown")

        await recorder.stop()
        await ws.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
