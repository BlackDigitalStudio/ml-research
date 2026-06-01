"""Project entrypoint for the recorder-v2 (Chronos) VM deployment.

A thin wrapper over ``chronos.Gateway`` tuned for THIS project (the
upstream ``scripts/record_data.py`` hard-codes L100 + Coinbase/Deribit):

- Binance USDⓈ-M futures — the 8 base coins (BNB, BTC, DOGE, ETH, LINK,
  LTC, SOL, XRP) on **both USDT and USDC** margin = 16 contracts — **L20**
  depth snapshots (parity with Cryptolake/Tardis ``raw/book``), maintained
  book + REST reconcile, ``@forceOrder`` liquidations, derivatives poll.
- Cross-venue trades: Bybit, OKX, Bitget, Gate.io — all 8 base coins
  (USDT perps) on each venue (bonus beyond Cryptolake, which is Binance-only).
  Enables same-asset cross-exchange lead-lag (e.g. Bybit DOGE -> Binance DOGE).
- Coinbase / Deribit deliberately OFF — out of the Cryptolake-equivalent
  scope (and Deribit ``.raw`` would need API keys).

All public market data — no API keys required. Config via env
(set by systemd EnvironmentFile): ``CHRONOS_ROOT``, ``RECORDER_HOST_ID``,
``CHRONOS_HEALTH_FILE``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# This file is dropped at the crypto-market-recorder repo root so the
# `chronos` package resolves on import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos import Recorder  # noqa: E402
from chronos.gateway import Gateway  # noqa: E402

logger = logging.getLogger("chronos")

DEPTH_LEVELS = 20  # L20 — Cryptolake / Tardis raw/book parity

# The 8 Cryptolake base coins, recorded on BOTH USDT and USDC margin.
# All 16 are TRADING PERPETUAL on Binance USDⓈ-M (fapi) — verified via
# exchangeInfo 2026-06-01. USDT first (Cryptolake parity), then USDC.
BASE_COINS = ("BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP")
BINANCE_SYMBOLS = tuple(
    f"{coin}{quote}" for quote in ("USDT", "USDC") for coin in BASE_COINS
)


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    repo_root = Path(__file__).resolve().parent
    root = Path(os.environ.get("CHRONOS_ROOT", str(repo_root / "data")))
    root.mkdir(parents=True, exist_ok=True)
    health = os.environ.get("CHRONOS_HEALTH_FILE", "/tmp/chronos_health")
    logger.info("chronos(project) root=%s health=%s depth=L%d", root, health, DEPTH_LEVELS)

    recorder = Recorder(root, health_file=health)
    gateway = Gateway(recorder)

    # Binance core — maintained book + L20 snapshots + liquidations + OI poll.
    for symbol in BINANCE_SYMBOLS:
        gateway.add_binance_futures(
            symbol,
            snapshot_levels=DEPTH_LEVELS,
            maintain_book=True,
            # Cap maintained depth at 100/side: we only emit L20, and Binance
            # @depth diffs cover the full book (unbounded RSS otherwise). 100
            # is a deep buffer for correct top-20 under churn. Seed REST at 100.
            book_max_levels=100,
            seed_limit=100,
            subscribe_force_order=True,
            subscribe_secondary_endpoint=False,  # fstream replica 302s from Tokyo
            reconcile_interval_sec=900.0,
            derivatives_poll_interval_sec=15.0,
        )

    # Cross-venue trades — all 8 base coins (USDT perps) on each venue.
    # Verified present on every venue via their REST instrument lists
    # (2026-06-01). One WS connection per (venue, symbol) = 8x4 = 32.
    for coin in BASE_COINS:
        gateway.add_bybit_trades(f"{coin}USDT")
        gateway.add_okx_trades(f"{coin}-USDT-SWAP")
        gateway.add_bitget_trades(f"{coin}USDT")
        gateway.add_gateio_trades(f"{coin}_USDT")
    # Coinbase / Deribit intentionally omitted.

    await gateway.start()

    shutdown = asyncio.Event()

    def _stop() -> None:
        logger.info("shutdown signal received")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    logger.info("chronos(project) running — Ctrl+C / SIGTERM to stop")
    await shutdown.wait()

    logger.info("stopping gateway + recorder")
    await gateway.stop()
    logger.info("done")


if __name__ == "__main__":
    asyncio.run(_main())
