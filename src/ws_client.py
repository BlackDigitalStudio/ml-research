from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any, Callable, Coroutine

import aiohttp
import orjson

from src.config import Config

logger = logging.getLogger(__name__)

Callback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class BinanceWSClient:
    """Manages multiple WebSocket streams to Binance Futures with auto-reconnect."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._session: aiohttp.ClientSession | None = None
        self._listen_key: str = ""
        self._listen_key_task: asyncio.Task | None = None
        self._streams: dict[str, _StreamHandler] = {}
        self._on_depth: Callback | None = None
        self._on_aggtrade: Callback | None = None
        self._on_markprice: Callback | None = None
        self._on_user_data: Callback | None = None
        self._on_disconnect: Callback | None = None
        # Secondary instrument streams (ETH for leading signal)
        self._on_secondary_depth: Callback | None = None
        self._on_secondary_aggtrade: Callback | None = None
        self._on_bybit_aggtrade: Callback | None = None
        self.last_message_time: float = 0.0
        self.rate_limit_weight: int = 0

    def on_depth(self, cb: Callback) -> None:
        self._on_depth = cb

    def on_aggtrade(self, cb: Callback) -> None:
        self._on_aggtrade = cb

    def on_markprice(self, cb: Callback) -> None:
        self._on_markprice = cb

    def on_user_data(self, cb: Callback) -> None:
        self._on_user_data = cb

    def on_disconnect(self, cb: Callback) -> None:
        self._on_disconnect = cb

    def on_secondary_depth(self, cb: Callback) -> None:
        self._on_secondary_depth = cb

    def on_secondary_aggtrade(self, cb: Callback) -> None:
        self._on_secondary_aggtrade = cb

    def on_bybit_aggtrade(self, cb: Callback) -> None:
        self._on_bybit_aggtrade = cb

    async def start(self) -> None:
        tcp = aiohttp.TCPConnector(
            limit=10,
            enable_cleanup_closed=True,
            force_close=False,
        )
        self._session = aiohttp.ClientSession(
            connector=tcp,
            headers={"X-MBX-APIKEY": self._cfg.api_key},
        )

        # Market data streams
        self._streams["depth"] = _StreamHandler(
            name="depth",
            url=self._cfg.ws_depth_url,
            callback=self._dispatch_depth,
            client=self,
        )
        self._streams["aggtrade"] = _StreamHandler(
            name="aggTrade",
            url=self._cfg.ws_aggtrade_url,
            callback=self._dispatch_aggtrade,
            client=self,
        )
        self._streams["markprice"] = _StreamHandler(
            name="markPrice",
            url=self._cfg.ws_markprice_url,
            callback=self._dispatch_markprice,
            client=self,
        )

        # Secondary instrument streams (ETH for leading signal)
        self._streams["secondary_depth"] = _StreamHandler(
            name="secondary_depth",
            url=self._cfg.ws_secondary_depth_url,
            callback=self._dispatch_secondary_depth,
            client=self,
        )
        self._streams["secondary_aggtrade"] = _StreamHandler(
            name="secondary_aggTrade",
            url=self._cfg.ws_secondary_aggtrade_url,
            callback=self._dispatch_secondary_aggtrade,
            client=self,
        )

        for handler in self._streams.values():
            asyncio.create_task(handler.run())

        # User data stream
        asyncio.create_task(self._run_user_data_stream())

        # OI + long/short ratio polling
        asyncio.create_task(self._poll_derivatives_data())

        # Bybit cross-exchange signal
        asyncio.create_task(self._run_bybit_stream())

    async def stop(self) -> None:
        for handler in self._streams.values():
            handler.stop()
        if self._listen_key_task:
            self._listen_key_task.cancel()
        if self._session:
            await self._session.close()

    # --- REST helpers ---

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        qs = urllib.parse.urlencode(params)
        sig = hmac.new(
            self._cfg.api_secret.encode(), qs.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    async def rest_get(self, path: str, params: dict | None = None, signed: bool = False) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        url = f"{self._cfg.rest_base}{path}"
        t0 = time.monotonic()
        async with self._session.get(url, params=params) as r:
            self.rate_limit_weight = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0"))
            if self.rate_limit_weight > 1000:
                logger.warning("Rate limit approaching: %d/1200", self.rate_limit_weight)
            data = await r.json(loads=orjson.loads)
            latency = (time.monotonic() - t0) * 1000
            if r.status != 200:
                logger.error("REST GET %s → %d: %s (%.1fms)", path, r.status, data, latency)
            return data

    async def rest_post(self, path: str, params: dict | None = None, signed: bool = True) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        url = f"{self._cfg.rest_base}{path}"
        t0 = time.monotonic()
        async with self._session.post(url, params=params) as r:
            self.rate_limit_weight = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0"))
            if self.rate_limit_weight > 1000:
                logger.warning("Rate limit approaching: %d/1200", self.rate_limit_weight)
            data = await r.json(loads=orjson.loads)
            latency = (time.monotonic() - t0) * 1000
            if r.status != 200:
                logger.error("REST POST %s → %d: %s (%.1fms)", path, r.status, data, latency)
            else:
                logger.debug("REST POST %s → 200 (%.1fms)", path, latency)
            return data

    async def rest_delete(self, path: str, params: dict | None = None, signed: bool = True) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        url = f"{self._cfg.rest_base}{path}"
        t0 = time.monotonic()
        async with self._session.delete(url, params=params) as r:
            self.rate_limit_weight = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0"))
            if self.rate_limit_weight > 1000:
                logger.warning("Rate limit approaching: %d/1200", self.rate_limit_weight)
            data = await r.json(loads=orjson.loads)
            latency = (time.monotonic() - t0) * 1000
            if r.status != 200:
                logger.error("REST DELETE %s → %d: %s (%.1fms)", path, r.status, data, latency)
            return data

    @property
    def rest_latency_ms(self) -> float:
        return getattr(self, "_last_rest_latency", 0.0)

    @property
    def is_rate_limited(self) -> bool:
        return self.rate_limit_weight > 1100

    # --- User data stream ---

    async def _obtain_listen_key(self) -> str:
        data = await self.rest_post("/fapi/v1/listenKey", signed=False)
        key = data.get("listenKey", "")
        if key:
            logger.info("Obtained listenKey: %s...", key[:8])
        return key

    async def _keepalive_listen_key(self) -> None:
        while True:
            await asyncio.sleep(30 * 60)  # every 30 min
            try:
                await self.rest_post("/fapi/v1/listenKey", signed=False)
                logger.debug("listenKey keepalive sent")
            except Exception as e:
                logger.error("listenKey keepalive failed: %s", e)

    async def _run_user_data_stream(self) -> None:
        backoff = 1
        while True:
            try:
                self._listen_key = await self._obtain_listen_key()
                if not self._listen_key:
                    logger.error("Failed to obtain listenKey, retry in %ds", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue

                backoff = 1
                self._listen_key_task = asyncio.create_task(self._keepalive_listen_key())

                url = f"{self._cfg.ws_base}/ws/{self._listen_key}"
                async with self._session.ws_connect(url) as ws:
                    logger.info("User data stream connected")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = orjson.loads(msg.data)
                            self.last_message_time = time.monotonic()
                            if self._on_user_data:
                                await self._on_user_data(data)
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break

            except Exception as e:
                logger.error("User data stream error: %s", e)

            if self._listen_key_task:
                self._listen_key_task.cancel()
            logger.warning("User data stream disconnected, reconnecting in %ds", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    # --- Dispatch ---

    async def _dispatch_depth(self, data: dict) -> None:
        self.last_message_time = time.monotonic()
        if self._on_depth:
            await self._on_depth(data)

    async def _dispatch_aggtrade(self, data: dict) -> None:
        self.last_message_time = time.monotonic()
        if self._on_aggtrade:
            await self._on_aggtrade(data)

    async def _dispatch_markprice(self, data: dict) -> None:
        self.last_message_time = time.monotonic()
        if self._on_markprice:
            await self._on_markprice(data)

    async def _dispatch_secondary_depth(self, data: dict) -> None:
        if self._on_secondary_depth:
            await self._on_secondary_depth(data)

    async def _dispatch_secondary_aggtrade(self, data: dict) -> None:
        if self._on_secondary_aggtrade:
            await self._on_secondary_aggtrade(data)

    # --- Derivatives data polling (OI, long/short ratio) ---

    # Latest polled values, read by FeatureEngine
    open_interest: float = 0.0
    open_interest_prev: float = 0.0
    long_short_ratio: float = 1.0

    async def _poll_derivatives_data(self) -> None:
        """Poll OI and long/short ratio every 15 seconds."""
        await asyncio.sleep(3)  # let WS connect first
        while True:
            try:
                # Open Interest
                oi_data = await self.rest_get(
                    "/fapi/v1/openInterest",
                    params={"symbol": self._cfg.symbol},
                )
                new_oi = float(oi_data.get("openInterest", 0))
                if self.open_interest > 0:
                    self.open_interest_prev = self.open_interest
                self.open_interest = new_oi

                # Long/Short ratio (top traders)
                ls_data = await self.rest_get(
                    "/futures/data/topLongShortAccountRatio",
                    params={"symbol": self._cfg.symbol, "period": "5m", "limit": 1},
                )
                if isinstance(ls_data, list) and ls_data:
                    self.long_short_ratio = float(ls_data[0].get("longShortRatio", 1.0))

            except Exception as e:
                logger.debug("Derivatives poll error: %s", e)

            await asyncio.sleep(15)


    async def _run_bybit_stream(self) -> None:
        """Bybit BTCUSDT aggTrade -- leads Binance by 100-500ms."""
        backoff = 1
        while True:
            try:
                async with self._session.ws_connect("wss://stream.bybit.com/v5/public/linear") as ws:
                    await ws.send_json({"op": "subscribe", "args": ["publicTrade.BTCUSDT"]})
                    logger.info("Bybit WS connected")
                    backoff = 1
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            raw = orjson.loads(msg.data)
                            if raw.get("topic") == "publicTrade.BTCUSDT" and self._on_bybit_aggtrade:
                                for t in raw.get("data", []):
                                    await self._on_bybit_aggtrade({
                                        "T": t.get("T", 0),
                                        "p": t.get("p", "0"),
                                        "q": t.get("v", "0"),
                                        "m": t.get("S") == "Sell",
                                    })
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bybit WS error: %s", e)
            logger.warning("Bybit WS disconnected, reconnect in %ds", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


class _StreamHandler:
    """Manages a single WebSocket stream with auto-reconnect."""

    def __init__(self, name: str, url: str, callback: Callback, client: BinanceWSClient) -> None:
        self.name = name
        self.url = url
        self.callback = callback
        self._client = client
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        backoff = 1
        while self._running:
            try:
                async with self._client._session.ws_connect(self.url) as ws:
                    logger.info("WS %s connected", self.name)
                    backoff = 1
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = orjson.loads(msg.data)
                            await self.callback(data)
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("WS %s error: %s", self.name, e)

            if not self._running:
                break

            logger.warning("WS %s disconnected, reconnecting in %ds", self.name, backoff)

            if self._client._on_disconnect:
                await self._client._on_disconnect({"stream": self.name})

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
