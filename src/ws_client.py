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
        # Cross-exchange trade callbacks (5 additional exchanges)
        self._on_exchange_trade: dict[str, Callback] = {}
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

    def on_exchange_trade(self, exchange: str, cb: Callback) -> None:
        """Register callback for cross-exchange trades (okx, bitget, gateio)."""
        self._on_exchange_trade[exchange] = cb

    async def start(self) -> None:
        # We hold ~10 sustained WebSocket connections (BTC depth/agg/mark,
        # ETH depth/agg, user data, Bybit, OKX, Bitget, Gate.io) plus
        # occasional REST polling. limit=10 was too small and starved the
        # later streams; bumped to 64 with per-host headroom.
        tcp = aiohttp.TCPConnector(
            limit=64,
            limit_per_host=8,
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

        # Cross-exchange signals (Bybit + 3 exchanges).
        # HTX and Deribit removed: both proved structurally unstable from this
        # Tokyo VPS (synchronized timeouts every 3-4 min in production despite
        # standalone probes succeeding). Both are Cloudflare-fronted; root
        # cause not isolated. Cost/benefit didn't justify keeping flaky data.
        asyncio.create_task(self._run_bybit_stream())
        asyncio.create_task(self._run_okx_stream())
        asyncio.create_task(self._run_bitget_stream())
        asyncio.create_task(self._run_gateio_stream())

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
        # Explicit short timeout: aiohttp's default is 5 min, which is far too
        # long when a keepalive connection goes stale — snapshot refetches then
        # block the OrderBook resync_lock for minutes while depth data goes dark.
        timeout = aiohttp.ClientTimeout(total=10, connect=3, sock_read=5)
        async with self._session.get(url, params=params, timeout=timeout) as r:
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
                logger.error("listenKey keepalive failed: %r", e)

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
                logger.error("User data stream error: %r", e)

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


    # --- Cross-exchange WS helpers ---

    async def _data_staleness_watchdog(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        last_data_ts: list[float],
        idle_max: float,
        name: str,
    ) -> None:
        """Force-close ws if no DATA message arrives for `idle_max` seconds.

        This catches half-open / "server still responds to PING but stopped
        sending data" failures that aiohttp's WS-level heartbeat misses.
        Closing the ws makes the outer reader loop exit and the caller
        reconnects with backoff.
        """
        check_interval = max(5.0, idle_max / 4)
        try:
            while not ws.closed:
                await asyncio.sleep(check_interval)
                if ws.closed:
                    return
                idle = time.monotonic() - last_data_ts[0]
                if idle > idle_max:
                    logger.warning(
                        "%s WS idle %.0fs (no data) — forcing reconnect",
                        name, idle,
                    )
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return
        except asyncio.CancelledError:
            pass

    async def _dispatch_exchange(self, exchange: str, ts: int, price: str, qty: str, is_sell: bool) -> None:
        cb = self._on_exchange_trade.get(exchange)
        if cb:
            await cb({"T": ts, "p": price, "q": qty, "m": is_sell, "exchange": exchange})

    async def _run_bybit_stream(self) -> None:
        """Bybit BTCUSDT aggTrade -- leads Binance by 100-500ms."""
        backoff = 1
        seen_ids: set[str] = set()  # dedup Bybit trade IDs within session
        while True:
            watchdog: asyncio.Task | None = None
            try:
                seen_ids.clear()
                async with self._session.ws_connect("wss://stream.bybit.com/v5/public/linear") as ws:
                    await ws.send_json({"op": "subscribe", "args": ["publicTrade.BTCUSDT"]})
                    logger.info("Bybit WS connected")
                    backoff = 1
                    last_data_ts = [time.monotonic()]
                    watchdog = asyncio.create_task(
                        self._data_staleness_watchdog(ws, last_data_ts, 60.0, "Bybit")
                    )
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            raw = orjson.loads(msg.data)
                            if raw.get("topic") == "publicTrade.BTCUSDT" and self._on_bybit_aggtrade:
                                last_data_ts[0] = time.monotonic()
                                for t in raw.get("data", []):
                                    tid = t.get("i", "")
                                    if tid and tid in seen_ids:
                                        continue  # skip duplicate
                                    if tid:
                                        seen_ids.add(tid)
                                        if len(seen_ids) > 100_000:
                                            seen_ids = set(list(seen_ids)[-50_000:])
                                    await self._on_bybit_aggtrade({
                                        "T": t.get("T", 0),
                                        "p": t.get("p", "0"),
                                        "q": t.get("v", "0"),
                                        "m": t.get("S") == "Sell",
                                    })
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                if watchdog:
                    watchdog.cancel()
                break
            except Exception as e:
                logger.error("Bybit WS error: %r", e)
            finally:
                if watchdog:
                    watchdog.cancel()
            logger.warning("Bybit WS disconnected, reconnect in %ds", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)

    # --- Cross-exchange streams (3 exchanges: OKX, Bitget, Gate.io) ---

    async def _run_okx_stream(self) -> None:
        backoff = 1
        while True:
            watchdog: asyncio.Task | None = None
            try:
                async with self._session.ws_connect("wss://ws.okx.com:8443/ws/v5/public") as ws:
                    await ws.send_json({"op": "subscribe", "args": [{"channel": "trades", "instId": "BTC-USDT-SWAP"}]})
                    logger.info("OKX WS connected")
                    backoff = 1
                    last_data_ts = [time.monotonic()]
                    watchdog = asyncio.create_task(
                        self._data_staleness_watchdog(ws, last_data_ts, 60.0, "OKX")
                    )
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            raw = orjson.loads(msg.data)
                            data = raw.get("data", [])
                            if data:
                                last_data_ts[0] = time.monotonic()
                            for t in data:
                                await self._dispatch_exchange(
                                    "okx", int(t.get("ts", 0)), t.get("px", "0"),
                                    t.get("sz", "0"), t.get("side") == "sell")
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                if watchdog:
                    watchdog.cancel()
                break
            except Exception as e:
                logger.error("OKX WS error: %r", e)
            finally:
                if watchdog:
                    watchdog.cancel()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)

    async def _run_bitget_stream(self) -> None:
        backoff = 1
        while True:
            ping_task: asyncio.Task | None = None
            watchdog: asyncio.Task | None = None
            try:
                async with self._session.ws_connect("wss://ws.bitget.com/v2/ws/public") as ws:
                    await ws.send_json({"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "trade", "instId": "BTCUSDT"}]})
                    logger.info("Bitget WS connected")
                    backoff = 1
                    last_data_ts = [time.monotonic()]

                    async def _bitget_ping() -> None:
                        # Bitget v2 public: server closes idle >30s. Plain text "ping" -> "pong".
                        try:
                            while not ws.closed:
                                await asyncio.sleep(20)
                                if ws.closed:
                                    break
                                await ws.send_str("ping")
                        except Exception:
                            pass

                    ping_task = asyncio.create_task(_bitget_ping())
                    watchdog = asyncio.create_task(
                        self._data_staleness_watchdog(ws, last_data_ts, 60.0, "Bitget")
                    )

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            # Plain "pong" reply, not JSON
                            if msg.data == "pong":
                                continue
                            raw = orjson.loads(msg.data)
                            data = raw.get("data", [])
                            if data:
                                last_data_ts[0] = time.monotonic()
                            for t in data:
                                await self._dispatch_exchange(
                                    "bitget", int(t.get("ts", 0)), t.get("price", "0"),
                                    t.get("size", "0"), t.get("side") == "sell")
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                if ping_task:
                    ping_task.cancel()
                if watchdog:
                    watchdog.cancel()
                break
            except Exception as e:
                logger.error("Bitget WS error: %r", e)
            finally:
                if ping_task:
                    ping_task.cancel()
                if watchdog:
                    watchdog.cancel()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)

    async def _run_gateio_stream(self) -> None:
        backoff = 1
        while True:
            watchdog: asyncio.Task | None = None
            try:
                async with self._session.ws_connect("wss://fx-ws.gateio.ws/v4/ws/usdt") as ws:
                    await ws.send_json({"channel": "futures.trades", "event": "subscribe", "payload": ["BTC_USDT"], "time": int(time.time())})
                    logger.info("Gate.io WS connected")
                    backoff = 1
                    last_data_ts = [time.monotonic()]
                    watchdog = asyncio.create_task(
                        self._data_staleness_watchdog(ws, last_data_ts, 60.0, "Gate.io")
                    )
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            raw = orjson.loads(msg.data)
                            if raw.get("channel") == "futures.trades" and raw.get("event") == "update":
                                result = raw.get("result", [])
                                if result:
                                    last_data_ts[0] = time.monotonic()
                                for t in result:
                                    ts = int(float(t.get("create_time_ms", t.get("create_time", 0) * 1000)))
                                    await self._dispatch_exchange(
                                        "gateio", ts, str(t.get("price", "0")),
                                        str(t.get("size", "0")), t.get("size", 0) < 0)
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                if watchdog:
                    watchdog.cancel()
                break
            except Exception as e:
                logger.error("Gate.io WS error: %r", e)
            finally:
                if watchdog:
                    watchdog.cancel()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)



class _StreamHandler:
    """Manages a single WebSocket stream with auto-reconnect + staleness watchdog."""

    IDLE_MAX = 60.0  # force-reconnect if no data for this many seconds

    def __init__(self, name: str, url: str, callback: Callback, client: BinanceWSClient) -> None:
        self.name = name
        self.url = url
        self.callback = callback
        self._client = client
        self._running = True
        self.last_data_time: float = time.monotonic()

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        backoff = 1
        while self._running:
            watchdog: asyncio.Task | None = None
            try:
                async with self._client._session.ws_connect(self.url) as ws:
                    logger.info("WS %s connected", self.name)
                    backoff = 1
                    self.last_data_time = time.monotonic()
                    watchdog = asyncio.create_task(
                        self._client._data_staleness_watchdog(
                            ws, [self.last_data_time], self.IDLE_MAX, self.name
                        )
                    )
                    last_ts_ref = [self.last_data_time]
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            now = time.monotonic()
                            self.last_data_time = now
                            last_ts_ref[0] = now
                            data = orjson.loads(msg.data)
                            await self.callback(data)
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                if watchdog:
                    watchdog.cancel()
                break
            except Exception as e:
                logger.error("WS %s error: %r", self.name, e)
            finally:
                if watchdog:
                    watchdog.cancel()

            if not self._running:
                break

            logger.warning("WS %s disconnected, reconnecting in %ds", self.name, backoff)

            if self._client._on_disconnect:
                await self._client._on_disconnect({"stream": self.name})

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
