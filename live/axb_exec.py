#!/usr/bin/env python3
"""axb_exec — order-execution sidecar for the Rust axb_engine.

Owns everything that talks money: the battle-tested maker trade lifecycle
(GTX entry at touch / hold from fill / pegged maker exit / catastrophic-only
backstops) copied verbatim from live/axb_live.py, its own DOGEUSDC bookTicker
WS feed, and the hourly GCS upload of the engine's decision logs.

Interface: Unix socket WORKDIR/exec.sock; one JSON line per command
{"side":"long"|"short","score":x} -> reply {"executed":true|false}\n.
Guards (busy / day-loss halt / day-trade halt / stale touch) are inside
Executor.maybe_trade, exactly as before.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import websockets
from google.cloud import storage

log = logging.getLogger("axb_exec")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJ = "project-0998ac51-36ba-445c-bc7"
MKT = "market-data-0998ac51"
SYMK = os.environ.get("SYMK", "DOGE")
EXEC_SYM = os.environ.get("EXEC_SYM", "DOGEUSDC")
WORK = os.environ.get("WORKDIR", "/home/delmi/axb_h150")
SHADOW_GCS = os.environ.get("SHADOW_GCS", f"research_runs/axb_shadow_h150/{SYMK}")
MODE = os.environ.get("MODE", "shadow")
TRADE_BUDGET = float(os.environ.get("TRADE_BUDGET", "5"))
SIZE_FRAC = float(os.environ.get("SIZE_FRAC", "1.0"))
LEVERAGE = 2
ENTRY_WIN_S = float(os.environ.get("ENTRY_WIN_S", "60.0"))
HOLD_S = float(os.environ.get("HOLD_S", "150.0"))
PX_DEC = {"DOGEUSDC": 5, "BTCUSDC": 1, "XRPUSDC": 4}.get(EXEC_SYM, 5)
QTY_DEC = {"DOGEUSDC": 0, "BTCUSDC": 3, "XRPUSDC": 1}.get(EXEC_SYM, 0)
EXIT_MAX_S = float(os.environ.get("EXIT_MAX_S", "86400"))
HARD_LOSS_BP = float(os.environ.get("HARD_LOSS_BP", "300"))
DAY_LOSS_HALT = 0.05
DAY_TRADES_HALT = 40
REST_BASE = "https://fapi.binance.com"
WS_BASE = "wss://fstream.binance.com"


class Rest:
    """Minimal signed Binance USDS-M futures REST client (blocking)."""

    def __init__(self) -> None:
        import hashlib
        import hmac
        import urllib.parse
        import urllib.request
        self._h = hashlib
        self._hmac = hmac
        self._up = urllib.parse
        self._ur = urllib.request
        cfg = {}
        with open(f"{WORK}/config.env") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.strip().partition("=")
                    cfg[k.strip()] = v.strip()
        self.key = cfg["BINANCE_API_KEY"]
        self.sec = cfg["BINANCE_API_SECRET"].encode()

    def call(self, method: str, path: str, params: dict | None = None, signed: bool = True) -> dict | list:
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
        qs = self._up.urlencode(params)
        if signed:
            qs += "&signature=" + self._hmac.new(self.sec, qs.encode(), self._h.sha256).hexdigest()
        url = f"{REST_BASE}{path}?{qs}"
        req = self._ur.Request(url, headers={"X-MBX-APIKEY": self.key}, method=method)
        try:
            with self._ur.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except Exception as ex:
            body = ""
            if hasattr(ex, "read"):
                try:
                    body = ex.read().decode()[:400]
                except Exception:
                    pass
            return {"_error": str(ex), "_body": body}


def _err(r) -> str | None:
    return (r.get("_body") or r.get("_error")) if isinstance(r, dict) and ("_error" in r) else None


class Executor:
    """One trade at a time: GTX entry at touch -> hold -> pegged reduce-only maker exit.
    Verbatim trade lifecycle from live/axb_live.py (first-trade execution parity
    verified against the offline simulator 2026-07-08)."""

    def __init__(self, rest: Rest) -> None:
        self.rest = rest
        self.busy = False
        self.halted = ""
        self.day = ""
        self.day_pnl = 0.0
        self.day_trades = 0
        self.day_bal0 = 0.0
        self.usdc_bid = 0.0
        self.usdc_ask = 0.0
        self.usdc_ts = 0.0
        os.makedirs(f"{WORK}/trades", exist_ok=True)

    def on_book_ticker(self, d: dict) -> None:
        self.usdc_bid = float(d["b"])
        self.usdc_ask = float(d["a"])
        self.usdc_ts = time.time()

    def touch_ok(self) -> bool:
        return self.usdc_bid > 0 and self.usdc_ask > self.usdc_bid and time.time() - self.usdc_ts < 5

    def _roll_day(self) -> None:
        d = datetime.now(timezone.utc).strftime("%Y%m%d")
        if d != self.day:
            self.day = d
            self.day_pnl = 0.0
            self.day_trades = 0
            self.day_bal0 = 0.0
            if self.halted.startswith("day"):
                self.halted = ""

    def _tlog(self, rec: dict) -> None:
        rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with open(f"{WORK}/trades/{self.day}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("TRADE %s", json.dumps(rec))

    def _avail_usdc(self) -> float:
        r = self.rest.call("GET", "/fapi/v2/balance")
        if isinstance(r, list):
            for b in r:
                if b["asset"] == "USDC":
                    return float(b["availableBalance"])
        return 0.0

    def _order(self, **p) -> dict:
        return self.rest.call("POST", "/fapi/v1/order", p)

    def _get_order(self, oid: int) -> dict:
        return self.rest.call("GET", "/fapi/v1/order", {"symbol": EXEC_SYM, "orderId": oid})

    def _cancel(self, oid: int) -> dict:
        return self.rest.call("DELETE", "/fapi/v1/order", {"symbol": EXEC_SYM, "orderId": oid})

    def _position_amt(self) -> float:
        r = self.rest.call("GET", "/fapi/v2/positionRisk", {"symbol": EXEC_SYM})
        if isinstance(r, list) and r:
            return float(r[0].get("positionAmt", 0))
        return 0.0

    def _place_gtx(self, side: str, qty: float, price: float, reduce_only: bool) -> dict:
        p = {"symbol": EXEC_SYM, "side": side, "type": "LIMIT", "timeInForce": "GTX",
             "quantity": f"{qty:.{QTY_DEC}f}", "price": f"{price:.{PX_DEC}f}"}
        if reduce_only:
            p["reduceOnly"] = "true"
        return self._order(**p)

    def _market_close(self, side: str, qty: float) -> dict:
        return self._order(symbol=EXEC_SYM, side=side, type="MARKET",
                           quantity=f"{qty:.{QTY_DEC}f}", reduceOnly="true")

    def startup_recover(self) -> None:
        self.rest.call("DELETE", "/fapi/v1/allOpenOrders", {"symbol": EXEC_SYM})
        amt = self._position_amt()
        if amt != 0:
            side = "SELL" if amt > 0 else "BUY"
            r = self._market_close(side, abs(amt))
            self._tlog({"ev": "recover_close", "amt": amt, "resp_err": _err(r)})
        r = self.rest.call("POST", "/fapi/v1/leverage", {"symbol": EXEC_SYM, "leverage": LEVERAGE})
        log.info("executor ready: leverage resp=%s", r if _err(r) else r.get("leverage"))

    def run_trade(self, decision_ts: float, side_long: bool, score: float) -> None:
        try:
            self._trade(decision_ts, side_long, score)
        except Exception:
            log.exception("trade lifecycle crashed — emergency flatten")
            try:
                self.rest.call("DELETE", "/fapi/v1/allOpenOrders", {"symbol": EXEC_SYM})
                amt = self._position_amt()
                if amt != 0:
                    self._market_close("SELL" if amt > 0 else "BUY", abs(amt))
                    self._tlog({"ev": "crash_flatten", "amt": amt})
            except Exception:
                log.exception("emergency flatten failed")
        finally:
            self.busy = False

    def _trade(self, decision_ts: float, side_long: bool, score: float) -> None:
        entry_side = "BUY" if side_long else "SELL"
        exit_side = "SELL" if side_long else "BUY"
        avail = self._avail_usdc()
        if self.day_bal0 <= 0:
            self.day_bal0 = max(avail, 1e-9)
        px = self.usdc_bid if side_long else self.usdc_ask
        raw = avail * SIZE_FRAC / px * (1 - 1e-3)
        step = 10 ** QTY_DEC
        qty = float(np.floor(raw * step)) / step
        if qty * px < 5.0:
            self._tlog({"ev": "skip_small", "avail": avail, "qty": qty})
            return
        r = self._place_gtx(entry_side, qty, px, reduce_only=False)
        if _err(r) or r.get("status") == "EXPIRED":
            px = self.usdc_bid if side_long else self.usdc_ask
            r = self._place_gtx(entry_side, qty, px, reduce_only=False)
            if _err(r) or r.get("status") == "EXPIRED":
                self._tlog({"ev": "entry_reject", "err": _err(r) or r.get("status"), "px": px})
                return
        oid = int(r["orderId"])
        self._tlog({"ev": "entry_placed", "oid": oid, "side": entry_side, "qty": qty,
                    "px": px, "score": score})
        filled = 0.0
        entry_px = px
        status = "NEW"
        deadline = decision_ts + ENTRY_WIN_S
        while time.time() < deadline and status not in ("FILLED", "CANCELED", "EXPIRED"):
            time.sleep(0.7)
            o = self._get_order(oid)
            if _err(o):
                continue
            status = o.get("status", "NEW")
            filled = float(o.get("executedQty", 0) or 0)
            if status == "FILLED":
                entry_px = float(o.get("avgPrice") or px)
        if status != "FILLED":
            self._cancel(oid)
            o = self._get_order(oid)
            if not _err(o):
                filled = float(o.get("executedQty", 0) or 0)
                if filled > 0:
                    entry_px = float(o.get("avgPrice") or px)
        if filled == 0:
            self._tlog({"ev": "entry_miss", "oid": oid})
            return
        fill_ts = time.time()
        self._tlog({"ev": "entry_fill", "oid": oid, "filled": filled, "avg_px": entry_px})
        time.sleep(max(0.0, fill_ts + HOLD_S - time.time()))
        t_exit0 = time.time()
        exit_oid = 0
        last_oid = 0
        exit_px = 0.0
        backstop = ""
        while True:
            now = time.time()
            mid = (self.usdc_bid + self.usdc_ask) / 2
            unreal_bp = ((mid - entry_px) / entry_px if side_long else (entry_px - mid) / entry_px) * 1e4
            if now - t_exit0 > EXIT_MAX_S:
                backstop = "time"
            elif unreal_bp < -HARD_LOSS_BP:
                backstop = "loss"
            if backstop:
                if exit_oid:
                    self._cancel(exit_oid)
                amt = self._position_amt()
                if amt != 0:
                    self._market_close(exit_side, abs(amt))
                break
            want = self.usdc_ask if side_long else self.usdc_bid
            if exit_oid:
                o = self._get_order(exit_oid)
                st = o.get("status", "")
                if st == "FILLED":
                    exit_px = float(o["avgPrice"])
                    break
                adverse = (want < exit_px - 1e-9) if side_long else (want > exit_px + 1e-9)
                if adverse and st in ("NEW", "PARTIALLY_FILLED"):
                    self._cancel(exit_oid)
                    last_oid = exit_oid
                    exit_oid = 0
            if not exit_oid:
                amt = abs(self._position_amt())
                if amt == 0:
                    o = self._get_order(last_oid) if last_oid else {}
                    exit_px = float(o.get("avgPrice") or want) if not _err(o) else want
                    break
                r = self._place_gtx(exit_side, amt, want, reduce_only=True)
                if not _err(r) and r.get("status") != "EXPIRED":
                    exit_oid = int(r["orderId"])
                    exit_px = want
            time.sleep(0.7)
        if backstop:
            time.sleep(1.0)
            inc = self.rest.call("GET", "/fapi/v1/userTrades",
                                 {"symbol": EXEC_SYM, "startTime": int(t_exit0 * 1000)})
            exit_px = (float(inc[-1]["price"]) if isinstance(inc, list) and inc else
                       (self.usdc_bid if side_long else self.usdc_ask))
        pnl_bp = ((exit_px - entry_px) / entry_px if side_long else (entry_px - exit_px) / entry_px) * 1e4
        pnl_usd = pnl_bp / 1e4 * entry_px * filled
        if backstop:
            pnl_usd -= 4e-4 * exit_px * filled
        self.day_pnl += pnl_usd
        self.day_trades += 1
        self._tlog({"ev": "closed", "side": "long" if side_long else "short", "qty": filled,
                    "entry_px": entry_px, "exit_px": exit_px, "pnl_bp": round(pnl_bp, 2),
                    "pnl_usd": round(pnl_usd, 5), "backstop": backstop,
                    "exit_chase_s": round(time.time() - t_exit0, 1),
                    "day_pnl": round(self.day_pnl, 5), "day_trades": self.day_trades})
        if self.day_pnl < -DAY_LOSS_HALT * self.day_bal0:
            self.halted = "day_loss"
            self._tlog({"ev": "halt", "why": "day_loss", "day_pnl": self.day_pnl})
        if self.day_trades >= DAY_TRADES_HALT:
            self.halted = "day_trades"
            self._tlog({"ev": "halt", "why": "day_trades", "n": self.day_trades})

    def maybe_trade(self, side_long: bool, score: float) -> bool:
        self._roll_day()
        if self.busy or self.halted or not self.touch_ok():
            return False
        self.busy = True
        decision_ts = time.time()
        threading.Thread(target=self.run_trade, args=(decision_ts, side_long, score),
                         daemon=True).start()
        return True


EXEC: Executor | None = None


async def book_ticker_ws() -> None:
    url = f"{WS_BASE}/public/stream?streams={EXEC_SYM.lower()}@bookTicker"
    while True:
        try:
            async with websockets.connect(url, ping_interval=180, ping_timeout=600) as ws:
                log.info("bookTicker WS connected (%s)", EXEC_SYM)
                async for raw in ws:
                    m = json.loads(raw)
                    d = m.get("data", m)
                    if d.get("e") == "bookTicker" and EXEC is not None:
                        EXEC.on_book_ticker(d)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            log.warning("bookTicker WS dropped: %s — reconnect in 2s", ex)
            await asyncio.sleep(2)


async def handle_cmd(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        cmd = json.loads(line)
        executed = False
        if MODE == "live" and EXEC is not None:
            executed = EXEC.maybe_trade(cmd.get("side") == "long", float(cmd.get("score", 0.0)))
        writer.write((json.dumps({"executed": executed}) + "\n").encode())
        await writer.drain()
    except Exception as ex:
        log.warning("cmd failed: %s", ex)
    finally:
        writer.close()


async def uploader() -> None:
    cl = storage.Client(project=PROJ)
    mkt = cl.bucket(MKT)
    while True:
        await asyncio.sleep(3600)
        try:
            ddir = f"{WORK}/decisions"
            for fn in sorted(os.listdir(ddir)):
                mkt.blob(f"{SHADOW_GCS}/decisions/{fn}").upload_from_filename(f"{ddir}/{fn}")
            log.info("decisions uploaded")
        except Exception as ex:
            log.warning("decision upload failed: %s", ex)


async def main() -> None:
    global EXEC
    sock = f"{WORK}/exec.sock"
    if os.path.exists(sock):
        os.remove(sock)
    if MODE == "live":
        EXEC = Executor(Rest())
        await asyncio.to_thread(EXEC.startup_recover)
        log.info("MODE=live: exec %s budget=t%d size_frac=%.2f entry=%.0fs hold=%.0fs",
                 EXEC_SYM, int(TRADE_BUDGET), SIZE_FRAC, ENTRY_WIN_S, HOLD_S)
    else:
        log.info("MODE=shadow: commands acknowledged but not executed")
    server = await asyncio.start_unix_server(handle_cmd, path=sock)
    os.chmod(sock, 0o600)
    async with server:
        await asyncio.gather(server.serve_forever(), book_ticker_ws(), uploader())


if __name__ == "__main__":
    asyncio.run(main())
