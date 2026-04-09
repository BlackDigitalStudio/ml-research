"""Check Binance Futures API connectivity and account setup."""
import hashlib
import hmac
import time
import urllib.parse
import asyncio
import aiohttp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config

cfg = load_config("config.env")
API_KEY = cfg.api_key
API_SECRET = cfg.api_secret
SYMBOL = cfg.symbol
BASE = cfg.rest_base


def sign(params: dict) -> dict:
    params["timestamp"] = int(time.time() * 1000)
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


async def main():
    headers = {"X-MBX-APIKEY": API_KEY}
    async with aiohttp.ClientSession(headers=headers) as s:
        # 1. Ping
        t0 = time.monotonic()
        async with s.get(f"{BASE}/fapi/v1/ping") as r:
            ms = (time.monotonic() - t0) * 1000
            print(f"[1] Ping: {r.status} ({ms:.1f}ms)")

        # 2. Server time
        async with s.get(f"{BASE}/fapi/v1/time") as r:
            data = await r.json()
            drift = int(time.time() * 1000) - data.get("serverTime", 0)
            print(f"[2] Time drift: {drift}ms")

        # 3. Symbol info
        async with s.get(f"{BASE}/fapi/v1/exchangeInfo") as r:
            data = await r.json()
            for sym in data["symbols"]:
                if sym["symbol"] == SYMBOL and sym["contractType"] == "PERPETUAL":
                    filters = {f["filterType"]: f for f in sym["filters"]}
                    pf = filters["PRICE_FILTER"]
                    lf = filters["LOT_SIZE"]
                    mn = filters["MIN_NOTIONAL"]
                    print(f"[3] {SYMBOL} Perpetual:")
                    print(f"    Status:       {sym['status']}")
                    print(f"    Tick size:    {pf['tickSize']}")
                    print(f"    Step size:    {lf['stepSize']}")
                    print(f"    Min qty:      {lf['minQty']}")
                    print(f"    Min notional: {mn['notional']}")
                    break

        # 4. Balance
        params = sign({})
        async with s.get(f"{BASE}/fapi/v2/balance", params=params) as r:
            data = await r.json()
            if r.status == 200:
                for b in data:
                    if float(b.get("balance", 0)) > 0:
                        print(f"[4] Balance: {b['asset']} = {b['balance']} (available: {b['availableBalance']})")

        # 5. Set isolated margin for primary symbol
        params = sign({"symbol": SYMBOL, "marginType": "ISOLATED"})
        async with s.post(f"{BASE}/fapi/v1/marginType", params=params) as r:
            data = await r.json()
            code = data.get("code", 200)
            if r.status == 200 or code == -4046:
                print(f"[5] {SYMBOL} margin: ISOLATED (OK)")
            else:
                print(f"[5] Margin error: [{code}] {data.get('msg')}")

        # 6. Set leverage
        params = sign({"symbol": SYMBOL, "leverage": cfg.leverage})
        async with s.post(f"{BASE}/fapi/v1/leverage", params=params) as r:
            data = await r.json()
            if r.status == 200:
                print(f"[6] {SYMBOL} leverage: x{data.get('leverage')}")
            else:
                print(f"[6] Leverage error: {data}")

        # 7. Current price
        async with s.get(f"{BASE}/fapi/v1/ticker/price", params={"symbol": SYMBOL}) as r:
            data = await r.json()
            price = float(data.get("price", 0))
            sl_usd = price * cfg.stop_loss_pct / 100
            tp_usd = price * cfg.take_profit_pct / 100
            print(f"[7] {SYMBOL} price: ${price:,.2f}")
            print(f"    SL = {cfg.stop_loss_pct}% = ${sl_usd:.2f}")
            print(f"    TP = {cfg.take_profit_pct}% = ${tp_usd:.2f}")
            print(f"    TP:SL = {cfg.take_profit_pct/cfg.stop_loss_pct:.0f}:1")


if __name__ == "__main__":
    asyncio.run(main())
