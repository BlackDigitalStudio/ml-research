#!/usr/bin/env python3
"""Offline mock test for the axb_exec v3 slot-pool executor (no network, no keys).

Covers the concurrency semantics that changed vs the single-position v2:
  1. MAX_CONC=1 keeps legacy behavior (second signal rejected while busy);
  2. MAX_CONC=3 stacks same-side trades, rejects the 4th and any opposite-side;
  3. each trade exits its OWN filled quantity via reduce-only (never positionAmt);
  4. MAX_CONC>1 sizing = wallet*SIZE_FRAC*NOTIONAL_MULT/MAX_CONC per trade.

Run: python live/test_axb_exec_slots.py   (exits 0 on pass)
"""
import importlib
import os
import sys
import tempfile
import threading
import time

WORK = tempfile.mkdtemp(prefix="axb_exec_test_")
BASE_ENV = {
    "MODE": "shadow", "WORKDIR": WORK, "EXEC_SYM": "DOGEUSDC", "SYMK": "DOGE",
    "ENTRY_WIN_S": "3.0", "HOLD_S": "0.3", "SIZE_FRAC": "0.5",
}


class FakeRest:
    """Scripted Binance REST: entries and reduce-only exits fill on first poll."""

    def __init__(self, wallet=20.0):
        self.wallet = wallet
        self.orders = {}
        self.next_oid = 1000
        self.pos = 0.0            # net position (signed)
        self.log = []             # every mutating call, for asserts
        self._mu = threading.Lock()

    def call(self, method, path, params=None, signed=True):
        params = dict(params or {})
        with self._mu:
            if path == "/fapi/v2/balance":
                return [{"asset": "USDC", "balance": f"{self.wallet}",
                         "availableBalance": f"{self.wallet}"}]
            if path == "/fapi/v2/positionRisk":
                return [{"positionAmt": f"{self.pos}"}]
            if path == "/fapi/v1/leverage" or path == "/fapi/v1/allOpenOrders":
                return {"leverage": params.get("leverage", 0)}
            if path == "/fapi/v1/userTrades":
                return []
            if path == "/fapi/v1/order" and method == "POST":
                oid = self.next_oid
                self.next_oid += 1
                qty = float(params["quantity"])
                o = {"orderId": oid, "status": "NEW", "executedQty": "0",
                     "avgPrice": "0", "side": params["side"],
                     "reduceOnly": params.get("reduceOnly") == "true",
                     "type": params["type"], "qty": qty,
                     "price": float(params.get("price", 0) or 0)}
                self.orders[oid] = o
                self.log.append(("place", o["type"], params["side"], qty,
                                 o["reduceOnly"]))
                if params["type"] == "MARKET":
                    self._fill(o)
                return dict(o)
            if path == "/fapi/v1/order" and method == "GET":
                o = self.orders[int(params["orderId"])]
                if o["status"] == "NEW":
                    self._fill(o)
                return dict(o)
            if path == "/fapi/v1/order" and method == "DELETE":
                o = self.orders[int(params["orderId"])]
                if o["status"] == "NEW":
                    o["status"] = "CANCELED"
                return dict(o)
        raise AssertionError(f"unexpected call {method} {path}")

    def _fill(self, o):
        o["status"] = "FILLED"
        o["executedQty"] = f"{o['qty']}"
        o["avgPrice"] = f"{o['price'] if o['price'] else 0.2}"
        signed_q = o["qty"] if o["side"] == "BUY" else -o["qty"]
        self.pos += signed_q
        self.log.append(("fill", o["type"], o["side"], o["qty"], o["reduceOnly"]))


def load(env):
    os.environ.update(BASE_ENV)
    os.environ.update(env)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import axb_exec
    return importlib.reload(axb_exec)


def mk_exec(mod, rest):
    ex = mod.Executor(rest)
    ex.on_book_ticker({"b": "0.20000", "a": "0.20001"})
    return ex


def wait_idle(ex, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with ex._mu:
            if not ex.active:
                return True
        time.sleep(0.1)
    return False


def test_m1_legacy():
    mod = load({"MAX_CONC": "1"})
    rest = FakeRest()
    ex = mk_exec(mod, rest)
    assert ex.maybe_trade(True, 0.9) is True
    time.sleep(0.2)
    assert ex.maybe_trade(True, 0.9) is False, "second signal must be rejected at M=1"
    assert wait_idle(ex)
    assert ex.maybe_trade(True, 0.9) is True, "slot must free after lifecycle"
    assert wait_idle(ex)
    closes = [l for l in rest.log if l[0] == "fill" and l[4]]
    assert len(closes) == 2, f"each trade must exit once, got {closes}"
    print("  ok: M=1 legacy busy semantics")


def test_m3_stack_and_opposite():
    mod = load({"MAX_CONC": "3"})
    rest = FakeRest(wallet=60.0)  # 60*0.5/3 = 10 USDC per trade, above minNotional
    ex = mk_exec(mod, rest)
    assert ex.maybe_trade(True, 0.9) is True
    assert ex.maybe_trade(True, 0.9) is True
    assert ex.maybe_trade(False, 0.9) is False, "opposite side must be skipped"
    assert ex.maybe_trade(True, 0.9) is True
    assert ex.maybe_trade(True, 0.9) is False, "4th signal must be rejected at M=3"
    with ex._mu:
        assert len(ex.active) == 3
    assert wait_idle(ex)
    entries = [l for l in rest.log if l[0] == "place" and not l[4]]
    exits = [l for l in rest.log if l[0] == "fill" and l[4]]
    assert len(entries) == 3 and len(exits) == 3, (entries, exits)
    assert abs(rest.pos) < 1e-9, f"flat after all exits, pos={rest.pos}"
    print("  ok: M=3 same-side stacking, opposite-side skip, slot cap")


def test_exit_own_qty():
    mod = load({"MAX_CONC": "2"})
    rest = FakeRest(wallet=40.0)  # 40*0.5/2 = 10 USDC per trade
    ex = mk_exec(mod, rest)
    assert ex.maybe_trade(True, 0.9) is True
    time.sleep(0.15)
    assert ex.maybe_trade(True, 0.9) is True
    assert wait_idle(ex)
    entry_qty = [l[3] for l in rest.log if l[0] == "place" and not l[4]]
    exit_qty = [l[3] for l in rest.log if l[0] == "place" and l[4]]
    assert sorted(entry_qty) == sorted(exit_qty), (entry_qty, exit_qty)
    # per-trade exit must equal the per-trade entry, not the (2x) net position
    assert all(abs(q - entry_qty[0]) < 1e-9 for q in exit_qty), exit_qty
    print("  ok: exits sized by own filled qty (not positionAmt)")


def test_sizing_m4():
    mod = load({"MAX_CONC": "4", "NOTIONAL_MULT": "1.0"})
    rest = FakeRest(wallet=48.0)
    ex = mk_exec(mod, rest)
    assert ex.maybe_trade(True, 0.9) is True
    assert wait_idle(ex)
    q = [l[3] for l in rest.log if l[0] == "place" and not l[4]][0]
    # 48 * 0.5 / 4 = 6.0 USDC notional -> /0.2 px * (1-1e-3) floored = 29
    assert q == 29, f"expected 29, got {q}"
    # NB: per-trade notional AT the minNotional boundary (e.g. 5.00) gets shaved
    # under it by the (1-1e-3) haircut and floored qty -> always skip_small.
    # Deploy configs must keep wallet*SIZE_FRAC*MULT/MAX_CONC >= ~5.5 (DOGE/XRP).
    print("  ok: M>1 per-trade notional = wallet*SIZE_FRAC*MULT/MAX_CONC")


if __name__ == "__main__":
    test_m1_legacy()
    test_m3_stack_and_opposite()
    test_exit_own_qty()
    test_sizing_m4()
    print("ALL PASS")
