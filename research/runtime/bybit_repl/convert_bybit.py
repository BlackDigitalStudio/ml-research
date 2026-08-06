#!/usr/bin/env python3
"""Bybit free-archive day -> Cryptolake raw layout (flat schema the frozen pipeline reads).

Per (symbol, day) produces, under $LOCAL_GCS_ROOT/market-data-0998ac51/:
  raw/book/exchange=BINANCE_FUTURES/symbol={SYM}-USDT-PERP/dt={day}/1.parquet
      <- quote-saver.bycsi.com ob500/ob200 stream replay, top-20 levels per message,
         timestamp ms->ns.  (path says BINANCE_FUTURES only because the frozen build
         script hard-codes that prefix; the DATA IS BYBIT LINEAR — see VENUE_NOTE.)
  raw/trades/.../dt={day}/1.parquet  <- public.bybit.com tick trades, side buy/sell,
         amount=size, ts s.4 -> ns, id = per-day row index (already unique, dedup no-op).
  raw/funding/.../dt={day}/1.parquet <- 1-min grid: rate = last SETTLED Binance funding
         rate <= t (venue proxy, causal: settled-at-t known at t), mark_price = last
         Bybit trade price <= t.  Bybit REST is geo-fenced from US egress; archive has
         no funding stream. col44 (basis) is zeroed by the anchored intervention anyway.
  raw/open_interest/.../dt={day}/1.parquet <- Binance USDT-M metrics 5-min
         sum_open_interest (venue proxy; CL had ~4s Bybit-free... Binance 4s equiv).
  feats_sub60/BTC-USDT-PERP/{day}.npz <- td/mid = BTC last trade price on 1s grid
         (proxy for CL BTC mids; btc_ret windows are 5/30/60s so trade-price path is
         adequate at that horizon).

Env: DAY (YYYY-MM-DD), LOCAL_GCS_ROOT, WORK (scratch), SKIP_BOOK=1 (trades/aux only).
Aux inputs (funding rates json + oi parquet) are prepared once by fetch_aux.py under
$LOCAL_GCS_ROOT/market-data-0998ac51/bybit_aux/.
"""
import gzip
import io
import json
import os
import sys
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NS = 1_000_000_000
ROOT = os.environ.get("LOCAL_GCS_ROOT", "/vol/gcs")
BUCKET = os.path.join(ROOT, "market-data-0998ac51")
WORK = os.environ.get("WORK", "/tmp/conv")
os.makedirs(WORK, exist_ok=True)
DEPTH = 20

OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{sym}/{day}_{sym}_{kind}.data.zip"
TR_URL = "https://public.bybit.com/trading/{sym}/{sym}{day}.csv.gz"


def fetch(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    for att in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as f:
                want = int(r.headers.get("Content-Length") or -1)
                got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
            if want in (-1, got):
                return True
            print(f"  fetch short {url}: {got}/{want} att{att}", flush=True)
        except Exception as e:
            if "404" in str(e):
                return False
            print(f"  fetch fail {url}: {e} att{att}", flush=True)
        import time as _t
        _t.sleep(2 * (att + 1))
    return False


def out_path(stream, symf, day):
    p = os.path.join(BUCKET, f"raw/{stream}/exchange=BINANCE_FUTURES/symbol={symf}/dt={day}")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "1.snappy.parquet")


def convert_book(sym, symf, day):
    dst = out_path("book", symf, day)
    if os.path.exists(dst):
        return "book-done"
    import zipfile
    try:
        import orjson as oj
        loads = oj.loads
    except ImportError:
        loads = json.loads
    from sortedcontainers import SortedDict
    zp = os.path.join(WORK, f"{sym}_{day}_ob.zip")
    ok = False
    for kind in ("ob500", "ob200"):
        if fetch(OB_URL.format(sym=sym, day=day, kind=kind), zp):
            ok = True
            break
    if not ok:
        return "book-missing"
    bids = SortedDict()   # price -> qty (neg key for desc? use peekitem from end)
    asks = SortedDict()
    ts_l, rows = [], []
    with zipfile.ZipFile(zp) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            for line in fh:
                o = loads(line)
                d = o["data"]
                if o["type"] == "snapshot":
                    bids.clear(); asks.clear()
                    for p, q in d["b"]:
                        bids[float(p)] = float(q)
                    for p, q in d["a"]:
                        asks[float(p)] = float(q)
                else:
                    for p, q in d.get("b", ()):
                        p = float(p); q = float(q)
                        if q == 0.0:
                            bids.pop(p, None)
                        else:
                            bids[p] = q
                    for p, q in d.get("a", ()):
                        p = float(p); q = float(q)
                        if q == 0.0:
                            asks.pop(p, None)
                        else:
                            asks[p] = q
                if len(bids) < DEPTH or len(asks) < DEPTH:
                    continue
                row = np.empty(4 * DEPTH)
                bk = bids.keys(); ak = asks.keys()
                nb = len(bk)
                for k in range(DEPTH):
                    bp = bk[nb - 1 - k]
                    ap = ak[k]
                    row[k] = bp; row[DEPTH + k] = bids[bp]
                    row[2 * DEPTH + k] = ap; row[3 * DEPTH + k] = asks[ap]
                ts_l.append(o["ts"]); rows.append(row)
    os.remove(zp)
    if len(rows) < 1000:
        return f"book-thin {len(rows)}"
    M = np.asarray(rows)
    ts = np.asarray(ts_l, np.int64) * 1_000_000  # ms -> ns
    o = np.argsort(ts, kind="stable")
    M = M[o]; ts = ts[o]
    cols = {"timestamp": pa.array(ts, pa.int64())}
    for k in range(DEPTH):
        cols[f"bid_{k}_price"] = pa.array(M[:, k], pa.float64())
        cols[f"bid_{k}_size"] = pa.array(M[:, DEPTH + k], pa.float64())
        cols[f"ask_{k}_price"] = pa.array(M[:, 2 * DEPTH + k], pa.float64())
        cols[f"ask_{k}_size"] = pa.array(M[:, 3 * DEPTH + k], pa.float64())
    pq.write_table(pa.table(cols), dst, compression="snappy")
    return f"book ok n={len(ts)} dens={len(ts)/max((ts[-1]-ts[0])/NS,1):.1f}/s"


def load_trades_csv(sym, day):
    gz = os.path.join(WORK, f"{sym}_{day}_t.csv.gz")
    if not fetch(TR_URL.format(sym=sym, day=day), gz):
        return None
    import pyarrow.csv as pc
    with gzip.open(gz, "rb") as f:
        data = f.read()
    os.remove(gz)
    t = pc.read_csv(io.BytesIO(data))
    return t


def convert_trades(sym, symf, day):
    dst = out_path("trades", symf, day)
    if os.path.exists(dst):
        return "trades-done", None
    t = load_trades_csv(sym, day)
    if t is None:
        return "trades-missing", None
    ts = (np.asarray(t["timestamp"], np.float64) * 1e9).round().astype(np.int64)
    side = np.asarray(t["side"])  # "Buy"/"Sell" = taker side
    amount = np.asarray(t["size"], np.float64)
    price = np.asarray(t["price"], np.float64)
    o = np.argsort(ts, kind="stable")
    ts, amount, price = ts[o], amount[o], price[o]
    side = np.array([s.lower() for s in side[o]], dtype=object)
    tbl = pa.table({
        "side": pa.array(side, pa.string()),
        "amount": pa.array(amount, pa.float64()),
        "price": pa.array(price, pa.float64()),
        "id": pa.array(np.arange(len(ts), dtype=np.int64), pa.int64()),
        "timestamp": pa.array(ts, pa.int64()),
    })
    pq.write_table(tbl, dst, compression="snappy")
    return f"trades ok n={len(ts)}", (ts, price)


def btc_mid_npz(day, ts_price):
    d = os.path.join(BUCKET, "feats_sub60/BTC-USDT-PERP")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f"{day}.npz")
    if os.path.exists(dst):
        return "btc-done"
    ts, price = ts_price
    # 1s last-price grid (CL feats_sub60 mids proxy; btc_ret windows are 5/30/60s)
    t0 = ts[0] - ts[0] % NS
    grid = np.arange(t0, ts[-1], NS)
    idx = np.clip(np.searchsorted(ts, grid, "right") - 1, 0, len(ts) - 1)
    np.savez_compressed(dst + ".tmp.npz", td=grid.astype(np.int64), mid=price[idx].astype(np.float64))
    os.replace(dst + ".tmp.npz", dst)
    return f"btc ok n={len(grid)}"


def convert_funding(symf, day, doge_ts_price):
    dst = out_path("funding", symf, day)
    if os.path.exists(dst):
        return "funding-done"
    fr = json.load(open(os.path.join(BUCKET, "bybit_aux/funding_rates.json")))
    fts = np.array([x[0] for x in fr], np.int64) * 1_000_000  # ms->ns
    fv = np.array([x[1] for x in fr], np.float64)
    day0 = np.datetime64(day).astype("datetime64[s]").astype(np.int64) * NS
    grid = np.arange(day0, day0 + 86400 * NS, 60 * NS)
    j = np.clip(np.searchsorted(fts, grid, "right") - 1, 0, len(fts) - 1)  # last settled <= t
    rate = fv[j]
    if doge_ts_price is not None:
        ts, price = doge_ts_price
        k = np.clip(np.searchsorted(ts, grid, "right") - 1, 0, len(ts) - 1)
        mark = price[k]
    else:
        mark = np.zeros(len(grid))
    tbl = pa.table({"rate": pa.array(rate, pa.float64()),
                    "mark_price": pa.array(mark, pa.float64()),
                    "timestamp": pa.array(grid, pa.int64())})
    pq.write_table(tbl, dst, compression="snappy")
    return "funding ok"


def convert_oi(symf, day):
    dst = out_path("open_interest", symf, day)
    if os.path.exists(dst):
        return "oi-done"
    src = os.path.join(BUCKET, "bybit_aux/oi_5m.parquet")
    t = pq.read_table(src)
    ts = np.asarray(t["timestamp"], np.int64)
    oi = np.asarray(t["open_interest"], np.float64)
    day0 = np.datetime64(day).astype("datetime64[s]").astype(np.int64) * NS
    m = (ts >= day0) & (ts < day0 + 86400 * NS)
    if not m.any():
        return "oi-empty"
    tbl = pa.table({"open_interest": pa.array(oi[m], pa.float64()),
                    "timestamp": pa.array(ts[m], pa.int64())})
    pq.write_table(tbl, dst, compression="snappy")
    return f"oi ok n={int(m.sum())}"


def run_day(day):
    log = [day]
    st, doge_tp = convert_trades("DOGEUSDT", "DOGE-USDT-PERP", day)
    log.append(st)
    log.append(convert_book("DOGEUSDT", "DOGE-USDT-PERP", day))
    st_eth, _ = convert_trades("ETHUSDT", "ETH-USDT-PERP", day)
    log.append(st_eth)
    st_btc, btc_tp = convert_trades("BTCUSDT", "BTC-USDT-PERP", day)
    log.append(st_btc)
    if btc_tp is not None:
        log.append(btc_mid_npz(day, btc_tp))
    log.append(convert_funding("DOGE-USDT-PERP", day, doge_tp))
    log.append(convert_oi("DOGE-USDT-PERP", day))
    return " | ".join(log)


if __name__ == "__main__":
    day = os.environ.get("DAY") or sys.argv[1]
    print(run_day(day), flush=True)
