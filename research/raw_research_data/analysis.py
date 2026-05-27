#!/usr/bin/env python3
"""Market analysis for the sub-60s holding-time hypothesis.

Runs on a GCE VM co-located with gs://blackdigital-scalper-data (europe-west1).
Reads raw parquet directly from GCS with column projection (no bulk download).

Produces MEASURED quantities only (distributions, exceedance fractions, rank-IC).
No verdicts, no pass/fail, no extrapolation.

Blocks:
  A. Volatility: distribution of |mid return| over horizons vs round-trip cost
     floors {4,7,10,13} bp; top-of-book spread distribution.
  B. Ingest latency: receipt_timestamp - exchange timestamp, per symbol/stream.
  C. Edge decay: rank-IC of order-book imbalance (OBI) and trade-flow
     imbalance (TFI) vs forward mid return across horizons -> signal half-life.
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

BUCKET = "gs://blackdigital-scalper-data/raw"
EXCH = "BINANCE_FUTURES"
SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "LINK", "LTC"]
N_DAYS = 30

GRID = "250ms"
STEP_S = 0.25
HORIZONS_VOL = [1, 5, 10, 15, 30, 45, 60]          # seconds
HORIZONS_EDGE = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60]
COST_FLOORS_BP = [4, 7, 10, 13]                     # maker-maker, mixed, taker-taker, +slippage
FFILL_LIMIT = 20                                     # 20 * 250ms = 5s max bridge over gaps
LAT_SAMPLE_PER_DAY = 50_000                          # subsample for latency quantiles


def _fs():
    import gcsfs
    return gcsfs.GCSFileSystem()


def list_days(sym):
    fs = _fs()

    def days_for(stream):
        base = f"blackdigital-scalper-data/raw/{stream}/exchange={EXCH}/symbol={sym}-USDT-PERP/"
        return {p.rstrip("/").split("dt=")[-1] for p in fs.ls(base) if "dt=" in p}

    common = sorted(days_for("book") & days_for("trades"))
    return common[-N_DAYS:]


def read_dir(stream, sym, day, cols):
    fs = _fs()
    prefix = f"blackdigital-scalper-data/raw/{stream}/exchange={EXCH}/symbol={sym}-USDT-PERP/dt={day}"
    files = fs.glob(prefix + "/*.parquet")
    if not files:
        raise FileNotFoundError(prefix)
    parts = [pd.read_parquet("gs://" + f, columns=cols) for f in files]
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def read_book(sym, day):
    cols = ["timestamp", "receipt_timestamp",
            "bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size"]
    df = read_dir("book", sym, day, cols)
    df = df[df["bid_0_price"] > 0]
    df = df[df["ask_0_price"] > df["bid_0_price"]]
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
    df["mid"] = (df["bid_0_price"] + df["ask_0_price"]) / 2.0
    df["spread_bp"] = (df["ask_0_price"] - df["bid_0_price"]) / df["mid"] * 1e4
    bs, as_ = df["bid_0_size"], df["ask_0_size"]
    df["obi"] = (bs - as_) / (bs + as_)
    df["lat_ms"] = (df["receipt_timestamp"] - df["timestamp"]) / 1e6
    return df.set_index("ts").sort_index()


def read_trades(sym, day):
    cols = ["timestamp", "receipt_timestamp", "side", "amount"]
    df = read_dir("trades", sym, day, cols)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
    df["lat_ms"] = (df["receipt_timestamp"] - df["timestamp"]) / 1e6
    df["signed"] = np.where(df["side"].str.lower() == "buy", 1.0, -1.0) * df["amount"]
    return df.set_index("ts").sort_index()


def grid_day(book, trades):
    """Build a regular GRID series of mid, obi, tfi for one day."""
    mid = book["mid"].resample(GRID).last().ffill(limit=FFILL_LIMIT)
    obi = book["obi"].resample(GRID).last().ffill(limit=FFILL_LIMIT)
    tfi = trades["signed"].resample(GRID).sum().reindex(mid.index).fillna(0.0)
    return pd.DataFrame({"mid": mid, "obi": obi, "tfi": tfi})


def rank_ic(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 100:
        return np.nan
    xr = x[m].rank().to_numpy()
    yr = y[m].rank().to_numpy()
    if xr.std() == 0 or yr.std() == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def process_symbol(sym):
    days = list_days(sym)
    abs_ret = {h: [] for h in HORIZONS_VOL}      # pooled |return| bp
    spreads, lat_book, lat_trades = [], [], []
    ic_obi = {h: [] for h in HORIZONS_EDGE}       # per-day IC
    ic_tfi = {h: [] for h in HORIZONS_EDGE}
    n_ok = 0
    rng = np.random.default_rng(0)

    def sub(a):
        a = a[np.isfinite(a)]
        if len(a) > LAT_SAMPLE_PER_DAY:
            a = rng.choice(a, LAT_SAMPLE_PER_DAY, replace=False)
        return a

    for day in days:
        try:
            book = read_book(sym, day)
            trades = read_trades(sym, day)
        except Exception:
            continue
        spreads.append(sub(book["spread_bp"].to_numpy()))
        lat_book.append(sub(book["lat_ms"].to_numpy()))
        lat_trades.append(sub(trades["lat_ms"].to_numpy()))

        g = grid_day(book, trades)
        mid = g["mid"]
        for h in HORIZONS_VOL:
            k = max(1, round(h / STEP_S))
            r = (mid.shift(-k) / mid - 1.0) * 1e4
            abs_ret[h].append(np.abs(r.dropna().to_numpy()))
        for h in HORIZONS_EDGE:
            k = max(1, round(h / STEP_S))
            fwd = (mid.shift(-k) / mid - 1.0)
            ic_obi[h].append(rank_ic(g["obi"], fwd))
            ic_tfi[h].append(rank_ic(g["tfi"], fwd))
        n_ok += 1

    def qstats(arrs, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
        a = np.concatenate(arrs) if arrs else np.array([0.0])
        out = {"n": int(len(a)), "mean": float(np.mean(a)), "std": float(np.std(a))}
        for q in qs:
            out[f"p{int(q*100)}"] = float(np.quantile(a, q))
        return out

    res = {"symbol": sym, "days": days, "n_days_ok": n_ok}
    # A: volatility + exceedance
    res["abs_ret_bp"] = {h: qstats(abs_ret[h]) for h in HORIZONS_VOL}
    res["exceed_frac"] = {}
    for h in HORIZONS_VOL:
        a = np.concatenate(abs_ret[h]) if abs_ret[h] else np.array([0.0])
        res["exceed_frac"][h] = {f"ge_{f}bp": float(np.mean(a >= f)) for f in COST_FLOORS_BP}
    res["spread_bp"] = qstats(spreads)
    # B: latency
    res["lat_ms"] = {"book": qstats(lat_book), "trades": qstats(lat_trades)}
    # C: edge decay (mean IC across days, +sd for stability)
    def icstat(d):
        return {h: {"ic_mean": float(np.nanmean(d[h])),
                    "ic_sd": float(np.nanstd(d[h])),
                    "n_days": int(np.sum(np.isfinite(d[h])))} for h in HORIZONS_EDGE}
    res["ic_obi"] = icstat(ic_obi)
    res["ic_tfi"] = icstat(ic_tfi)
    return res


def main():
    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_symbol, s): s for s in SYMBOLS}
        for f in as_completed(futs):
            s = futs[f]
            try:
                results[s] = f.result()
                print(f"[done] {s}: {results[s]['n_days_ok']} days", flush=True)
            except Exception as e:
                print(f"[FAIL] {s}: {e}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- compact stdout summary ----
    print("\n================ A. VOLATILITY: |mid return| (bp) by horizon ================")
    print(f"{'sym':5} {'h(s)':>5} {'p50':>7} {'p90':>7} {'p99':>7} | "
          f"{'>=4bp':>6} {'>=7bp':>6} {'>=10bp':>7} {'>=13bp':>7}")
    for s in SYMBOLS:
        if s not in results:
            continue
        r = results[s]
        for h in HORIZONS_VOL:
            q = r["abs_ret_bp"][h]; e = r["exceed_frac"][h]
            print(f"{s:5} {h:5d} {q['p50']:7.2f} {q['p90']:7.2f} {q['p99']:7.2f} | "
                  f"{e['ge_4bp']:6.3f} {e['ge_7bp']:6.3f} {e['ge_10bp']:7.3f} {e['ge_13bp']:7.3f}")

    print("\n================ spread (bp) ================")
    print(f"{'sym':5} {'p50':>7} {'p90':>7} {'p99':>7}")
    for s in SYMBOLS:
        if s in results:
            sp = results[s]["spread_bp"]
            print(f"{s:5} {sp['p50']:7.3f} {sp['p90']:7.3f} {sp['p99']:7.3f}")

    print("\n================ B. INGEST LATENCY receipt-exchange (ms) ================")
    print(f"{'sym':5} {'stream':7} {'p50':>8} {'p90':>8} {'p99':>8}")
    for s in SYMBOLS:
        if s not in results:
            continue
        for st in ("book", "trades"):
            L = results[s]["lat_ms"][st]
            print(f"{s:5} {st:7} {L['p50']:8.1f} {L['p90']:8.1f} {L['p99']:8.1f}")

    print("\n================ C. EDGE DECAY: rank-IC vs forward mid return ================")
    for sig in ("ic_obi", "ic_tfi"):
        print(f"\n--- {sig} (mean across days) ---")
        print("sym  " + " ".join(f"{h:>7}" for h in HORIZONS_EDGE))
        for s in SYMBOLS:
            if s not in results:
                continue
            row = results[s][sig]
            print(f"{s:5}" + " ".join(f"{row[h]['ic_mean']:7.3f}" for h in HORIZONS_EDGE))
    print("\nDONE")


if __name__ == "__main__":
    main()
