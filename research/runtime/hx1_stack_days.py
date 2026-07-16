#!/usr/bin/env python3
"""HX1 rev5 — day builder for the A0/A1 retrain arms: per (SYM, day) recorder
day -> F71 features + tb3s-h150 maker labels + decision grid-sec map, saved to
gs://.../research_runs/hx1_stack/days/{SYM}/{day}.npz.

Replicates the recorder-EV per-day stage (trading_algorithm
validation/subs60_recorder_ev_h150.py) with the SAME frozen binaries
(FB_BIN/BS_BIN/GRID_BIN — the July-2026 parity-proven builds), FUNDING_MODE=
anchor, calendar-UTC-midnight 3s grid, W=50 H=6000, entry 60s / hold 150s /
chase 300s / always-last / 0 fee. gcloud-cp based IO (no storage lib dep).

PARITY: for symbols with an existing _recev_h150anch2_{SYM} prefix (DOGE, XRP)
the rebuilt netl/nets arrays are compared against the stored D_{day}.npz —
any mismatch beyond float storage noise fails the day loudly.

Env: SYMS(DOGE,XRP,BTC,ETH) DAY0(20260628) DAYN(20260714) NPROC(8)
     BINS(/home/delmi/xsym/bins) OUT(gs://market-data-0998ac51/research_runs/hx1_stack)
"""
import io
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("DAY0", "20260628")
os.environ.setdefault("DAYN", "20260714")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx1_oos import day_list  # noqa: E402

MKT = "gs://market-data-0998ac51"
REC = "gs://recorder-data-asia-0998ac51/chronos/scalper-recorder/binance_futures"
OUT = os.environ.get("OUT", f"{MKT}/research_runs/hx1_stack")
SYMS = os.environ.get("SYMS", "DOGE,XRP,BTC,ETH").split(",")
BINS = os.environ.get("BINS", "/home/delmi/xsym/bins")
FB = f"{BINS}/fb_target/release/feature_builder"
BS = f"{BINS}/husdc_target/release/build_samples"
GRID = f"{BINS}/husdc_target/release/grid_sim_exitdbg"
PARITY_REF = {"DOGE": "_recev_h150anch2_DOGE", "XRP": "_recev_h150anch2_XRP"}
NS = 1_000_000_000
LV, W, H, STEP_S = 20, 50, 6000, 3
ENTRY_MS, CHASE_MS, HOLD_MS = 60_000, 300_000, 150_000
CFG = [{"tp": 50.0, "sl": 50.0, "to": 282, "to_ms": float(HOLD_MS),
        "par": False, "tr": False}]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1500:]}")
    return r.stdout


def fetch(prefix, day, td, sub):
    os.makedirs(f"{td}/{sub}", exist_ok=True)
    try:
        sh(f"gcloud storage cp '{prefix}/{day}_*.parquet' {td}/{sub}/ -q")
    except RuntimeError:
        return []
    return sorted(os.path.join(td, sub, f) for f in os.listdir(f"{td}/{sub}"))


def _flat(col, k=LV):
    arr = col.combine_chunks()
    vals = arr.values.to_numpy(zero_copy_only=False).astype(np.float64)
    if hasattr(arr, "offsets"):
        off = arr.offsets.to_numpy().astype(np.int64)
        m = np.zeros((len(off) - 1, k))
        for i in range(len(off) - 1):
            n = min(off[i + 1] - off[i], k)
            m[i, :n] = vals[off[i]:off[i] + n]
        return m
    return vals.reshape(-1, arr.type.list_size)[:, :k]


def book_cl(files, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ts_l, cols_l = [], []
    for f in files:
        t = pq.read_table(f, columns=["exchange_event_ts_us", "bid_prices",
                                      "bid_qtys", "ask_prices", "ask_qtys"])
        ets = t["exchange_event_ts_us"].to_numpy().astype(np.float64)
        m = ~np.isnan(ets)
        if not m.any():
            continue
        ts_l.append((ets[m].astype(np.int64)) * 1000)
        cols_l.append([_flat(t[c])[m] for c in
                       ("bid_prices", "bid_qtys", "ask_prices", "ask_qtys")])
    if not ts_l:
        return 0, None
    ts = np.concatenate(ts_l)
    bp, bq, ap, aq = (np.concatenate([c[i] for c in cols_l]) for i in range(4))
    o = np.argsort(ts, kind="stable")
    ts, bp, bq, ap, aq = ts[o], bp[o], bq[o], ap[o], aq[o]
    data = {"timestamp": ts, "receipt_timestamp": ts,
            "sequence_number": np.arange(len(ts), dtype=np.int64)}
    for i in range(LV):
        data[f"bid_{i}_price"] = bp[:, i]
        data[f"bid_{i}_size"] = bq[:, i]
        data[f"ask_{i}_price"] = ap[:, i]
        data[f"ask_{i}_size"] = aq[:, i]
    import pyarrow as pa
    pq.write_table(pa.table(data), out)
    return len(ts), ts


def trades_cl(files, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    acc = {k: [] for k in ("ts", "px", "q", "ibm", "tid")}
    for f in files:
        t = pq.read_table(f, columns=["exchange_event_ts_us", "local_ts_us",
                                      "trade_id", "price", "qty", "is_buyer_maker"])
        ets = t["exchange_event_ts_us"].to_numpy().astype(np.float64)
        m = ~np.isnan(ets)
        acc["ts"].append(ets[m].astype(np.int64) * 1000)
        acc["px"].append(t["price"].to_numpy()[m])
        acc["q"].append(t["qty"].to_numpy()[m])
        acc["ibm"].append(np.asarray(t["is_buyer_maker"].to_numpy(zero_copy_only=False))[m])
        acc["tid"].append(t["trade_id"].to_numpy(zero_copy_only=False)[m])
    if not acc["ts"] or not sum(len(a) for a in acc["ts"]):
        return 0
    a = {k: np.concatenate(v) for k, v in acc.items()}
    _, ui = np.unique(a["tid"].astype(np.int64), return_index=True)
    ui = np.sort(ui)
    t = pa.table({"side": np.where(a["ibm"][ui].astype(bool), "sell", "buy"),
                  "amount": a["q"][ui].astype(np.float64),
                  "price": a["px"][ui].astype(np.float64),
                  "id": a["tid"][ui].astype(np.int64),
                  "timestamp": a["ts"][ui],
                  "receipt_timestamp": a["ts"][ui]}).sort_by("timestamp")
    pq.write_table(t, out)
    return t.num_rows


def liq_cl(files, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    acc = {k: [] for k in ("ts", "side", "q", "px")}
    for f in files:
        t = pq.read_table(f, columns=["exchange_event_ts_us", "side",
                                      "original_qty", "price"])
        ets = t["exchange_event_ts_us"].to_numpy().astype(np.float64)
        m = ~np.isnan(ets)
        acc["ts"].append(ets[m].astype(np.int64) * 1000)
        acc["side"].append(np.array([str(s).lower() for s in
                                     np.asarray(t["side"].to_pylist())[m]]))
        acc["q"].append(t["original_qty"].to_numpy()[m])
        acc["px"].append(t["price"].to_numpy()[m])
    n = sum(len(x) for x in acc["ts"]) if acc["ts"] else 0
    if not n:
        return 0
    a = {k: np.concatenate(v) for k, v in acc.items()}
    t = pa.table({"side": a["side"], "quantity": a["q"].astype(np.float64),
                  "price": a["px"].astype(np.float64),
                  "id": np.arange(n, dtype=np.int64),
                  "status": np.array(["filled"] * n),
                  "timestamp": a["ts"], "receipt_timestamp": a["ts"]}
                 ).sort_by("timestamp")
    pq.write_table(t, out)
    return t.num_rows


def funding_cl_anchor(files, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ets_l, fr_l = [], []
    for f in files:
        t = pq.read_table(f, columns=["exchange_event_ts_us", "funding_rate"])
        e = t["exchange_event_ts_us"].to_numpy().astype(np.float64)
        m = ~np.isnan(e)
        ets_l.append(e[m].astype(np.int64))
        fr_l.append(np.nan_to_num(t["funding_rate"].to_numpy().astype(np.float64))[m])
    if not ets_l or not sum(len(x) for x in ets_l):
        return 0
    ets = np.concatenate(ets_l)
    fr = np.concatenate(fr_l)
    i = int(np.argmin(ets))
    t = pa.table({"funding_rate": np.array([fr[i]], np.float64),
                  "mark_price": np.array([0.0], np.float64),
                  "timestamp": np.array([1], np.int64)})
    pq.write_table(t, out)
    return 1


def oi_cl(files, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ts_l, oi_l = [], []
    for f in files:
        t = pq.read_table(f, columns=["local_ts_us", "open_interest"])
        lt = t["local_ts_us"].to_numpy().astype(np.float64)
        o = t["open_interest"].to_numpy().astype(np.float64)
        m = ~np.isnan(lt) & ~np.isnan(o)
        ts_l.append(lt[m].astype(np.int64) * 1000)
        oi_l.append(o[m])
    n = sum(len(x) for x in ts_l) if ts_l else 0
    if not n:
        return 0
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({"open_interest": np.concatenate(oi_l),
                  "timestamp": np.concatenate(ts_l)}).sort_by("timestamp")
    pq.write_table(t, out)
    return n


def btc_mid(day, td):
    import pyarrow.parquet as pq
    files = fetch(f"{REC}/BTCUSDT/depth_snapshot", day, td, "btc")
    ts_l, mid_l = [], []
    for f in files:
        t = pq.read_table(f, columns=["exchange_event_ts_us", "bid_prices",
                                      "ask_prices"])
        e = t["exchange_event_ts_us"].to_numpy().astype(np.float64)
        m = ~np.isnan(e)
        b0 = _flat(t["bid_prices"], 1)[:, 0][m]
        a0 = _flat(t["ask_prices"], 1)[:, 0][m]
        ts_l.append(e[m].astype(np.int64) * 1000)
        mid_l.append((b0 + a0) / 2)
    if not ts_l:
        return np.array([]), np.array([])
    ts = np.concatenate(ts_l)
    mid = np.concatenate(mid_l)
    o = np.argsort(ts)
    return ts[o], mid[o]


def feat71(dtd, X, bts, bm):
    nb = len(bts)
    i = (np.clip(np.searchsorted(bts, dtd, "right") - 1, 0, nb - 1)
         if nb else np.zeros(len(dtd), int))
    bl = []
    for Wd in (5, 30, 60):
        if nb:
            j = np.clip(np.searchsorted(bts, dtd - int(Wd * NS), "right") - 1, 0, nb - 1)
            a = bm[j]
            b = bm[i]
            bl.append(np.where((a > 0) & (b > 0),
                               np.log(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0)),
                               0.0) * 1e4)
        else:
            bl.append(np.zeros(len(dtd)))
    h = ((dtd / NS) % 86400.0) / 3600.0
    hf = h % 8.0
    tod = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
           np.sin(2 * np.pi * hf / 8), np.cos(2 * np.pi * hf / 8)]
    return np.concatenate([X, np.stack(bl, 1), np.stack(tod, 1)], axis=1).astype(np.float32)


def build_day(args):
    sym, day = args
    out_url = f"{OUT}/days/{sym}/{day}.npz"
    if subprocess.run(f"gcloud storage ls {out_url}", shell=True,
                      capture_output=True).returncode == 0:
        return f"{sym} {day} SKIP"
    try:
        with tempfile.TemporaryDirectory() as td:
            nb, bt = book_cl(fetch(f"{REC}/{sym}USDT/depth_snapshot", day, td, "b"),
                             f"{td}/book.parquet")
            ntr = trades_cl(fetch(f"{REC}/{sym}USDT/agg_trade", day, td, "t"),
                            f"{td}/trades.parquet")
            if not nb or not ntr:
                return f"{sym} {day} SKIP thin (book={nb} tr={ntr})"
            nliq = liq_cl(fetch(f"{REC}/{sym}USDT/liquidation", day, td, "l"),
                          f"{td}/liq.parquet")
            nfd = funding_cl_anchor(fetch(f"{REC}/{sym}USDT/mark_price", day, td, "f"),
                                    f"{td}/fund.parquet")
            noi = oi_cl(fetch(f"{REC}/{sym}USDT/derivatives_poll", day, td, "o"),
                        f"{td}/oi.parquet")
            neth = trades_cl(fetch(f"{REC}/ETHUSDT/agg_trade", day, td, "e"),
                             f"{td}/eth.parquet")
            n = len(bt)
            if n < W + H + 100:
                return f"{sym} {day} SKIP thin n={n}"
            mid0 = int(datetime.strptime(day, "%Y%m%d")
                       .replace(tzinfo=timezone.utc).timestamp()) * NS
            grid = np.arange(mid0, bt[-1], STEP_S * NS)
            grid = grid[grid >= bt[0]]
            e = np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1)
            ends, first_i = np.unique(e, return_index=True)
            keep = (ends >= W - 1) & (ends < n - H - 1)
            ends, first_i = ends[keep], first_i[keep]
            sec = ((grid[first_i] - mid0) // NS).astype(np.int64)
            np.save(f"{td}/ends.npy", ends.astype(np.int64))
            od = f"{td}/bs"
            os.makedirs(od, exist_ok=True)
            sh(f"{BS} --depth {td}/book.parquet --trades {td}/trades.parquet "
               f"--out-dir {od} --window {W} --horizon {H} "
               f"--sample-ends {td}/ends.npy --skip-xlob")
            se = np.load(f"{od}/end_indices.npy").astype(np.int64)
            if len(se) != len(ends) or not np.array_equal(se, ends):
                return f"{sym} {day} FAIL se!=ends ({len(se)} vs {len(ends)})"
            with open(f"{td}/cfg.json", "w") as f:
                json.dump(CFG, f)
            g = f"{td}/g"
            sh(f"{GRID} --entry-long {od}/entry_long.npy --entry-short {od}/entry_short.npy "
               f"--mid-paths {od}/mid_paths.npy --book-paths {od}/book_paths.npy "
               f"--entry-book {od}/entry_book.npy --flow-paths {od}/flow_paths.npy "
               f"--entry-q {od}/entry_q.npy --configs {td}/cfg.json --out-prefix {g} "
               f"--queue-mult 1.0 --exit-queue-mult 1.0 --ts-paths {od}/ts_paths.npy "
               f"--sample-ts {od}/sample_ts.npy --entry-window-ms {ENTRY_MS} "
               f"--chase-ms {CHASE_MS} --entry-window-ticks 120 --maker-offset-frac 0 "
               f"--commission-win-pct 0 --commission-loss-pct 0")
            PL = np.load(f"{g}_pnl_long.npy")[0] * 100.0
            PS = np.load(f"{g}_pnl_short.npy")[0] * 100.0
            FL = np.load(f"{g}_filled_long.npy").astype(bool)
            FS = np.load(f"{g}_filled_short.npy").astype(bool)
            np.save(f"{td}/idx.npy", se)
            fcmd = (f"{FB} --depth {td}/book.parquet --indices {td}/idx.npy "
                    f"--out {td}/f.npy --trades {td}/trades.parquet")
            if nfd:
                fcmd += f" --funding {td}/fund.parquet"
            if nliq:
                fcmd += f" --liquidations {td}/liq.parquet"
            if noi:
                fcmd += f" --open-interest {td}/oi.parquet"
            if neth:
                fcmd += f" --eth {td}/eth.parquet"
            sh(fcmd)
            X = np.load(f"{td}/f.npy").astype(np.float64)
            bts, bm = btc_mid(day, td)
            F = feat71(bt[se], X, bts, bm)
            parity = ""
            if sym in PARITY_REF:
                sh(f"gcloud storage cp {MKT}/research_runs/{PARITY_REF[sym]}/D_{day}.npz "
                   f"{td}/D.npz -q")
                d = np.load(f"{td}/D.npz")
                if len(d["netl"]) != len(PL):
                    return f"{sym} {day} FAIL parity len {len(d['netl'])} vs {len(PL)}"
                if not (np.array_equal(np.isnan(d["netl"]), np.isnan(PL))
                        and np.array_equal(np.isnan(d["nets"]), np.isnan(PS))):
                    return f"{sym} {day} FAIL parity NaN-pattern mismatch"
                dl_ = np.nanmax(np.abs(d["netl"] - PL.astype(np.float32)))
                ds_ = np.nanmax(np.abs(d["nets"] - PS.astype(np.float32)))
                if not (dl_ < 1e-3 and ds_ < 1e-3):
                    return f"{sym} {day} FAIL parity netl dmax={dl_:.4g} nets dmax={ds_:.4g}"
                parity = f" parity(netl/nets dmax {dl_:.2g}/{ds_:.2g})"
            buf = io.BytesIO()
            np.savez_compressed(
                buf, F=F, netl=PL.astype(np.float32), nets=PS.astype(np.float32),
                FL=FL, FS=FS, sec=sec, nfd=np.int64(nfd), nliq=np.int64(nliq),
                noi=np.int64(noi), neth=np.int64(neth))
            lf = f"{td}/out.npz"
            open(lf, "wb").write(buf.getvalue())
            sh(f"gcloud storage cp {lf} {out_url} -q")
            return (f"{sym} {day} OK n={len(se)} F={F.shape[1]}col "
                    f"fd/liq/oi/eth={nfd}/{nliq}/{noi}/{neth}{parity}")
    except Exception as ex:  # noqa: BLE001
        return f"{sym} {day} FAIL {type(ex).__name__}: {str(ex)[:300]}"


def main():
    jobs = [(s, d) for s in SYMS for d in day_list()]
    nproc = int(os.environ.get("NPROC", "8"))
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for msg in ex.map(build_day, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
