#!/usr/bin/env python3
"""End-to-end test of the maker-fill integration (runs the built Rust binaries).

Two deterministic checks with KNOWN expected outputs:

  TEST B (build_samples): write tiny synthetic NESTED depth+trades parquets,
    run build_samples, assert the NEW outputs:
      - flow_paths.npy shape (ns,h,2) and a specific cell == injected trade vol
      - entry_q.npy   shape (ns,2)   and == top-1 bid/ask qty at entry
  TEST A (grid_sim maker): craft synthetic .npy inputs for 3 hand-designed
    samples and run grid_sim --flow-paths, asserting:
      - sample0: sells at our level + price->TP  => filled, pnl_long>0
      - sample1: price runs up, no sells<=level  => MISS (filled=0, pnl NaN)
      - sample2: fill then price drops to SL      => filled, pnl_long<0
    and that LEGACY mode (no --flow-paths) still produces a finite assumed-entry pnl.

Usage (on the build VM):
  python3 husdc_e2e_test.py --bins /opt/build/rust_ingest/target/release --work /tmp/e2e
"""
import argparse, json, os, subprocess, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OKS, FAILS = [], []
def chk(name, cond, detail=""):
    (OKS if cond else FAILS).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


def write_npy(path, arr):
    np.save(path, arr)


def test_grid_sim_maker(bins, work):
    print("\n=== TEST A: grid_sim maker mode (synthetic .npy) ===", flush=True)
    # H must exceed timeout_limit_ticks (20) + the config timeout so the exit
    # sim has a monitoring window (simulate_trade_book: max_mon = len - 20).
    N, H = 3, 60
    bid0 = 100.0
    spread = 0.02
    # book_paths[i,t]=[bid,ask]; flow_paths[i,t]=[buy_vol,sell_vol]
    book = np.zeros((N, H, 2))
    flow = np.zeros((N, H, 2))
    # sample 0: at level with sells, then bid climbs above tp (100*1.002=100.2) -> TP
    for t in range(H):
        b = 100.0 if t == 0 else 100.0 + t * 0.05  # 100.05,0.10,0.15,0.20,0.25...
        book[0, t] = [b, b + spread]
        flow[0, t] = [0.0, 1.0]  # sells present
    # sample 1: bid stays ABOVE our level (100) -> maker buy never fills (runaway)
    for t in range(H):
        book[1, t] = [100.5, 100.5 + spread]
        flow[1, t] = [0.0, 5.0]
    # sample 2: fill at tick0 (bid at level, sells), then bid DROPS below SL (100*0.998=99.8)
    for t in range(H):
        b = 100.0 if t == 0 else 100.0 - t * 0.05
        book[2, t] = [b, b + spread]
        flow[2, t] = [0.0, 1.0]
    entry_book = np.array([[100.0, 100.02]] * N)
    entry_long = entry_book[:, 0].copy()
    entry_short = entry_book[:, 1].copy()
    entry_q = np.array([[5.0, 7.0]] * N)
    mid_paths = book.mean(axis=2)  # (N,H) — not used in book mode but required arg
    cfgs = [{"tp": 0.20, "sl": 0.20, "to": 10, "par": False, "tr": False}]

    os.makedirs(work, exist_ok=True)
    p = lambda f: os.path.join(work, f)
    write_npy(p("el.npy"), entry_long); write_npy(p("es.npy"), entry_short)
    write_npy(p("mid.npy"), mid_paths)
    write_npy(p("book.npy"), book); write_npy(p("flow.npy"), flow.astype(np.float32))
    write_npy(p("eb.npy"), entry_book); write_npy(p("eq.npy"), entry_q)
    json.dump(cfgs, open(p("cfg.json"), "w"))

    # MAKER run
    cmd = [os.path.join(bins, "grid_sim"),
           "--entry-long", p("el.npy"), "--entry-short", p("es.npy"),
           "--mid-paths", p("mid.npy"), "--book-paths", p("book.npy"),
           "--entry-book", p("eb.npy"), "--flow-paths", p("flow.npy"),
           "--entry-q", p("eq.npy"), "--configs", p("cfg.json"),
           "--out-prefix", p("mk"), "--commission-win-pct", "0",
           "--commission-loss-pct", "0", "--entry-window-ticks", "60",
           "--queue-mult", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("  grid_sim(maker) rc=", r.returncode, r.stderr.strip()[-300:], flush=True)
    chk("grid_sim maker rc==0", r.returncode == 0)
    if r.returncode != 0:
        return
    pl = np.load(p("mk_pnl_long.npy"))      # (1, N)
    fl = np.load(p("mk_filled_long.npy"))   # (N,)
    print(f"  filled_long={fl.tolist()}  pnl_long={pl[0].tolist()}", flush=True)
    chk("s0 filled", fl[0] == 1)
    chk("s0 pnl>0 (TP)", pl[0, 0] > 0, f"{pl[0,0]:.4f}")
    chk("s1 MISS (filled=0)", fl[1] == 0)
    chk("s1 pnl NaN", not np.isfinite(pl[0, 1]), f"{pl[0,1]}")
    chk("s2 filled", fl[2] == 1)
    chk("s2 pnl<0 (SL/adverse)", pl[0, 2] < 0, f"{pl[0,2]:.4f}")

    # LEGACY run (no --flow-paths) must still work + be finite (assumed entry).
    cmd2 = [os.path.join(bins, "grid_sim"),
            "--entry-long", p("el.npy"), "--entry-short", p("es.npy"),
            "--mid-paths", p("mid.npy"), "--book-paths", p("book.npy"),
            "--entry-book", p("eb.npy"), "--configs", p("cfg.json"),
            "--out-prefix", p("lg"), "--commission-win-pct", "0",
            "--commission-loss-pct", "0"]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    chk("grid_sim legacy rc==0", r2.returncode == 0, r2.stderr.strip()[-200:])
    if r2.returncode == 0:
        pl2 = np.load(p("lg_pnl_long.npy"))
        chk("legacy all finite (assumed entry)", np.isfinite(pl2).all())
        chk("legacy no _filled file (maker off)", not os.path.exists(p("lg_filled_long.npy")))


def test_build_samples(bins, work):
    print("\n=== TEST B: build_samples flow_paths + entry_q (synthetic parquet) ===", flush=True)
    try:
        import pyarrow as pa, pyarrow.parquet as pq
    except Exception as e:
        chk("pyarrow available", False, str(e)); return
    nrows = 30
    ts = list(range(nrows))  # ms 0..29
    def fsl(per_row_first, fill=0.01):
        rows = [[float(v)] + [fill] * 19 for v in per_row_first]
        return pa.array(rows, type=pa.list_(pa.float64(), 20))
    bid_p = [100.0] * nrows
    ask_p = [100.02] * nrows
    bid_q = [5.0] * nrows   # top-1 bid qty -> entry_q[:,0]
    ask_q = [7.0] * nrows   # top-1 ask qty -> entry_q[:,1]
    depth = pa.table({
        "timestamp": pa.array(ts, pa.int64()),
        "bid_prices": fsl(bid_p), "bid_qtys": fsl(bid_q),
        "ask_prices": fsl(ask_p), "ask_qtys": fsl(ask_q),
    })
    os.makedirs(work, exist_ok=True)
    dpath = os.path.join(work, "depth.parquet")
    pq.write_table(depth, dpath)
    # Trades: inject a SELL qty=3.0 at ts=3 and a BUY qty=2.0 at ts=2.
    trades = pa.table({
        "timestamp": pa.array([2, 3], pa.int64()),
        "price": pa.array([100.0, 100.0], pa.float64()),
        "quantity": pa.array([2.0, 3.0], pa.float64()),
        "is_buyer_maker": pa.array([False, True], pa.bool_()),  # ts2=buy, ts3=sell
    })
    tpath = os.path.join(work, "trades.parquet")
    pq.write_table(trades, tpath)

    outdir = os.path.join(work, "bs_out")
    cmd = [os.path.join(bins, "build_samples"), "--depth", dpath, "--trades", tpath,
           "--out-dir", outdir, "--window", "2", "--horizon", "5", "--step", "1",
           "--max-samples", "1000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("  build_samples rc=", r.returncode, r.stderr.strip()[-300:], flush=True)
    chk("build_samples rc==0", r.returncode == 0)
    if r.returncode != 0:
        return
    flow = np.load(os.path.join(outdir, "flow_paths.npy"))
    eq = np.load(os.path.join(outdir, "entry_q.npy"))
    ns = flow.shape[0]
    chk("flow_paths shape (ns,5,2)", flow.shape == (ns, 5, 2), str(flow.shape))
    chk("entry_q shape (ns,2)", eq.shape == (ns, 2), str(eq.shape))
    chk("entry_q == [5,7] top-1", np.allclose(eq, [5.0, 7.0]), str(eq[0].tolist()))
    # sample 0: start=0, end=1, forward rows = end+1+k = 2,3,4,5,6.
    #   k=0 -> row2: buy_vol 2.0 ; k=1 -> row3: sell_vol 3.0
    chk("flow s0 k0 buy_vol==2.0", abs(flow[0, 0, 0] - 2.0) < 1e-6, str(flow[0, 0, 0]))
    chk("flow s0 k1 sell_vol==3.0", abs(flow[0, 1, 1] - 3.0) < 1e-6, str(flow[0, 1, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--work", default="/tmp/e2e")
    a = ap.parse_args()
    print(f"HUSDC e2e test bins={a.bins}", flush=True)
    test_grid_sim_maker(a.bins, a.work)
    test_build_samples(a.bins, a.work)
    print(f"\nE2E RESULT: {len(OKS)} OK, {len(FAILS)} FAIL", flush=True)
    if FAILS:
        print("FAILED:", FAILS, flush=True)
    print("E2E_DONE", "PASS" if not FAILS else "FAIL", flush=True)


if __name__ == "__main__":
    main()
