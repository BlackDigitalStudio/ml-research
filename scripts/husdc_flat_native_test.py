#!/usr/bin/env python3
"""Validate NATIVE flat-schema reading in build_samples == the proven shim path.

For one real SOL L2 day: run build_samples TWICE on identical underlying data:
  NATIVE: feed the raw FLAT parquet directly (new code path).
  SHIM  : convert flat->nested (FixedSizeList) then feed (proven path).
Assert the 6 output arrays are element-wise identical. If equal, native flat
reading is correct (it reproduces the trusted nested path bit-for-bit).
"""
import argparse, glob, os, subprocess, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LV = 20
RAW = "gs://market-data-0998ac51/raw"
OK, FAIL = [], []
def chk(n, c, d=""):
    (OK if c else FAIL).append(n); print(f"  [{'OK' if c else 'FAIL'}] {n} {d}", flush=True)


def gather_flat(dt_dir, out_flat):
    """concat all flat parquet(s) in dt_dir into one flat parquet (native input)."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(dt_dir, "*.parquet")))
    if not files:
        return 0
    t = pq.read_table(files)
    pq.write_table(t, out_flat)
    return t.num_rows


def to_nested_book(flat_path, out_path):
    import pyarrow as pa, pyarrow.parquet as pq
    t = pq.read_table(flat_path)
    ts_ms = (t.column("timestamp").to_numpy() // 1_000_000).astype(np.int64)
    def fsl(pre, suf):
        cols = [t.column(f"{pre}_{k}_{suf}").to_numpy().astype(np.float64) for k in range(LV)]
        return pa.FixedSizeListArray.from_arrays(pa.array(np.stack(cols, 1).reshape(-1)), LV)
    pq.write_table(pa.table({
        "timestamp": pa.array(ts_ms, pa.int64()),
        "bid_prices": fsl("bid", "price"), "bid_qtys": fsl("bid", "size"),
        "ask_prices": fsl("ask", "price"), "ask_qtys": fsl("ask", "size"),
    }), out_path)


def to_nested_trades(flat_path, out_path):
    import pyarrow as pa, pyarrow.parquet as pq
    t = pq.read_table(flat_path)
    ts_ms = (t.column("timestamp").to_numpy() // 1_000_000).astype(np.int64)
    side = t.column("side").to_pylist()
    pq.write_table(pa.table({
        "timestamp": pa.array(ts_ms, pa.int64()),
        "price": pa.array(t.column("price").to_numpy().astype(np.float64), pa.float64()),
        "quantity": pa.array(t.column("amount").to_numpy().astype(np.float64), pa.float64()),
        "is_buyer_maker": pa.array(np.array([s == "sell" for s in side]), pa.bool_()),
    }), out_path)


def run_bs(bins, depth, trades, outdir):
    cmd = [os.path.join(bins, "build_samples"), "--depth", depth, "--trades", trades,
           "--out-dir", outdir, "--window", "20", "--horizon", "120", "--step", "2",
           "--max-samples", "30000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--symbol", default="SOL-USDT-PERP")
    ap.add_argument("--day", default="2026-05-07")
    ap.add_argument("--work", default="/tmp/flatv")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    print(f"native-flat test {a.symbol} {a.day}", flush=True)
    subprocess.run(["gsutil", "-m", "-q", "cp", "-r",
                    f"{RAW}/book/exchange=BINANCE_FUTURES/symbol={a.symbol}/dt={a.day}/",
                    os.path.join(a.work, "book_flat_dir")], check=False)
    subprocess.run(["gsutil", "-m", "-q", "cp", "-r",
                    f"{RAW}/trades/exchange=BINANCE_FUTURES/symbol={a.symbol}/dt={a.day}/",
                    os.path.join(a.work, "trades_flat_dir")], check=False)
    bdir = glob.glob(os.path.join(a.work, "book_flat_dir", "**", "*.parquet"), recursive=True)
    tdir = glob.glob(os.path.join(a.work, "trades_flat_dir", "**", "*.parquet"), recursive=True)
    chk("book parquet downloaded", len(bdir) > 0, f"{len(bdir)} files")
    chk("trades parquet downloaded", len(tdir) > 0, f"{len(tdir)} files")
    if not bdir or not tdir:
        return
    bflat = os.path.join(a.work, "book_flat.parquet")
    tflat = os.path.join(a.work, "trades_flat.parquet")
    nb = gather_flat(os.path.dirname(bdir[0]), bflat)
    gather_flat(os.path.dirname(tdir[0]), tflat)
    print(f"  flat book rows={nb:,}", flush=True)

    # NATIVE: build_samples on flat parquet directly (new code path)
    rn = run_bs(a.bins, bflat, tflat, os.path.join(a.work, "native"))
    chk("build_samples NATIVE flat rc==0", rn.returncode == 0, rn.stderr.strip()[-250:])

    # SHIM: convert flat->nested then build_samples (proven path)
    bnest = os.path.join(a.work, "book_nested.parquet")
    tnest = os.path.join(a.work, "trades_nested.parquet")
    to_nested_book(bflat, bnest); to_nested_trades(tflat, tnest)
    rs = run_bs(a.bins, bnest, tnest, os.path.join(a.work, "shim"))
    chk("build_samples SHIM nested rc==0", rs.returncode == 0, rs.stderr.strip()[-250:])
    if rn.returncode != 0 or rs.returncode != 0:
        return

    arrs = ["entry_long", "entry_short", "mid_paths", "book_paths", "flow_paths", "entry_q", "sample_ts"]
    for nm in arrs:
        na = np.load(os.path.join(a.work, "native", nm + ".npy"))
        sa = np.load(os.path.join(a.work, "shim", nm + ".npy"))
        same_shape = na.shape == sa.shape
        eq = same_shape and np.allclose(na, sa, rtol=0, atol=0, equal_nan=True)
        chk(f"NATIVE==SHIM {nm}", eq, f"native{na.shape} shim{sa.shape}"
            + ("" if eq else f" maxdiff={np.nanmax(np.abs(na.astype(float)-sa.astype(float))) if same_shape else 'shape'}"))
    # also sanity: native outputs non-degenerate
    fl = np.load(os.path.join(a.work, "native", "flow_paths.npy"))
    eq2 = np.load(os.path.join(a.work, "native", "entry_q.npy"))
    chk("native flow_paths non-degenerate", float((fl > 0).mean()) > 0.05, f"nz={float((fl>0).mean()):.3f}")
    chk("native entry_q positive", float((eq2 > 0).mean()) > 0.5, f"pos={float((eq2>0).mean()):.3f}")

    print(f"\nFLATV RESULT: {len(OK)} OK, {len(FAIL)} FAIL", flush=True)
    if FAIL:
        print("FAILED:", FAIL, flush=True)
    print("FLATV_DONE", "PASS" if not FAIL else "FAIL", flush=True)


if __name__ == "__main__":
    main()
