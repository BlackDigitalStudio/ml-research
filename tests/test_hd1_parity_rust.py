#!/usr/bin/env python3
"""HD1 rev25 Rust↔Python parity gate.

The heavy data path is rust_ingest/src/bin/hd1_seq_build (Python only
orchestrates). This asserts that binary is BIT-EXACT to the frozen
Python contract for the epistemically-binding parts (decision points,
first-passage label, t0, rH) and fp-equivalent for the §3 feature
windows. Divergence == bug, not DOF. The Modal driver runs this before
any sweep; the sweep must not run unless it passes.

Runs WITHOUT GCS/Modal: synthetic flat-schema cryptolake book parquet.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.ha5_screen import _first_passage as FROZEN_FP  # noqa: E402
from scripts import hd1_seq_core as C  # noqa: E402

BIN = REPO / "rust_ingest" / "target" / "release" / "hd1_seq_build"
RNG = np.random.default_rng(20260518)


def _make_book(n=9000):
    """Synthetic 20-level flat cryptolake book + strictly-sorted ns ts
    (with duplicate stamps to exercise searchsorted-left)."""
    mid = np.maximum(100.0 + RNG.normal(0, 0.03, n).cumsum(), 1.0)
    half = 0.01 + RNG.random(n) * 0.02
    cols = {"timestamp": None}
    for k in range(C.N_LEVELS):
        cols[f"bid_{k}_price"] = (mid - half * (k + 1)).astype(np.float64)
        cols[f"ask_{k}_price"] = (mid + half * (k + 1)).astype(np.float64)
        cols[f"bid_{k}_size"] = (RNG.random(n) * 10 + 0.1).astype(np.float64)
        cols[f"ask_{k}_size"] = (RNG.random(n) * 10 + 0.1).astype(np.float64)
    dt = RNG.integers(1, 200_000_000, n).astype(np.int64)
    dt[RNG.integers(0, n, 30)] = 0
    cols["timestamp"] = np.cumsum(dt).astype(np.int64)
    return cols["timestamp"], cols


def _stack(cols):
    bp = np.column_stack([cols[f"bid_{k}_price"] for k in range(C.N_LEVELS)])
    bs = np.column_stack([cols[f"bid_{k}_size"] for k in range(C.N_LEVELS)])
    ap = np.column_stack([cols[f"ask_{k}_price"] for k in range(C.N_LEVELS)])
    az = np.column_stack([cols[f"ask_{k}_size"] for k in range(C.N_LEVELS)])
    return bp, bs, ap, az


def test_rust_matches_frozen_python():
    assert BIN.exists(), (f"{BIN} not built — run `cd rust_ingest && "
                          f"cargo build --release --bin hd1_seq_build`")
    n, L = 9000, 64
    ts, cols = _make_book(n)
    base = np.sort(RNG.choice(np.arange(1, n - 1), 2600, replace=False))
    idx = np.concatenate([[0, -5, n - 1], base, [0, n - 1]]).astype(np.int64)

    with tempfile.TemporaryDirectory() as td:
        pqf, idxf, out = (os.path.join(td, x) for x in
                          ("b.parquet", "idx.npy", "out"))
        pq.write_table(pa.table({k: pa.array(v) for k, v in cols.items()}),
                       pqf)
        np.save(idxf, idx)
        r = subprocess.run([str(BIN), "--book", pqf, "--indices", idxf,
                            "--out-dir", out, "--max-l", str(L)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"rust failed: {r.stderr}\n{r.stdout}"
        rX = np.load(f"{out}/X.npy")
        ri = np.load(f"{out}/i.npy")
        rt0 = np.load(f"{out}/t0.npy")
        ry0 = {H: np.load(f"{out}/y0_{H}.npy") for H in C.HS}
        rrh = {H: np.load(f"{out}/rH_{H}.npy") for H in C.HS}

    bp, bs, ap, az = _stack(cols)
    mid = 0.5 * (bp[:, 0] + ap[:, 0])
    sel, i_ref = C.select_decision_idx(idx, n)
    t0_ref = ts[i_ref].astype(np.int64)

    # decision points / t0 — BIT-EXACT
    assert np.array_equal(ri, i_ref), "decision indices differ"
    assert np.array_equal(rt0, t0_ref), "t0 differs"

    # §3 feature windows — both compute f64 -> f32 (HD1 rev26: f32
    # storage, no f16 pack); expect near bit-exact.
    tf = C.tick_features(bp, bs, ap, az)
    win, _ = C.gather_windows(tf, i_ref, L=L)
    assert win.dtype == np.float32, win.dtype
    assert rX.shape == win.shape, (rX.shape, win.shape)
    np.testing.assert_allclose(rX, win, rtol=1e-5, atol=1e-5)

    # first-passage label + rH — BIT-EXACT vs the ACTUAL frozen function
    for H in C.HS:
        y0c, rHc, _, _ = C.labels_for_H(ts, mid, i_ref, t0_ref, H)
        jH = np.minimum(np.searchsorted(ts, t0_ref + H * C.NS, "left"),
                        len(mid) - 1)
        y0_frozen = FROZEN_FP(mid, i_ref, jH, mid[i_ref], C.F_T0)
        assert np.array_equal(ry0[H].astype(np.int8),
                              y0_frozen.astype(np.int8)), \
            f"y0 H={H}: Rust != FROZEN ha5._first_passage"
        assert np.array_equal(ry0[H].astype(np.int8),
                              y0c.astype(np.int8)), \
            f"y0 H={H}: Rust != hd1_seq_core"
        fin = np.isfinite(rHc)
        np.testing.assert_array_equal(np.isfinite(rrh[H]), fin)
        np.testing.assert_allclose(rrh[H][fin], rHc[fin].astype(np.float32),
                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    import traceback
    try:
        test_rust_matches_frozen_python()
        print("PASS test_rust_matches_frozen_python")
        print("1/1 passed — Rust heavy path == frozen Python contract")
    except Exception:
        traceback.print_exc()
        print("FAILED")
        raise SystemExit(1)
