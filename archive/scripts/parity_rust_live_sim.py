#!/usr/bin/env python3
"""Parity harness for Rust `sim_labels` vs Python `live_sim.simulate_trade`.

Generates N synthetic samples with realistic BTC mid paths + tp/sl/timeout,
runs Python reference for LONG & SHORT, runs Rust batch, and diffs outputs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_sim import LiveSimConfig, SimDirection, label_from_outcomes, simulate_trade, TradeOutcome  # noqa: E402


REASONS = TradeOutcome.REASONS


def synth_samples(n: int, horizon: int = 1300, seed: int = 42):
    rng = np.random.default_rng(seed)
    entry = 67000.0 + rng.normal(0, 50, size=n)
    # Random-walk mid paths; stddev per step tuned so we sometimes hit TP/SL at 0.2%.
    steps = rng.normal(0, entry[:, None] * 2e-5, size=(n, horizon))
    mid = entry[:, None] + np.cumsum(steps, axis=1)
    entry_long = entry - 0.05
    entry_short = entry + 0.05
    tp = np.full(n, 0.20)
    sl = np.full(n, 0.20)
    timeout = np.full(n, 600, dtype=np.int64)
    return entry_long, entry_short, mid, tp, sl, timeout


def python_ref(entry_long, entry_short, mid, tp, sl, timeout,
               partial_enabled=True, trailing_enabled=True,
               comm_win=0.04, comm_loss=0.07, fill_ms=150.0):
    n = len(entry_long)
    y = np.zeros(n, dtype=np.uint8)
    target = np.zeros(n, dtype=np.float64)
    rl = np.zeros(n, dtype=np.uint8)
    rs = np.zeros(n, dtype=np.uint8)
    pl = np.zeros(n, dtype=np.float64)
    ps = np.zeros(n, dtype=np.float64)
    for i in range(n):
        cfg = LiveSimConfig(
            tp_pct=float(tp[i]),
            sl_pct=float(sl[i]),
            timeout_ticks=int(timeout[i]),
            commission_win_pct=comm_win,
            commission_loss_pct=comm_loss,
            partial_enabled=partial_enabled,
            trailing_enabled=trailing_enabled,
        )
        lo = simulate_trade(SimDirection.LONG, float(entry_long[i]), mid[i], cfg, fill_ms)
        so = simulate_trade(SimDirection.SHORT, float(entry_short[i]), mid[i], cfg, fill_ms)
        lbl, tgt = label_from_outcomes(lo, so)
        y[i] = lbl
        target[i] = tgt
        rl[i] = REASONS.index(lo.exit_reason)
        rs[i] = REASONS.index(so.exit_reason)
        pl[i] = lo.net_pnl_pct
        ps[i] = so.net_pnl_pct
    return y, target, rl, rs, pl, ps


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--rust-bin", default=str(Path(__file__).resolve().parents[1]
                                             / "rust_ingest/target/release/sim_labels"))
    args = p.parse_args()

    entry_long, entry_short, mid, tp, sl, timeout = synth_samples(args.n)

    print(f"[parity] computing Python reference for {args.n} samples...")
    y_py, tgt_py, rl_py, rs_py, pl_py, ps_py = python_ref(
        entry_long, entry_short, mid, tp, sl, timeout)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.save(td / "el.npy", entry_long)
        np.save(td / "es.npy", entry_short)
        np.save(td / "mid.npy", mid)
        np.save(td / "tp.npy", tp)
        np.save(td / "sl.npy", sl)
        np.save(td / "to.npy", timeout)
        print("[parity] invoking Rust sim_labels...")
        subprocess.run([
            args.rust_bin,
            "--entry-long", str(td / "el.npy"),
            "--entry-short", str(td / "es.npy"),
            "--mid-paths", str(td / "mid.npy"),
            "--tp-pct", str(td / "tp.npy"),
            "--sl-pct", str(td / "sl.npy"),
            "--timeout-ticks", str(td / "to.npy"),
            "--out-prefix", str(td / "out"),
        ], check=True)
        y_rs = np.load(td / "out_y.npy")
        tgt_rs = np.load(td / "out_target_pnl.npy")
        rl_rs = np.load(td / "out_reason_long.npy")
        rs_rs = np.load(td / "out_reason_short.npy")
        pl_rs = np.load(td / "out_pnl_long.npy")
        ps_rs = np.load(td / "out_pnl_short.npy")

    ok = True
    def _check(name, a, b, atol):
        nonlocal ok
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        mx = float(diff.max()) if len(a) else 0.0
        mean = float(diff.mean()) if len(a) else 0.0
        tag = "OK" if mx <= atol else "FAIL"
        if mx > atol:
            ok = False
        print(f"  {name:18s}  max={mx:.3e}  mean={mean:.3e}  atol={atol:.0e}  {tag}")

    _check("y (label)", y_py, y_rs, 0.0)
    _check("target_pnl", tgt_py, tgt_rs, 1e-9)
    _check("reason_long", rl_py, rl_rs, 0.0)
    _check("reason_short", rs_py, rs_rs, 0.0)
    _check("pnl_long", pl_py, pl_rs, 1e-9)
    _check("pnl_short", ps_py, ps_rs, 1e-9)

    if ok:
        print("[parity] PASS — Rust live_sim matches Python")
        return 0
    print("[parity] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
