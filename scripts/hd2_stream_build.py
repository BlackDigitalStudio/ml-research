#!/usr/bin/env python3
"""HD2 streaming-stateful builder — per-(symbol,day) continuous tick-stream
for the Mamba-2 tier (HD2 rev1 frozen spec).

Output (one .npz per symbol-day), all decision-point arrays aligned to t0:
  stream    (n_ticks, 80) f16   raw 20-level L2, per-channel normalized
                                 [bid_p0..19, bid_s0..19, ask_p0..19, ask_s0..19]
                                 prices (p-mid)/mid ; sizes sign*log1p(|s|)
  t0        (n_dp,)       i64    decision-point BOOK INDICES (== tick positions),
                                 STRIDE=1 features_v1 grid, boundary-filtered
  globals   (n_dp, 6)     f32    frozen hd1_seq_core.tick_features cols [40:46]
                                 at t0 (last-tick): logret, spread, L5, L20, OFI, micro
  y_{H}     (n_dp,)       i8     first-passage label (F_T0=0.0013), per H
  rH_{H}    (n_dp,)       f32    forward log-return at H
  rc_{H}    (n_dp,)       bool   reached (|barrier| hit & finite)
  meta      json          symbol, day, n_ticks, n_dp, Hs

Parity: labels/globals come from the FROZEN hd1_seq_core verbatim (STRIDE
forced to 1). The standardization of `stream` is deferred to the trainer
(fit-region mean/std only). f16 storage is safe: (p-mid)/mid ~1e-3 and
sign*log1p(|s|) <~ 15 are both well inside f16 range (unlike the raw Cont-OFI
in the 46-ch transform, which is why globals stay f32).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hd1_seq_core as core  # noqa: E402

core.STRIDE = 1  # HD2: full features_v1 24s grid, no decimation
N_LEVELS = 20
HS_DEFAULT = (180, 600, 1800)


def _read_book(parquet_path: str):
    """-> ts(n,) i64 ns, bid_p/bid_s/ask_p/ask_s (n,20) f64, mid(n,) f64."""
    cols = ["timestamp"]
    for side in ("bid", "ask"):
        for k in range(N_LEVELS):
            cols += [f"{side}_{k}_price", f"{side}_{k}_size"]
    t = pq.read_table(parquet_path, columns=cols)
    ts = t["timestamp"].to_numpy().astype(np.int64)
    n = len(ts)
    bid_p = np.empty((n, N_LEVELS)); bid_s = np.empty((n, N_LEVELS))
    ask_p = np.empty((n, N_LEVELS)); ask_s = np.empty((n, N_LEVELS))
    for k in range(N_LEVELS):
        bid_p[:, k] = t[f"bid_{k}_price"].to_numpy()
        bid_s[:, k] = t[f"bid_{k}_size"].to_numpy()
        ask_p[:, k] = t[f"ask_{k}_price"].to_numpy()
        ask_s[:, k] = t[f"ask_{k}_size"].to_numpy()
    mid = 0.5 * (bid_p[:, 0] + ask_p[:, 0])
    return ts, bid_p, bid_s, ask_p, ask_s, mid


def _lob_stream_80(bid_p, bid_s, ask_p, ask_s, mid):
    """Raw 20-level L2 -> (n, 80) f16. Order [bid_p|bid_s|ask_p|ask_s]."""
    safe_mid = np.where(mid > 0, mid, 1.0)[:, None]
    bp = (bid_p - mid[:, None]) / safe_mid
    ap = (ask_p - mid[:, None]) / safe_mid
    bs = np.sign(bid_s) * np.log1p(np.abs(bid_s))
    as_ = np.sign(ask_s) * np.log1p(np.abs(ask_s))
    out = np.concatenate([bp, bs, ap, as_], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)


def build_day(parquet_path: str, indices_path: str, out_path: str,
              symbol: str, day: str, Hs=HS_DEFAULT) -> dict:
    ts, bid_p, bid_s, ask_p, ask_s, mid = _read_book(parquet_path)
    n_ticks = len(ts)

    # decision points: features_v1 indices, frozen boundary filter + STRIDE=1
    idx = np.load(indices_path).astype(np.int64)
    sel, i = core.select_decision_idx(idx, n_ticks)   # i = book indices of t0
    t0_ns = ts[i]

    # 80-ch raw LOB stream (per-channel norm; standardization in trainer)
    stream = _lob_stream_80(bid_p, bid_s, ask_p, ask_s, mid)

    # 6 globals from the FROZEN 46-ch transform, cols [40:46], at t0 (last-tick)
    tf = core.tick_features(bid_p, bid_s, ask_p, ask_s)   # (n,46) f32, frozen
    globals6 = tf[i, 40:46].astype(np.float32)

    out = {"stream": stream, "t0": i.astype(np.int64), "globals": globals6}
    label_summary = {}
    for H in Hs:
        y0, rH, reached, _up = core.labels_for_H(ts, mid, i, t0_ns, H)
        out[f"y_{H}"] = y0.astype(np.int8)
        out[f"rH_{H}"] = rH.astype(np.float32)
        out[f"rc_{H}"] = reached
        label_summary[H] = {
            "reached_frac": float(reached.mean()),
            "up_frac_of_reached": float((y0[reached] == 1).mean()) if reached.any() else None,
            "rH_std": float(np.nanstd(rH[reached])) if reached.any() else None,
        }
    meta = {"symbol": symbol, "day": day, "n_ticks": int(n_ticks),
            "n_dp": int(len(i)), "Hs": list(Hs), "stride": core.STRIDE,
            "label_summary": label_summary}
    out["meta"] = json.dumps(meta)
    np.savez(out_path, **out)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol", default="SOL-USDT-PERP")
    ap.add_argument("--day", default="2025-05-13")
    a = ap.parse_args()
    meta = build_day(a.parquet, a.indices, a.out, a.symbol, a.day)
    print(json.dumps(meta, indent=2))
