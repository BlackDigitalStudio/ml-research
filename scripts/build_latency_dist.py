#!/usr/bin/env python3
"""Generate a per-sample fill-latency array for the book-aware simulator.

Models Binance Futures WS→REST round-trip: lognormal distribution with
median ~110 ms and a long tail. Parameters sourced from public benchmarks
and our own host-to-Tokyo/Singapore pings (Contabo → Binance AWS regions):

    p50 ≈  110 ms
    p95 ≈  350 ms
    p99 ≈  800 ms
    max ~ 2500 ms (rare)

Shape: lognormal(mu, sigma) in log-ms space where mu = ln(110), sigma = 0.55.
That gives median 110, p95 ≈ 330, p99 ≈ 520 — realistic without over-modelling
the extreme tail (which would distort grid results via rare fast-fill
adverses).

When Binance empirical timing-stats become available (future work: parse
recorder health-file deltas), replace this synthetic dist with the real
dataset by passing --empirical <path>.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sample_latency(n: int, seed: int = 42,
                   median_ms: float = 110.0, sigma: float = 0.55,
                   floor_ms: float = 20.0, cap_ms: float = 3000.0) -> np.ndarray:
    """Draw n latency samples (ms) from lognormal with given parameters.

    `floor` and `cap` clip the pathological tail — below 20 ms is network-
    impossible from any region to Binance; above 3 s is almost certainly a
    disconnect that the bot handles separately.
    """
    rng = np.random.default_rng(seed)
    mu = np.log(median_ms)
    raw = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return np.clip(raw, floor_ms, cap_ms).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True,
                    help="Number of samples (match cache N)")
    ap.add_argument("--out", required=True,
                    help="Output .npy path (float64, shape (N,))")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--median-ms", type=float, default=110.0)
    ap.add_argument("--sigma", type=float, default=0.55)
    args = ap.parse_args()

    arr = sample_latency(args.n, seed=args.seed,
                          median_ms=args.median_ms, sigma=args.sigma)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)

    q = np.quantile(arr, [0.05, 0.50, 0.90, 0.95, 0.99])
    print(f"[latency] wrote {args.out}  n={len(arr)}  "
          f"p05={q[0]:.0f} p50={q[1]:.0f} p90={q[2]:.0f} "
          f"p95={q[3]:.0f} p99={q[4]:.0f}  (ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
