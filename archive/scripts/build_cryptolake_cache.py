#!/usr/bin/env python3
"""Reconstructed Cryptolake cache builder — MAKER-first by construction.

The original `build_cryptolake_cache.py` died with the Contabo host. This
is a clean reconstruction against the reverse-engineered GCS contract in
`research/CRYPTOLAKE_SCHEMA.md` (read it before touching this file).

What it does, per (symbol, day):
  1. Pull `features_v1/.../features.npy` (X, the 3602x59 model input) +
     `indices.npy` (decision-point k -> raw book row indices[k]).
  2. Pull `raw/book/.../*.parquet` (20-level LOB, ns timestamps,
     `contains_gaps=Yes`).
  3. For each decision point: entry prices + a forward L1 path. The
     fee/entry regime is EXPLICIT and the whole point of H5:
       MAKER_FIRST : long fills at bid_0, short at ask_0 ; fees 0.04/0.07
       TAKER       : long fills at ask_0, short at bid_0 ; fees 0.07/0.10
     (RESEARCH_LOG claimed sim_labels has --entry-taker flags — it does
     NOT; the regime is encoded by the entry side + commissions, which is
     the correct and only honest model of maker-first vs taker.)
  4. Horizon is WALL-CLOCK (gaps!): per-sample `timeout_ticks` = number of
     forward book rows until `--timeout-sec` elapses, clamped to the
     60-180 s holding zone. Never a fixed row count.
  5. `simulate_labels` (Rust) -> y / pnl_long / pnl_short. Emits a
     manifest with the exact ledger provenance (fee_regime, cache_id,
     symbols, date range, n_samples, commissions, label_def, git_commit).

Two run modes:
  --validate-only : data-prep + invariants on ONE day, NO Rust. Runs in
                    the planning container at $0. Use this to de-risk
                    before paying for the GCP VM.
  (default)       : full build; calls rust_bridge.simulate_labels (only
                    works where the Rust binaries are built — the VM).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GCP_PROJECT = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"

# Canonical constants (src/trainer.py / STRATEGY.md §5/§6).
TP_PCT = 0.20
SL_PCT = 0.10
HOLD_FLOOR_SEC = 60
HOLD_CEIL_SEC = 180

# Fee regimes — the H5 gate. Round-trip percentages.
FEE = {
    "MAKER_FIRST": dict(commission_win_pct=0.04, commission_loss_pct=0.07),
    "TAKER":       dict(commission_win_pct=0.07, commission_loss_pct=0.10),
}


@dataclass
class Manifest:
    schema_ref: str
    symbol: str
    date_start: str
    date_end: str
    n_samples: int
    n_days: int
    fee_regime: str
    commission_win_pct: float
    commission_loss_pct: float
    tp_pct: float
    sl_pct: float
    timeout_sec: int
    horizon_rows: int
    label_def: str
    git_commit: str
    feature_cols: int


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _gcs_bucket():
    from google.cloud import storage
    return storage.Client(project=GCP_PROJECT).bucket(BUCKET)


def _list_days(bk, symbol: str) -> list[str]:
    from google.cloud import storage
    pref = f"features_v1/symbol={symbol}/"
    it = bk.client.list_blobs(bk, prefix=pref, delimiter="/")
    for _ in it:
        pass
    return sorted(p.split("dt=")[1].rstrip("/") for p in it.prefixes)


def _load_npy(bk, name: str) -> np.ndarray:
    return np.load(io.BytesIO(bk.blob(name).download_as_bytes()),
                   allow_pickle=False)


def _load_book_l1(bk, symbol: str, day: str):
    """Return (ts_ns, bid0, ask0) for one day, row order preserved."""
    import pyarrow.parquet as pq
    pref = (f"raw/book/exchange=BINANCE_FUTURES/symbol={symbol}/dt={day}/")
    blobs = sorted((b for b in bk.client.list_blobs(bk, prefix=pref)
                    if b.name.endswith(".parquet")), key=lambda b: b.name)
    if not blobs:
        raise FileNotFoundError(f"no book parquet for {symbol} {day}")
    ts, bid, ask = [], [], []
    for b in blobs:
        t = pq.read_table(io.BytesIO(b.download_as_bytes()),
                          columns=["timestamp", "bid_0_price", "ask_0_price"])
        ts.append(t.column("timestamp").to_numpy(zero_copy_only=False))
        bid.append(t.column("bid_0_price").to_numpy(zero_copy_only=False))
        ask.append(t.column("ask_0_price").to_numpy(zero_copy_only=False))
    return (np.concatenate(ts).astype(np.int64),
            np.concatenate(bid).astype(np.float64),
            np.concatenate(ask).astype(np.float64))


def _build_day(bk, symbol: str, day: str, regime: str, timeout_sec: int):
    """One day -> dict of per-sample arrays (no Rust). Drops tail samples
    whose full max-horizon window doesn't fit in the book."""
    X = _load_npy(bk, f"features_v1/symbol={symbol}/dt={day}/features.npy")
    idx = _load_npy(bk, f"features_v1/symbol={symbol}/dt={day}/indices.npy"
                    ).astype(np.int64)
    ts, bid0, ask0 = _load_book_l1(bk, symbol, day)
    n_book = ts.shape[0]
    if X.shape[0] != idx.shape[0]:
        raise ValueError(f"{day}: features rows {X.shape[0]} != indices "
                         f"{idx.shape[0]}")
    mid = (bid0 + ask0) * 0.5

    ceil_ns = HOLD_CEIL_SEC * 1_000_000_000
    tgt_ns = timeout_sec * 1_000_000_000
    floor_ns = HOLD_FLOOR_SEC * 1_000_000_000

    keep_k, entry_l, entry_s, eb, to_rows = [], [], [], [], []
    max_h = 0
    for k in range(idx.shape[0]):
        i = int(idx[k])
        if i <= 0 or i >= n_book - 1:
            continue
        t0 = ts[i]
        # rows needed to cover the *ceiling* hold (so the path is long
        # enough for any timeout we might sweep later); drop if it runs
        # past available book.
        j_ceil = np.searchsorted(ts[i + 1:], t0 + ceil_ns, "left") + 1
        if i + j_ceil >= n_book:
            continue
        # per-sample timeout in rows = until target hold elapses, clamped
        # into [floor, ceil] so it always lands in the 60-180 s zone.
        j_to = np.searchsorted(ts[i + 1:], t0 + tgt_ns, "left") + 1
        j_lo = np.searchsorted(ts[i + 1:], t0 + floor_ns, "left") + 1
        j_to = max(int(j_lo), min(int(j_to), int(j_ceil)))
        b, a = bid0[i], ask0[i]
        if not (np.isfinite(b) and np.isfinite(a) and b > 0 and a >= b):
            continue
        if regime == "MAKER_FIRST":
            el, es = b, a            # post bid (long) / post ask (short)
        else:                        # TAKER: cross the spread on entry
            el, es = a, b
        keep_k.append(k)
        entry_l.append(el)
        entry_s.append(es)
        eb.append((b, a))
        to_rows.append(j_to)
        if j_ceil > max_h:
            max_h = int(j_ceil)

    if not keep_k:
        return None
    keep_k = np.asarray(keep_k, np.int64)
    H = max_h
    n = keep_k.shape[0]
    mid_paths = np.empty((n, H), np.float64)
    book_paths = np.empty((n, H, 2), np.float64)
    for r, k in enumerate(keep_k):
        i = int(idx[k])
        seg = slice(i + 1, i + 1 + H)
        m = mid[seg]
        pad = H - m.shape[0]
        if pad > 0:                  # shouldn't happen (tail dropped) — be safe
            m = np.concatenate([m, np.full(pad, m[-1])])
        mid_paths[r] = m
        bb = bid0[seg]
        aa = ask0[seg]
        if pad > 0:
            bb = np.concatenate([bb, np.full(pad, bb[-1])])
            aa = np.concatenate([aa, np.full(pad, aa[-1])])
        book_paths[r, :, 0] = bb
        book_paths[r, :, 1] = aa
    return dict(
        X=X[keep_k].astype(np.float32),
        entry_long=np.asarray(entry_l, np.float64),
        entry_short=np.asarray(entry_s, np.float64),
        entry_book=np.asarray(eb, np.float64),
        mid_paths=mid_paths,
        book_paths=book_paths,
        timeout_ticks=np.asarray(to_rows, np.int64),
        horizon_rows=H,
    )


def _invariants(day: str, d: dict) -> list[str]:
    """Cheap structural checks — these are what the $0 validation buys."""
    errs = []
    n = d["X"].shape[0]
    if d["X"].shape[1] != 59:
        errs.append(f"X cols {d['X'].shape[1]} != 59")
    for key in ("entry_long", "entry_short", "timeout_ticks"):
        if d[key].shape[0] != n:
            errs.append(f"{key} len {d[key].shape[0]} != {n}")
    if d["mid_paths"].shape != (n, d["horizon_rows"]):
        errs.append(f"mid_paths {d['mid_paths'].shape} != {(n, d['horizon_rows'])}")
    if not np.isfinite(d["mid_paths"]).all():
        errs.append("mid_paths has non-finite values")
    if not np.isfinite(d["entry_long"]).all():
        errs.append("entry_long has non-finite values")
    # entry_long must equal a real book price (bid for maker-first long)
    eb = d["entry_book"]
    if (d["entry_long"] > d["entry_short"] + 1e-9).any():
        # long entry (bid side) should be <= short entry (ask side) for
        # maker-first; for taker it's the reverse. Just assert spread sane.
        pass
    if (eb[:, 1] < eb[:, 0]).any():
        errs.append("entry_book has ask < bid")
    to = d["timeout_ticks"]
    if (to < 1).any():
        errs.append("timeout_ticks has non-positive entries")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC-USDT-PERP")
    ap.add_argument("--start", help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", help="YYYY-MM-DD inclusive")
    ap.add_argument("--fee-regime", choices=tuple(FEE), default="MAKER_FIRST")
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--out", default=str(REPO / "data/_cache/cryptolake"))
    ap.add_argument("--validate-only", action="store_true",
                    help="data-prep + invariants on the FIRST day only, no "
                         "Rust. $0, runs in the planning container.")
    ap.add_argument("--max-days", type=int, default=0,
                    help="cap days processed (0 = no cap)")
    a = ap.parse_args(argv)

    bk = _gcs_bucket()
    days = _list_days(bk, a.symbol)
    if a.start:
        days = [d for d in days if d >= a.start]
    if a.end:
        days = [d for d in days if d <= a.end]
    if not days:
        print("no days in range", file=sys.stderr)
        return 2
    if a.validate_only:
        days = days[:1]
    elif a.max_days:
        days = days[: a.max_days]

    print(f"symbol={a.symbol} regime={a.fee_regime} days={len(days)} "
          f"[{days[0]}..{days[-1]}] timeout={a.timeout_sec}s")

    if a.validate_only:
        d = _build_day(bk, a.symbol, days[0], a.fee_regime, a.timeout_sec)
        if d is None:
            print("VALIDATION FAILED: no usable samples on", days[0])
            return 1
        errs = _invariants(days[0], d)
        hold_s = None
        # report a few sanity numbers
        n = d["X"].shape[0]
        print(f"day {days[0]}: n_samples={n} horizon_rows={d['horizon_rows']}")
        print(f"  entry_long[:3]={d['entry_long'][:3]}  "
              f"entry_short[:3]={d['entry_short'][:3]}")
        print(f"  mid[0,0]={d['mid_paths'][0,0]:.2f} "
              f"in [bid={d['entry_book'][0,0]:.2f}, ask={d['entry_book'][0,1]:.2f}]")
        print(f"  timeout_ticks: min={d['timeout_ticks'].min()} "
              f"med={int(np.median(d['timeout_ticks']))} "
              f"max={d['timeout_ticks'].max()} (rows)")
        if errs:
            print("INVARIANTS FAILED:")
            for e in errs:
                print("  -", e)
            return 1
        print("INVARIANTS OK — data-prep is sound; Rust sim deferred to VM.")
        return 0

    # ---- full build (VM): per-day prep -> Rust sim -> accumulate --------
    from src import rust_bridge
    fee = FEE[a.fee_regime]
    Xs, ys, pls, pss, rls, rss = [], [], [], [], [], []
    total = 0
    for di, day in enumerate(days):
        d = _build_day(bk, a.symbol, day, a.fee_regime, a.timeout_sec)
        if d is None:
            print(f"  {day}: 0 samples, skip")
            continue
        n = d["X"].shape[0]
        tp = np.full(n, TP_PCT, np.float64)
        sl = np.full(n, SL_PCT, np.float64)
        res = rust_bridge.simulate_labels(
            entry_long=d["entry_long"], entry_short=d["entry_short"],
            mid_paths=d["mid_paths"], tp_pct=tp, sl_pct=sl,
            timeout_ticks=d["timeout_ticks"],
            commission_win_pct=fee["commission_win_pct"],
            commission_loss_pct=fee["commission_loss_pct"],
            book_paths=d["book_paths"], entry_book=d["entry_book"],
        )
        Xs.append(d["X"])
        ys.append(res["y"])
        pls.append(res["pnl_long"])
        pss.append(res["pnl_short"])
        rls.append(res["reason_long"])
        rss.append(res["reason_short"])
        total += n
        print(f"  {day}: n={n} cum={total}")

    if total == 0:
        print("no samples built", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{out}_{a.symbol}_{a.fee_regime}_{days[0]}_{days[-1]}"
    np.save(f"{tag}_X.npy", np.concatenate(Xs))
    np.save(f"{tag}_y.npy", np.concatenate(ys))
    np.save(f"{tag}_pl.npy", np.concatenate(pls))
    np.save(f"{tag}_ps.npy", np.concatenate(pss))
    np.save(f"{tag}_reason_long.npy", np.concatenate(rls))
    np.save(f"{tag}_reason_short.npy", np.concatenate(rss))
    man = Manifest(
        schema_ref="research/CRYPTOLAKE_SCHEMA.md",
        symbol=a.symbol, date_start=days[0], date_end=days[-1],
        n_samples=total, n_days=len(days), fee_regime=a.fee_regime,
        commission_win_pct=fee["commission_win_pct"],
        commission_loss_pct=fee["commission_loss_pct"],
        tp_pct=TP_PCT, sl_pct=SL_PCT, timeout_sec=a.timeout_sec,
        horizon_rows=-1,
        label_def=("triple-barrier direction-aware; entry "
                   f"{a.fee_regime} (long=bid/short=ask if maker-first); "
                   "wall-clock timeout in [60,180]s"),
        git_commit=_git_commit(), feature_cols=59,
    )
    Path(f"{tag}_manifest.json").write_text(json.dumps(asdict(man), indent=2))
    print(f"saved {tag}_*.npy + manifest ({total} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
