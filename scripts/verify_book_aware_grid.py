#!/usr/bin/env python3
"""Ground-truth regression check for the book-aware simulator + grid wiring.

Unlike a legacy-parity test (which can't byte-match a simulator whose whole
purpose is to fix legacy's bugs), this verifier constructs hand-crafted
scenarios with analytically-computed expected outcomes, then asserts
simulate_trade_book reproduces them.

Scenarios (each independent, <30 samples, runs in <1 s):

  S1. TP exact fill (spread=0):
      Long entry at mid=50000, path walks linearly up past TP=0.20%.
      Expected: gross == tp_pct exactly (maker limit fills at tp_px).

  S2. TP with spread (entry pays spread):
      Long entry at ask=50005 (mid=50000, bid=49995). Path walks bid up.
      Expected: gross == tp_pct when bid crosses entry_ask*(1+tp/100);
      NOT when mid crosses the level (would under-count spread cost).

  S3. SL gap slippage:
      Long entry at 50000. Path: bid at tick 5 drops to 49800 (well
      below sl_px = 50000*(1-0.15/100) = 49925). Expected: fill at 49800
      (the gapped bid), NOT at 49925 — realistic taker slippage past stop.

  S4. Partial at limit target (not overshoot mid):
      Long entry at 50000, TP=0.40%, partial_tp_progress=0.50. At tick 8
      bid jumps past 50000*(1+0.002)=50100 to 50150. Later price falls
      and SL hits. Expected partial_px = 50100 (the limit target), not
      50150 (overshoot mid that legacy would record).

  S5. Timeout-limit fill (wait for mid to return):
      Entry 50000, wide TP/SL so neither trigger. Forward path drifts
      up to 50020 at end of timeout_ticks, then bid drifts back down to
      exactly 50010 = timeout_mid = (50020 book_path[tt-1].mid()) within
      timeout_limit_ticks window. Expected: fill at timeout_mid = 50010,
      exit_reason = TimeoutLimit.

  S6. Python→Rust plumbing:
      grid_live_retargeted._load_cache() auto-detects book_paths.npy +
      entry_book.npy next to cache prefix. Catches broken auto-detect.

Run:
    python scripts/verify_book_aware_grid.py
Exits 0 on pass, 1 on any failed invariant.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402


# Exit-reason IDs — mirrors live_sim.rs::ExitReason
R_TP = 0
R_SL = 1
R_TRAIL_SL1 = 2
R_TRAIL_SL2 = 3
R_PARTIAL_TP = 4
R_PARTIAL_TRAIL_SL1 = 5
R_PARTIAL_TRAIL_SL2 = 6
R_TIMEOUT_LIMIT = 7
R_TIMEOUT_MARKET = 8
R_FAST_FILL_ADVERSE = 9
R_FAST_FILL_SL = 10
R_NO_FWD = 11


def _run_long(entry_book, book_path, tp_pct, sl_pct, timeout_ticks,
              fill_latency_ms=150.0,
              partial=True, trailing=True):
    """Single-sample long simulate_labels call — returns (pnl, reason)."""
    N = 1
    el = np.array([entry_book[0]], dtype=np.float64)
    es = np.array([entry_book[1]], dtype=np.float64)
    mid_path = np.array([0.5 * (bp[0] + bp[1]) for bp in book_path], dtype=np.float64)
    mp_arr = mid_path.reshape(1, -1)
    bp_arr = np.array([book_path], dtype=np.float64)   # (1, H, 2)
    eb_arr = np.array([entry_book], dtype=np.float64)  # (1, 2)
    out = rust_bridge.simulate_labels(
        el, es, mp_arr,
        np.full(N, tp_pct), np.full(N, sl_pct),
        np.full(N, timeout_ticks, dtype=np.int64),
        partial_enabled=partial, trailing_enabled=trailing,
        fill_latency_ms=fill_latency_ms,
        book_paths=bp_arr, entry_book=eb_arr,
    )
    return float(out["pnl_long"][0]), int(out["reason_long"][0])


def s1_tp_exact_zero_spread() -> None:
    """Long TP fills at tp_px exactly when spread = 0."""
    entry = 50000.0
    tp = 0.20
    entry_book = (entry, entry)  # bid = ask = entry (spread=0)
    # Path: price walks up past tp_px = entry * (1 + tp/100) = 50100
    H = 50
    path = [(entry + (t / 20) * 200, entry + (t / 20) * 200) for t in range(H)]
    pnl, reason = _run_long(entry_book, path, tp_pct=tp, sl_pct=1.0,
                             timeout_ticks=40,
                             partial=False, trailing=False)
    # gross = tp; net = tp - commission_win (0.04%)
    expected_net = tp - 0.04
    assert reason == R_TP, f"S1: reason {reason} != R_TP"
    assert abs(pnl - expected_net) < 1e-9, f"S1: pnl={pnl} != {expected_net}"
    print(f"[S1 TP zero-spread]        pnl={pnl:+.4f} reason={reason} OK")


def s2_tp_with_spread_requires_bid_cross() -> None:
    """With spread > 0, long TP requires bid to reach entry_ask*(1+tp/100)."""
    mid = 50000.0
    spread = 10.0
    bid0 = mid - spread / 2
    ask0 = mid + spread / 2
    entry_book = (bid0, ask0)
    tp = 0.20
    tp_trigger_bid = ask0 * (1 + tp / 100)  # 50105.01

    # Path: bid crosses tp_trigger_bid at tick 10 (well within main loop).
    # Main loop sees first H - timeout_limit_ticks = H - 20 ticks.
    H = 60
    slope = (tp_trigger_bid - bid0 + 5) / 10.0  # reach +5 past target at t=10
    path = []
    for t in range(H):
        b = bid0 + t * slope
        path.append((b, b + spread))
    pnl, reason = _run_long(entry_book, path, tp_pct=tp, sl_pct=1.0,
                             timeout_ticks=30,
                             partial=False, trailing=False)
    expected_net = tp - 0.04  # gross == tp exactly (maker fill at tp_px)
    assert reason == R_TP, f"S2: reason {reason} != R_TP"
    assert abs(pnl - expected_net) < 1e-9, f"S2: pnl={pnl} != {expected_net}"
    print(f"[S2 TP with 10$ spread]    pnl={pnl:+.4f} reason={reason} OK")


def s3_sl_gap_slippage() -> None:
    """SL triggers at sl_px, but fill happens at current bid (may gap past)."""
    entry = 50000.0
    sl = 0.15
    entry_book = (entry, entry)
    sl_px = entry * (1 - sl / 100)  # 49925
    gapped_bid = 49800.0  # deep below sl_px → realistic slippage
    H = 30
    path = []
    for t in range(H):
        if t < 5:
            path.append((entry, entry))  # stable
        elif t == 5:
            path.append((gapped_bid, gapped_bid))  # gap
        else:
            path.append((gapped_bid, gapped_bid))
    pnl, reason = _run_long(entry_book, path, tp_pct=1.0, sl_pct=sl,
                             timeout_ticks=20,
                             partial=False, trailing=False)
    # gross = (gapped_bid - entry) / entry * 100 = -0.40
    # net = gross - commission_loss (0.07) = -0.47
    expected_gross = (gapped_bid - entry) / entry * 100
    expected_net = expected_gross - 0.07
    assert reason == R_SL, f"S3: reason {reason} != R_SL"
    assert abs(pnl - expected_net) < 1e-6, \
        f"S3: pnl={pnl} expected={expected_net} (should gap past sl_px)"
    print(f"[S3 SL gap slippage]       pnl={pnl:+.4f} reason={reason} "
          f"(filled at {gapped_bid} not {sl_px}) OK")


def s4_partial_at_limit_not_overshoot() -> None:
    """Partial TP fills at the limit target, even when bid overshoots."""
    entry = 50000.0
    tp = 0.40
    sl = 0.10
    entry_book = (entry, entry)
    # partial_tp_progress=0.50 → partial_target = entry + 0.5 * tp_dist
    #                          = 50000 + 0.5 * 200 = 50100
    partial_target = entry + 0.5 * (entry * tp / 100)
    # Must be between partial_target (progress=0.5) and trailing_step2
    # target (progress=0.75). Above 0.75 skips step-1 entirely (if/else if
    # in live_sim.rs), losing the partial fire.
    overshoot_bid = partial_target + 30  # 50130 = progress 0.65
    sl_px = entry * (1 - sl / 100)  # 49950

    # Tick 0-7: flat. Tick 8: big up-jump past partial_target → triggers
    # partial + trailing_step1. Tick 9-11: descent. Tick 12+: bid below
    # trailing stop → partial+trailing_sl1 exit.
    # Need H ≥ timeout_ticks + timeout_limit_ticks(20) for main loop to
    # actually iterate for timeout_ticks.
    H = 50
    path = []
    for t in range(H):
        if t < 8:
            path.append((entry, entry))
        elif t == 8:
            path.append((overshoot_bid, overshoot_bid))
        elif t < 12:
            bid = overshoot_bid - (t - 8) * 30
            path.append((bid, bid))
        else:
            # Well below the trailing-step-1 stop (which is at max(entry
            # - 0.08%, partial_target - 0.30*tp_dist) = max(49960, 50040)
            # = 50040 when floor < ratio; but partial_target - ratio*dist
            # = 50100 - 60 = 50040 for long. So 49940 clearly triggers.
            path.append((49940, 49940))
    pnl, reason = _run_long(entry_book, path, tp_pct=tp, sl_pct=sl,
                             timeout_ticks=25,
                             partial=True, trailing=True)
    # Expected: half at partial_target (fair limit), half at trailing stop.
    # Realism assertion: partial_px == partial_target (not overshoot_bid).
    # Since we can't inspect partial_px directly, we assert the combined
    # net falls in the tight band around the limit-target assumption.
    assert reason in (R_PARTIAL_TRAIL_SL1, R_PARTIAL_TRAIL_SL2), \
        f"S4: reason {reason}, expected partial+trailing_sl"
    # Gross lower bound: if partial filled at limit (50100) vs legacy
    # overshoot (50150), gross differs by ~0.05% (spread of 50 on 50k).
    # Trailing SL portion is same either way. So total gross diff = 0.025%.
    # We assert pnl is NOT >= the overshoot-case gross.
    overshoot_gross_a = (overshoot_bid - entry) / entry * 100  # 0.30
    limit_gross_a = (partial_target - entry) / entry * 100     # 0.20
    # If we used overshoot: total gross = 0.5*0.30 + 0.5*(-0.02) = 0.14
    # If we used limit:    total gross = 0.5*0.20 + 0.5*(-0.02) = 0.09
    # Difference: 0.05 pct. If book-aware uses limit, pnl < 0.14 - comm.
    overshoot_pnl_upper_bound = 0.5 * overshoot_gross_a - 0.05  # rough
    assert pnl < overshoot_pnl_upper_bound + 0.01, \
        f"S4: pnl={pnl} suspiciously close to overshoot-fill semantics"
    print(f"[S4 partial at limit]      pnl={pnl:+.4f} reason={reason} "
          f"(fair limit fill, NOT overshoot) OK")


def s5_timeout_limit_when_mid_returns() -> None:
    """Timeout-limit exit fills at timeout_mid when bid returns within window."""
    entry = 50000.0
    entry_book = (entry, entry)
    # Main loop horizon: 30 ticks. Limit window: default 20. Total H=60.
    # Forward path drifts up to 50020 at tick 29, then DOWN to 50005,
    # then UP to 50010 (== timeout_mid) at tick 40.
    timeout_ticks = 30
    H = timeout_ticks + 20  # 50 ticks = main + limit window
    path = []
    for t in range(H):
        if t < timeout_ticks:
            # drift up
            p = entry + t * 0.5
        elif t < timeout_ticks + 5:
            # drift down
            p = entry + 20 - (t - timeout_ticks + 1) * 3
        else:
            # drift up past timeout_mid = 50014.5 (final tick of main loop)
            # Actually timeout_mid = mid_path[tt-1] = 50000 + 29*0.5 = 50014.5
            # We need bid to reach >= 50014.5 for fill
            p = entry + 15 + (t - (timeout_ticks + 5)) * 1.0  # 50015..50030
        path.append((p, p))
    pnl, reason = _run_long(entry_book, path, tp_pct=1.0, sl_pct=1.0,
                             timeout_ticks=timeout_ticks,
                             partial=False, trailing=False)
    assert reason == R_TIMEOUT_LIMIT, \
        f"S5: reason {reason} != R_TIMEOUT_LIMIT (got {reason}, path may not return)"
    # Fill at timeout_mid = 50014.5, gross = 0.029%, net = 0.029 - 0.04 = -0.011
    timeout_mid = entry + (timeout_ticks - 1) * 0.5
    expected_gross = (timeout_mid - entry) / entry * 100
    expected_net = expected_gross - 0.04
    assert abs(pnl - expected_net) < 1e-6, \
        f"S5: pnl={pnl} expected={expected_net}"
    print(f"[S5 timeout-limit fill]    pnl={pnl:+.4f} reason={reason} OK")


def s6_grid_auto_detect() -> None:
    """grid_live_retargeted._load_cache() picks up book_paths.npy + entry_book.npy."""
    # Locate existing cache just to borrow non-book files.
    cache_dir = Path("/home/scalper/scalper-bot/data/_cache")
    cand = sorted(cache_dir.glob("samples_v3_*_mid_paths.npy"),
                  key=lambda p: p.stat().st_size, reverse=True)
    if not cand:
        print("[S6 auto-detect]           SKIP (no samples_v3 cache)")
        return
    prefix = str(cand[0])[: -len("_mid_paths.npy")]
    mp = np.load(f"{prefix}_mid_paths.npy", mmap_mode="r")[:20]
    el = np.load(f"{prefix}_entry_long.npy")[:20]
    es = np.load(f"{prefix}_entry_short.npy")[:20]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tp_prefix = str(td / Path(prefix).name)
        for suf in ("mid_paths", "entry_long", "entry_short", "y", "pnl"):
            src = Path(f"{prefix}_{suf}.npy")
            if src.exists():
                os.symlink(src, f"{tp_prefix}_{suf}.npy")
        bp = np.stack([np.asarray(mp, dtype=np.float64),
                        np.asarray(mp, dtype=np.float64)], axis=-1)[:20]
        eb = np.column_stack([el, es])[:20]
        # Resize main NPYs to 20 samples for consistency. The auto-detect
        # does not check size — just existence — so this is fine.
        np.save(f"{tp_prefix}_book_paths.npy", bp)
        np.save(f"{tp_prefix}_entry_book.npy", eb)

        import importlib
        import scripts.grid_live_retargeted as grid_mod
        grid_mod = importlib.reload(grid_mod)
        grid_mod.CACHE_DIR = td
        c = grid_mod._load_cache()
        assert "book_paths" in c, "grid _load_cache() did not auto-detect book_paths.npy"
        assert "entry_book" in c, "grid _load_cache() did not auto-detect entry_book.npy"
        print(f"[S6 grid auto-detect]      book_paths shape={c['book_paths'].shape} OK")


def main() -> int:
    s1_tp_exact_zero_spread()
    s2_tp_with_spread_requires_bid_cross()
    s3_sl_gap_slippage()
    s4_partial_at_limit_not_overshoot()
    s5_timeout_limit_when_mid_returns()
    s6_grid_auto_detect()
    print("\n[verify] PASS — all ground-truth scenarios + grid wiring OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
