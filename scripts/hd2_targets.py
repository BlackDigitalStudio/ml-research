#!/usr/bin/env python3
"""HD2 target-form labelers (rev5 objective round 1). Given the per-tick mid+ts
(from /cache/midts) and decision-point book indices t0, produce (y, rH, reached,
up) for a swept target-form `spec`. Symmetric first-passage reuses the FROZEN
hd1_seq_core.first_passage_fast (parity); asym/vol/terminal/deadband/volnorm are
thin, well-defined extensions on the same mid-path + terminal-return primitives.

spec forms:
  {"form":"fp","f":0.0013}                  symmetric first-passage at +-f
  {"form":"fp_asym","f_up":..,"f_dn":..}    asymmetric first-passage
  {"form":"fp_vol","k":1.0,"win":1000}      vol-scaled FP: f_i = k*sigma_H(i)
  {"form":"terminal"}                        y=sign(r_H), all reached
  {"form":"deadband","delta":0.0013}         y=sign(r_H) iff |r_H|>delta
  {"form":"volnorm","win":1000}              y=sign(r_H/sigma_i)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hd1_seq_core as core  # noqa: E402

NS = core.NS


def _jH(ts, t0_ns, H, n):
    return np.minimum(np.searchsorted(ts, t0_ns + H * NS, "left"), n - 1)


def first_passage_asym(mid, i0, jH, m0, f_up, f_dn):
    """Asymmetric barriers: up=m0(1+f_up), dn=m0(1-f_dn). Same tie rule as the
    frozen core.first_passage_fast (+1 iff d<0 or (u>=0 and u<=d))."""
    n = i0.shape[0]
    out = np.zeros(n, np.int8)
    if n == 0:
        return out
    up = m0 * (1.0 + f_up)
    dn = m0 * (1.0 - f_dn)
    lo = i0 + 1
    hi = jH + 1
    length = np.maximum(hi - lo, 0)
    Lmax = int(length.max())
    if Lmax == 0:
        return out
    cols = np.arange(Lmax)
    idx = lo[:, None] + cols[None, :]
    valid = cols[None, :] < length[:, None]
    idx = np.clip(idx, 0, mid.shape[0] - 1)
    paths = mid[idx]
    hit_u = (paths >= up[:, None]) & valid
    hit_d = (paths <= dn[:, None]) & valid
    any_u = hit_u.any(1)
    any_d = hit_d.any(1)
    fu = np.where(any_u, hit_u.argmax(1), np.iinfo(np.int64).max)
    fd = np.where(any_d, hit_d.argmax(1), np.iinfo(np.int64).max)
    none = ~any_u & ~any_d
    up_first = (~any_d) | (any_u & (fu <= fd))
    out[:] = np.where(none, 0, np.where(up_first, 1, -1)).astype(np.int8)
    return out


def trailing_sigma(mid, t0, win=1000):
    """Per-decision-point trailing std of per-tick mid log-returns over the last
    `win` ticks ending at t0 (book index). Vectorized via cumulative sums."""
    n = mid.shape[0]
    lr = np.zeros(n, np.float64)
    pos = mid > 0
    ok = pos[1:] & pos[:-1]
    lr[1:] = np.where(ok, np.log(np.where(pos[:-1], mid[1:] / np.where(pos[:-1], mid[:-1], 1.0), 1.0)), 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(lr)])
    c2 = np.concatenate([[0.0], np.cumsum(lr * lr)])
    hi = t0 + 1                       # inclusive of t0
    lo = np.maximum(hi - win, 1)      # lr[0]=0 placeholder, start at 1
    cnt = np.maximum(hi - lo, 1)
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
    return np.sqrt(var)              # per-tick sigma at each t0


def make_target(mid, ts, t0, H, spec):
    """-> (y int8 in {-1,0,1}, rH float32, reached bool, up int). t0 = book idx."""
    n = len(mid)
    t0 = t0.astype(np.int64)
    t0_ns = ts[t0]
    jH = _jH(ts, t0_ns, H, n)
    m0 = mid[t0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rH = np.log(np.where(m0 > 0, mid[jH] / m0, np.nan))
    finite = np.isfinite(rH)
    form = spec["form"]

    if form == "fp":
        y = core.first_passage_fast(mid, t0, jH, m0, spec["f"])
        reached = (y != 0) & finite
    elif form == "fp_asym":
        y = first_passage_asym(mid, t0, jH, m0, spec["f_up"], spec["f_dn"])
        reached = (y != 0) & finite
    elif form == "fp_vol":
        sig = trailing_sigma(mid, t0, spec.get("win", 1000))
        Hticks = np.maximum(jH - t0, 1)
        f_i = spec["k"] * sig * np.sqrt(Hticks)           # ~k-sigma over H
        f_i = np.clip(f_i, 1e-5, 0.05)                    # sane bounds
        y = core.first_passage_fast(mid, t0, jH, m0, f_i)
        reached = (y != 0) & finite
    elif form == "fp_sqrt":
        # barrier grows with sqrt(H) to match the ~sqrt(t) expected move:
        # f = f0 * sqrt(H / H0). Keeps reached-rate ~constant across H and
        # makes the label genuinely H-horizon (not the first-wiggle direction).
        f = spec["f0"] * np.sqrt(H / spec.get("H0", 180))
        y = core.first_passage_fast(mid, t0, jH, m0, f)
        reached = (y != 0) & finite
    elif form == "terminal":
        y = np.sign(np.nan_to_num(rH)).astype(np.int8)
        reached = finite & (y != 0)
    elif form == "deadband":
        d = spec["delta"]
        y = np.where(rH > d, 1, np.where(rH < -d, -1, 0)).astype(np.int8)
        reached = (np.abs(rH) > d) & finite
    elif form == "volnorm":
        sig = trailing_sigma(mid, t0, spec.get("win", 1000))
        z = rH / np.where(sig > 1e-9, sig, 1.0)
        y = np.sign(np.nan_to_num(z)).astype(np.int8)
        reached = finite & (y != 0)
    else:
        raise ValueError(f"unknown target form {form!r}")

    rH = np.where(finite, rH, 0.0).astype(np.float32)
    up = (y == 1).astype(int)
    return y.astype(np.int8), rH, reached, up


def _pnl_dir(rf, tp, sl, cw, cl):
    """Net PnL% of a directional trade given favorable-direction return series rf
    (rf = forward return for long; pass -r for short). TP/SL first-passage, else
    mark-to-last at timeout. Commissions: cw win-side, cl loss-side (taker)."""
    if rf.size == 0:
        return -cw
    tp_mask = rf >= tp
    sl_mask = rf <= -sl
    tb = int(np.argmax(tp_mask)) if tp_mask.any() else -1
    sb = int(np.argmax(sl_mask)) if sl_mask.any() else -1
    if tb != -1 and (sb == -1 or tb <= sb):
        return tp * 100.0 - cw
    if sb != -1 and (tb == -1 or sb < tb):
        return -sl * 100.0 - cl
    return float(rf[-1]) * 100.0 - cw          # timeout: mark-to-last, pay exit


def make_target_profit3(mid, ts, t0, H, tp_pct=0.20, sl_pct=0.10, cw=0.07, cl=0.10):
    """Legacy 3-class triple-barrier PROFITABILITY label (reproduces the old-repo
    target that gave standalone Mamba prec_NF~0.33 = ~1.5x base on the 20/19/61
    class split). For each decision point, simulate long & short over the
    [t0, t0+H seconds] window with TP/SL first-passage (timeout -> mark-to-mid),
    NET of taker commissions, then:
      y3 = 0 (UP) iff pnl_long  > 0 and pnl_long  > pnl_short
           1 (DN) iff pnl_short > 0 and pnl_short > pnl_long
           2 (FL) otherwise  (the abstain class — majority ~60%).
    FL is a TRAINED class (NO masking), unlike the binary fp/terminal path.
    Returns y3 int8 (n,)."""
    n = len(mid)
    t0 = t0.astype(np.int64)
    t0_ns = ts[t0]
    jH = _jH(ts, t0_ns, H, n)
    m0 = mid[t0].astype(np.float64)
    tp = tp_pct / 100.0
    sl = sl_pct / 100.0
    y3 = np.full(len(t0), 2, np.int8)               # default FL
    for k in range(len(t0)):
        a = int(t0[k]); b = int(jH[k])
        if b <= a or m0[k] <= 0:
            continue
        r = mid[a + 1:b + 1] / m0[k] - 1.0          # forward returns vs entry
        pl = _pnl_dir(r, tp, sl, cw, cl)            # long
        ps = _pnl_dir(-r, tp, sl, cw, cl)           # short (favorable = -r)
        if pl > 0.0 and pl > ps:
            y3[k] = 0
        elif ps > 0.0 and ps > pl:
            y3[k] = 1
    return y3


# canonical rev5 variant set (name -> spec); H-agnostic (applied per H)
VARIANTS = {
    "fp_0.08":   {"form": "fp", "f": 0.0008},
    "fp_0.13":   {"form": "fp", "f": 0.0013},   # control == current HD2 label
    "fp_0.20":   {"form": "fp", "f": 0.0020},
    "fp_0.30":   {"form": "fp", "f": 0.0030},
    "fp_0.40":   {"form": "fp", "f": 0.0040},   # wider: lower breakeven hit-rate
    "fp_0.50":   {"form": "fp", "f": 0.0050},
    "fp_0.75":   {"form": "fp", "f": 0.0075},
    "fp_asym_up": {"form": "fp_asym", "f_up": 0.0013, "f_dn": 0.0020},
    "fp_asym_dn": {"form": "fp_asym", "f_up": 0.0020, "f_dn": 0.0013},
    "fp_vol_1.0": {"form": "fp_vol", "k": 1.0, "win": 1000},
    "fp_vol_1.5": {"form": "fp_vol", "k": 1.5, "win": 1000},
    "fp_sqrt_0.13": {"form": "fp_sqrt", "f0": 0.0013, "H0": 180},  # H-matched barrier
    "fp_sqrt_0.20": {"form": "fp_sqrt", "f0": 0.0020, "H0": 180},
    "terminal":  {"form": "terminal"},
    "deadband_0.13": {"form": "deadband", "delta": 0.0013},
    "volnorm":   {"form": "volnorm", "win": 1000},
}
