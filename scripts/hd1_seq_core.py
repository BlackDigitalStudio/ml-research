#!/usr/bin/env python3
"""HD1-seq numeric core — frozen contract, optimized path + parity ref.

Pre-registered & FROZEN under HD1 rev25 (hypotheses.jsonl, freeze commit
recorded by the runner). This module is a NUMERICALLY-EQUIVALENT,
PARITY-GATED reimplementation of the frozen hr1_screen/ha5_screen numeric
core (decision points, first-passage label, >=cost scope, honest split,
R1 weights, AUC, single-sample block-bootstrap SE). It is NOT a slow
verbatim fork: the per-tick L2 feature transform and the window gather
are vectorized for the raw 20-level LOB / GPU regime.

The ONLY things HD1-seq varies vs HM6 baseline_ref are `model_family`
(tcn) and `input_repr` (raw_l2_20lvl_eventclock). Everything in this
module that touches the label / scope / split / weighting / metric is a
bit-exact mirror of the frozen logic; tests/test_hd1_parity.py asserts
that against the actual frozen functions before any sweep runs.
"""
from __future__ import annotations

import numpy as np

# --- frozen constants (verbatim: hr1_screen.py / ha5_screen.py) ----------
STRIDE = 4
SEED = 42
NS = 1_000_000_000
F_T0 = 0.0013          # 0.13% strict round-trip floor (>=cost scope)
EMB = 64               # honest_val_test embargo (rows)
TRAIN_FRAC = 0.70
HS = (180, 300, 600)
N_TR_FLOOR, N_OOS_FLOOR = 300, 200

# --- frozen pre-registered HD1-seq design (HD1 rev25, §3/§4) -------------
N_LEVELS = 20
MAX_L = 512                       # cache stores windows at max context
L_GRID = (64, 128, 256, 512)      # primary sweep axis (HA3 locus probe)
D_GRID = (4, 6)                   # secondary: TCN depth
W_FIXED = 64                      # channel width (fixed a priori)
WD_FIXED = 1e-4                   # weight decay (fixed a priori)
SEED_GRID = (0, 1, 2)
VAL_FRAC_OF_TRAIN = 0.20          # val carved from tail of the 70% train


# =========================================================================
# First-passage label  (FROZEN — must equal ha5_screen._first_passage)
# =========================================================================
def first_passage_ref(mid, i0, jH, m0, f):
    """Verbatim copy of ha5_screen._first_passage (parity reference)."""
    up = m0 * (1.0 + f)
    dn = m0 * (1.0 - f)
    out = np.zeros(i0.shape[0], np.int8)
    for r in range(i0.shape[0]):
        s = mid[i0[r] + 1: jH[r] + 1]
        if s.size == 0:
            continue
        u = np.argmax(s >= up[r]) if (s >= up[r]).any() else -1
        d = np.argmax(s <= dn[r]) if (s <= dn[r]).any() else -1
        if u < 0 and d < 0:
            continue
        if d < 0 or (u >= 0 and u <= d):
            out[r] = 1
        else:
            out[r] = -1
    return out


def first_passage_fast(mid, i0, jH, m0, f):
    """Vectorized first-passage. Bit-identical to first_passage_ref.

    Per entry r: scan mid[i0[r]+1 : jH[r]+1]; first index hitting the up
    barrier m0*(1+f) vs the down barrier m0*(1-f); tie or up-not-later
    -> +1, else -1, none -> 0. The frozen tie rule is `d<0 or
    (u>=0 and u<=d)` => +1. Implemented with searchsorted-free running
    comparisons over a ragged set via per-entry slicing but with the
    inner argmax replaced by the first-True trick on contiguous views;
    windows are short (<= a few thousand ticks) so this stays O(sum len)
    with no Python-level barrier scan branching.
    """
    mid = np.asarray(mid)
    n = i0.shape[0]
    out = np.zeros(n, np.int8)
    up = m0 * (1.0 + f)
    dn = m0 * (1.0 - f)
    lo = i0 + 1
    hi = jH + 1                      # exclusive
    length = np.maximum(hi - lo, 0)
    if n == 0:
        return out
    Lmax = int(length.max())
    if Lmax == 0:
        return out
    # Build a (n, Lmax) padded matrix of the per-entry forward paths.
    # Index clamp keeps it in-bounds; the validity mask zeroes padding so
    # padded cells never trigger a barrier (up>mid_pad, dn<mid_pad false).
    cols = np.arange(Lmax)
    idx = lo[:, None] + cols[None, :]
    valid = cols[None, :] < length[:, None]
    idx = np.clip(idx, 0, mid.shape[0] - 1)
    paths = mid[idx]
    hit_u = (paths >= up[:, None]) & valid
    hit_d = (paths <= dn[:, None]) & valid
    any_u = hit_u.any(1)
    any_d = hit_d.any(1)
    # argmax on bool -> first True index; meaningless if any_*==False
    fu = np.where(any_u, hit_u.argmax(1), np.iinfo(np.int64).max)
    fd = np.where(any_d, hit_d.argmax(1), np.iinfo(np.int64).max)
    none = ~any_u & ~any_d
    # frozen rule: +1 iff (d<0) or (u>=0 and u<=d); else -1
    up_first = (~any_d) | (any_u & (fu <= fd))
    out[:] = np.where(none, 0, np.where(up_first, 1, -1)).astype(np.int8)
    return out


# =========================================================================
# Decision points / split / labels / R1 weights  (FROZEN — == hr1_screen)
# =========================================================================
def select_decision_idx(idx, n_ticks):
    """sel into idx, then book indices i.  == hr1_screen.run lines 120-121
    (and ha5_screen 178-180): ok=(idx>0)&(idx<n-1); [::STRIDE]."""
    ok = (idx > 0) & (idx < n_ticks - 1)
    sel = np.where(ok)[0][::STRIDE]
    return sel, idx[sel].astype(np.int64)


def honest_split(n):
    """Global honest_val_test masks. == hr1_screen.run lines 135-138:
    n_tr=int(n*0.70); tr[:n_tr-EMB]=True; te[n_tr:]=True."""
    n_tr = int(n * TRAIN_FRAC)
    tr = np.zeros(n, bool)
    te = np.zeros(n, bool)
    tr[:n_tr - EMB] = True
    te[n_tr:] = True
    return tr, te, n_tr


def train_val_split(tr_mask):
    """Carve a chronological val from the tail of the 70% train region
    for early-stop + config selection (FROZEN HD1 rev25 §3: 'model
    selection by VALIDATION only, test untouched'). 80/20 of tr rows,
    contiguous, with an EMB-row gap so val never leaks into fit. The OOS
    test mask is UNCHANGED (== HM6), so delta_ic stays comparable."""
    tr_pos = np.where(tr_mask)[0]
    if tr_pos.size == 0:
        return tr_mask.copy(), np.zeros_like(tr_mask)
    cut = int(tr_pos.size * (1.0 - VAL_FRAC_OF_TRAIN))
    fit = np.zeros_like(tr_mask)
    val = np.zeros_like(tr_mask)
    fit[tr_pos[:max(0, cut - EMB)]] = True
    val[tr_pos[cut:]] = True
    return fit, val


def labels_for_H(ts, mid, i, t0_ns, H, fp=first_passage_fast):
    """Per-H first-passage label + forward log-return, FROZEN logic
    (hr1_screen.run lines 150-154): jH=min(searchsorted(ts,t0+H*NS,
    'left'), len(mid)-1); m0=mid[i]; y0=first_passage; rH=log(mid[jH]/m0).
    """
    jH = np.minimum(np.searchsorted(ts, t0_ns + H * NS, "left"),
                    len(mid) - 1)
    m0 = mid[i]
    y0 = fp(mid, i, jH, m0, F_T0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rH = np.log(np.where(m0 > 0, mid[jH] / m0, np.nan))
    reached = (y0 != 0) & np.isfinite(rH)
    up = (y0 == 1).astype(int)
    return y0, rH, reached, up


def r1_weights(rH, fit_reached_mask):
    """R1 sample weights (HM5 rev3 / hr1_screen.run lines 168-170):
    amove=|rH|; wcap=nanquantile(amove[train&reached],0.99);
    w1=clip(amove,0,wcap).  Train-split stats only (no leakage)."""
    amove = np.abs(rH)
    wcap = np.nanquantile(amove[fit_reached_mask], 0.99)
    return np.clip(amove, 0, wcap)


def block_size(H):
    """== hr1_screen.run line 159: max(1, ceil(H/(STRIDE*24)))."""
    return max(1, int(np.ceil(H / (STRIDE * 24))))


# =========================================================================
# Metric / placebo / bootstrap  (FROZEN — == ha5._auc / HM5 rev3 SE)
# =========================================================================
def auc(y, p):
    """== ha5_screen._auc (sklearn roc_auc_score, 0.5 if degenerate)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, p))


def placebo_auc(yv, p, seed=SEED):
    """== hr1/ha5: _auc(rng.permutation(yv), p), rng=default_rng(SEED)."""
    rng = np.random.default_rng(seed)
    return auc(rng.permutation(yv), p)


def block_bootstrap_auc_se(yv, p, block, seed=SEED, B=300):
    """HM5 rev3 single-sample block-bootstrap SE of R1-AUC (NOT paired
    vs R0 — HM5 rev3 dropped that). Same block construction as
    hr1_screen._paired_boot (contiguous blocks, n//block draws)."""
    yv = np.asarray(yv)
    n = len(yv)
    if n < block * 2 or yv.min() == yv.max():
        return float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block + 1)
    nb = max(1, n // block)
    vals = []
    for _ in range(B):
        s = rng.choice(starts, nb, replace=True)
        ix = np.concatenate([np.arange(x, x + block) for x in s])
        yi = yv[ix]
        if yi.min() == yi.max():
            continue
        vals.append(auc(yi, p[ix]))
    return float(np.std(vals)) if len(vals) > 30 else float("nan")


# =========================================================================
# Per-tick L2 feature transform  (FROZEN HD1 rev25 §3)
# =========================================================================
# Deterministic, no learned preprocessing. Builder stores RAW per-tick
# features; standardization (train-split mean/std) is applied at train
# time in the trainer so "train-split stats only" holds exactly.
#
# Per tick, in this fixed order (F = 2*N_LEVELS + 6 = 46):
#   [0:20]   per-level signed log size imbalance:
#              sign*log1p(|bid_k_size - ask_k_size|), sign of (bid-ask)
#   [20:40]  per-level price offset from mid (mid-relative):
#              ((bid_k+ask_k)/2 - mid) / mid   (book is symmetric around
#              mid; this encodes the per-level price ladder shape)
#   [40]     mid log-return since previous tick (0 at first tick)
#   [41]     relative spread (ask_0 - bid_0) / mid
#   [42]     L5 depth imbalance  (sum size lvls 0..4)
#   [43]     L20 depth imbalance (sum size lvls 0..19)
#   [44]     OFI increment (Cont et al. top-of-book order-flow imbalance)
#   [45]     microprice - mid, mid-relative
N_TICK_FEAT = 2 * N_LEVELS + 6


def tick_features(bid_p, bid_s, ask_p, ask_s):
    """Vectorized per-tick L2 transform. Inputs are (n_ticks, N_LEVELS)
    float64 arrays of the raw 20-level book; returns (n_ticks, 46)
    float32 RAW features (standardized later, train-split only)."""
    nt = bid_p.shape[0]
    b0, a0 = bid_p[:, 0], ask_p[:, 0]
    mid = 0.5 * (b0 + a0)
    safe_mid = np.where(mid > 0, mid, 1.0)
    F = np.empty((nt, N_TICK_FEAT), np.float64)

    d = bid_s - ask_s
    F[:, 0:N_LEVELS] = np.sign(d) * np.log1p(np.abs(d))
    lvl_mid = 0.5 * (bid_p + ask_p)
    F[:, N_LEVELS:2 * N_LEVELS] = (lvl_mid - mid[:, None]) / safe_mid[:, None]

    lr = np.zeros(nt)
    lr[1:] = np.log(np.where((mid[1:] > 0) & (mid[:-1] > 0),
                             mid[1:] / np.where(mid[:-1] > 0, mid[:-1], 1.0),
                             1.0))
    F[:, 40] = lr
    F[:, 41] = (a0 - b0) / safe_mid

    bs5, as5 = bid_s[:, :5].sum(1), ask_s[:, :5].sum(1)
    bs20, as20 = bid_s.sum(1), ask_s.sum(1)
    F[:, 42] = (bs5 - as5) / np.where(bs5 + as5 > 0, bs5 + as5, 1.0)
    F[:, 43] = (bs20 - as20) / np.where(bs20 + as20 > 0, bs20 + as20, 1.0)

    # Cont OFI (top of book): contribution from bid/ask level-0 moves.
    ofi = np.zeros(nt)
    db = np.diff(b0)
    da = np.diff(a0)
    bs0, as0 = bid_s[:, 0], ask_s[:, 0]
    e = np.zeros(nt - 1)
    e += np.where(db > 0, bs0[1:], np.where(db < 0, -bs0[:-1], bs0[1:] - bs0[:-1]))
    e -= np.where(da < 0, as0[1:], np.where(da > 0, -as0[:-1], as0[1:] - as0[:-1]))
    ofi[1:] = e
    F[:, 44] = ofi

    micro = np.where(bs0 + as0 > 0,
                     (b0 * as0 + a0 * bs0) / np.where(bs0 + as0 > 0,
                                                      bs0 + as0, 1.0),
                     mid)
    F[:, 45] = (micro - mid) / safe_mid
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def gather_windows(tickfeat, i, L=MAX_L):
    """Causal window gather: for each decision book-index i, the last L
    ticks with ts<=t0 == tickfeat[i-L+1 .. i] (inclusive of i). Left-pad
    with zeros when i<L-1; returns (n_dp, L, F) float32 and a (n_dp, L)
    bool valid mask. Single fancy-index op — no Python loop.

    f32 (not f16): the §3 transform produces values (e.g. raw Cont-OFI,
    feat [44]) outside float16 range for high-notional symbols; f16
    overflows to ±inf -> NaN. f32 is exact for the deterministic
    transform (HD1 rev26 defect-fix; storage-only, no math change)."""
    n_dp = i.shape[0]
    F = tickfeat.shape[1]
    offs = np.arange(-(L - 1), 1)
    rows = i[:, None] + offs[None, :]            # (n_dp, L)
    valid = rows >= 0
    rows = np.clip(rows, 0, tickfeat.shape[0] - 1)
    win = tickfeat[rows]                         # (n_dp, L, F)
    win[~valid] = 0.0
    return win.astype(np.float32), valid


# =========================================================================
# Frozen §5 decision gate (BINDING) — applied at ingest across symbols
# =========================================================================
def cell_delta_ic(rank_ic_oos, baseline_ref_ric):
    return round(rank_ic_oos - baseline_ref_ric, 5)


def gate_cell(delta_ic, placebo_ric, boot_se):
    """Per-cell HM1 robustness, frozen HD1 rev25 §5 (a)&(b). (c)
    cross-symbol consistency is decided at ingest over the 4 symbols at
    matched H. Returns (passes_ab, reason)."""
    if delta_ic is None:
        return False, "delta_ic=None"
    a = delta_ic > placebo_ric           # (a) > R1-placebo upper band
    b = (boot_se is not None and np.isfinite(boot_se)
         and delta_ic > boot_se)         # (b) outside +-1*block-boot SE
    return bool(a and b), f"a={a} b={b} d={delta_ic} plac={placebo_ric} se={boot_se}"


def status_for_cell(passes_ab, cross_symbol_ok):
    """Frozen §5 status map. Never auto-'confirmed' (HM1; economic
    deploy-gate). 'refuted' := delta within noise/placebo at that scope."""
    if passes_ab and cross_symbol_ok:
        return "suspect"
    if passes_ab:
        return "exploratory"
    return "refuted"
