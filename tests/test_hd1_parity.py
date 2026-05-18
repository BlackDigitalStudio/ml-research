#!/usr/bin/env python3
"""HD1 rev25 parity gate — the optimized HD1-seq numeric core must be
bit-for-bit equivalent to the FROZEN hr1_screen/ha5_screen logic for
everything that touches label / scope / split / R1 weights / metric.

This test runs WITHOUT GCS/Modal (synthetic + adversarial book paths).
It is the pre-registered safety gate: divergence == bug, not DOF. The
sweep must not run unless this passes (the Modal driver invokes it).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ha5_screen import _auc as FROZEN_AUC  # noqa: E402
from scripts.ha5_screen import _first_passage as FROZEN_FP  # noqa: E402
from scripts import hd1_seq_core as C  # noqa: E402

RNG = np.random.default_rng(12345)


def _rand_book(n=4000):
    mid = 100.0 + np.cumsum(RNG.normal(0, 0.02, n))
    return np.maximum(mid, 1e-6)


def test_first_passage_matches_frozen():
    """fast & ref vs the ACTUAL frozen ha5_screen._first_passage, over
    random + adversarial (ties, none, up-only, down-only, empty win)."""
    for trial in range(200):
        mid = _rand_book(RNG.integers(500, 5000))
        nt = mid.shape[0]
        k = RNG.integers(20, 200)
        i0 = np.sort(RNG.choice(np.arange(1, nt - 2), k, replace=False))
        # mix of normal, zero-length and full-length forward windows
        span = RNG.integers(0, 400, k)
        jH = np.minimum(i0 + span, nt - 1)
        m0 = mid[i0]
        f = float(RNG.choice([0.0013, 0.0008, 0.003]))
        ref = FROZEN_FP(mid, i0, jH, m0, f)
        a = C.first_passage_ref(mid, i0, jH, m0, f)
        b = C.first_passage_fast(mid, i0, jH, m0, f)
        assert np.array_equal(ref, a), f"ref!=frozen trial {trial}"
        assert np.array_equal(ref, b), (
            f"FAST!=frozen trial {trial}: "
            f"{np.where(ref != b)[0][:10]}")


def test_first_passage_adversarial_ties():
    """Exact-tie barrier hits (u==d) must resolve +1 per the frozen rule
    `d<0 or (u>=0 and u<=d)`."""
    mid = np.array([100, 100, 100.13, 99.87, 100, 100, 100], np.float64)
    i0 = np.array([0], np.int64)
    jH = np.array([6], np.int64)
    m0 = mid[i0]
    f = 0.0013
    ref = FROZEN_FP(mid, i0, jH, m0, f)
    assert C.first_passage_fast(mid, i0, jH, m0, f)[0] == ref[0]


def test_select_decision_idx_and_split():
    """sel/i and honest split == hr1_screen.run inline logic."""
    n_ticks = 50000
    idx = np.sort(RNG.choice(np.arange(-3, n_ticks + 3),
                             8000, replace=False)).astype(np.int64)
    ok = (idx > 0) & (idx < n_ticks - 1)
    sel_ref = np.where(ok)[0][::C.STRIDE]
    i_ref = idx[sel_ref].astype(np.int64)
    sel, i = C.select_decision_idx(idx, n_ticks)
    assert np.array_equal(sel, sel_ref) and np.array_equal(i, i_ref)

    n = i.shape[0]
    n_tr_ref = int(n * 0.70)
    tr_ref = np.zeros(n, bool); tr_ref[:n_tr_ref - 64] = True
    te_ref = np.zeros(n, bool); te_ref[n_tr_ref:] = True
    tr, te, n_tr = C.honest_split(n)
    assert (np.array_equal(tr, tr_ref) and np.array_equal(te, te_ref)
            and n_tr == n_tr_ref)


def test_labels_and_r1_weights_match_frozen():
    """labels_for_H + r1_weights reproduce hr1_screen.run lines 150-170."""
    mid = _rand_book(60000)
    ts = np.sort(RNG.integers(0, 4_000_000_000, mid.shape[0])
                 ).astype(np.int64)
    idx = np.sort(RNG.choice(np.arange(1, mid.shape[0] - 2),
                             9000, replace=False)).astype(np.int64)
    sel, i = C.select_decision_idx(idx, mid.shape[0])
    t0 = ts[i].astype(np.int64)
    n = i.shape[0]
    tr, te, _ = C.honest_split(n)
    for H in C.HS:
        # frozen reference (hr1_screen.run inline)
        jH = np.minimum(np.searchsorted(ts, t0 + H * C.NS, "left"),
                        len(mid) - 1)
        m0 = mid[i]
        y0_ref = FROZEN_FP(mid, i, jH, m0, C.F_T0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rH_ref = np.log(np.where(m0 > 0, mid[jH] / m0, np.nan))
        reached_ref = (y0_ref != 0) & np.isfinite(rH_ref)

        y0, rH, reached, up = C.labels_for_H(ts, mid, i, t0, H)
        assert np.array_equal(y0, y0_ref), f"y0 H={H}"
        np.testing.assert_array_equal(np.isfinite(rH),
                                      np.isfinite(rH_ref))
        m = np.isfinite(rH_ref)
        np.testing.assert_allclose(rH[m], rH_ref[m], rtol=0, atol=0)
        assert np.array_equal(reached, reached_ref)

        s_tr = tr & reached_ref
        if s_tr.sum() < 10:
            continue
        amove = np.abs(rH_ref)
        wcap = np.nanquantile(amove[s_tr], 0.99)
        w1_ref = np.clip(amove, 0, wcap)
        w1 = C.r1_weights(rH, s_tr)
        np.testing.assert_allclose(w1, w1_ref, rtol=0, atol=0)
        assert C.block_size(H) == max(1, int(np.ceil(H / (C.STRIDE * 24))))


def test_auc_and_placebo_match_frozen():
    for _ in range(50):
        n = RNG.integers(300, 3000)
        y = RNG.integers(0, 2, n)
        p = RNG.random(n)
        assert C.auc(y, p) == FROZEN_AUC(y, p)
        # placebo: frozen uses rng=default_rng(SEED); _auc(perm(yv), p)
        rng = np.random.default_rng(C.SEED)
        ref = FROZEN_AUC(rng.permutation(y), p)
        assert C.placebo_auc(y, p) == ref
    # degenerate -> 0.5 (frozen contract)
    assert C.auc(np.ones(10), RNG.random(10)) == 0.5


def test_tick_features_shape_and_finite():
    nt, nl = 1000, C.N_LEVELS
    bp = 100 - np.cumsum(RNG.random((nt, nl)) * 0.01, axis=1)
    ap = 100 + np.cumsum(RNG.random((nt, nl)) * 0.01, axis=1)
    bs = RNG.random((nt, nl)) * 10
    as_ = RNG.random((nt, nl)) * 10
    F = C.tick_features(bp, bs, ap, as_)
    assert F.shape == (nt, C.N_TICK_FEAT) and np.isfinite(F).all()
    win, valid = C.gather_windows(F, np.array([0, 5, 999]), L=64)
    assert win.shape == (3, 64, C.N_TICK_FEAT)
    assert not valid[0, :-1].any() and valid[0, -1]   # i=0 -> only last
    assert valid[2].all()                              # i=999 -> full


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fail = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            fail += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fail}/{len(fns)} passed")
    raise SystemExit(1 if fail else 0)
