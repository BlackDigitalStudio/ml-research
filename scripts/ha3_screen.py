#!/usr/bin/env python3
"""HA3 — pre-registered FEATURE-LOCALIZATION screen (where does the signal
physically live; concentrated or diffuse).

Pre-registered & FROZEN 2026-05-17 (HA3 rev2; user AskUserQuestion:
method = univariate-IC + grouped-ablation; z-window DEFERRED — features_v1
is a pre-normalized .npy, honest re-normalization needs a cache rebuild,
a separate cost tier). NO post-hoc DOF.

Open axis (HM3/HM5/HD1): the snapshot directional negatives are
representation-bound, but NOBODY has measured WHICH of the ~46 live
features_v1 cols carry the weak directional signal. This is pure
direction-finding ($0 GPU, CPU on the existing features_v1 cache) and it
de-risks/cheapens the decided sequence build (HD1) by telling it which
inputs matter.

TESTBED — IDENTICAL to HM6 `baseline_ref` (mandatory; HM5 caveat: an
R1-only-vs-baseline_ref comparison is valid ONLY when the testbed
matches HM6). The data assembly, >=cost scope, up-first label, honest
70/emb/30 split and the R1 objective below are the FROZEN
``hr1_screen.run`` numeric core, replicated VERBATIM and reduced to R1
only (HM5 rev3: new runners are R1-only, no R0-R4 grid). Set =
{SOL,BTC,ETH,LTC}-USDT-PERP, common-aligned calendar window
(``baseline_360._window``, canon HM6 rev2). The ONLY thing varied vs
HM6 is the SET OF FEATURE COLUMNS fed to the model.

Per (sym in {SOL,BTC,ETH,LTC}, H in {180,300,600}s):

  FULL  = R1-XGB on all live cols (== HM6 baseline_ref cell; the
          |AUC_full - HM6_R1_cell| < 0.02 sentinel makes the HM5
          testbed-match caveat executable — drift => NOT comparable).
  (a) UNIVARIATE map: per live col j, OOS Spearman(col_j, r_H) +
      permutation placebo. Descriptive direction-finding (NOT gated).
  (b) GROUPED ABLATION: for each named feature group g (decoded
      col->name map, research/CRYPTOLAKE_SCHEMA.md):
        ONLY-g  R1-XGB on group-g cols alone   -> AUC_only, placebo
        DROP-g  R1-XGB on all-live minus g     -> AUC_drop
        dAUC_drop = AUC_full - AUC_drop  (>0 => g carries signal),
        paired block-bootstrap SE of (AUC_full - AUC_drop), z, placebo.

PRE-REGISTERED BAR (HM1-standard, FROZEN, NOT economic-gated, NEVER
auto-'confirmed' — auto-status caps at 'suspect'):
  * a GROUP is a robust signal carrier iff DROP-g lowers AUC vs FULL
    with paired-boot z>2 AND dAUC>0 AND ONLY-g AUC>0.5 with
    |placebo-0.5|<0.02, AND the sign is consistent across ALL 4
    symbols (cross-symbol agreement = post-run ledger decision).
  * verdict CONCENTRATED iff one group's DROP accounts for the bulk of
    (AUC_full-0.5) on >=3/4 symbols; DIFFUSE iff no single DROP-g
    clears z>2 on >=3/4 symbols. (post-run ledger reading; the run
    only records the grid + flags.)
  * `refuted` for HA3 := no group is a robust cross-symbol carrier AND
    the univariate map shows no col with |IC| robust vs placebo —
    i.e. the ~0.51 edge is diffuse/placebo-thin, not localizable
    (Δ-within-noise, NOT economic_pass_strict=0; SELECTION POLICY).

Execution: 96-vCPU VM via scripts/gcp_bootstrap.sh (NOT the ledger
sandbox). Results -> gs://.../research_runs/{run_id}/results.json
-> appended to research/experiments.jsonl -> research.db.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_cryptolake_cache import (  # noqa: E402
    _gcs_bucket, _list_days, _load_npy, _load_book_l1)
from scripts.ha5_screen import (  # noqa: E402  (one source of truth)
    _auc, _cond_feats, _first_passage, _load_events, _xgbc)
from scripts.baseline_360 import _window  # noqa: E402  (HM6 canon window)

# --- FROZEN constants, byte-identical to hr1_screen / HM6 ------------------
HS = (180, 300, 600)
F_T0 = 0.0013                       # 0.13% strict round-trip floor (scope)
STRIDE = 4
SEED = 42
NS = 1_000_000_000
NV1 = 59                            # features_v1 column count
B_BOOT = 300
N_TR_FLOOR, N_OOS_FLOOR = 300, 200
EMB = 64

# --- HM6 canonical baseline_ref: frozen 12-cell R1 AUC (HM6 rev4) ---------
# The testbed-match sentinel. Source: hypotheses.jsonl HM6 / RESEARCH_LOG.
HM6_R1 = {
    "SOL-USDT-PERP": {180: 0.514, 300: 0.515, 600: 0.513},
    "BTC-USDT-PERP": {180: 0.542, 300: 0.525, 600: 0.514},
    "ETH-USDT-PERP": {180: 0.523, 300: 0.516, 600: 0.507},
    "LTC-USDT-PERP": {180: 0.533, 300: 0.534, 600: 0.526},
}
HM6_RUN = "phaseb-20260517-203822"   # baseline_ref experiment_id stem

# --- FROZEN feature groups: decoded features_v1 cols 0-58 -----------------
# research/CRYPTOLAKE_SCHEMA.md "features_v1 (the model input X)". Every
# one of the 59 cols is assigned exactly once; trailing HA5 cond columns
# (index >= NV1) form 'events_cond' (they are part of HM6's FULL F).
GROUPS = {
    "ofi_flow":      [0, 6, 9, 26, 27, 28, 29, 40, 41, 42, 46],
    "book_liq":      [1, 2, 3, 4, 25, 31, 32, 33, 45, 49],
    "trade_intens":  [7, 8, 22, 47, 48],
    "momentum":      [11, 12, 34, 35, 36],
    "volatility":    [10, 21, 23, 37, 38, 39],
    "funding_basis": [13, 43, 44],
    "events_ext":    [17, 56, 57, 58],
    "cross_asset":   [14, 15, 16, 30, 50, 51, 52, 53, 54, 55],
    "dead_const":    [5, 18, 19, 20, 24],
}
COLNAME = {
    0: "ofi", 1: "imbalance_ratio", 2: "imbalance_velocity", 3: "spread",
    4: "depth_ratio_l5", 5: "large_order", 6: "trade_flow_imbalance",
    7: "trade_intensity", 8: "large_trade", 9: "cvd", 10: "volatility_1s",
    11: "vwap_deviation", 12: "momentum_5s", 13: "funding_rate",
    14: "eth_momentum_1s", 15: "eth_ofi", 16: "eth_leading_signal",
    17: "open_interest_delta", 18: "long_short_ratio",
    19: "liquidation_proximity", 20: "spoof_score", 21: "volatility_ratio",
    22: "trade_intensity_ratio", 23: "hurst", 24: "sweep_intensity",
    25: "cancel_rate_diff", 26: "ofi_1s", 27: "ofi_5s", 28: "ofi_30s",
    29: "ofi_divergence", 30: "cross_exch_mom_500ms", 31: "queue_pressure",
    32: "top3_asymmetry", 33: "effective_spread_ratio", 34: "momentum_30s",
    35: "momentum_60s", 36: "momentum_120s", 37: "realized_vol_60s",
    38: "realized_vol_120s", 39: "bipower_var_120s", 40: "ofi_60s",
    41: "ofi_120s", 42: "trade_flow_imbalance_60s",
    43: "funding_time_to_next_min", 44: "funding_basis_bps",
    45: "microprice_deviation", 46: "ofi_top5_weighted",
    47: "kyle_lambda_60s", 48: "vpin_60s", 49: "cancel_to_trade_ratio_30s",
    50: "bybit_lead_lag_corr_30s", 51: "okx_net_flow_30s",
    52: "bitget_net_flow_30s", 53: "gateio_net_flow_30s",
    54: "eth_momentum_60s", 55: "eth_btc_corr_30s",
    56: "cl_ext_56", 57: "cl_ext_57", 58: "cl_ext_58",
}


def _rank(a):
    o = a.argsort()
    r = np.empty(len(a), np.float64)
    r[o] = np.arange(len(a))
    return r


def _spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return 0.0
    rx, ry = _rank(x[m]), _rank(y[m])
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def _fit_r1(F, y, w, tr_cols, s_tr, s_te):
    """R1 == HM5 default: logloss + sample_weight=clip(|r_H|,p99). The
    ONLY objective (HM5 rev3 — no R0-R4 grid)."""
    m = _xgbc()
    m.fit(F[s_tr][:, tr_cols], y[s_tr], sample_weight=w[s_tr])
    return m.predict_proba(F[s_te][:, tr_cols])[:, 1]


def _paired_boot(y, p_a, p_b, block, rng):
    """Block-bootstrap SE of (AUC_a - AUC_b) on the SAME resamples
    (verbatim shape of hr1_screen._paired_boot)."""
    n = len(y)
    if n < block * 2 or y.min() == y.max():
        return float("nan")
    starts = np.arange(0, n - block + 1)
    nb = max(1, n // block)
    d = []
    for _ in range(B_BOOT):
        s = rng.choice(starts, nb, replace=True)
        ix = np.concatenate([np.arange(x, x + block) for x in s])
        yi = y[ix]
        if yi.min() == yi.max():
            continue
        d.append(_auc(yi, p_a[ix]) - _auc(yi, p_b[ix]))
    return float(np.std(d)) if len(d) > 30 else float("nan")


def run(bk, sc, sym, days, run_id, git):
    # ---- data assembly: VERBATIM from hr1_screen.run -------------------
    feats, meta, dayid, books = [], [], [], {}
    for di, day in enumerate(days):
        try:
            X = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/features.npy")
            idx = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/indices.npy"
                            ).astype(np.int64)
            ts, b, a = _load_book_l1(bk, sym, day)
            ev = _load_events(bk, sc, sym, day)
        except Exception as e:                       # noqa: BLE001
            print(f"  {sym} {day}: skip ({type(e).__name__})", flush=True)
            continue
        mid = (b + a) * 0.5
        ntk = ts.shape[0]
        sel = np.where((idx > 0) & (idx < ntk - 1))[0][::STRIDE]
        i = idx[sel]
        cf, _ = _cond_feats(ts[i], ev)
        feats.append(np.hstack([X[sel].astype(np.float64), cf]))
        meta.append(np.stack([np.full(i.size, di), i,
                              ts[i].astype(np.float64)], 1))
        dayid.append(np.full(i.size, di, np.int32))
        books[di] = (ts, mid)
    if not feats:
        return [{"symbol": sym, "error": "no data"}]
    ncol = min(x.shape[1] for x in feats)
    F = np.concatenate([x[:, :ncol] for x in feats])
    M = np.concatenate(meta)
    dy = np.concatenate(dayid)
    n = F.shape[0]
    n_tr = int(n * 0.70)
    tr = np.zeros(n, bool); tr[:n_tr - EMB] = True
    te = np.zeros(n, bool); te[n_tr:] = True
    print(f"[{sym}] n={n} feat={ncol} days={len(days)} "
          f"tr={int(tr.sum())} te={int(te.sum())}", flush=True)

    # ---- live-column detection (empirical dead/constant on TRAIN) ------
    cols = np.arange(ncol)
    std_tr = F[tr].std(axis=0)
    live = cols[std_tr > 1e-12]
    dead = sorted(int(c) for c in cols if c not in set(live.tolist()))
    dead_named = {int(c): COLNAME.get(int(c), f"col{c}") for c in dead}

    recs = []
    for H in HS:
        # ---- scope + label + R1 weight: VERBATIM from hr1_screen.run --
        y0 = np.zeros(n, np.int8)
        rH = np.full(n, np.nan)
        for di, (ts, mid) in books.items():
            mr = np.where(dy == di)[0]
            bi = M[mr, 1].astype(np.int64)
            t0 = M[mr, 2].astype(np.int64)
            jH = np.minimum(np.searchsorted(ts, t0 + H * NS, "left"),
                            len(mid) - 1)
            m0 = mid[bi]
            y0[mr] = _first_passage(mid, bi, jH, m0, F_T0)
            rH[mr] = np.log(np.where(m0 > 0, mid[jH] / m0, np.nan))
        reached = (y0 != 0) & np.isfinite(rH)
        up = (y0 == 1).astype(int)
        s_tr, s_te = tr & reached, te & reached
        ntr, nte = int(s_tr.sum()), int(s_te.sum())
        block = max(1, int(np.ceil(H / (STRIDE * 24))))
        rng = np.random.default_rng(SEED)

        if (ntr < N_TR_FLOOR or nte < N_OOS_FLOOR
                or up[s_tr].min() == up[s_tr].max()):
            recs.append({"symbol": sym, "horizon_sec": H,
                         "error": f"underpowered n_tr={ntr} n_oos={nte}"})
            print(f"  H={H}s underpowered n_tr={ntr} n_oos={nte}", flush=True)
            continue

        amove = np.abs(rH)
        wcap = np.nanquantile(amove[s_tr], 0.99)
        w1 = np.clip(amove, 0, wcap)                  # R1 weight (HM5)
        yv = up[s_te]

        live_l = [int(c) for c in live]
        # ---- FULL (== HM6 baseline_ref cell) --------------------------
        p_full = _fit_r1(F, up, w1, live_l, s_tr, s_te)
        auc_full = _auc(yv, p_full)
        plac_full = _auc(rng.permutation(yv), p_full)
        hm6 = HM6_R1.get(sym, {}).get(H)
        d_hm6 = None if hm6 is None else round(auc_full - hm6, 4)
        testbed_ok = int(hm6 is not None and abs(auc_full - hm6) < 0.02)

        # ---- (a) UNIVARIATE map (descriptive, NOT gated) --------------
        uni = []
        for c in live_l:
            ic = _spearman(F[s_te][:, c], rH[s_te])
            ic_p = _spearman(F[s_te][:, c],
                             rng.permutation(rH[s_te]))
            uni.append({"col": c, "name": COLNAME.get(c, f"col{c}"),
                        "ic": round(ic, 5), "placebo_ic": round(ic_p, 5)})
        uni.sort(key=lambda r: -abs(r["ic"]))
        uni_top = uni[:12]

        # ---- (b) GROUPED ABLATION -------------------------------------
        liveset = set(live_l)
        grid = []
        for gname, gcols in GROUPS.items():
            g_live = sorted(liveset.intersection(gcols))
            rest = sorted(liveset.difference(gcols))
            if not g_live:
                grid.append({"group": gname, "n_live": 0,
                             "note": "all dead/absent this build"})
                continue
            try:
                p_only = _fit_r1(F, up, w1, g_live, s_tr, s_te)
                auc_only = _auc(yv, p_only)
                plac_only = _auc(rng.permutation(yv), p_only)
            except Exception as e:                   # noqa: BLE001
                grid.append({"group": gname, "err": repr(e)[:90]})
                continue
            if rest:
                p_drop = _fit_r1(F, up, w1, rest, s_tr, s_te)
                auc_drop = _auc(yv, p_drop)
                d_drop = round(auc_full - auc_drop, 4)
                se = _paired_boot(yv, p_full, p_drop, block, rng)
                zd = (d_drop / se) if (se and se > 0) else None
            else:                                    # group == all live
                auc_drop = 0.5
                d_drop = round(auc_full - 0.5, 4)
                se, zd = None, None
            robust = int(d_drop is not None and d_drop > 0
                         and zd is not None and zd > 2
                         and auc_only > 0.5
                         and abs(plac_only - 0.5) < 0.02)
            grid.append({
                "group": gname, "n_live": len(g_live),
                "cols": g_live,
                "auc_only": round(auc_only, 4),
                "placebo_only": round(plac_only, 4),
                "auc_drop": round(auc_drop, 4),
                "d_drop_vs_full": d_drop,
                "se_d": None if not se else round(se, 5),
                "z_drop": None if zd is None else round(zd, 3),
                "robust_local": robust,
            })
        any_robust = any(g.get("robust_local") for g in grid)
        status = "suspect" if any_robust else "exploratory"

        recs.append({
            "experiment_id": f"{run_id}_HA3_{sym}_H{H}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git, "author": "claude(vm)",
            "hypothesis_id": "HA3", "status": status, "kind": "alpha",
            "setup": (f"HA3 pre-reg feature-localization (univariate-IC + "
                      f"grouped-ablation) R1-only XGB ({sym}, H={H}s, "
                      f">=cost scope; testbed==HM6)"),
            "model_family": "xgb",
            "params": {
                "H": H, "stride": STRIDE, "seed": SEED, "block": block,
                "n_tr": ntr, "n_oos": nte, "objective": "R1 (HM5 default)",
                "auc_full": round(auc_full, 4),
                "placebo_full": round(plac_full, 4),
                "hm6_baseline_r1": hm6, "d_vs_hm6": d_hm6,
                "testbed_match_ok": testbed_ok,
                "n_live": len(live_l), "dead_cols": dead_named,
                "ablation_grid": grid, "univariate_top": uni_top,
                "judge": ("HM1: group robust iff DROP z>2 & d>0 & "
                          "ONLY>0.5 & placebo clean & sign consistent "
                          "ALL 4 sym; no auto-confirmed"),
            },
            "data_source": "cryptolake",
            "cache_id": (f"cryptolake_{sym}_features_v1+events_"
                         f"{days[0]}_{days[-1]}"),
            "symbols": [sym], "date_range_start": days[0],
            "date_range_end": days[-1], "n_samples": int(n),
            "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
            "commission_loss_pct": 0.07, "split_method": "honest_val_test",
            "embargo": str(EMB),
            "label_def": ("head2 up-first on >=cost subset (==HM6); R1 "
                          "objective FIXED; ONLY the feature column set "
                          "varied (FULL / per-group ONLY / per-group DROP)"),
            "alpha_target": "feature_localization_dir", "horizon_sec": H,
            "rank_ic_oos": round(auc_full - 0.5, 5),
            "auc_oos": round(auc_full, 4),
            "baseline_ref": (f"HM6 R1 cell {HM6_RUN}_BASELINE_{sym}_H{H} "
                             f"(AUC {hm6}); HM5 testbed-match sentinel"),
            "delta_ic": d_hm6,
            "top_decile_absmove_pct": F_T0 * 100,
            "bot_decile_absmove_pct": F_T0 * 100,
            "cost_floor_pct": F_T0 * 100, "decile_monotonic": None,
            "economic_pass_loose": 0, "economic_pass_strict": 0,
            "n_eff": nte,
            "repro_cmd": (f"python scripts/ha3_screen.py --run-id {run_id} "
                          f"(git {git})"),
            "artifact_path": (f"gs://blackdigital-scalper-data/"
                              f"research_runs/{run_id}/"),
            "note": (f"HA3 feature-localization. FULL AUC={round(auc_full,4)} "
                     f"(HM6 R1 {hm6}, d={d_hm6}, testbed_ok={testbed_ok}). "
                     f"any group local-robust={any_robust}. CONCENTRATED vs "
                     f"DIFFUSE + cross-symbol agreement = post-run ledger "
                     f"decision, NEVER auto-confirmed (HM1). De-risks the "
                     f"HD1 sequence build's input design; not a lever."),
        })
        rb = sum(g.get("robust_local", 0) for g in grid)
        print(f"  H={H}s n_oos={nte} FULL={round(auc_full,4)} "
              f"(HM6 {hm6} d={d_hm6} ok={testbed_ok}) "
              f"top_ic={uni_top[0]['name']}={uni_top[0]['ic']} "
              f"robust_groups={rb} {status}", flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--git-commit", default="unknown")
    ap.add_argument("--symbols", nargs="+",
                    default=["SOL-USDT-PERP", "BTC-USDT-PERP",
                             "ETH-USDT-PERP", "LTC-USDT-PERP"])
    a = ap.parse_args(argv)

    bk = _gcs_bucket()
    from google.cloud import storage
    sc = storage.Client(project="project-26a24ad0-1059-4f73-93b")

    winlo, winhi, psd, avail = _window(bk)
    if not winlo or not any(psd.values()):
        print("FATAL: empty common calendar window", file=sys.stderr)
        return 2
    n_by = {s: len(psd.get(s, [])) for s in a.symbols}
    print(f"[HA3] HM6-aligned calendar window {winlo}..{winhi}; "
          f"per-symbol days={n_by}; full-history={avail}", flush=True)

    out = {"run_id": a.run_id, "screen": "HA3",
           "symbol_set": a.symbols, "window": [winlo, winhi],
           "n_days_per_symbol": n_by,
           "window_rule": "HM6 rev2 common-calendar-range (baseline_360)",
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "records": []}
    for sym in a.symbols:
        try:
            out["records"] += run(bk, sc, sym, psd.get(sym, []),
                                   a.run_id, a.git_commit)
        except Exception as e:                       # noqa: BLE001
            out["records"].append({"symbol": sym, "error": repr(e),
                                   "trace": traceback.format_exc()})
            print(f"[{sym}] ERROR {e}", flush=True)
        bk.blob(f"research_runs/{a.run_id}/results.json"
                ).upload_from_string(json.dumps(out, indent=2, default=str))
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bk.blob(f"research_runs/{a.run_id}/results.json"
            ).upload_from_string(json.dumps(out, indent=2, default=str))
    print("PHASE_B_DONE", a.run_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
