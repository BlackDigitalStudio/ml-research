#!/usr/bin/env python3
"""HA7 — pre-registered SCOPE sweep for head2 direction (strict bar).

Pre-registered 2026-05-17 (RESEARCH_LOG "HA7"; user design + AskUserQuestion
choices: all 3 axes, strict Bonferroni + both-symbol + placebo). FROZEN —
no post-hoc degrees of freedom.

Premise (user): scope is a DIRECT lever on conditional predictability, not
reducible to objective tuning. HA1/HA5/HA2 tested essentially ONE scope
point (pooled >=cost subset). Prior caveat (recorded, not a kill): for a
GBT, "include C as a feature" >= "hard-subset on C" (the tree can carve the
conditional), and HA5/HA2 already included liq/OI/funding/features_v1 as
features at ~chance dir-AUC -> the expected-positive case is NARROW:
(A) a directionally-predictable sub-regime too rare to move the pooled
logloss, (B) a different target/barrier definition, (C) head1->head2
cascade coupling. HA7 tests exactly those three uncovered axes.

Per (sym in {LINK,SOL}, H in {180,300,600}s), F = features_v1(59) +
HA5 causal event conds(7). Honest time split (train70 / emb / OOS30) on
the full strided timeline; regime/target thresholds from TRAIN ONLY.
Control T0 = up-first +-0.13% within H on the reached subset (== HA1/HA2/
HA5 pooled-head2 baseline; emitted in-run so delta_ic is marginal vs it).

  AXIS A  regime-bucket head2 (heterogeneity): 11 cells from causal conds
          + ts-hour — liq-burst(2), |basis| tercile(3), OI-shock(2),
          time-of-day(4). head2 trained & eval WITHIN each cell.
  AXIS B  alternative barrier targets (label-scope): T0 +-0.13%,
          T1 +-0.25% (large move), T2 asym +0.13/-0.20, T3 signed r_H
          with |r_H|>0.13% deadband. 4 cells.
  AXIS C  head1-gated cascade: head1(reached) trained on TRAIN; operating
          point = train reached-rate quantile; head2 trained & eval on
          head1's PREDICTED-feasible set (realistic deploy scope). 1 cell.

Strict pre-registered bar (no auto-'confirmed', HM1): a cell PASSES iff
block-bootstrap |AUC-0.5|/SE exceeds the Bonferroni z* (alpha 0.05 / M,
M = #cells meeting the n-floor n_tr>=300 & n_oos>=200 for this sym,H),
placebo within +-0.02 of 0.5, AUC>0.5 — AND the SAME (axis,cell) passes
on BOTH symbols (cross-symbol agreement is a post-run ledger decision,
NEVER auto-set per-row). Auto-status caps at 'suspect'.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from math import sqrt
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_cryptolake_cache import (  # noqa: E402
    _gcs_bucket, _list_days, _load_npy, _load_book_l1)
from scripts.ha5_screen import (  # noqa: E402  (one source of truth)
    COND_KEYS, _auc, _cond_feats, _first_passage, _load_events, _xgbc)

HS = (180, 300, 600)
F_T0 = 0.0013          # 0.13% strict round-trip floor (control barrier)
F_T1 = 0.0025          # 0.25% large-move barrier
F_T2U, F_T2D = 0.0013, 0.0020   # asymmetric up/down barrier
DEADBAND = 0.0013      # T3 |r_H| deadband
STRIDE = 4
SEED = 42
NS = 1_000_000_000
NV1 = 59               # features_v1 width (cond block follows)
N_TR_FLOOR, N_OOS_FLOOR = 300, 200
B_BOOT = 300
ALPHA_FAM = 0.05


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam) — stdlib-only, no scipy on the VM."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    pl = 0.02425
    if p < pl:
        q = sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= 1 - pl:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                                + b[4]) * r + 1)
    q = sqrt(-2 * np.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _fp_asym(mid, i0, jH, m0, fu, fd):
    """First-passage with asymmetric up/down barriers: +1 up-first
    (>= m0*(1+fu)), -1 down-first (<= m0*(1-fd)), 0 neither within H."""
    up = m0 * (1.0 + fu)
    dn = m0 * (1.0 - fd)
    out = np.zeros(i0.shape[0], np.int8)
    for r in range(i0.shape[0]):
        s = mid[i0[r] + 1: jH[r] + 1]
        if s.size == 0:
            continue
        u = np.argmax(s >= up[r]) if (s >= up[r]).any() else -1
        d = np.argmax(s <= dn[r]) if (s <= dn[r]).any() else -1
        if u < 0 and d < 0:
            continue
        out[r] = 1 if (d < 0 or (u >= 0 and u <= d)) else -1
    return out


def _boot_auc_se(y, p, block, rng):
    """Block-bootstrap SE of AUC (autocorr-aware). NaN if degenerate."""
    y = np.asarray(y)
    n = len(y)
    if n < block * 2 or y.min() == y.max():
        return float("nan")
    starts = np.arange(0, n - block + 1)
    nb = max(1, n // block)
    vals = []
    for _ in range(B_BOOT):
        s = rng.choice(starts, nb, replace=True)
        ix = np.concatenate([np.arange(x, x + block) for x in s])
        yi = y[ix]
        if yi.min() == yi.max():
            continue
        vals.append(_auc(yi, p[ix]))
    return float(np.std(vals)) if len(vals) > 30 else float("nan")


def _eval_cell(F, y, tr_m, te_m, block, rng):
    """Train head2 (logloss XGB) on tr_m, eval te_m. -> dict or None
    (under-powered / degenerate)."""
    ntr, nte = int(tr_m.sum()), int(te_m.sum())
    if ntr < N_TR_FLOOR or nte < N_OOS_FLOOR:
        return {"n_tr": ntr, "n_oos": nte, "auc": None,
                "placebo": None, "se": None, "z": None,
                "passed_local": 0, "powered": 0}
    ytr = y[tr_m]
    if ytr.min() == ytr.max():
        return {"n_tr": ntr, "n_oos": nte, "auc": None, "placebo": None,
                "se": None, "z": None, "passed_local": 0, "powered": 0}
    m = _xgbc()
    m.fit(F[tr_m], ytr)
    p = m.predict_proba(F[te_m])[:, 1]
    yv = y[te_m]
    auc = _auc(yv, p)
    plac = _auc(rng.permutation(yv), p)
    se = _boot_auc_se(yv, p, block, rng)
    z = (abs(auc - 0.5) / se) if (se and se > 0) else None
    return {"n_tr": ntr, "n_oos": nte, "auc": round(auc, 4),
            "placebo": round(plac, 4),
            "se": None if not se else round(se, 5),
            "z": None if z is None else round(z, 3),
            "passed_local": 0, "powered": 1}


def run(bk, sc, sym, days, run_id, git):
    # ---- gather X+cond, per-day book, meta (HA5 pattern) ----
    feats, meta, dayid, books = [], [], [], {}
    for di, day in enumerate(days):
        try:
            X = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/features.npy")
            idx = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/indices.npy"
                            ).astype(np.int64)
            ts, b, a = _load_book_l1(bk, sym, day)
            ev = _load_events(bk, sc, sym, day)
        except Exception as e:
            print(f"  {sym} {day}: skip ({type(e).__name__})", flush=True)
            continue
        mid = (b + a) * 0.5
        n = ts.shape[0]
        sel = np.where((idx > 0) & (idx < n - 1))[0][::STRIDE]
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
    emb = 64
    tr = np.zeros(n, bool); tr[:n_tr - emb] = True
    te = np.zeros(n, bool); te[n_tr:] = True
    # cond columns by name (block starts at NV1)
    ci = {k: NV1 + j for j, k in enumerate(COND_KEYS)}
    liq30 = np.abs(F[:, ci["liq_sq_30s"]])
    basis = np.abs(F[:, ci["fu_basis"]])
    oid = np.abs(F[:, ci["oi_d_120s"]])
    hour = ((M[:, 2].astype(np.int64) // NS) % 86400) // 3600
    tod = (hour // 6).astype(int)              # 0..3
    print(f"[{sym}] n={n} feat={ncol} days={len(days)} "
          f"tr={int(tr.sum())} te={int(te.sum())}", flush=True)

    recs = []
    for H in HS:
        # per-day labels: y0(+-0.13), y1(+-0.25), y2(asym), rH
        y0 = np.zeros(n, np.int8); y1 = np.zeros(n, np.int8)
        y2 = np.zeros(n, np.int8); rH = np.full(n, np.nan)
        for di, (ts, mid) in books.items():
            mr = np.where(dy == di)[0]
            bi = M[mr, 1].astype(np.int64)
            t0 = M[mr, 2].astype(np.int64)
            jH = np.minimum(np.searchsorted(ts, t0 + H * NS, "left"),
                            len(mid) - 1)
            m0 = mid[bi]
            y0[mr] = _first_passage(mid, bi, jH, m0, F_T0)
            y1[mr] = _first_passage(mid, bi, jH, m0, F_T1)
            y2[mr] = _fp_asym(mid, bi, jH, m0, F_T2U, F_T2D)
            rH[mr] = np.log(np.where(m0 > 0, mid[jH] / m0, np.nan))
        block = max(1, int(np.ceil(H / (STRIDE * 24))))
        rng = np.random.default_rng(SEED)

        reached0 = y0 != 0
        up0 = (y0 == 1).astype(int)
        tr2, te2 = tr & reached0, te & reached0
        grid = []

        # ---- AXIS B: alternative barrier targets ----
        for tag, sub, yy in (
                ("T0_pm0.13", reached0, up0),
                ("T1_pm0.25", y1 != 0, (y1 == 1).astype(int)),
                ("T2_asym0.13/0.20", y2 != 0, (y2 == 1).astype(int)),
                ("T3_signed_db", np.isfinite(rH) & (np.abs(rH) > DEADBAND),
                 (rH > 0).astype(int))):
            c = _eval_cell(F, yy, tr & sub, te & sub, block, rng)
            c.update(axis="B_target", cell=tag)
            grid.append(c)
        t0_auc = next((g["auc"] for g in grid
                       if g["cell"] == "T0_pm0.13"), None)

        # ---- AXIS A: regime-bucket head2 (thresholds TRAIN-only) ----
        def _p(v, q):
            return float(np.quantile(v[tr2], q)) if tr2.sum() else float("nan")
        regimes = {
            "liqburst": np.where(liq30 > _p(liq30, 0.80), 1, 0),
            "oishock": np.where(oid > _p(oid, 0.80), 1, 0),
            "tod": tod,
        }
        bq = np.quantile(basis[tr2], [1 / 3, 2 / 3]) if tr2.sum() else [0, 0]
        regimes["basis3"] = np.digitize(basis, bq)        # 0,1,2
        for rname, lab in regimes.items():
            for bct in np.unique(lab):
                cell = (tr2 & (lab == bct), te2 & (lab == bct))
                c = _eval_cell(F, up0, cell[0], cell[1], block, rng)
                c.update(axis="A_regime", cell=f"{rname}={int(bct)}")
                grid.append(c)

        # ---- AXIS C: head1-gated cascade (realistic deploy scope) ----
        try:
            h1 = _xgbc(); h1.fit(F[tr], reached0[tr].astype(int))
            p1 = h1.predict_proba(F)[:, 1]
            thr = float(np.quantile(p1[tr], 1.0 - reached0[tr].mean()))
            P = p1 >= thr
            c = _eval_cell(F, up0, tr & P, te & P, block, rng)
        except Exception as e:
            c = {"n_tr": 0, "n_oos": 0, "auc": None, "placebo": None,
                 "se": None, "z": None, "passed_local": 0, "powered": 0,
                 "err": repr(e)[:80]}
        c.update(axis="C_cascade", cell="head1_pred_feasible")
        grid.append(c)

        # ---- Bonferroni over powered cells (this sym,H) ----
        powered = [g for g in grid if g.get("powered")]
        Mb = max(1, len(powered))
        z_star = _norm_ppf(1.0 - ALPHA_FAM / (2.0 * Mb))
        for g in grid:
            if (g.get("powered") and g["auc"] is not None
                    and g["z"] is not None and g["auc"] > 0.5
                    and g["z"] > z_star
                    and g["placebo"] is not None
                    and abs(g["placebo"] - 0.5) < 0.02):
                g["passed_local"] = 1
        best = max((g for g in grid if g["auc"] is not None),
                   key=lambda g: g["auc"], default=None)
        any_pass = any(g["passed_local"] for g in grid)
        best_auc = best["auc"] if best else 0.5
        d_ic = (None if (best is None or t0_auc is None)
                else round(best_auc - t0_auc, 4))
        # HM1: never auto-'confirmed'. Cross-symbol agreement is a post-run
        # ledger decision. Single-symbol Bonferroni pass caps at 'suspect'.
        status = "suspect" if any_pass else "exploratory"
        n_oos_ge = int(te2.sum())
        recs.append({
            "experiment_id": f"{run_id}_HA7_{sym}_H{H}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git, "author": "claude(vm)",
            "hypothesis_id": "HA7", "status": status, "kind": "alpha",
            "setup": f"HA7 pre-reg scope sweep (3 axes) XGB ({sym}, H={H}s)",
            "model_family": "xgb",
            "params": {"H": H, "stride": STRIDE, "seed": SEED,
                       "block": block, "M_bonferroni": Mb,
                       "z_star": round(z_star, 4),
                       "control_T0_auc": t0_auc,
                       "best_cell": None if best is None else best["cell"],
                       "best_auc": best_auc, "any_pass_local": int(any_pass),
                       "n_floor": [N_TR_FLOOR, N_OOS_FLOOR],
                       "grid": grid},
            "data_source": "cryptolake",
            "cache_id": f"cryptolake_{sym}_features_v1+events_{days[0]}_{days[-1]}",
            "symbols": [sym], "date_range_start": days[0],
            "date_range_end": days[-1], "n_samples": int(n),
            "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
            "commission_loss_pct": 0.07, "split_method": "honest_val_test",
            "embargo": str(emb),
            "label_def": ("head2 up-first; SCOPE swept 3 axes "
                          "(A regime-bucket / B alt-barrier / C head1-gated); "
                          "directional logloss; train-only thresholds"),
            "alpha_target": "scope_sweep_dir", "horizon_sec": H,
            "rank_ic_oos": round(best_auc - 0.5, 5),
            "auc_oos": best_auc,
            "baseline_ref": (f"in-run T0 pooled >=cost head2 ({t0_auc}) "
                             f"== HA1/HA2 ~0.51 ceiling"),
            "delta_ic": d_ic,
            "top_decile_absmove_pct": F_T0 * 100,
            "bot_decile_absmove_pct": F_T0 * 100,
            "cost_floor_pct": F_T0 * 100, "decile_monotonic": None,
            "economic_pass_loose": 0, "economic_pass_strict": 0,
            "n_eff": n_oos_ge,
            "repro_cmd": f"python scripts/ha7_screen.py (run {run_id} git {git})",
            "artifact_path": f"gs://blackdigital-scalper-data/research_runs/{run_id}/",
            "note": (f"HA7 scope sweep. M={Mb} powered cells, z*={z_star:.3f} "
                     f"(Bonferroni a={ALPHA_FAM}). T0 baseline AUC={t0_auc}. "
                     f"best={None if best is None else best['cell']} "
                     f"AUC={best_auc} (delta vs T0 {d_ic}). "
                     f"any local pass={any_pass}. STRICT bar: 'confirmed' "
                     f"requires the SAME cell pass on BOTH LINK&SOL + placebo "
                     f"clean — post-run ledger decision, never auto (HM1)."),
        })
        np_pass = sum(g["passed_local"] for g in grid)
        print(f"  H={H}s M={Mb} z*={z_star:.2f} T0={t0_auc} "
              f"best={None if best is None else best['cell']}={best_auc} "
              f"dIC={d_ic} pass={np_pass} {status}", flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+",
                    default=["LINK-USDT-PERP", "SOL-USDT-PERP"])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--git-commit", default="unknown")
    a = ap.parse_args(argv)
    bk = _gcs_bucket()
    from google.cloud import storage
    sc = storage.Client(project="project-26a24ad0-1059-4f73-93b")
    out = {"run_id": a.run_id, "records": [],
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for sym in a.symbols:
        try:
            days = _list_days(bk, sym)[-a.days:]
            out["records"] += run(bk, sc, sym, days, a.run_id, a.git_commit)
        except Exception as e:
            out["records"].append({"symbol": sym, "error": repr(e),
                                    "trace": traceback.format_exc()})
            print(f"[{sym}] ERROR {e}", flush=True)
        bk.blob(f"research_runs/{a.run_id}/results.json").upload_from_string(
            json.dumps(out, indent=2, default=str))
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bk.blob(f"research_runs/{a.run_id}/results.json").upload_from_string(
        json.dumps(out, indent=2, default=str))
    print("PHASE_B_DONE", a.run_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
