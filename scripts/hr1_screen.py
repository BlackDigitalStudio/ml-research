#!/usr/bin/env python3
"""HR1 — pre-registered REWARD/LOSS-STRUCTURE sweep (HM4 open axis).

Pre-registered 2026-05-17 (RESEARCH_LOG "HM4"; user AskUserQuestion:
all four R1-R4, HM1-standard bar). FROZEN — no post-hoc DOF.

HM4: the training reward/loss structure (how prediction error is scored
& weighted) is a DISTINCT axis from input representation (HM3) and from
scope-as-subset/label/gating (HA7), and was sampled at only ~3 points
(MSE-return HA1, logloss-sign HA2a, MSE-volnorm HA2b); HA7 added ZERO.
This screen sweeps the genuinely-uncovered reward points on the SAME
testbed and economically-relevant scope (>=cost subset) as HA2/HA7 R0.

Per (sym in {LINK,SOL}, H in {180,300,600}s): F = features_v1(59) +
HA5 causal conds(7). Honest time split (train70/emb/OOS30). Scope =
the >=cost subset (reached at +-0.13% within H) — where direction is
economically relevant (HA5/HA7). Target = up-first (== HA2a/HA7 R0).
The ONLY thing varied is the reward/loss structure:

  R0  plain logloss, no weights         (== HA2a/HA7 baseline; anchor)
  R1  logloss, sample_weight = |r_H|     (clip p99) — punish wrong
      direction proportional to the move that mattered
  R2  logloss, sample_weight = max(|r_H|-cost,0) — economic: sub-cost
      moves get ~0 weight ("only learn from trades that would pay")
  R3  rank:pairwise on up label (XGBRanker, 1 group) — reward correct
      ORDER of P(up) (the IC/AUC surrogate HM2-rev1 named, never run)
  R4  logloss, asymmetric class weight up:down = 0.20:0.13 (the T2_asym
      ratio — HA7's lone non-null thread, put INTO the loss)

HM1-standard bar (pre-registered, NOT economic-gated, no auto-
'confirmed'): a reward point is a ROBUST marginal iff paired block-
bootstrap (AUC_R - AUC_R0) > 2*SE AND > 0 AND placebo within +-0.02 of
0.5 AND it beats R0 by >noise on BOTH symbols. Cross-symbol agreement
is a post-run ledger decision; auto-status caps at 'suspect'.
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

HS = (180, 300, 600)
F_T0 = 0.0013          # 0.13% strict round-trip floor (scope + R2 cost)
R4_UP, R4_DN = 0.0013, 0.0020   # T2_asym barriers -> R4 weight ratio
STRIDE = 4
SEED = 42
NS = 1_000_000_000
NV1 = 59
B_BOOT = 300
N_TR_FLOOR, N_OOS_FLOOR = 300, 200


def _xgb_ranker():
    from xgboost import XGBRanker
    return XGBRanker(objective="rank:pairwise", n_estimators=300,
                     max_depth=4, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, n_jobs=-1,
                     random_state=SEED, verbosity=0)


def _fit_score(reward, F, y, w, tr, te):
    """Train one reward variant on tr, return OOS scores on te."""
    if reward == "R3":
        m = _xgb_ranker()
        m.fit(F[tr], y[tr], group=[int(tr.sum())])
        return m.predict(F[te])
    m = _xgbc()
    if w is None:
        m.fit(F[tr], y[tr])
    else:
        m.fit(F[tr], y[tr], sample_weight=w[tr])
    return m.predict_proba(F[te])[:, 1]


def _paired_boot(y, p_r, p_0, block, rng):
    """Block-bootstrap SE of (AUC_R - AUC_R0) on the SAME resamples."""
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
        d.append(_auc(yi, p_r[ix]) - _auc(yi, p_0[ix]))
    return float(np.std(d)) if len(d) > 30 else float("nan")


def run(bk, sc, sym, days, run_id, git):
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
    emb = 64
    tr = np.zeros(n, bool); tr[:n_tr - emb] = True
    te = np.zeros(n, bool); te[n_tr:] = True
    print(f"[{sym}] n={n} feat={ncol} days={len(days)} "
          f"tr={int(tr.sum())} te={int(te.sum())}", flush=True)

    recs = []
    for H in HS:
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

        if ntr < N_TR_FLOOR or nte < N_OOS_FLOOR or up[s_tr].min() == up[s_tr].max():
            recs.append({"symbol": sym, "horizon_sec": H,
                         "error": f"underpowered n_tr={ntr} n_oos={nte}"})
            print(f"  H={H}s underpowered n_tr={ntr} n_oos={nte}", flush=True)
            continue

        amove = np.abs(rH)
        wcap = np.nanquantile(amove[s_tr], 0.99)
        w1 = np.clip(amove, 0, wcap)
        w2 = np.maximum(amove - F_T0, 0.0) + 1e-9
        w4 = np.where(up == 1, R4_DN / R4_UP, 1.0).astype(float)
        weights = {"R0": None, "R1": w1, "R2": w2, "R3": None, "R4": w4}

        yv = up[s_te]
        scores, grid = {}, []
        for R in ("R0", "R1", "R2", "R3", "R4"):
            try:
                p = _fit_score(R, F, up, weights[R], s_tr, s_te)
            except Exception as e:
                grid.append({"reward": R, "auc": None, "err": repr(e)[:90]})
                continue
            scores[R] = p
            auc = _auc(yv, p)
            plac = _auc(rng.permutation(yv), p)
            grid.append({"reward": R, "auc": round(auc, 4),
                         "placebo": round(plac, 4)})
        a0 = next((g["auc"] for g in grid if g["reward"] == "R0"), None)
        for g in grid:
            R = g["reward"]
            if R == "R0" or g.get("auc") is None or a0 is None:
                g.update(d_auc=None, se_d=None, z_d=None, robust_local=0)
                continue
            d = round(g["auc"] - a0, 4)
            se = _paired_boot(yv, scores[R], scores["R0"], block, rng)
            zd = (d / se) if (se and se > 0) else None
            g.update(d_auc=d,
                     se_d=None if not se else round(se, 5),
                     z_d=None if zd is None else round(zd, 3),
                     robust_local=int(d > 0 and zd is not None and zd > 2
                                      and g.get("placebo") is not None
                                      and abs(g["placebo"] - 0.5) < 0.02))
        best = max((g for g in grid if g.get("auc") is not None),
                   key=lambda g: g["auc"], default=None)
        any_robust = any(g.get("robust_local") for g in grid)
        best_auc = best["auc"] if best else 0.5
        best_R = best["reward"] if best else None
        d_best = (None if (best is None or a0 is None or best_R == "R0")
                  else round(best_auc - a0, 4))
        status = "suspect" if any_robust else "exploratory"
        recs.append({
            "experiment_id": f"{run_id}_HR1_{sym}_H{H}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git, "author": "claude(vm)",
            "hypothesis_id": "HR1", "status": status, "kind": "alpha",
            "setup": (f"HR1 pre-reg reward/loss sweep R0-R4 XGB "
                      f"({sym}, H={H}s, >=cost scope)"),
            "model_family": "xgb",
            "params": {"H": H, "stride": STRIDE, "seed": SEED,
                       "block": block, "n_tr": ntr, "n_oos": nte,
                       "R0_auc": a0, "best_reward": best_R,
                       "best_auc": best_auc, "any_robust_local": int(any_robust),
                       "judge": "HM1 paired-boot d_auc>2*SE & both-sym",
                       "grid": grid},
            "data_source": "cryptolake",
            "cache_id": f"cryptolake_{sym}_features_v1+events_{days[0]}_{days[-1]}",
            "symbols": [sym], "date_range_start": days[0],
            "date_range_end": days[-1], "n_samples": int(n),
            "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
            "commission_loss_pct": 0.07, "split_method": "honest_val_test",
            "embargo": str(emb),
            "label_def": ("head2 up-first on >=cost subset; REWARD/LOSS "
                          "swept R0 logloss / R1 |move|-wt / R2 econ-wt / "
                          "R3 rank:pairwise / R4 asym-cost; same features"),
            "alpha_target": "reward_loss_sweep_dir", "horizon_sec": H,
            "rank_ic_oos": round(best_auc - 0.5, 5),
            "auc_oos": best_auc,
            "baseline_ref": (f"in-run R0 logloss head2 ({a0}) == "
                             f"HA2a/HA7 ~0.51; HA1 baseline_ref"),
            "delta_ic": d_best,
            "top_decile_absmove_pct": F_T0 * 100,
            "bot_decile_absmove_pct": F_T0 * 100,
            "cost_floor_pct": F_T0 * 100, "decile_monotonic": None,
            "economic_pass_loose": 0, "economic_pass_strict": 0,
            "n_eff": nte,
            "repro_cmd": f"python scripts/hr1_screen.py (run {run_id} git {git})",
            "artifact_path": f"gs://blackdigital-scalper-data/research_runs/{run_id}/",
            "note": (f"HR1 reward/loss sweep. R0(logloss)={a0}. "
                     f"best={best_R} AUC={best_auc} (d vs R0 {d_best}). "
                     f"any local robust={any_robust}. HM1 bar: 'confirmed' "
                     f"needs the SAME R beat R0 by >noise on BOTH LINK&SOL "
                     f"+ placebo clean — post-run ledger decision, never "
                     f"auto. Tests HM4's open reward axis, NOT scope/repr."),
        })
        rb = sum(g.get("robust_local", 0) for g in grid)
        print(f"  H={H}s n_oos={nte} R0={a0} best={best_R}={best_auc} "
              f"dR0={d_best} robust={rb} {status}", flush=True)
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
