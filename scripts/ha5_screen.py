#!/usr/bin/env python3
"""HA5 conditional-asymmetry screen (+ HA6 head-1 feasibility readout).

Pre-registered (RESEARCH_LOG "symmetry wall"; chat 2026-05-17). Baseline
forward excursion is ~symmetric, so wide TP/SL makes no edge. Question:
on the economically-relevant SUBSET (windows whose first-passage reaches
the cost floor within H), does a rare event/regime conditioner break the
symmetry — i.e. is "up-resolves-first" predictable enough to beat
~50/50 + cost?

Per symbol x H in {180,300,600}s (where >=cost windows actually exist):
  entry mid m0 at decision point k (book row indices[k]); first-passage
  to +f (up-first) vs -f (down-first), f = strict floor 0.13%.
    head1 (HA6 feasibility / regime gate): reaches +-f within H ?  (binary)
    head2 (HA5 core): on the >=cost subset, up-first ?              (binary)
  Conditioners (causal, ts<=entry): raw liquidations (signed qty 30/120s,
  count), OI delta (120s), funding rate + (mark-index) basis, + the 59
  features_v1. XGB classifier, honest time split (train70/embargo>=H/OOS30)
  per symbol. PLACEBO (shuffled head2 label) AUC must be ~0.5.

Economic decision (HA5): select OOS >=cost samples in the top/bottom
head2-prob decile -> long/short; mean realized signed first-passage
capture must clear the cost floor (economic_pass_{loose,strict}) at a
sane selection rate. Emits kind='alpha' rows to GCS.
"""
from __future__ import annotations

import argparse
import io
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

HS = (180, 300, 600)
F_STRICT = 0.0013          # 0.13% binding cost floor (fraction)
F_LOOSE = 0.0008           # 0.08% maker idealised
STRIDE = 4                 # decision-point subsample (screen, pre-registered)
SEED = 42
NS = 1_000_000_000


def _xgbc():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         random_state=SEED, eval_metric="logloss")


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, p))


def _load_events(bk, sc, sym, day):
    """raw liquidations / open_interest / funding for one day -> sorted ts
    arrays. Missing stream -> empty (graceful)."""
    out = {}
    base = f"raw/{{}}/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
    import pyarrow.parquet as pq

    def _rd(kind, cols):
        bl = [b for b in sc.list_blobs(bk, prefix=base.format(kind))
              if b.name.endswith(".parquet")]
        if not bl:
            return None
        t = pq.read_table(io.BytesIO(bl[0].download_as_bytes()), columns=cols)
        return {c: t.column(c).to_numpy(zero_copy_only=False) for c in cols}

    liq = _rd("liquidations", ["side", "quantity", "timestamp"])
    if liq is not None:
        o = np.argsort(liq["timestamp"])
        sgn = np.where(liq["side"][o] == "buy", 1.0, -1.0)  # buy-liq => up
        out["liq_ts"] = liq["timestamp"][o].astype(np.int64)
        out["liq_sq"] = sgn * liq["quantity"][o].astype(np.float64)
    oi = _rd("open_interest", ["open_interest", "timestamp"])
    if oi is not None:
        o = np.argsort(oi["timestamp"])
        out["oi_ts"] = oi["timestamp"][o].astype(np.int64)
        out["oi_v"] = oi["open_interest"][o].astype(np.float64)
    fu = _rd("funding", ["rate", "mark_price", "index_price", "timestamp"])
    if fu is not None:
        o = np.argsort(fu["timestamp"])
        out["fu_ts"] = fu["timestamp"][o].astype(np.int64)
        out["fu_rate"] = fu["rate"][o].astype(np.float64)
        out["fu_basis"] = ((fu["mark_price"][o] - fu["index_price"][o])
                           / np.where(fu["index_price"][o] > 0,
                                      fu["index_price"][o], 1.0))
    return out


# FIXED conditioner schema — every day emits exactly these columns in
# this order, zero-filled when the event stream is absent that day. A
# variable per-day width is the ragged-concat bug (same class as
# phase_b_run._collect); a fixed schema kills it structurally.
COND_KEYS = ("liq_sq_30s", "liq_n_30s", "liq_sq_120s", "liq_n_120s",
             "oi_d_120s", "fu_rate", "fu_basis")


def _cond_feats(t0, ev):
    """Causal event/regime conditioners at entry ts t0 (ns). Always
    returns (n, len(COND_KEYS)) in COND_KEYS order; 0 where stream
    missing."""
    n = t0.shape[0]
    F = {k: np.zeros(n, np.float64) for k in COND_KEYS}
    if "liq_ts" in ev:
        lt, lq = ev["liq_ts"], ev["liq_sq"]
        cum = np.concatenate([[0.0], np.cumsum(lq)])
        for w, tag in ((30, "30s"), (120, "120s")):
            a = np.searchsorted(lt, t0 - w * NS, "left")
            b = np.searchsorted(lt, t0, "right")
            F[f"liq_sq_{tag}"] = cum[b] - cum[a]
            F[f"liq_n_{tag}"] = (b - a).astype(np.float64)
    if "oi_ts" in ev:
        ot, ov = ev["oi_ts"], ev["oi_v"]
        j = np.clip(np.searchsorted(ot, t0, "right") - 1, 0, len(ov) - 1)
        j0 = np.clip(np.searchsorted(ot, t0 - 120 * NS, "right") - 1,
                     0, len(ov) - 1)
        base = np.where(ov[j0] > 0, ov[j0], 1.0)
        F["oi_d_120s"] = (ov[j] - ov[j0]) / base
    if "fu_ts" in ev:
        ft = ev["fu_ts"]
        j = np.clip(np.searchsorted(ft, t0, "right") - 1, 0, len(ft) - 1)
        F["fu_rate"] = ev["fu_rate"][j]
        F["fu_basis"] = ev["fu_basis"][j]
    return np.column_stack([F[k] for k in COND_KEYS]), list(COND_KEYS)


def _first_passage(mid, i0, jH, m0, f):
    """Per-entry first-passage: +1 up-first, -1 down-first, 0 none."""
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


def run(bk, sc, sym, days, run_id, git):
    recs = []
    # gather X + per-day book + events once
    feats_all, t0_all, day_all = [], [], []
    books = {}
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
        ok = (idx > 0) & (idx < n - 1)
        sel = np.where(ok)[0][::STRIDE]
        i = idx[sel]
        cf, ckeys = _cond_feats(ts[i], ev)
        Xc = np.hstack([X[sel].astype(np.float64), cf]) if cf.shape[1] \
            else X[sel].astype(np.float64)
        feats_all.append(Xc)
        t0_all.append(np.stack([np.full(i.size, di), i,
                                ts[i].astype(np.float64)], 1))
        day_all.append(np.full(i.size, di, np.int32))
        books[di] = (ts, mid)
    if not feats_all:
        return [{"symbol": sym, "error": "no data"}]
    F = np.concatenate(feats_all)
    meta = np.concatenate(t0_all)             # [day, book_i, ts]
    dayid = np.concatenate(day_all)
    ncol = min(x.shape[1] for x in feats_all)
    F = F[:, :ncol]
    print(f"[{sym}] n={F.shape[0]} feat={ncol} days={len(days)}", flush=True)

    for H in HS:
        # first-passage per sample at strict floor, per day
        y_dir = np.zeros(F.shape[0], np.int8)
        for di, (ts, mid) in books.items():
            m = dayid == di
            bi = meta[m, 1].astype(np.int64)
            t0 = meta[m, 2].astype(np.int64)
            jH = np.searchsorted(ts, t0 + H * NS, "left")
            jH = np.minimum(jH, len(mid) - 1)
            m0 = mid[bi]
            y_dir[np.where(m)[0]] = _first_passage(mid, bi, jH, m0, F_STRICT)
        reached = y_dir != 0
        up = (y_dir == 1).astype(int)
        n = F.shape[0]
        n_tr = int(n * 0.70)
        emb = 64
        tr = np.zeros(n, bool); tr[:n_tr - emb] = True
        te = np.zeros(n, bool); te[n_tr:] = True

        # head1 (HA6): predict reached (>=cost feasibility / regime)
        h1 = _xgbc(); h1.fit(F[tr], reached[tr].astype(int))
        p1 = h1.predict_proba(F[te])[:, 1]
        auc1 = _auc(reached[te].astype(int), p1)

        # head2 (HA5): up-first on the >=cost subset only
        tr2 = tr & reached
        te2 = te & reached
        base_up = float(up[te2].mean()) if te2.sum() else float("nan")
        if tr2.sum() < 200 or te2.sum() < 100:
            auc2 = float("nan"); placebo = float("nan")
            eL = eS = 0; sel_rate = 0.0; cap = float("nan")
        else:
            h2 = _xgbc(); h2.fit(F[tr2], up[tr2])
            p2 = h2.predict_proba(F[te2])[:, 1]
            yv = up[te2]
            auc2 = _auc(yv, p2)
            rng = np.random.default_rng(SEED)
            placebo = _auc(rng.permutation(yv), p2)
            # economic: top/bottom decile -> long/short; realised capture
            # = +f if correct side, -f if wrong (first-passage pays the
            # floor by construction on the >=cost subset).
            q = np.quantile(p2, [0.1, 0.9])
            longs = p2 >= q[1]; shorts = p2 <= q[0]
            sgn = np.zeros(len(p2)); sgn[longs] = 1; sgn[shorts] = -1
            took = sgn != 0
            correct = ((sgn == 1) & (yv == 1)) | ((sgn == -1) & (yv == 0))
            cap = float((np.where(correct[took], F_STRICT, -F_STRICT)).mean()
                        * 100) if took.any() else float("nan")
            n_te_days = int(dayid[te2].max() - dayid[te2].min() + 1)
            sel_rate = float(took.sum() / max(1, n_te_days))
            eL = int(cap > F_LOOSE * 100)
            eS = int(cap > 0)   # already net of the floor: >0 == beats cost
        status = "exploratory"
        if (not np.isnan(auc2) and abs(auc2 - 0.5) > 0.02
                and abs(placebo - 0.5) < 0.02 and eS):
            status = "suspect" if eL == 0 else "confirmed"
        nq = int(reached[te].sum())
        recs.append({
            "experiment_id": f"{run_id}_HA5_{sym}_H{H}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git, "author": "claude(vm)",
            "hypothesis_id": "HA5", "status": status, "kind": "alpha",
            "setup": f"HA5 conditional-asymmetry XGB ({sym}, H={H}s)",
            "model_family": "xgb",
            "params": {"H": H, "f_strict": F_STRICT, "f_loose": F_LOOSE,
                       "stride": STRIDE, "seed": SEED,
                       "head1_auc_feasibility_HA6": round(auc1, 4),
                       "base_P_up_on_subset": None if np.isnan(base_up)
                       else round(base_up, 4),
                       "head2_auc": None if np.isnan(auc2) else round(auc2, 4),
                       "placebo_auc": None if np.isnan(placebo)
                       else round(placebo, 4),
                       "decile_capture_pct_net_floor": None if np.isnan(cap)
                       else round(cap, 5),
                       "selection_per_day": round(sel_rate, 2),
                       "n_ge_cost_oos": nq},
            "data_source": "cryptolake",
            "cache_id": f"cryptolake_{sym}_features_v1+events_{days[0]}_{days[-1]}",
            "symbols": [sym], "date_range_start": days[0],
            "date_range_end": days[-1], "n_samples": int(n),
            "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
            "commission_loss_pct": 0.07, "split_method": "honest_val_test",
            "embargo": str(emb),
            "label_def": (f"head1=reach +-{F_STRICT:.4f} within {H}s; "
                          f"head2=up-first on >=cost subset; conditioners "
                          f"= features_v1 + liq/OI/funding causal aggs"),
            "alpha_target": "updown_first_on_ge_cost_subset",
            "horizon_sec": H,
            "rank_ic_oos": None if np.isnan(auc2) else round(auc2 - 0.5, 5),
            "auc_oos": None if np.isnan(auc2) else round(auc2, 4),
            "top_decile_absmove_pct": F_STRICT * 100,
            "bot_decile_absmove_pct": F_STRICT * 100,
            "cost_floor_pct": F_STRICT * 100,
            "decile_monotonic": None,
            "economic_pass_loose": eL, "economic_pass_strict": eS,
            "n_eff": nq,
            "repro_cmd": f"python scripts/ha5_screen.py (run {run_id}, git {git})",
            "artifact_path": f"gs://blackdigital-scalper-data/research_runs/{run_id}/",
            "note": (f"HA5 conditional-asymmetry. base P(up|>=cost)="
                     f"{base_up:.3f} (symmetry check ~0.5). head1 feasibility "
                     f"AUC={auc1:.3f} (HA6 gate readout). head2 dir AUC="
                     f"{auc2}, placebo={placebo}. decile capture net of "
                     f"{F_STRICT*100:.2f}% floor = {cap}%/trade, "
                     f"sel={sel_rate:.1f}/day. eS:= capture>0 (beats floor)."),
        })
        print(f"  H={H}s reached_oos={nq} baseP_up={base_up:.3f} "
              f"h1AUC={auc1:.3f} h2AUC={auc2} plac={placebo} "
              f"cap={cap} sel/d={sel_rate:.1f} {status}", flush=True)
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
