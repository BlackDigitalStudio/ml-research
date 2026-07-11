#!/usr/bin/env python3
"""HA2 — directional-objective two-head screen (+BTC+ETH cross-asset).

Pre-registered 2026-05-17 (RESEARCH_LOG HM2; user design). Fixes every
identified flaw at once: OBJECTIVE rewards direction (not MSE-magnitude),
per-head training SCOPE differs, and the cross-asset dimension is real
input (BTC+ETH raw), not the dead features_v1 slots.

Per (alt sym in {LINK,SOL}, H in {180,300,600}s; strict cost floor
0.13%, where the >=cost subset meaningfully exists per the MFE study):
  first-passage up_first in {+1 up hits +f first, -1 down first, 0 none}.
  head1 "sufficient move": scope=ALL pts, target=reached(|fp|!=0),
        XGBClassifier logloss -> AUC (volatility gate; strong expected).
  head2 "direction": scope=>=cost subset ONLY (train & eval), target=
        up_first==+1, XGBClassifier logloss -> directional AUC. Two
        variants for clean cross-asset attribution (HM1): base =
        features_v1 only; xasset = +BTC-lead(8)+ETH-lead(8). placebo
        (shuffled) sentinel. economic capture recorded, NOT gating (HM1).
Baseline_ref = HA1 sign-AUC ceiling (~0.52). Pre-registered success:
xasset head2 AUC > 0.52 on BOTH symbols, placebo~0.5, delta vs base
> noise. Fixed 16-col xasset schema (0 if missing) -> no ragged concat.
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
F_LOOSE, F_STRICT = 0.08, 0.13          # % round-trip floors
F_FRAC = F_STRICT / 100.0
STRIDE = 4                              # decision-point subsample (pre-reg)
SEED = 42
NS = 1_000_000_000
XA = {"BTC": "BTC-USDT-PERP", "ETH": "ETH-USDT-PERP"}
LEAD_KEYS = ("ret_5s", "ret_30s", "ret_60s", "ret_120s",
             "flow_30s", "flow_60s", "rv_60s", "cumsgn_60s")


def _xgbc():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         random_state=SEED, eval_metric="logloss")


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return 0.5 if y.min() == y.max() else float(roc_auc_score(y, p))


def _load_trades(bk, sc, sym, day):
    import pyarrow.parquet as pq
    pref = f"raw/trades/exchange=BINANCE_FUTURES/symbol={sym}/dt={day}/"
    bl = [b for b in sc.list_blobs(bk, prefix=pref)
          if b.name.endswith(".parquet")]
    if not bl:
        return None
    t = pq.read_table(io.BytesIO(bl[0].download_as_bytes()),
                      columns=["side", "amount", "timestamp"])
    ts = t.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
    amt = t.column("amount").to_numpy(zero_copy_only=False).astype(np.float64)
    sgn = np.where(t.column("side").to_numpy(zero_copy_only=False) == "buy",
                   1.0, -1.0)
    o = np.argsort(ts)
    return ts[o], (amt * sgn)[o]


def _lead(t0, bts, bmid, tr):
    """Causal cross-asset lead features at alt entry ts t0. Fixed 8-col
    schema; zeros when that asset's data is missing for the day."""
    n = t0.shape[0]
    F = {k: np.zeros(n, np.float64) for k in LEAD_KEYS}
    if bts is not None and bmid is not None:
        j = np.clip(np.searchsorted(bts, t0, "right") - 1, 0, len(bmid) - 1)
        m_now = bmid[j]
        for w, key in ((5, "ret_5s"), (30, "ret_30s"),
                       (60, "ret_60s"), (120, "ret_120s")):
            j0 = np.clip(np.searchsorted(bts, t0 - w * NS, "right") - 1,
                         0, len(bmid) - 1)
            mb = np.where(bmid[j0] > 0, bmid[j0], np.nan)
            F[key] = np.nan_to_num(np.log(m_now / mb))
        for r in range(n):
            lo = np.searchsorted(bts, t0[r] - 60 * NS, "left")
            hi = np.searchsorted(bts, t0[r], "right")
            if hi - lo > 2:
                seg = bmid[lo:hi]
                F["rv_60s"][r] = np.std(np.diff(np.log(
                    np.where(seg > 0, seg, np.nan)))) if seg.size > 2 else 0.0
        F["rv_60s"] = np.nan_to_num(F["rv_60s"])
    if tr is not None:
        tt, tq = tr
        cum = np.concatenate([[0.0], np.cumsum(tq)])
        for w, key in ((30, "flow_30s"), (60, "flow_60s")):
            a = np.searchsorted(tt, t0 - w * NS, "left")
            b = np.searchsorted(tt, t0, "right")
            F[key] = cum[b] - cum[a]
        a6 = np.searchsorted(tt, t0 - 60 * NS, "left")
        b6 = np.searchsorted(tt, t0, "right")
        v = cum[b6] - cum[a6]
        F["cumsgn_60s"] = np.sign(v) * np.log1p(np.abs(v))
    return np.column_stack([F[k] for k in LEAD_KEYS])


def _first_passage(mid, i0, jH, m0, f):
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
        out[r] = 1 if (d < 0 or (u >= 0 and u <= d)) else -1
    return out


def run(bk, sc, sym, days, run_id, git):
    Xv, XB, XE, meta, dayid = [], [], [], [], []
    books = {}
    for di, day in enumerate(days):
        try:
            X = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/features.npy")
            idx = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/indices.npy"
                            ).astype(np.int64)
            ts, b, a = _load_book_l1(bk, sym, day)
        except Exception as e:
            print(f"  {sym} {day}: skip ({type(e).__name__})", flush=True)
            continue
        mid = (b + a) * 0.5
        n = ts.shape[0]
        sel = np.where((idx > 0) & (idx < n - 1))[0][::STRIDE]
        i = idx[sel]
        t0 = ts[i]
        leads = {}
        for tag, xsym in XA.items():
            try:
                xts, xb, xa = _load_book_l1(bk, xsym, day)
                xmid = (xb + xa) * 0.5
            except Exception:
                xts = xmid = None
            xtr = _load_trades(bk, sc, xsym, day)
            leads[tag] = _lead(t0, xts, xmid, xtr)
        Xv.append(X[sel].astype(np.float64))
        XB.append(leads["BTC"]); XE.append(leads["ETH"])
        meta.append(np.stack([np.full(i.size, di), i,
                              t0.astype(np.float64)], 1))
        dayid.append(np.full(i.size, di, np.int32))
        books[di] = (ts, mid)
    if not Xv:
        return [{"symbol": sym, "error": "no data"}]
    V = np.concatenate(Xv)
    Bx = np.concatenate(XB)
    Ex = np.concatenate(XE)
    M = np.concatenate(meta)
    dy = np.concatenate(dayid)
    xa_live = float((np.abs(np.hstack([Bx, Ex])).sum(0) > 0).mean())
    print(f"[{sym}] n={V.shape[0]} v1={V.shape[1]} +xa={Bx.shape[1]+Ex.shape[1]}"
          f" xa_live={xa_live:.2f}", flush=True)
    Vx = np.hstack([V, Bx, Ex])

    recs = []
    for H in HS:
        fp = np.zeros(V.shape[0], np.int8)
        for di, (ts, mid) in books.items():
            mrow = np.where(dy == di)[0]
            bi = M[mrow, 1].astype(np.int64)
            t0 = M[mrow, 2].astype(np.int64)
            jH = np.minimum(np.searchsorted(ts, t0 + H * NS, "left"),
                            len(mid) - 1)
            fp[mrow] = _first_passage(mid, bi, jH, mid[bi], F_FRAC)
        reached = fp != 0
        up = (fp == 1).astype(int)
        n = V.shape[0]
        ntr = int(n * 0.70)
        emb = max(32, int(np.ceil(H / 24)) * 4)
        tr = np.zeros(n, bool); tr[:ntr - emb] = True
        te = np.zeros(n, bool); te[ntr:] = True

        # head1 — sufficient-move feasibility, scope = ALL
        h1 = _xgbc(); h1.fit(V[tr], reached[tr].astype(int))
        auc1 = _auc(reached[te].astype(int), h1.predict_proba(V[te])[:, 1])

        # head2 — direction, scope = >=cost subset ONLY; base vs +xasset
        tr2, te2 = tr & reached, te & reached
        base_up = float(up[te2].mean()) if te2.sum() else float("nan")
        res = {}
        rng = np.random.default_rng(SEED)
        if tr2.sum() >= 200 and te2.sum() >= 100:
            yv = up[te2]
            for tag, FM in (("base", V), ("xasset", Vx)):
                m = _xgbc(); m.fit(FM[tr2], up[tr2])
                p = m.predict_proba(FM[te2])[:, 1]
                res[tag] = dict(auc=round(_auc(yv, p), 4))
                if tag == "xasset":
                    plc = round(_auc(rng.permutation(yv), p), 4)
                    q = np.quantile(p, [0.1, 0.9])
                    lo, hi = p <= q[0], p >= q[1]
                    took = lo | hi
                    corr = ((hi & (yv == 1)) | (lo & (yv == 0)))
                    cap = (float(np.where(corr[took], F_STRICT,
                           -F_STRICT).mean()) if took.any() else float("nan"))
                    ndays = int(dy[te2].max() - dy[te2].min() + 1)
                    sel_d = float(took.sum() / max(1, ndays))
            d_auc = round(res["xasset"]["auc"] - res["base"]["auc"], 4)
        else:
            res = {"base": {"auc": None}, "xasset": {"auc": None}}
            plc = cap = sel_d = d_auc = None
        xauc = res["xasset"]["auc"]
        eL = int(cap is not None and cap > F_LOOSE / 100)
        eS = int(cap is not None and cap > 0)   # capture already net of floor
        # HM1: economic must NOT drive status. Auto-status caps at
        # 'suspect'; 'confirmed' is a human/pre-registered both-symbol
        # decision, never from cap>0 at near-chance AUC (HA5/HA2 artifact).
        status = "exploratory"
        if (xauc is not None and xauc > 0.52 and plc is not None
                and abs(plc - 0.5) < 0.02 and (d_auc or 0) > 0):
            status = "suspect"
        recs.append({
            "experiment_id": f"{run_id}_HA2_{sym}_H{H}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git, "author": "claude(vm)",
            "hypothesis_id": "HA2", "status": status, "kind": "alpha",
            "setup": f"HA2 dir two-head +BTC+ETH ({sym}, H={H}s)",
            "model_family": "xgb",
            "params": {"H": H, "head1_auc_feasibility": round(auc1, 4),
                       "base_P_up": None if np.isnan(base_up) else round(base_up, 4),
                       "head2_base_auc": res["base"]["auc"],
                       "head2_xasset_auc": xauc, "delta_auc": d_auc,
                       "placebo_auc": plc, "decile_capture_net_floor": cap,
                       "sel_per_day": sel_d, "xa_cols_live": round(xa_live, 3),
                       "stride": STRIDE, "n_ge_cost_oos": int(te2.sum())},
            "data_source": "cryptolake",
            "cache_id": f"cryptolake_{sym}+BTC+ETH_{days[0]}_{days[-1]}",
            "symbols": [sym], "date_range_start": days[0],
            "date_range_end": days[-1], "n_samples": int(n),
            "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
            "commission_loss_pct": 0.07, "split_method": "honest_val_test",
            "embargo": str(emb),
            "label_def": (f"head1=reach +-{F_STRICT}% within {H}s (scope ALL); "
                          f"head2=up-first on >=cost subset ONLY; "
                          f"directional logloss objective; +BTC+ETH lead"),
            "alpha_target": "updown_first_dirobj_xasset", "horizon_sec": H,
            "rank_ic_oos": None if xauc is None else round(xauc - 0.5, 4),
            "auc_oos": xauc, "baseline_ref": f"HA1_sign_auc~0.52 / in-run base head2 ({res['base']['auc']})",
            "delta_ic": d_auc,
            "top_decile_absmove_pct": F_STRICT, "bot_decile_absmove_pct": F_STRICT,
            "cost_floor_pct": F_STRICT, "decile_monotonic": None,
            "economic_pass_loose": eL, "economic_pass_strict": eS,
            "n_eff": int(te2.sum() * 24 / H),
            "repro_cmd": f"python scripts/ha2_screen.py (run {run_id} git {git})",
            "artifact_path": f"gs://blackdigital-scalper-data/research_runs/{run_id}/",
            "note": (f"HA2 directional two-head, per-head scope, +BTC+ETH. "
                     f"head1(vol,all)AUC={auc1:.3f}; base P(up|>=cost)="
                     f"{base_up:.3f}; head2 dir AUC base={res['base']['auc']} "
                     f"-> +xasset={xauc} (delta {d_auc}); placebo={plc}; "
                     f"capture net floor={cap}; xa_live={xa_live:.2f}. "
                     f"Pre-reg success: xasset AUC>0.52 both sym, plac~0.5, "
                     f"delta>noise. Judge by delta vs HA1 baseline (HM1), "
                     f"NOT the economic gate."),
        })
        print(f"  H={H}s h1={auc1:.3f} baseP={base_up:.3f} "
              f"h2 base={res['base']['auc']} xa={xauc} d={d_auc} "
              f"plc={plc} cap={cap} {status}", flush=True)
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
