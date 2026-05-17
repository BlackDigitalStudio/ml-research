#!/usr/bin/env python3
"""H3 — BTC-lead cross-asset features (the never-populated dimension).

Feature decode (2026-05-17) showed every cross-asset/ETH slot in
features_v1 is 100% zero. BTC leads alts. This screen computes causal
BTC-lead features from in-bucket BTC raw (book + trades, ts <= alt
entry) and measures, per (alt symbol, horizon h), the OOS rank-IC and
economic decile capture of:
    base : XGB on features_v1 only            (reproduces HA1)
    btc  : XGB on features_v1 + 8 BTC-lead    (the H3 lift)
Pre-registered (mirrors HA1): honest train70/embargo/OOS30 per symbol,
target = alt forward mid log-return at h in {30,60,120,180}s, metric =
OOS rank-IC + economic decile vs cost floor (loose 0.08% / strict
0.13%), PLACEBO (shuffled y) leak sentinel. H3 is confirmed only if the
btc variant materially lifts rank-IC AND clears the strict floor on BOTH
symbols with clean placebo; else exploratory/refuted. Fixed BTC-lead
schema (zero when BTC missing) — no ragged-concat. kind='alpha' rows.
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

HS = (30, 60, 120, 180)
F_LOOSE, F_STRICT = 0.08, 0.13      # % round-trip floors
SEED = 42
NS = 1_000_000_000
BTC = "BTC-USDT-PERP"
BTC_KEYS = ("btc_ret_5s", "btc_ret_30s", "btc_ret_60s", "btc_ret_120s",
            "btc_flow_30s", "btc_flow_60s", "btc_rv_60s", "btc_cumsgn_60s")


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        random_state=SEED, objective="reg:squarederror")


def _rank(a):
    o = a.argsort(); r = np.empty(len(a), np.float64); r[o] = np.arange(len(a))
    return r


def _sp(x, y):
    rx, ry = _rank(x) - _rank(x).mean(), _rank(y) - _rank(y).mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def _load_btc_trades(bk, sc, day):
    import pyarrow.parquet as pq
    pref = f"raw/trades/exchange=BINANCE_FUTURES/symbol={BTC}/dt={day}/"
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


def _btc_lead(t0, bts, bmid, tr):
    """Causal BTC-lead features at alt entry ts t0 (ns). Fixed schema."""
    n = t0.shape[0]
    F = {k: np.zeros(n, np.float64) for k in BTC_KEYS}
    if bts is not None and bmid is not None:
        j = np.clip(np.searchsorted(bts, t0, "right") - 1, 0, len(bmid) - 1)
        m_now = bmid[j]
        for w, key in ((5, "btc_ret_5s"), (30, "btc_ret_30s"),
                       (60, "btc_ret_60s"), (120, "btc_ret_120s")):
            j0 = np.clip(np.searchsorted(bts, t0 - w * NS, "right") - 1,
                         0, len(bmid) - 1)
            mb = np.where(bmid[j0] > 0, bmid[j0], np.nan)
            F[key] = np.nan_to_num(np.log(m_now / mb))
    if tr is not None:
        tt, tq = tr
        cum = np.concatenate([[0.0], np.cumsum(tq)])
        for w, key in ((30, "btc_flow_30s"), (60, "btc_flow_60s")):
            a = np.searchsorted(tt, t0 - w * NS, "left")
            b = np.searchsorted(tt, t0, "right")
            F[key] = cum[b] - cum[a]
        a60 = np.searchsorted(tt, t0 - 60 * NS, "left")
        b60 = np.searchsorted(tt, t0, "right")
        F["btc_cumsgn_60s"] = np.sign(cum[b60] - cum[a60]) * np.log1p(
            np.abs(cum[b60] - cum[a60]))
    if bts is not None and bmid is not None:
        # realized vol of BTC mid over last 60s (std of 1s-spaced log-rets)
        for r in range(n):
            lo = np.searchsorted(bts, t0[r] - 60 * NS, "left")
            hi = np.searchsorted(bts, t0[r], "right")
            if hi - lo > 2:
                seg = bmid[lo:hi]
                F["btc_rv_60s"][r] = np.std(np.diff(np.log(
                    np.where(seg > 0, seg, np.nan)))) if seg.size > 2 else 0.0
        F["btc_rv_60s"] = np.nan_to_num(F["btc_rv_60s"])
    return np.column_stack([F[k] for k in BTC_KEYS])


def _econ(p, y, fl, fs):
    q = np.quantile(p, [0.1, 0.9])
    lo, hi = p <= q[0], p >= q[1]
    cap = []
    if hi.any():
        cap.append(np.sign(y[hi]).mean())   # long top decile
    if lo.any():
        cap.append(-np.sign(y[lo]).mean())  # short bottom decile
    # mean |move| in selected deciles (economic size, %)
    mv = np.abs(y[hi | lo]).mean() * 100 if (hi | lo).any() else 0.0
    return float(mv), int(mv > fl), int(mv > fs)


def run(bk, sc, sym, days, run_id, git):
    Xs, B, t0s, dayid = [], [], [], []
    books = {}
    for di, day in enumerate(days):
        try:
            X = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/features.npy")
            idx = _load_npy(bk, f"features_v1/symbol={sym}/dt={day}/indices.npy"
                            ).astype(np.int64)
            ts, b, a = _load_book_l1(bk, sym, day)
            try:
                bts, bb, ba = _load_book_l1(bk, BTC, day)
                bmid = (bb + ba) * 0.5
            except Exception:
                bts = bmid = None
            tr = _load_btc_trades(bk, sc, day)
        except Exception as e:
            print(f"  {sym} {day}: skip ({type(e).__name__})", flush=True)
            continue
        mid = (b + a) * 0.5
        n = ts.shape[0]
        ok = (idx > 0) & (idx < n - 1)
        sel = np.where(ok)[0]
        i = idx[sel]
        Xs.append(X[sel].astype(np.float64))
        B.append(_btc_lead(ts[i], bts, bmid, tr))
        t0s.append(np.stack([np.full(i.size, di), i,
                             ts[i].astype(np.float64)], 1))
        dayid.append(np.full(i.size, di, np.int32))
        books[di] = (ts, mid)
    if not Xs:
        return [{"symbol": sym, "error": "no data"}]
    Xb = np.concatenate(Xs)
    Bb = np.concatenate(B)
    meta = np.concatenate(t0s)
    dy = np.concatenate(dayid)
    btc_live = float((np.abs(Bb).sum(0) > 0).mean())
    print(f"[{sym}] n={Xb.shape[0]} feat={Xb.shape[1]} +btc={Bb.shape[1]} "
          f"btc_cols_live={btc_live:.2f}", flush=True)

    recs = []
    for H in HS:
        y = np.full(Xb.shape[0], np.nan)
        for di, (ts, mid) in books.items():
            m = np.where(dy == di)[0]
            bi = meta[m, 1].astype(np.int64)
            t0 = meta[m, 2].astype(np.int64)
            j = np.searchsorted(ts, t0 + H * NS, "left")
            ok = j < len(mid)
            m0 = mid[bi]
            yy = np.full(m.size, np.nan)
            yy[ok] = np.log(mid[j[ok]] / np.where(m0[ok] > 0, m0[ok], np.nan))
            y[m] = yy
        keep = np.isfinite(y)
        Xk, Bk, yk, dk = Xb[keep], Bb[keep], y[keep], dy[keep]
        nn = Xk.shape[0]
        ntr = int(nn * 0.70)
        emb = max(32, int(np.ceil(H / 24)) * 4)
        tr_m = np.zeros(nn, bool); tr_m[:ntr - emb] = True
        te_m = np.zeros(nn, bool); te_m[ntr:] = True
        out = {}
        for tag, M in (("base", Xk), ("btc", np.hstack([Xk, Bk]))):
            mdl = _xgb(); mdl.fit(M[tr_m], yk[tr_m])
            p = mdl.predict(M[te_m])
            yv = yk[te_m]
            ic = _sp(p, yv)
            rng = np.random.default_rng(SEED)
            plc = _sp(p, rng.permutation(yv))
            mv, eL, eS = _econ(p, yv, F_LOOSE, F_STRICT)
            out[tag] = dict(ic=round(ic, 5), placebo=round(plc, 5),
                            top_absmove_pct=round(mv, 4), eL=eL, eS=eS)
        n_te_days = int(dk[te_m].max() - dk[te_m].min() + 1)
        d_ic = out["btc"]["ic"] - out["base"]["ic"]
        status = "exploratory"
        if (out["btc"]["eS"] and abs(out["btc"]["ic"]) > abs(out["base"]["ic"])
                + 0.01 and abs(out["btc"]["placebo"]) < 0.02):
            status = "confirmed"
        elif d_ic > 0.005 and abs(out["btc"]["placebo"]) < 0.02:
            status = "suspect"
        recs.append({
            "experiment_id": f"{run_id}_H3_{sym}_h{H}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git, "author": "claude(vm)",
            "hypothesis_id": "H3", "status": status, "kind": "alpha",
            "setup": f"H3 BTC-lead vs base XGB ({sym}, h={H}s)",
            "model_family": "xgb",
            "params": {"h": H, "base": out["base"], "btc": out["btc"],
                       "delta_ic": round(d_ic, 5), "btc_cols_live": round(btc_live, 3),
                       "n_oos": int(te_m.sum()), "embargo": emb, "seed": SEED},
            "data_source": "cryptolake",
            "cache_id": f"cryptolake_{sym}+BTClead_{days[0]}_{days[-1]}",
            "symbols": [sym], "date_range_start": days[0],
            "date_range_end": days[-1], "n_samples": int(nn),
            "fee_regime": "MAKER_FIRST", "commission_win_pct": 0.04,
            "commission_loss_pct": 0.07, "split_method": "honest_val_test",
            "embargo": str(emb),
            "label_def": (f"alt fwd mid logret {H}s; +8 causal BTC-lead "
                          f"(ret 5/30/60/120s, flow 30/60s, rv60s, cumsgn60s)"),
            "alpha_target": "fwd_mid_logret_btclead", "horizon_sec": H,
            "rank_ic_oos": out["btc"]["ic"],
            "auc_oos": None,
            "top_decile_absmove_pct": out["btc"]["top_absmove_pct"],
            "bot_decile_absmove_pct": out["base"]["top_absmove_pct"],
            "cost_floor_pct": F_STRICT, "decile_monotonic": None,
            "economic_pass_loose": out["btc"]["eL"],
            "economic_pass_strict": out["btc"]["eS"],
            "n_eff": int(te_m.sum() * 24 / H),
            "repro_cmd": f"python scripts/h3_screen.py (run {run_id} git {git})",
            "artifact_path": f"gs://blackdigital-scalper-data/research_runs/{run_id}/",
            "note": (f"H3 BTC-lead. base IC={out['base']['ic']} -> +BTC IC="
                     f"{out['btc']['ic']} (delta {d_ic:+.4f}); btc top|mv|="
                     f"{out['btc']['top_absmove_pct']}% vs strict {F_STRICT}; "
                     f"placebo={out['btc']['placebo']}; btc_cols_live="
                     f"{btc_live:.2f}. Confirmed only if +BTC clears strict "
                     f"AND lifts IC >0.01 both sym, placebo~0."),
        })
        print(f"  h={H}s base_ic={out['base']['ic']} btc_ic={out['btc']['ic']} "
              f"d={d_ic:+.4f} btc|mv|={out['btc']['top_absmove_pct']}% "
              f"eS={out['btc']['eS']} plc={out['btc']['placebo']} {status}",
              flush=True)
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
