#!/usr/bin/env python3
"""HX1 rev4 — the registered OOS experiment of the frozen rev1 spec.

Frozen spec (hypotheses.jsonl HX1 rev1, 2026-06-02): cross-exchange signal
strength, recorder chronos/ data, stable window >= 2026-06-01T19:39Z.
SIGNALS per (coin, venue): basis_dm = log(venue_last_px) - log(binance_mid)
minus trailing-120s mean; flow_imb_h = sum(signed_qty)/sum(qty) trailing h,
sign +1 buyer-aggressor. COMBO = per-coin small model (logreg + GBM) on
[basis_dm, flow_imb_{15,30,60}] per-venue AND a parameter-free z-sum.
TARGET = Binance mid forward log-return, H in {15,30,60}s, 1s grid.
SPLIT = honest day-based OOS (early train, late test), embargo >= 60s
(day boundary + first-120s-of-day NaN mask => >=120s). rank_ic_oos primary;
n_eff decorrelated by horizon. BASELINE_REF = within-Binance OBI(L5) at
matched horizon; delta_ic = combo_IC - max(component_IC, OBI_IC).
PRIMARY DELIVERABLE = the conditional rank_IC surface (coin x venue x signal
x horizon), argmax, stability (per-day, split-half). SECONDARY (deploy
annotation only): top-decile conviction capture vs maker 4/7 bp.

Stages (env STAGE): grid | analyze | all. Grid stage is resumable per
(coin, day) via the npz marker in GCS (capture-everything).

Run recipe (same-region VM, asia-northeast1; see ledger HX1 rev4):
  STAGE=grid NPROC=6 python3 hx1_oos.py     # ~1-2 h
  STAGE=analyze python3 hx1_oos.py
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BUCKET_IN = "gs://recorder-data-asia-0998ac51/chronos/scalper-recorder"
BUCKET_OUT = os.environ.get("OUT", "gs://market-data-0998ac51/research_runs/hx1_oos")
COINS = os.environ.get("COINS", "BNB,BTC,DOGE,ETH,LINK,LTC,SOL,XRP").split(",")
VENUES = ["bybit", "okx", "bitget", "gateio"]
VSYM = {
    "bybit": "{c}USDT", "okx": "{c}-USDT-SWAP",
    "bitget": "{c}USDT", "gateio": "{c}_USDT",
}
DAY0 = os.environ.get("DAY0", "20260602")
TEST0 = os.environ.get("TEST0", "20260702")
DAYN = os.environ.get("DAYN", "20260714")
HORIZONS = [15, 30, 60]
FLOW_WINDOWS = [15, 30, 60]
BASIS_DEMEAN_S = 120
OBI_LEVELS = 5
SEC_DAY = 86400
TRAIN_STRIDE_S = int(os.environ.get("TRAIN_STRIDE_S", "5"))
GBM_SEEDS = [0, 1]


def day_list():
    import datetime as dt
    d0 = dt.datetime.strptime(DAY0, "%Y%m%d")
    d1 = dt.datetime.strptime(DAYN, "%Y%m%d")
    return [(d0 + dt.timedelta(days=i)).strftime("%Y%m%d")
            for i in range((d1 - d0).days + 1)]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed rc={r.returncode}: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def gcs_exists(url):
    return subprocess.run(f"gcloud storage ls {url}", shell=True,
                          capture_output=True).returncode == 0


# ---------------------------------------------------------------- stage 1
def _list_first_and_headsum(col, k):
    """pyarrow (FixedSize)ListArray -> (first element, sum of first k) per row."""
    arr = col.combine_chunks()
    vals = arr.values.to_numpy(zero_copy_only=False).astype(np.float64)
    if hasattr(arr, "offsets"):  # ragged ListArray
        off = arr.offsets.to_numpy().astype(np.int64)
        lens = off[1:] - off[:-1]
        first = np.full(len(lens), np.nan)
        nz = lens > 0
        first[nz] = vals[off[:-1][nz]]
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        hi = off[:-1] + np.minimum(lens, k)
        head = csum[hi] - csum[off[:-1]]
        return first, head
    size = arr.type.list_size  # FixedSizeListArray
    m = vals.reshape(-1, size)
    return m[:, 0].copy(), m[:, :k].sum(axis=1)


def snap_to_grid(pq_files, day_us0):
    """Binance depth_snapshot hour files -> per-second mid & OBI5 (last snapshot
    at or before the end of each 1s bucket; NaN until the first snapshot)."""
    import pyarrow.parquet as pq
    rows_ts, rows_mid, rows_obi = [], [], []
    for f in pq_files:
        t = pq.read_table(f, columns=["local_ts_us", "bid_prices", "bid_qtys",
                                      "ask_prices", "ask_qtys"])
        ts = t["local_ts_us"].to_numpy().astype(np.int64)
        bb, _ = _list_first_and_headsum(t["bid_prices"], 1)
        ba, _ = _list_first_and_headsum(t["ask_prices"], 1)
        _, sb = _list_first_and_headsum(t["bid_qtys"], OBI_LEVELS)
        _, sa = _list_first_and_headsum(t["ask_qtys"], OBI_LEVELS)
        m = (bb + ba) / 2.0
        tot = sb + sa
        o = np.where(tot > 0, (sb - sa) / np.where(tot > 0, tot, 1.0), 0.0)
        ok = ~np.isnan(m)
        rows_ts.append(ts[ok])
        rows_mid.append(m[ok])
        rows_obi.append(o[ok])
    ts = np.concatenate(rows_ts)
    m = np.concatenate(rows_mid)
    o = np.concatenate(rows_obi)
    order = np.argsort(ts, kind="stable")
    ts, m, o = ts[order], m[order], o[order]
    sec = (ts - day_us0) // 1_000_000
    keep = (sec >= 0) & (sec < SEC_DAY)  # drop pre-midnight flush tail (hour-00)
    sec, m, o = sec[keep], m[keep], o[keep]
    mid = np.full(SEC_DAY, np.nan)
    obi = np.full(SEC_DAY, np.nan)
    if len(sec):
        last = np.searchsorted(sec, np.arange(SEC_DAY), side="right") - 1
        valid = last >= 0
        mid[valid] = m[last[valid]]
        obi[valid] = o[last[valid]]
    return mid, obi


def trades_to_grid(pq_files, day_us0):
    """Venue trade hour files -> per-second last px (ffill), signed qty sum,
    total qty sum."""
    import pyarrow.parquet as pq
    px = np.full(SEC_DAY, np.nan)
    sgn = np.zeros(SEC_DAY)
    tot = np.zeros(SEC_DAY)
    for f in pq_files:
        t = pq.read_table(f, columns=["local_ts_us", "price", "qty",
                                      "is_buyer_maker"])
        ts = t["local_ts_us"].to_numpy()
        pr = t["price"].to_numpy()
        q = t["qty"].to_numpy().astype(np.float64)
        ibm = t["is_buyer_maker"].to_numpy()
        order = np.argsort(ts, kind="stable")
        ts, pr, q, ibm = ts[order], pr[order], q[order], ibm[order]
        sec = ((ts - day_us0) // 1_000_000).astype(np.int64)
        ok = (sec >= 0) & (sec < SEC_DAY)
        sec, pr, q, ibm = sec[ok], pr[ok], q[ok], ibm[ok]
        # aggressor buy = is_buyer_maker False -> +1 (unified chronos semantics)
        s = np.where(ibm, -1.0, 1.0) * q
        sgn += np.bincount(sec, weights=s, minlength=SEC_DAY)
        tot += np.bincount(sec, weights=q, minlength=SEC_DAY)
        # last trade price per second (rows sorted; later rows overwrite)
        px[sec] = pr
    # forward-fill px; NaN before the first trade of the day
    has = ~np.isnan(px)
    idx = np.where(has, np.arange(SEC_DAY), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = px[idx]
    if not has[0]:
        first = np.argmax(has) if has.any() else SEC_DAY
        filled[:first] = np.nan
    return filled, sgn, tot


def build_day(args):
    coin, day = args
    out_url = f"{BUCKET_OUT}/grid/{coin}/{day}.npz"
    if gcs_exists(out_url):
        return f"{coin} {day} SKIP (exists)"
    t0 = time.time()
    import datetime as dt
    day_us0 = int(dt.datetime.strptime(day, "%Y%m%d")
                  .replace(tzinfo=dt.timezone.utc).timestamp() * 1_000_000)
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/b", exist_ok=True)
        sh(f"gcloud storage cp '{BUCKET_IN}/binance_futures/{coin}USDT/"
           f"depth_snapshot/{day}_*.parquet' {td}/b/ -q")
        bfiles = sorted(os.path.join(td, "b", f) for f in os.listdir(f"{td}/b"))
        if len(bfiles) < 20:
            return f"{coin} {day} INCOMPLETE binance ({len(bfiles)}h) — skipped"
        mid, obi = snap_to_grid(bfiles, day_us0)
        out = {"mid": mid.astype(np.float32), "obi5": obi.astype(np.float32)}
        for v in VENUES:
            sym = VSYM[v].format(c=coin)
            os.makedirs(f"{td}/{v}", exist_ok=True)
            sh(f"gcloud storage cp '{BUCKET_IN}/{v}/{sym}/trade/{day}_*.parquet'"
               f" {td}/{v}/ -q")
            vfiles = sorted(os.path.join(td, v, f) for f in os.listdir(f"{td}/{v}"))
            px, sgn, tot = trades_to_grid(vfiles, day_us0)
            out[f"{v}_px"] = px.astype(np.float32)
            out[f"{v}_sgn"] = sgn.astype(np.float32)
            out[f"{v}_tot"] = tot.astype(np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, **out)
        lf = os.path.join(td, "grid.npz")
        open(lf, "wb").write(buf.getvalue())
        sh(f"gcloud storage cp {lf} {out_url} -q")
    return f"{coin} {day} OK ({time.time()-t0:.0f}s)"


# ---------------------------------------------------------------- stage 2
def roll_sum(x, w):
    c = np.cumsum(np.nan_to_num(x, nan=0.0))
    out = np.full_like(c, np.nan, dtype=np.float64)
    out[w:] = c[w:] - c[:-w]
    return out


def day_signals(g):
    """Per-day signal dict from a grid npz (trailing windows within-day;
    first max(120s, w) of each day stays NaN = built-in embargo)."""
    mid = g["mid"].astype(np.float64)
    lm = np.log(mid)
    sig = {"obi5": g["obi5"].astype(np.float64)}
    flows = []
    for v in VENUES:
        px = g[f"{v}_px"].astype(np.float64)
        d = np.log(px) - lm
        valid = ~np.isnan(d)
        cnt = roll_sum(valid.astype(float), BASIS_DEMEAN_S)
        m120 = roll_sum(d, BASIS_DEMEAN_S) / np.where(cnt > 0, cnt, 1.0)
        basis = d - m120
        basis[np.isnan(d) | (cnt < BASIS_DEMEAN_S * 0.8)] = np.nan
        sig[f"basis_{v}"] = basis
        for w in FLOW_WINDOWS:
            s = roll_sum(g[f"{v}_sgn"].astype(np.float64), w)
            q = roll_sum(g[f"{v}_tot"].astype(np.float64), w)
            f = np.where(q > 0, s / np.where(q > 0, q, 1.0), 0.0)
            f[np.isnan(s)] = np.nan
            sig[f"flow{w}_{v}"] = f
    for w in FLOW_WINDOWS:
        pool = np.nanmean(
            np.stack([sig[f"flow{w}_{v}"] for v in VENUES]), axis=0)
        sig[f"flow{w}_pool"] = pool
    sig["basis_pool"] = np.nanmean(
        np.stack([sig[f"basis_{v}"] for v in VENUES]), axis=0)
    tgt = {}
    for h in HORIZONS:
        r = np.full(SEC_DAY, np.nan)
        r[:-h] = lm[h:] - lm[:-h]
        tgt[h] = r
    return sig, tgt


def rank_ic(x, y):
    from scipy.stats import rankdata
    ok = ~(np.isnan(x) | np.isnan(y))
    n = int(ok.sum())
    if n < 1000:
        return np.nan, n
    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    return float(np.corrcoef(rx, ry)[0, 1]), n


def analyze():
    from scipy.stats import rankdata
    days = day_list()
    train_days = [d for d in days if d < TEST0]
    test_days = [d for d in days if d >= TEST0]
    os.makedirs("hx1_local", exist_ok=True)
    surface = []
    combo_rows = []
    for coin in COINS:
        per_day_sig, per_day_tgt, day_ok = {}, {}, []
        for day in days:
            lf = f"hx1_local/{coin}_{day}.npz"
            if not os.path.exists(lf):
                try:
                    sh(f"gcloud storage cp {BUCKET_OUT}/grid/{coin}/{day}.npz {lf} -q")
                except RuntimeError:
                    continue
            g = np.load(lf)
            per_day_sig[day], per_day_tgt[day] = day_signals(g)
            day_ok.append(day)
        tr = [d for d in day_ok if d < TEST0]
        te = [d for d in day_ok if d >= TEST0]
        print(f"[{coin}] days train={len(tr)} test={len(te)}", flush=True)
        signames = list(per_day_sig[day_ok[0]].keys())

        def cat(names_days, key, is_tgt=False, h=None):
            src = per_day_tgt if is_tgt else per_day_sig
            return np.concatenate(
                [src[d][h] if is_tgt else src[d][key] for d in names_days])

        for h in HORIZONS:
            y_te = cat(te, None, True, h)
            y_tr = cat(tr, None, True, h)
            for sname in signames:
                x_te = cat(te, sname)
                ic, n = rank_ic(x_te, y_te)
                ic_tr, n_tr = rank_ic(cat(tr, sname), y_tr)
                # stability: per-test-day ICs + split-half
                dics = []
                for d in te:
                    di, dn = rank_ic(per_day_sig[d][sname], per_day_tgt[d][h])
                    if not np.isnan(di):
                        dics.append(di)
                half = len(te) // 2
                ic_h1, _ = rank_ic(cat(te[:half], sname), cat(te[:half], None, True, h))
                ic_h2, _ = rank_ic(cat(te[half:], sname), cat(te[half:], None, True, h))
                n_eff = n / h if n else 0
                surface.append(dict(
                    coin=coin, signal=sname, horizon=h,
                    ic_oos=ic, n=n, n_eff=n_eff,
                    se=(1 / np.sqrt(n_eff) if n_eff else np.nan),
                    ic_train=ic_tr,
                    day_ic_mean=float(np.mean(dics)) if dics else np.nan,
                    day_ic_sd=float(np.std(dics)) if dics else np.nan,
                    days_pos=int(sum(d > 0 for d in dics)), days_n=len(dics),
                    ic_half1=ic_h1, ic_half2=ic_h2))
            # ---- combos (fit on TRAIN, score TEST) ----
            feats = [f"basis_{v}" for v in VENUES] + \
                    [f"flow{w}_{v}" for w in FLOW_WINDOWS for v in VENUES]
            Xtr = np.column_stack([cat(tr, f) for f in feats])
            Xte = np.column_stack([cat(te, f) for f in feats])
            ytr, yte = y_tr, y_te
            ok_tr = ~(np.isnan(Xtr).any(1) | np.isnan(ytr))
            ok_tr[np.arange(len(ok_tr)) % TRAIN_STRIDE_S != 0] = False
            ok_te = ~(np.isnan(Xte).any(1) | np.isnan(yte))
            mu = Xtr[ok_tr].mean(0)
            sd = Xtr[ok_tr].std(0) + 1e-12
            # (a) parameter-free z-sum on pooled components
            zfeats = ["basis_pool"] + [f"flow{w}_pool" for w in FLOW_WINDOWS]
            Ztr = np.column_stack([cat(tr, f) for f in zfeats])
            Zte = np.column_stack([cat(te, f) for f in zfeats])
            zok = ~np.isnan(Ztr).any(1)
            zmu, zsd = Ztr[zok].mean(0), Ztr[zok].std(0) + 1e-12
            zsum = ((Zte - zmu) / zsd).sum(1)
            arms = {"zsum_pool": (zsum, ~np.isnan(Zte).any(1))}
            # (b) logreg
            try:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=200, C=1.0)
                lr.fit((Xtr[ok_tr] - mu) / sd, (ytr[ok_tr] > 0).astype(int))
                arms["logreg"] = (lr.predict_proba((np.nan_to_num(Xte) - mu) / sd)[:, 1], ok_te)
            except Exception as e:  # noqa: BLE001
                print("logreg failed:", e)
            # (c) GBM x seeds
            try:
                import xgboost as xgb
                for s in GBM_SEEDS:
                    m = xgb.XGBClassifier(
                        n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        random_state=s, n_jobs=int(os.environ.get("NPROC", "4")),
                        eval_metric="logloss")
                    m.fit(Xtr[ok_tr], (ytr[ok_tr] > 0).astype(int))
                    arms[f"gbm_s{s}"] = (m.predict_proba(np.nan_to_num(Xte))[:, 1], ok_te)
            except Exception as e:  # noqa: BLE001
                print("gbm failed:", e)
            comp_best = max(
                (r["ic_oos"] for r in surface
                 if r["coin"] == coin and r["horizon"] == h
                 and r["signal"] != "obi5" and not np.isnan(r["ic_oos"])),
                default=np.nan)
            obi_ic = next(
                (r["ic_oos"] for r in surface
                 if r["coin"] == coin and r["horizon"] == h
                 and r["signal"] == "obi5"), np.nan)
            for aname, (score, okm) in arms.items():
                x = np.where(okm, score, np.nan)
                ic, n = rank_ic(x, yte)
                # conviction annotation (top-decile |centered score|)
                med = np.nanmedian(x)
                conv = np.abs(x - med)
                thr = np.nanquantile(conv, 0.9)
                sel = (conv >= thr) & ~np.isnan(x) & ~np.isnan(yte)
                cap_bp = float(np.nanmean(
                    np.sign(x[sel] - med) * yte[sel]) * 1e4) if sel.any() else np.nan
                dics = []
                off = 0
                for d in te:
                    seg = slice(off, off + SEC_DAY)
                    di, _ = rank_ic(x[seg], yte[seg])
                    if not np.isnan(di):
                        dics.append(di)
                    off += SEC_DAY
                combo_rows.append(dict(
                    coin=coin, arm=aname, horizon=h, ic_oos=ic, n=n,
                    delta_vs_best=(ic - np.nanmax([comp_best, obi_ic])),
                    comp_best=comp_best, obi_ic=obi_ic,
                    day_ic_mean=float(np.mean(dics)) if dics else np.nan,
                    days_pos=int(sum(d > 0 for d in dics)), days_n=len(dics),
                    conv_top10_capture_bp=cap_bp))
                print(f"[{coin}] h{h} {aname}: ic={ic:+.4f} "
                      f"delta={ic - np.nanmax([comp_best, obi_ic]):+.4f} "
                      f"cap={cap_bp:+.2f}bp", flush=True)
    out = dict(
        spec="HX1 rev1 frozen OOS", grid="1s local_ts",
        window=dict(day0=DAY0, test0=TEST0, dayn=DAYN),
        train_stride_s=TRAIN_STRIDE_S, obi_levels=OBI_LEVELS,
        surface=surface, combos=combo_rows)
    with open("hx1_local/HX1_OOS_RESULT.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    sh(f"gcloud storage cp hx1_local/HX1_OOS_RESULT.json {BUCKET_OUT}/ -q")
    print("saved", f"{BUCKET_OUT}/HX1_OOS_RESULT.json")


def main():
    stage = os.environ.get("STAGE", "all")
    if stage in ("grid", "all"):
        jobs = [(c, d) for c in COINS for d in day_list()]
        nproc = int(os.environ.get("NPROC", "6"))
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            for msg in ex.map(build_day, jobs):
                print(msg, flush=True)
    if stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
