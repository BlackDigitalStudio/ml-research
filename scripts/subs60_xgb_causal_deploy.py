#!/usr/bin/env python3
"""CAUSAL (real-time-deployable) deploy policy vs the post-hoc top-N, DOGE walk-forward.

The post-hoc policy (daily top-N by pct_rank(pA)^wA * pct_rank(|pB-0.5|)^wB) ranks within
the day / over the test pool -> LOOK-AHEAD, not deployable. The causal version:
  - fix the pA and |pB-0.5| -> percentile maps on the fold's TRAIN distribution;
  - calibrate a score threshold tau on TRAIN to yield ~target trades/day;
  - LIVE: trade any window whose train-mapped score >= tau (causal; variable trades/day).
Reuses the SAVED per-fold weights (research_runs/wf_models/DOGE_adaptive_W200T30/) -- NO training.
Optional daily cap = trade at most N/day among threshold-passers, in TIME order (also causal).
"""
import io, json, os, tempfile
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
MODELS_DIR = "research_runs/wf_models/DOGE_adaptive_W200T30"
W, T, EMB = 200, 30, 2
bk = storage.Client(project=PROJ).bucket(BUCKET)


def load_rr(sub, symk):
    d = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pl": d["pnl_long"].astype(np.float32), "ps": d["pnl_short"].astype(np.float32),
            "fl": d["fill_long"].astype(bool), "fs": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "fee": m["maker_rt_fee_pct"] * 100.0}


def load_model(blobname):
    p = tempfile.mktemp(suffix=".json"); bk.blob(blobname).download_to_filename(p)
    b = xgb.Booster(); b.load_model(p); os.remove(p); return b


def pct_rank(x): o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score, n=1):
    order = np.lexsort((-score, day)); ds = day[order]
    st = np.zeros(len(order), bool); st[0] = True; st[1:] = ds[1:] != ds[:-1]
    si = np.where(st)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


def cdf_map(x, sorted_ref):
    return np.searchsorted(sorted_ref, x, side="right") / max(len(sorted_ref), 1)


SYMK = "DOGE"
E = load_rr("maker_labels_rr", SYMK)
F = E["F"]; rH = E["rH"]; day = E["day"]; fee = E["fee"]; ndays = E["ndays"]
fl = E["fl"][0]; fs = E["fs"][0]
netl = E["pl"][:, 0, :].astype(np.float64) * 100.0 - fee
nets = E["ps"][:, 0, :].astype(np.float64) * 100.0 - fee

# ---- load saved per-fold models, predict train+test (NO training) ----
folds = []; ts = W + EMB; fi = 0
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() < 50 or trn.sum() < 5000:
        ts += T; continue
    A = load_model(f"{MODELS_DIR}/fold{fi}_A.json"); B = load_model(f"{MODELS_DIR}/fold{fi}_B.json")
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    folds.append({
        "pA_tr": A.predict(xgb.DMatrix(F[tri])), "pB_tr": B.predict(xgb.DMatrix(F[tri])), "day_tr": day[tri],
        "pA_te": A.predict(xgb.DMatrix(F[tei])), "pB_te": B.predict(xgb.DMatrix(F[tei])), "day_te": day[tei],
        "fl": fl[tei], "fs": fs[tei], "nl": netl[0][tei], "ns": nets[0][tei],
        "ndays_tr": len(set(day[tri].tolist())), "ndays_te": len(set(day[tei].tolist()))})
    fi += 1; ts += T
tot_te_days = sum(f["ndays_te"] for f in folds)
print(f"[loaded {fi} folds of saved weights, NO training] OOS test = {tot_te_days} days\n", flush=True)


def realize(idx_sel, fd):
    side = fd["pB_te"][idx_sel] >= 0.5
    net = np.where(side, fd["nl"][idx_sel], fd["ns"][idx_sel])
    fc = np.where(side, fd["fl"][idx_sel], fd["fs"][idx_sel])
    ex = fc & np.isfinite(net); return net[ex]


def posthoc(fd, wA, wB, budget):
    a = pct_rank(fd["pA_te"]); b = pct_rank(np.abs(fd["pB_te"] - 0.5))
    score = (a ** wA) * (b ** wB) if (wA or wB) else np.ones_like(a)
    return realize(daily_pick(fd["day_te"], score, budget), fd)


def causal(fd, wA, wB, target_tpd, cap=None):
    sA = np.sort(fd["pA_tr"]); sB = np.sort(np.abs(fd["pB_tr"] - 0.5))
    # train self-score to calibrate tau for the target trades/day
    sc_tr = (cdf_map(fd["pA_tr"], sA) ** wA) * (cdf_map(np.abs(fd["pB_tr"] - 0.5), sB) ** wB)
    k = int(round(target_tpd * fd["ndays_tr"]))
    tau = np.partition(sc_tr, -k)[-k] if 0 < k <= len(sc_tr) else -np.inf
    # causal test score from TRAIN maps
    sc_te = (cdf_map(fd["pA_te"], sA) ** wA) * (cdf_map(np.abs(fd["pB_te"] - 0.5), sB) ** wB)
    passers = np.where(sc_te >= tau)[0]
    if cap is not None and len(passers):              # optional daily cap, in TIME order (causal)
        keep = []; seen = {}
        order = passers[np.argsort(fd["day_te"][passers], kind="stable")]
        for j in order:
            d = int(fd["day_te"][j])
            if seen.get(d, 0) < cap:
                keep.append(j); seen[d] = seen.get(d, 0) + 1
        passers = np.array(keep)
    return realize(passers, fd)


def causal_rolling(fd, wA, wB, target_tpd, Kdays=30):
    """Causal: train cdf-maps for the score, but tau recalibrated DAILY from a trailing
    K-day buffer of recent scores to hit ~target trades/day (regime-adaptive deploy gate)."""
    sA = np.sort(fd["pA_tr"]); sB = np.sort(np.abs(fd["pB_tr"] - 0.5))
    sc_tr = (cdf_map(fd["pA_tr"], sA) ** wA) * (cdf_map(np.abs(fd["pB_tr"] - 0.5), sB) ** wB)
    sc_te = (cdf_map(fd["pA_te"], sA) ** wA) * (cdf_map(np.abs(fd["pB_te"] - 0.5), sB) ** wB)
    days = sorted(set(fd["day_te"].tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - target_tpd / max(wpd, 1.0))
    tr_days = sorted(set(fd["day_tr"].tolist()))
    buf = list(sc_tr[np.isin(fd["day_tr"], tr_days[-Kdays:])]); cap = max(int(Kdays * wpd), 1)
    sel = []
    for d in days:
        idx = np.where(fd["day_te"] == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist())
        buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    return realize(np.array(sel, dtype=int), fd)


def stat(nets):
    nets = np.asarray(nets); n = len(nets)
    ev = float(nets.mean()) if n else float("nan"); tpd = n / max(tot_te_days, 1)
    daily = ev * tpd * 0.01; cum = float(nets.sum()) * 0.01
    return f"trd/day={tpd:4.1f} EV/trd={ev*0.01:+.4f}% daily={daily:+.3f}% OOS-{tot_te_days}d={cum:+5.1f}%"


def statf(nets):
    nets = np.asarray(nets); n = len(nets)
    if not n:
        return "n=0 (no trades)"
    ev = float(nets.mean()); se = float(nets.std() / np.sqrt(n)); tpd = n / max(tot_te_days, 1)
    return f"trd/day={tpd:4.1f} EV/trd={ev*0.01:+.4f}%+-{se*0.01:.4f} OOS={float(nets.sum())*0.01:+6.1f}%"


def report(name, fn, configs):
    print(f"\n=== {name} ===")
    for wlab, wA, wB, tgt in configs:
        pf = [fn(fd, wA, wB, tgt) for fd in folds]
        print(f"  {wlab} ~{tgt}/d | {statf(np.concatenate(pf) if pf else np.array([]))}")
        print(f"      per-fold OOS%: {[(f'{p.sum()*0.01:+.1f}(n{len(p)})') for p in pf]}")


CFG = [("A=B", 1, 1, 1), ("A2", 2, 1, 1), ("A2", 2, 1, 5), ("A2", 2, 1, 10)]
report("POST-HOC top-N (look-ahead upper bound) — product", lambda fd, wA, wB, b: posthoc(fd, wA, wB, b), CFG)
report("CAUSAL FROZEN (train thresholds) — product", causal, CFG)
report("CAUSAL ROLLING (regime-adaptive, deployable) — product", causal_rolling, CFG)
