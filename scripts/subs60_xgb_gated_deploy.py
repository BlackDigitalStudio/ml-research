#!/usr/bin/env python3
"""GATED two-stage cascade (hard A-gate then hard B-gate) vs the AB-product score, DOGE WF.

Deploy = trade windows that pass BOTH gates:  pA >= thrA  AND  |pB-0.5| >= thrB.
  thrA keeps the top-qA% by A confidence (the strong/primary filter, user prior);
  thrB keeps, among the A-survivors, enough by B confidence to hit ~target trades/day.
Three regimes: post-hoc (look-ahead pool ranks, upper bound), causal-frozen (train thresholds),
causal-rolling (thresholds recalibrated daily from a K-day trailing buffer -> regime-adaptive,
the deployable one). Reuses SAVED per-fold weights (wf_models/) -- NO training.
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


SYMK = "DOGE"
E = load_rr("maker_labels_rr", SYMK)
F = E["F"]; rH = E["rH"]; day = E["day"]; fee = E["fee"]; ndays = E["ndays"]
fl = E["fl"][0]; fs = E["fs"][0]
netl = E["pl"][:, 0, :].astype(np.float64) * 100.0 - fee
nets = E["ps"][:, 0, :].astype(np.float64) * 100.0 - fee

folds = []; ts = W + EMB; fi = 0
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() < 50 or trn.sum() < 5000:
        ts += T; continue
    A = load_model(f"{MODELS_DIR}/fold{fi}_A.json"); B = load_model(f"{MODELS_DIR}/fold{fi}_B.json")
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    folds.append({"pA_tr": A.predict(xgb.DMatrix(F[tri])), "pB_tr": B.predict(xgb.DMatrix(F[tri])), "day_tr": day[tri],
                  "pA_te": A.predict(xgb.DMatrix(F[tei])), "pB_te": B.predict(xgb.DMatrix(F[tei])), "day_te": day[tei],
                  "fl": fl[tei], "fs": fs[tei], "nl": netl[0][tei], "ns": nets[0][tei],
                  "ndays_te": len(set(day[tei].tolist()))})
    fi += 1; ts += T
tot_te_days = sum(f["ndays_te"] for f in folds)
print(f"[loaded {fi} folds of saved weights, NO training] OOS test = {tot_te_days} days\n", flush=True)


def realize(idx_sel, fd):
    if not len(idx_sel):
        return np.array([])
    side = fd["pB_te"][idx_sel] >= 0.5
    net = np.where(side, fd["nl"][idx_sel], fd["ns"][idx_sel])
    fc = np.where(side, fd["fl"][idx_sel], fd["fs"][idx_sel])
    ex = fc & np.isfinite(net); return net[ex]


def gated_posthoc(fd, qA, target_tpd):
    pA = fd["pA_te"]; b = np.abs(fd["pB_te"] - 0.5)
    thrA = np.quantile(pA, 1 - qA); amask = pA >= thrA
    nd = len(set(fd["day_te"].tolist())); tgt = int(round(target_tpd * nd))
    if 0 < tgt < amask.sum():
        thrB = np.partition(b[amask], -tgt)[-tgt]
    else:
        thrB = -np.inf
    return realize(np.where(amask & (b >= thrB))[0], fd)


def gated_frozen(fd, qA, target_tpd):
    pAtr = fd["pA_tr"]; btr = np.abs(fd["pB_tr"] - 0.5)
    thrA = np.quantile(pAtr, 1 - qA); aT = pAtr >= thrA
    wpd = len(fd["pA_tr"]) / max(len(set(fd["day_tr"].tolist())), 1)
    qB = min(1.0, target_tpd / max(qA * wpd, 1e-9))
    thrB = np.quantile(btr[aT], 1 - qB) if aT.any() else 0.0
    b = np.abs(fd["pB_te"] - 0.5)
    return realize(np.where((fd["pA_te"] >= thrA) & (b >= thrB))[0], fd)


def gated_rolling(fd, qA, target_tpd, Kdays=30):
    days = sorted(set(fd["day_te"].tolist())); wpd = len(fd["pA_te"]) / max(len(days), 1)
    qB = min(1.0, target_tpd / max(qA * wpd, 1e-9))
    tr_days = sorted(set(fd["day_tr"].tolist())); seed = np.isin(fd["day_tr"], tr_days[-Kdays:])
    bufA = list(fd["pA_tr"][seed])
    thrA0 = np.quantile(fd["pA_tr"][seed], 1 - qA) if seed.any() else 0.0
    sA = seed & (fd["pA_tr"] >= thrA0)
    bufB = list(np.abs(fd["pB_tr"][sA] - 0.5)); capA = max(int(Kdays * wpd), 1); capB = max(int(Kdays * wpd * qA), 50)
    sel = []
    for d in days:
        idx = np.where(fd["day_te"] == d)[0]
        thrA = float(np.quantile(bufA, 1 - qA)) if bufA else 0.0
        thrB = float(np.quantile(bufB, 1 - qB)) if bufB else 0.0
        am = fd["pA_te"][idx] >= thrA; b = np.abs(fd["pB_te"][idx] - 0.5)
        sel.extend(idx[am & (b >= thrB)].tolist())
        bufA.extend(fd["pA_te"][idx].tolist()); bufA = bufA[-capA:]
        bufB.extend(b[am].tolist()); bufB = bufB[-capB:]
    return realize(np.array(sel, dtype=int), fd)


def stat(nets):
    nets = np.asarray(nets); n = len(nets)
    ev = float(nets.mean()) if n else float("nan"); tpd = n / max(tot_te_days, 1)
    return f"trd/day={tpd:4.1f} EV/trd={ev*0.01:+.4f}% daily={ev*tpd*0.01:+.3f}% OOS-{tot_te_days}d={float(nets.sum())*0.01:+5.1f}%"


def statf(nets):
    nets = np.asarray(nets); n = len(nets)
    if not n:
        return "n=0 (no trades)"
    ev = float(nets.mean()); se = float(nets.std() / np.sqrt(n)); tpd = n / max(tot_te_days, 1)
    return (f"trd/day={tpd:4.1f} EV/trd={ev*0.01:+.4f}%+-{se*0.01:.4f} "
            f"daily={ev*tpd*0.01:+.3f}% OOS={float(nets.sum())*0.01:+6.1f}%")


print("=== GATED causal-rolling: tight A-gate x budget (deployable) + per-fold robustness ===")
GRID2 = [("qA1.0%", 0.010, 1), ("qA0.5%", 0.005, 1), ("qA1.0%", 0.010, 5), ("qA0.5%", 0.005, 5),
         ("qA1.0%", 0.010, 10), ("qA0.5%", 0.005, 10)]
for lab, qA, tgt in GRID2:
    pf = [gated_rolling(fd, qA, tgt) for fd in folds]
    alln = np.concatenate(pf) if pf else np.array([])
    print(f"  {lab} {tgt:>2}/d | {statf(alln)}")
    print(f"        per-fold OOS%: {[(f'{p.sum()*0.01:+.1f}(n{len(p)})') for p in pf]}")
print("\n(ref single-run: gated qA1% 5/day +21.7%; AB-product causal-rolling A2~1/day +11.8%)")
