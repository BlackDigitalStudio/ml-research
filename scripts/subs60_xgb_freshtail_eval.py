#!/usr/bin/env python3
"""Fresh-tail validation of the DOGE apred cascade (+5.73), with PERIOD-ADAPTIVE
Model-A vol-gate threshold.

Model A's label = |rH60| >= train-p95(|rH60|) (the "5% most-volatile deviation").
This threshold is PERIOD-DEPENDENT and must be recomputed per training window.
We compare three trainings (all on clean data only; fresh tail held out):
  stale          : train May..Nov 2025, thr=p95 of that window   (= the +5.73 model)
  current(own)   : train all clean thru 05-08, thr=p95 of all     (period-correct)
  current(stale-thr): train all clean, thr=p95 of May..Nov 2025   (the earlier mismatch)
Eval = net maker EV, 1 trade/day, hold-60s, maker 4bp, on maker_labels_rr_freshtail.
"""
import io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05; GATE_PCT = 5.0
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def load_rr(sub, symk):
    d = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "pl": d["pnl_long"].astype(np.float32), "ps": d["pnl_short"].astype(np.float32),
            "fl": d["fill_long"].astype(bool), "fs": d["fill_short"].astype(bool),
            "ndays": m["n_days"], "fee": m["maker_rt_fee_pct"] * 100.0}


def split(day, ndays):
    cut = int(ndays * SPLIT[0]); emb = int(ndays * SPLIT[1]); tr = day < cut
    td = sorted(set(day[tr].tolist())); vcut = td[int(len(td) * SPLIT[2])] if td else cut
    return (tr & (day < vcut)), (day >= emb)


def pct_rank(x): o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score, n=1):
    order = np.lexsort((-score, day)); ds = day[order]
    st = np.zeros(len(order), bool); st[0] = True; st[1:] = ds[1:] != ds[:-1]
    si = np.where(st)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


def fit(hp, niter, X, y, w=None, spw=None):
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    if spw is not None:
        base["scale_pos_weight"] = spw
    return xgb.train(dict(base, **hp), xgb.DMatrix(X, label=y, weight=w), num_boost_round=max(1, niter + 1))


def oof_pA(F, yA, trn, day, hpA, k=5):
    tdays = sorted(set(day[trn].tolist())); fold = {d: i % k for i, d in enumerate(tdays)}
    fday = np.array([fold.get(int(d), -1) for d in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], hpA["best_iter"], F[trk], yA[trk], spw=spwk)
        oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


SYMK = "DOGE"
hpA = jload(f"{MAIN}/A_{SYMK}.json"); hpB = jload(f"{MAIN}/B_pool.json")
E = load_rr("maker_labels_rr", SYMK)
F = E["F"]; rH = E["rH"]; day = E["day"]; fee = E["fee"]
trn_stale, te = split(day, E["ndays"])
trn_curr = np.ones(len(day), bool)
fl = E["fl"][0]; fs = E["fs"][0]
netl = E["pl"][:, 0, :].astype(np.float64) * 100.0 - fee
nets = E["ps"][:, 0, :].astype(np.float64) * 100.0 - fee


def train_model(trn, thr_mask=None, lab=""):
    tm = trn if thr_mask is None else thr_mask
    thr_t = float(np.quantile(np.abs(rH[tm]), 1 - NF_RATE))
    yA = (np.abs(rH) >= thr_t).astype(int)
    print(f"    [{lab}] A vol-gate thr=p95|rH60|={thr_t:.2f}bp  non-flat_rate(on train)={yA[trn].mean():.3f}")
    spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
    bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw)
    oof = oof_pA(F, yA, trn, day, hpA); valid = trn & np.isfinite(oof)
    thr_oof = float(np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0)); gate = valid & (oof >= thr_oof)
    keep = gate & (fl | fs); nl = netl[0]; ns = nets[0]
    yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int); both = fl & fs
    w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
    wc = np.clip(w[keep], 0, np.quantile(w[keep][w[keep] > 0], 0.99))
    bstB = fit(hpB["best_params"], hpB["best_iter"], F[keep], yB[keep], w=wc)
    return bstA, bstB


def evalp(bstA, bstB, F2, day2, fl2, fs2, nl2, ns2, feec, lab):
    pA = bstA.predict(xgb.DMatrix(F2)); pB = bstB.predict(xgb.DMatrix(F2))
    score = pct_rank(pA) * pct_rank(np.abs(pB - 0.5)); sel = daily_pick(day2, score, 1)
    side = pB[sel] >= 0.5; net = np.where(side, nl2[0][sel], ns2[0][sel]); gross = net + feec
    fc = np.where(side, fl2[sel], fs2[sel]); ex = fc & np.isfinite(net)
    ev = float(net[ex].mean()) if ex.any() else float("nan")
    gr = float(gross[ex].mean()) if ex.any() else float("nan")
    md = float(np.median(net[ex])) if ex.any() else float("nan")
    se = float(np.std(net[ex]) / max(np.sqrt(ex.sum()), 1)) if ex.any() else float("nan")
    print(f"  {lab}: n={int(ex.sum())} net={ev:+.2f}±{se:.1f}bp (gross {gr:+.2f}) median={md:+.2f} pos={int((net[ex]>0).sum())}/{int(ex.sum())}")
    return ev


# period volatility context
print(f"[vol context] p95|rH60|:  stale(May-Nov2025)={np.quantile(np.abs(rH[trn_stale]),0.95):.2f}bp  "
      f"all-thru-0508={np.quantile(np.abs(rH[trn_curr]),0.95):.2f}bp")
G = load_rr("maker_labels_rr_freshtail", SYMK)
print(f"              fresh-tail(05-09..06-01)={np.quantile(np.abs(G['rH']),0.95):.2f}bp\n")

print("[training]")
As, Bs = train_model(trn_stale, lab="stale")
Acf, Bcf = train_model(trn_curr, lab="current(own-thr)")
Acb, Bcb = train_model(trn_curr, thr_mask=trn_stale, lab="current(stale-thr)")

print("\n[sanity] stale on ORIGINAL test (expect ~+5.73):")
evalp(As, Bs, F[te], day[te], fl[te], fs[te], netl[:, te], nets[:, te], fee, "orig-test")

Ff = G["F"]; dayf = G["day"]; flf = G["fl"][0]; fsf = G["fs"][0]
nlf = G["pl"][:, 0, :].astype(np.float64) * 100.0 - G["fee"]
nsf = G["ps"][:, 0, :].astype(np.float64) * 100.0 - G["fee"]
print("\n[FRESH-TAIL 2026-05-09..06-01 — held out, pessimistic]:")
evalp(As, Bs, Ff, dayf, flf, fsf, nlf, nsf, G["fee"], "stale            (May-Nov thr)")
evalp(Acf, Bcf, Ff, dayf, flf, fsf, nlf, nsf, G["fee"], "current own-thr  (period-correct)")
evalp(Acb, Bcb, Ff, dayf, flf, fsf, nlf, nsf, G["fee"], "current stale-thr(the mismatch)")
