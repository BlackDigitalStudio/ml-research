#!/usr/bin/env python3
"""Do we need the vol-gate (Model A)?  no-A (B on FULL DOGE universe, deploy by B-confidence,
NO vol-gate) vs A x B (gated), ZERO maker fee, short horizons (15s/30s). One symbol (DOGE).

Tests whether at zero fee a high-frequency directional maker (predict 15-30s side, post the
favourable side, catch small moves) beats the selective A-gated vol-harvester. B-full trained on
ALL filled train windows (payoff-weighted, no apred gate). Deploy no-A by |pB-0.5| at 1/5/20/50
trades/day; reference A x B at 1/5. net = GROSS (zero fee). Reports net bp/trade, total bp/day,
B dir-AUC, per-fold. Reads research_runs/maker_labels_h/DOGE.npz.
"""
import io, json
import numpy as np
from google.cloud import storage
import xgboost as xgb

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; MAIN = "research_runs/xgb_maker"
W, T, EMB = 200, 30, 2; NF_RATE = 0.05; GATE_PCT = 5.0
HORIZONS = [("15s", 0, "rH15"), ("30s", 1, "rH30")]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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


def oof_pA(F, yA, trn, day, hpA, k=4):
    tdays = sorted(set(day[trn].tolist())); fold = {dd: i % k for i, dd in enumerate(tdays)}
    fday = np.array([fold.get(int(dd), -1) for dd in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or trk.sum() < 500 or (yA[trk] == 1).sum() < 20:
            continue
        spwk = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = fit(hpA["best_params"], hpA["best_iter"], F[trk], yA[trk], spw=spwk); oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


hpA = jload(f"{MAIN}/A_DOGE.json"); hpB = jload(f"{MAIN}/B_pool.json")
d = np.load(io.BytesIO(bk.blob("research_runs/maker_labels_h/DOGE.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float32); day = d["day"]
rHk = {"rH15": d["rH15"].astype(np.float64), "rH30": d["rH30"].astype(np.float64), "rH60": d["rH60"].astype(np.float64)}
PL = d["pnl_long"].astype(np.float64); PS = d["pnl_short"].astype(np.float64)
fl = d["fill_long"].astype(bool)[0]; fs = d["fill_short"].astype(bool)[0]

FOLDS = []; ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays)
    trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    ts += T
tot_days = sum(len(set(day[tst].tolist())) for _, tst in FOLDS)
print(f"[no-A vs AxB | ZERO fee] {len(FOLDS)} folds, OOS={tot_days} days\n", flush=True)


def trainB(F_, netl, nets, mask):
    nl = netl; ns = nets
    yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int); both = fl & fs
    wq = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
    keep = mask & (fl | fs); pos = wq[keep][wq[keep] > 0]
    wc = np.clip(wq[keep], 0, np.quantile(pos, 0.99) if len(pos) else 1.0)
    return fit(hpB["best_params"], hpB["best_iter"], F_[keep], yB[keep], w=wc), keep


def realize(sel, pB, netl, nets, ti):
    side = pB[sel] >= 0.5; net = np.where(side, netl[ti][sel], nets[ti][sel])
    fc = np.where(side, fl[ti][sel], fs[ti][sel]); ex = fc & np.isfinite(net); return net[ex]


def run(cfgidx, rhkey):
    rH = rHk[rhkey]; netl = PL[cfgidx, 0, :] * 100.0; nets = PS[cfgidx, 0, :] * 100.0   # zero fee
    res = {f"AxB_{b}": [] for b in (1, 5)}; res.update({f"noA_{b}": [] for b in (1, 5, 20, 50)})
    bauc_full, bauc_gate = [], []
    for trn, tst in FOLDS:
        ti = np.where(tst)[0]
        # A + B-gated (reference)
        thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
        spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
        bstA = fit(hpA["best_params"], hpA["best_iter"], F[trn], yA[trn], spw=spw); pA = bstA.predict(xgb.DMatrix(F[ti]))
        oof = oof_pA(F, yA, trn, day, hpA); valid = trn & np.isfinite(oof)
        gate = valid & (oof >= np.nanquantile(oof[valid], 1 - GATE_PCT / 100.0))
        bBg, _ = trainB(F, netl, nets, gate); pBg = bBg.predict(xgb.DMatrix(F[ti]))
        # B-full (no gate)
        bBf, _ = trainB(F, netl, nets, trn); pBf = bBf.predict(xgb.DMatrix(F[ti]))
        # dir-AUC (better-side label) on test filled
        both_te = fl[ti] & fs[ti]; yb_te = (netl[ti] > nets[ti]).astype(int)
        bauc_full.append(auc(yb_te[both_te], pBf[both_te])); bauc_gate.append(auc(yb_te[both_te], pBg[both_te]))
        dyt = day[ti]
        for b in (1, 5):
            sc = pct_rank(pA) * pct_rank(np.abs(pBg - 0.5)); res[f"AxB_{b}"].append(realize(daily_pick(dyt, sc, b), pBg, netl, nets, ti))
        for b in (1, 5, 20, 50):
            sc = np.abs(pBf - 0.5); res[f"noA_{b}"].append(realize(daily_pick(dyt, sc, b), pBf, netl, nets, ti))
    return res, float(np.nanmean(bauc_full)), float(np.nanmean(bauc_gate))


def summ(pf):
    a = np.concatenate(pf) if pf else np.array([]); n = len(a)
    ev = float(a.mean()) if n else float("nan"); tpd = n / max(tot_days, 1)
    return ev, tpd, ev * tpd, [round(float(p.sum() * 0.01), 1) for p in pf]


OUT = {}
for hlabel, cfgidx, rhkey in HORIZONS:
    res, bf, bg = run(cfgidx, rhkey)
    print(f"=== {hlabel} | B dir-AUC: full={bf:.3f} gated={bg:.3f} | net=GROSS(zero fee) ===", flush=True)
    print(f"  {'policy':>10} {'trd/day':>8} {'net bp/trd':>11} {'tot bp/day':>11}  per-fold OOS%")
    for key in ["AxB_1", "AxB_5", "noA_1", "noA_5", "noA_20", "noA_50"]:
        ev, tpd, tot, pf = summ(res[key])
        print(f"  {key:>10} {tpd:>8.1f} {ev:>+10.2f} {tot:>+10.2f}   {pf}", flush=True)
        OUT[f"{hlabel}_{key}"] = {"net_bp_trd": ev, "trd_day": tpd, "tot_bp_day": tot, "perfold": pf}
    OUT[f"{hlabel}_Bauc"] = {"full": bf, "gated": bg}
    print(flush=True)
bk.blob("research_runs/maker_labels_h/NOA_TEST_RESULT.json").upload_from_string(json.dumps(OUT, default=float))
print("[saved] research_runs/maker_labels_h/NOA_TEST_RESULT.json", flush=True)
