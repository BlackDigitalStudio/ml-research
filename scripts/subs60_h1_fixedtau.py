#!/usr/bin/env python3
"""HD5 rev1 — user hypothesis 1: FIXED confidence threshold instead of the dynamic causal
tau. Rule (user-specified, frozen): on the validation window (last KDAYS=30 train days of
each fold — the same window that seeds the deployed rolling tau), take the post-factum
top-K scores per day (K=5 for t5, K=10 for t10), set tau_fixed = MEAN of all those top-K
scores; on the fold's test window select score >= tau_fixed, FROZEN for the whole fold
(no daily adaptation). Compare against the dynamic causal-rolling tau (deployed policy)
computed from the SAME PERFOLD artifacts — per-seed (4) and ensemble (mean rank-score,
majority side) variants, per symbol. Metrics: EV/tr, trades/day, bp/day (=EV*tpd, the
economic total), hit, per-fold, zero-trade-day share; ensemble cells get the REQUIRED
score-jitter perturbation gate (sd .02/.05 x 100 reps) for BOTH policies.
No retraining — pure re-analysis of saved fold scores (anchored h150, rev8/s22 artifacts).
Env: SYMS. Out: print + research_runs/h1_fixedtau/{SYM}_h1.json
"""
import io, json, os
import numpy as np
from google.cloud import storage

SYMS = os.environ.get("SYMS", "BTC,ETH,DOGE,XRP").split(",")
SUB = "research_runs/maker_labels_tb3s_h150anch"
OUT = "research_runs/h1_fixedtau"
KDAYS = 30
BUDGETS = [5, 10]
bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")


def trades(p, sel):
    sd_ = p["side"][sel]; net = np.where(sd_, p["nl"][sel], p["ns"][sel])
    fc = np.where(sd_, p["fl"][sel], p["fs"][sel])
    ex = fc & np.isfinite(net)
    return net[ex], p["day_te"][sel][ex]


def fixed_tau(p, K):
    """tau = mean of per-day top-K train scores over the last KDAYS train days."""
    trd = sorted(set(p["day_tr"].tolist()))[-KDAYS:]
    tops = []
    for d in trd:
        s = p["tr"][p["day_tr"] == d]
        if len(s):
            tops.append(np.sort(s)[-K:])
    return float(np.mean(np.concatenate(tops))) if tops else float("inf")


def fixq_tau(p, K):
    """FIXQ variant: tau = the val-window quantile matched to K/day selectivity,
    frozen for the whole fold test (hardcoded number, no daily adaptation)."""
    trd = sorted(set(p["day_tr"].tolist()))[-KDAYS:]
    m = np.isin(p["day_tr"], trd)
    s = p["tr"][m]; nd = len(trd)
    if not len(s):
        return float("inf")
    wpd = len(s) / max(nd, 1)
    q = max(0.0, 1.0 - K / max(wpd, 1.0))
    return float(np.quantile(s, q))


def sel_fixed(p, K, jit=0.0, rng=None, key="tauf"):
    tau = p[f"{key}{K}"]
    te = p["te"] + (rng.normal(0, jit, len(p["te"])) if jit > 0 else 0.0)
    return np.where(te >= tau)[0]


def sel_dyn(p, tgt, jit=0.0, rng=None):
    days = sorted(set(p["day_te"].tolist())); wpd = len(p["te"]) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(p["day_tr"].tolist())); seed = np.isin(p["day_tr"], trd[-KDAYS:])
    te = p["te"] + (rng.normal(0, jit, len(p["te"])) if jit > 0 else 0.0)
    buf = list(p["tr"][seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(p["day_te"] == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[te[idx] >= tau].tolist()); buf.extend(te[idx].tolist()); buf = buf[-cap:]
    return np.array(sel, dtype=int)


def evaluate(folds, selfn, tot_days):
    nets, days_sel, perfold = [], [], []
    for p in folds:
        s = selfn(p)
        n, d = trades(p, s)
        nets.append(n); days_sel.append(d); perfold.append(n)
    a = np.concatenate(nets)
    if not len(a):
        return dict(ev=float("nan"), n=0, tpd=0.0, bpd=0.0, hit=float("nan"), perfold=[], zshare=1.0)
    dsel = np.concatenate(days_sel)
    all_days = set()
    for p in folds:
        all_days.update(set(p["day_te"].tolist()))
    zshare = 1.0 - len(set(dsel.tolist())) / max(len(all_days), 1)
    ev = float(a.mean()); tpd = len(a) / max(tot_days, 1)
    return dict(ev=ev, n=int(len(a)), tpd=tpd, bpd=ev * tpd, hit=float((a > 0).mean()),
                perfold=[f"{p.mean():+.1f}({len(p)})" if len(p) else "n/a" for p in perfold],
                zshare=zshare)


for SYM in SYMS:
    nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SUB}/PERFOLD_S0_{SYM}_qm0_f")
             if b.name.endswith(".npz"))
    print(f"\n================ {SYM} — {nf} folds ================", flush=True)
    seedfolds = {s: [] for s in range(4)}; ensfolds = []
    for f in range(nf):
        zs = [np.load(io.BytesIO(bk.blob(f"{SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
              for s in range(4)]
        z0 = zs[0]
        base = dict(day_tr=z0["day_tr"], day_te=z0["day_te"], fl=z0["fl"], fs=z0["fs"],
                    nl=z0["netl"].astype(np.float64), ns=z0["nets"].astype(np.float64))
        for s in range(4):
            seedfolds[s].append(dict(base, tr=zs[s]["axb_tr"].astype(np.float64),
                                     te=zs[s]["axb_te"].astype(np.float64), side=zs[s]["side"]))
        votes = np.sum([z["side"].astype(int) for z in zs], 0)
        ensfolds.append(dict(base, tr=np.mean([z["axb_tr"].astype(np.float64) for z in zs], 0),
                             te=np.mean([z["axb_te"].astype(np.float64) for z in zs], 0),
                             side=votes >= 2))
    tot_days = sum(len(set(p["day_te"].tolist())) for p in ensfolds)
    for fl_ in list(seedfolds.values()) + [ensfolds]:
        for p in fl_:
            for K in BUDGETS:
                p[f"tauf{K}"] = fixed_tau(p, K)
                p[f"tauq{K}"] = fixq_tau(p, K)
    res = {"sym": SYM, "nf": nf, "tot_days": tot_days}
    for K in BUDGETS:
        print(f"-- budget K={K} (rule: tau = mean of val-window per-day top-{K}) --", flush=True)
        pseed_f, pseed_q, pseed_d = [], [], []
        for s in range(4):
            rf = evaluate(seedfolds[s], lambda p: sel_fixed(p, K), tot_days)
            rq = evaluate(seedfolds[s], lambda p: sel_fixed(p, K, key="tauq"), tot_days)
            rd = evaluate(seedfolds[s], lambda p: sel_dyn(p, K), tot_days)
            pseed_f.append(rf); pseed_q.append(rq); pseed_d.append(rd)
            print(f"  seed{s}: FIXED ev={rf['ev']:+6.2f} tpd={rf['tpd']:4.1f} bpd={rf['bpd']:+6.1f} hit={100*rf['hit']:4.1f}% z0d={rf['zshare']:.2f} "
                  f"| FIXQ ev={rq['ev']:+6.2f} tpd={rq['tpd']:4.1f} bpd={rq['bpd']:+6.1f} "
                  f"| DYN ev={rd['ev']:+6.2f} tpd={rd['tpd']:4.1f} bpd={rd['bpd']:+6.1f}", flush=True)
        ef = evaluate(ensfolds, lambda p: sel_fixed(p, K), tot_days)
        eq = evaluate(ensfolds, lambda p: sel_fixed(p, K, key="tauq"), tot_days)
        ed = evaluate(ensfolds, lambda p: sel_dyn(p, K), tot_days)
        print(f"  ENS  : FIXED ev={ef['ev']:+6.2f} tpd={ef['tpd']:4.1f} bpd={ef['bpd']:+6.1f} hit={100*ef['hit']:4.1f}% z0d={ef['zshare']:.2f} n={ef['n']}", flush=True)
        print(f"         perfold {ef['perfold']}", flush=True)
        print(f"  ENS  : FIXQ  ev={eq['ev']:+6.2f} tpd={eq['tpd']:4.1f} bpd={eq['bpd']:+6.1f} hit={100*eq['hit']:4.1f}% z0d={eq['zshare']:.2f} n={eq['n']}", flush=True)
        print(f"         perfold {eq['perfold']}", flush=True)
        print(f"  ENS  : DYN   ev={ed['ev']:+6.2f} tpd={ed['tpd']:4.1f} bpd={ed['bpd']:+6.1f} hit={100*ed['hit']:4.1f}% n={ed['n']}", flush=True)
        print(f"         perfold {ed['perfold']}", flush=True)
        jf = {}
        for sd_j in (0.02, 0.05):
            for tag, fn in (("FIXED", lambda p, j, r: sel_fixed(p, K, j, r)),
                            ("FIXQ", lambda p, j, r: sel_fixed(p, K, j, r, key="tauq")),
                            ("DYN", lambda p, j, r: sel_dyn(p, K, j, r))):
                rng = np.random.default_rng(0); r = []
                for rep in range(100):
                    a = np.concatenate([trades(p, fn(p, sd_j, rng))[0] for p in ensfolds])
                    r.append(a.mean() if len(a) else np.nan)
                r = np.array(r)
                p50 = np.nanquantile(r, .5); pos = 100 * np.nanmean(r > 0)
                jf[f"{tag}_sd{sd_j}"] = dict(p10=float(np.nanquantile(r, .1)), p50=float(p50), p90=float(np.nanquantile(r, .9)), Ppos=float(pos))
                print(f"  ENS jitter {tag} sd={sd_j}: p50={p50:+.2f} P(EV>0)={pos:.0f}%", flush=True)
        res[f"K{K}"] = dict(perseed_fixed=pseed_f, perseed_fixq=pseed_q, perseed_dyn=pseed_d,
                            ens_fixed=ef, ens_fixq=eq, ens_dyn=ed, jitter=jf,
                            tauf=[[p[f"tauf{K}"] for p in fl_] for fl_ in [ensfolds]],
                            tauq=[[p[f"tauq{K}"] for p in fl_] for fl_ in [ensfolds]])
    bk.blob(f"{OUT}/{SYM}_h1v2.json").upload_from_string(json.dumps(res, default=str))
    print(f"[saved] {OUT}/{SYM}_h1v2.json", flush=True)
