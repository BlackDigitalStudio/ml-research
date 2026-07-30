#!/usr/bin/env python3
"""Re-accounting of the published year cells under the STRICT entry-fill model.

OPS-FILLYEAR rev1 / cell A1. Nothing is retrained: the per-fold test SCORES and the
per-seed SIDES come from the existing PERFOLD_S* artifacts, so the models, the search,
the vol-norm, tau and the selection rule are byte-identical to the published cells. The
only thing swapped is the label pair (fill, pnl), from the frozen gap-through model to
the strict price-resolved one produced by research/runtime/strictfill_year.py.

Both label sets are evaluated in the SAME run, so every strict number has its frozen
twin computed by the identical code path — the difference is attributable to the fill
model and to nothing else. The frozen twin doubles as the validation: it must reproduce
the published cell (DOGE +10.51 / XRP +21.31 / BTC +12.27 / ETH +16.84).

Row join (the part that can silently go wrong, so it is gated):
  PERFOLD stores day_te but not the row index. The fold test set is a contiguous day
  range, and tei = np.where(tst)[0] is ascending, so idx = where(isin(day, uniq(day_te)))
  reconstructs it exactly. GATES: (1) day[idx] must equal day_te elementwise;
  (2) the FROZEN labels at idx must equal PERFOLD's stored netl/nets/fl/fs. Gate (2) is
  the strong one — it proves the strict npz is row-aligned with the score artifacts.

Policies (as deployed and published):
  fixq  frozen tau = quantile of the last-KDAYS train-window scores at K/day
        selectivity, frozen for the whole fold   (DOGE t10, XRP t5)
  dyn   causal rolling daily tau                 (BTC t5, ETH t5)
Ensemble = mean of the per-seed rank scores, side = seed majority vote (ties -> long,
i.e. votes >= ceil(n/2), the published rule at 4 seeds being votes >= 2).

Env: SYM, SCORE_SUB, STRICT_SUB, NSEED (4), K (5), POLICY (fixq|dyn), CFGIDX (1 = 150s
hold), OUT (gs key for the json), HARMONY (0|1, ETH safety form).
"""
import io, json, os, sys

import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
SCORE_SUB = os.environ.get("SCORE_SUB", "research_runs/maker_labels_tb3s_h150anch")
STRICT_SUB = os.environ.get("STRICT_SUB", "research_runs/maker_labels_tb3s_h150strict")
NSEED = int(os.environ.get("NSEED", "4"))
K = float(os.environ.get("K", "5"))
POLICY = os.environ.get("POLICY", "fixq")
CFGIDX = int(os.environ.get("CFGIDX", "1"))
HARMONY = os.environ.get("HARMONY", "0") == "1"
OUT = os.environ.get("OUT", f"research_runs/strictfill_cells/{SYM}_{POLICY}_t{int(K)}.json")
KDAYS = 30

bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s):
    print(s, flush=True)


# ---------------------------------------------------------------- label source
log(f"[load] {STRICT_SUB}/{SYM}.npz")
z = np.load(io.BytesIO(bk.blob(f"{STRICT_SUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
day_all = z["day"].astype(int)
LAB = {}
for tag, suf in (("strict", ""), ("frozen", "_frozen")):
    LAB[tag] = dict(
        nl=z[f"pnl_long{suf}"][CFGIDX, 0, :].astype(np.float64) * 100.0,
        ns=z[f"pnl_short{suf}"][CFGIDX, 0, :].astype(np.float64) * 100.0,
        fl=z[f"fill_long{suf}"][0].astype(bool),
        fs=z[f"fill_short{suf}"][0].astype(bool))
log(f"[load] N={len(day_all)} days={day_all.max()+1} "
    f"fill(frozen)={LAB['frozen']['fl'].mean():.4f} fill(strict)={LAB['strict']['fl'].mean():.4f}")

nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SCORE_SUB}/PERFOLD_S0_{SYM}_qm0_f")
         if b.name.endswith(".npz"))
log(f"[folds] {nf}  seeds={NSEED}  policy={POLICY} K={K} harmony={HARMONY}")

# ---------------------------------------------------------------- folds + gates
folds = []
for f in range(nf):
    zs = [np.load(io.BytesIO(bk.blob(f"{SCORE_SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
          for s in range(NSEED)]
    z0 = zs[0]
    day_te = z0["day_te"].astype(int)
    idx = np.where(np.isin(day_all, np.unique(day_te)))[0]
    if len(idx) != len(day_te) or not np.array_equal(day_all[idx], day_te):
        raise SystemExit(f"JOIN GATE 1 FAILED fold{f}: reconstructed index does not match day_te "
                         f"({len(idx)} vs {len(day_te)})")
    fz = LAB["frozen"]
    g2 = (np.array_equal(fz["fl"][idx], z0["fl"].astype(bool))
          and np.array_equal(fz["fs"][idx], z0["fs"].astype(bool))
          and np.allclose(fz["nl"][idx], z0["netl"].astype(np.float64), rtol=0, atol=2e-3, equal_nan=True)
          and np.allclose(fz["ns"][idx], z0["nets"].astype(np.float64), rtol=0, atol=2e-3, equal_nan=True))
    if not g2:
        d = np.abs(fz["nl"][idx] - z0["netl"].astype(np.float64))
        raise SystemExit(f"JOIN GATE 2 FAILED fold{f}: frozen labels != PERFOLD labels "
                         f"(max|dnetl|={np.nanmax(d):.6f}, fill mismatch "
                         f"{(fz['fl'][idx] != z0['fl'].astype(bool)).sum()})")
    p = dict(day_tr=z0["day_tr"].astype(int), day_te=day_te, idx=idx,
             tr=np.mean([x["axb_tr"].astype(np.float64) for x in zs], 0),
             te=np.mean([x["axb_te"].astype(np.float64) for x in zs], 0),
             side=np.sum([x["side"].astype(int) for x in zs], 0) >= int(np.ceil(NSEED / 2)),
             seed_tr=[x["axb_tr"].astype(np.float64) for x in zs],
             seed_te=[x["axb_te"].astype(np.float64) for x in zs],
             seed_side=[x["side"].astype(bool) for x in zs])
    folds.append(p)
    log(f"  fold{f}: n_te={len(day_te)} days={len(np.unique(day_te))} GATES OK")

tot_days = sum(len(np.unique(p["day_te"])) for p in folds)
log(f"[gates] all folds passed the row-join gates; test days total = {tot_days}")


# ---------------------------------------------------------------- selection
def fixq_tau(tr, day_tr, k):
    trd = sorted(set(day_tr.tolist()))[-KDAYS:]
    s = tr[np.isin(day_tr, trd)]
    if not len(s):
        return float("inf")
    wpd = len(s) / max(len(trd), 1)
    return float(np.quantile(s, max(0.0, 1.0 - k / max(wpd, 1.0))))


def sel_fixq(p, te, tr, k):
    return np.where(te >= fixq_tau(tr, p["day_tr"], k))[0]


def sel_dyn(p, te, tr, k):
    days = sorted(set(p["day_te"].tolist())); wpd = len(te) / max(len(days), 1)
    q = max(0.0, 1.0 - k / max(wpd, 1.0))
    trd = sorted(set(p["day_tr"].tolist())); seed = np.isin(p["day_tr"], trd[-KDAYS:])
    buf = list(tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        i = np.where(p["day_te"] == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(i[te[i] >= tau].tolist()); buf.extend(te[i].tolist()); buf = buf[-cap:]
    return np.array(sel, dtype=int)


SEL = {"fixq": sel_fixq, "dyn": sel_dyn}[POLICY]


def harmony_mask(p, k):
    """SAFETY form: keep only decisions every seed would also have taken on its OWN
    frozen tau (per-seed agreement), and on which the seeds agree on the side."""
    m = np.ones(len(p["te"]), bool)
    for s in range(NSEED):
        te_s, tr_s = p["seed_te"][s], p["seed_tr"][s]
        sel = np.zeros(len(te_s), bool)
        sel[SEL(p, te_s, tr_s, k)] = True
        m &= sel
        m &= (p["seed_side"][s] == p["side"])
    return m


def take(p, sel, lab):
    ii = p["idx"][sel]
    sd = p["side"][sel]
    net = np.where(sd, lab["nl"][ii], lab["ns"][ii])
    fc = np.where(sd, lab["fl"][ii], lab["fs"][ii])
    ex = fc & np.isfinite(net)
    return net[ex], p["day_te"][sel][ex]


def cell(sel_per_fold, lab):
    nets, days, per = [], [], []
    for p, sel in zip(folds, sel_per_fold):
        n, d = take(p, sel, lab)
        nets.append(n); days.append(d); per.append(n)
    a = np.concatenate(nets) if nets else np.array([])
    if not len(a):
        return dict(ev=float("nan"), n=0, tpd=0.0, bpd=0.0, hit=float("nan"), perfold=[]), None
    ev = float(a.mean()); tpd = len(a) / max(tot_days, 1)
    return (dict(ev=ev, n=int(len(a)), tpd=tpd, bpd=ev * tpd, hit=float((a > 0).mean()),
                 perfold=[f"{x.mean():+.2f}({len(x)})" if len(x) else "n/a" for x in per]),
            (a, np.concatenate(days)))


res = {"sym": SYM, "policy": POLICY, "K": K, "nseed": NSEED, "harmony": HARMONY,
       "score_sub": SCORE_SUB, "strict_sub": STRICT_SUB, "cfgidx": CFGIDX,
       "tot_days": tot_days, "nfolds": nf, "cells": {}, "trades": {}}

# ensemble selection is computed ONCE and reused for both label sets — identical
# decisions by construction, which is the whole point of the comparison
ens_sel = []
for p in folds:
    sel = SEL(p, p["te"], p["tr"], K)
    if HARMONY:
        hm = harmony_mask(p, K)
        sel = sel[hm[sel]]
    ens_sel.append(sel)

for tag in ("frozen", "strict"):
    c, tr_ = cell(ens_sel, LAB[tag])
    res["cells"][f"ens_{tag}"] = c
    if tr_ is not None:
        res["trades"][f"ens_{tag}"] = dict(net=tr_[0].tolist(), day=tr_[1].tolist())
    log(f"[ENS {tag:6s}] EV {c['ev']:+7.3f}bp  n={c['n']:5d}  tpd {c['tpd']:5.2f}  "
        f"bpd {c['bpd']:+7.2f}  hit {100*c['hit']:.1f}%  perfold {c['perfold']}")

for s in range(NSEED):
    ssel = []
    for p in folds:
        sel = SEL(p, p["seed_te"][s], p["seed_tr"][s], K)
        ssel.append(sel)
    # a per-seed cell uses that seed's OWN side, not the ensemble vote
    saved = [p["side"] for p in folds]
    for p in folds:
        p["side"] = p["seed_side"][s]
    for tag in ("frozen", "strict"):
        c, _ = cell(ssel, LAB[tag])
        res["cells"][f"seed{s}_{tag}"] = c
    for p, sv in zip(folds, saved):
        p["side"] = sv
    log(f"[seed{s}]   frozen EV {res['cells'][f'seed{s}_frozen']['ev']:+7.3f} "
        f"(n={res['cells'][f'seed{s}_frozen']['n']})   "
        f"strict EV {res['cells'][f'seed{s}_strict']['ev']:+7.3f} "
        f"(n={res['cells'][f'seed{s}_strict']['n']})")

ef, es = res["cells"]["ens_frozen"], res["cells"]["ens_strict"]
res["ratio_bpd"] = (es["bpd"] / ef["bpd"]) if ef["bpd"] else float("nan")
res["fill_rate_selected"] = {}
sel_n = sum(len(s) for s in ens_sel)
res["fill_rate_selected"] = {"frozen": ef["n"] / max(sel_n, 1), "strict": es["n"] / max(sel_n, 1),
                             "selected": sel_n}
log(f"[SUMMARY {SYM} {POLICY} t{int(K)}] selected {sel_n} | filled frozen {ef['n']} "
    f"({100*ef['n']/max(sel_n,1):.1f}%) strict {es['n']} ({100*es['n']/max(sel_n,1):.1f}%) | "
    f"EV {ef['ev']:+.2f} -> {es['ev']:+.2f} | bpd {ef['bpd']:+.2f} -> {es['bpd']:+.2f} "
    f"= {100*res['ratio_bpd']:.0f}% of the published cell")

bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
log(f"[saved] gs://{BUCKET}/{OUT}")
