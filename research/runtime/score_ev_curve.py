#!/usr/bin/env python3
"""Score -> EV/tr dependency for the four deployed symbols (MODELCAP, 2026-08-01).

Answers, on the population that produced the published cells: how much is a decision
worth as a function of how highly the ensemble scored it? Nothing is retrained and no
label is rebuilt — the per-fold test scores, the per-seed sides and the frozen labels all
come out of the existing PERFOLD artifacts, so the models, the vol-norm, the walk-forward
and the accounting are byte-identical to the cells.

Two curves per symbol, because they answer different questions:

  marginal    EV/tr inside a score-percentile band — what a decision at that level is
              worth on its own.
  cumulative  EV/tr of "trade everything above this threshold" as a function of the
              traded fraction q — the selectivity surface the policy sits on. Computed in
              TWO selection frames:
                per-fold  threshold set inside each fold, as FIXQ/DYN actually do;
                pooled    one global threshold over the whole test year.
              The score is a rank-CDF built on its OWN fold's train distribution, so its
              LEVELS are not comparable across folds; the gap between the two frames is
              exactly that incomparability and nothing else. On ETH it is the difference
              between a negative and a positive cell — see the ledger record.

Accounting matches research/runtime/strictfill_cells.py: ensemble score = mean of the
per-seed rank scores, side = seed majority vote (votes >= ceil(nseed/2)),
net = side ? netl : nets, and a row counts only where THAT side filled and net is finite.
Labels are the FROZEN ones carried by the PERFOLD artifacts (not the strict ones).

Input: a flat directory of PERFOLD_S{seed}_{SYM}_qm0_f{fold}.npz. Populate it with the
score_sub each cell records (strictfill_cells/{SYM}_*.json -> "score_sub"):

  B=gs://market-data-0998ac51/research_runs
  gcloud storage cp \
    "$B/maker_labels_tb3s_h150anch/PERFOLD_S[0-3]_DOGE_qm0_f*.npz" \
    "$B/maker_labels_tb3s_h150anch/PERFOLD_S[0-3]_XRP_qm0_f*.npz" \
    "$B/maker_labels_tb3s_h150d/PERFOLD_S[0-3]_BTC_qm0_f*.npz" \
    "$B/maker_labels_tb3s_h150danch_v2notod/PERFOLD_S[0-7]_ETH_qm0_f*.npz" <dir>/

Usage:  python3 score_ev_curve.py <perfold_dir> <out.json>
Cost:   ~3.2GiB of artifacts, a few minutes of CPU, no GPU, no retraining.

Consistency check built in: `EV(all filled)` and the filled/decision counts must
reproduce the untraded-population record (DOGE -1.824bp / 42.96% / 3,393,760 of
4,191,122) — if they do not, the join or the seed set is wrong.
"""
import glob
import json
import os
import sys

import numpy as np

PF = sys.argv[1]
OUT = sys.argv[2]

# cell_ev / cell_n are the published figures, carried only to place the deployed
# operating point on the curve — they are not recomputed here.
SYMS = {
    "DOGE": dict(nseed=4, policy="fixq", K=10, cell_ev=10.513, cell_n=1204, tot_days=169),
    "XRP":  dict(nseed=4, policy="fixq", K=5,  cell_ev=21.314, cell_n=401,  tot_days=163),
    "BTC":  dict(nseed=4, policy="dyn",  K=5,  cell_ev=12.268, cell_n=449,  tot_days=168),
    "ETH":  dict(nseed=8, policy="dyn",  K=5,  cell_ev=15.788, cell_n=430,  tot_days=159),
}

# marginal bands: coarse over the bulk, geometric refinement in the top 1% — the
# traded set lives inside the last band on every symbol
EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9, 99.99, 100]


def load_symbol(sym, cfg):
    nseed = cfg["nseed"]
    nf = len(glob.glob(os.path.join(PF, f"PERFOLD_S0_{sym}_qm0_f*.npz")))
    sc, net, fill, fold = [], [], [], []
    for f in range(nf):
        zs = [np.load(os.path.join(PF, f"PERFOLD_S{s}_{sym}_qm0_f{f}.npz")) for s in range(nseed)]
        te = np.mean([z["axb_te"].astype(np.float64) for z in zs], 0)
        votes = np.sum([z["side"].astype(int) for z in zs], 0)
        side = votes >= int(np.ceil(nseed / 2))
        z0 = zs[0]
        nl = z0["netl"].astype(np.float64); ns = z0["nets"].astype(np.float64)
        n = np.where(side, nl, ns)
        fc = np.where(side, z0["fl"].astype(bool), z0["fs"].astype(bool))
        sc.append(te); net.append(n); fill.append(fc & np.isfinite(n))
        fold.append(np.full(len(te), f, np.int8))
        for z in zs:
            z.close()
    return (np.concatenate(sc), np.concatenate(net), np.concatenate(fill),
            np.concatenate(fold), nf)


def _sweep(u, y, n):
    """u = fraction-from-the-top in [0,1) for each row; sweep the threshold."""
    o = np.argsort(u, kind="stable")
    yy = y[o]; uu = u[o]
    cum = np.cumsum(yy); win = np.cumsum(yy > 0)
    # floor the grid at 50 trades: below that the mean is noise, not a level
    ks = np.unique(np.clip(np.round(n * np.geomspace(50.0 / n, 1.0, 160)).astype(np.int64), 50, n))
    return [dict(q=float(k / n), k=int(k), ev=float(cum[k - 1] / k),
                 hit=float(win[k - 1] / k), thr=float(uu[k - 1])) for k in ks]


def curves(sc, net, fill, fold):
    """All statistics are on the FILLED population with the ensemble's own side —
    the same rows the cell accounting keeps."""
    s = sc[fill]; y = net[fill]; f = fold[fill]
    n = len(s)

    u_fold = np.empty(n)
    for k in np.unique(f):
        m = f == k
        r = np.argsort(np.argsort(-s[m], kind="stable"), kind="stable")
        u_fold[m] = r / m.sum()
    u_pool = np.argsort(np.argsort(-s, kind="stable"), kind="stable") / n

    cumulative = _sweep(u_fold, y, n)
    cumulative_pooled = _sweep(u_pool, y, n)

    marginal = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (u_fold >= 1 - hi / 100.0) & (u_fold < 1 - lo / 100.0 + (1e-12 if lo == 0 else 0))
        if m.sum() < 20:
            continue
        seg = y[m]
        marginal.append(dict(lo=lo, hi=hi, k=int(m.sum()), ev=float(seg.mean()),
                             hit=float((seg > 0).mean()),
                             se=float(seg.std(ddof=1) / np.sqrt(len(seg)))))

    # bulk association of score with realised net, on the same population — the
    # quantity a population-level metric would see
    rs = np.argsort(np.argsort(s)); ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rs, ry)[0, 1])
    pear = float(np.corrcoef(s, y)[0, 1])
    return marginal, cumulative, cumulative_pooled, dict(
        n_filled=n, ev_all=float(y.mean()), hit_all=float((y > 0).mean()),
        spearman=spear, pearson=pear)


res = {}
for sym, cfg in SYMS.items():
    sc, net, fill, fold, nf = load_symbol(sym, cfg)
    mar, cum, cumP, stats = curves(sc, net, fill, fold)
    dep_q = cfg["cell_n"] / stats["n_filled"]
    near = min(cum, key=lambda p: abs(p["q"] - dep_q))
    nearP = min(cumP, key=lambda p: abs(p["q"] - dep_q))
    res[sym] = dict(cfg=cfg, nfolds=nf, n_decisions=int(len(sc)), stats=stats,
                    marginal=mar, cumulative=cum, cumulative_pooled=cumP,
                    at_deployed_q=dict(perfold=near, pooled=nearP),
                    deployed=dict(q=dep_q, ev=cfg["cell_ev"], n=cfg["cell_n"]))
    print(f"{sym}: folds={nf} filled={stats['n_filled']:,} EV(all)={stats['ev_all']:+.3f}bp "
          f"hit={100*stats['hit_all']:.2f}% spearman={stats['spearman']:+.4f} || at q={100*dep_q:.4f}%: "
          f"per-fold {near['ev']:+.2f} | pooled {nearP['ev']:+.2f} | deployed cell {cfg['cell_ev']:+.2f}",
          flush=True)

json.dump(res, open(OUT, "w"))
print(f"[saved] {OUT}")
