#!/usr/bin/env python3
"""Why the seed-ensemble beats a single seed, and why seeds disagree so much.

MODELCAP rev1, part 2a — the part that needs no retraining: everything here comes from
the stored PERFOLD_S* artifacts (per-seed composite test score, per-seed side, and the
labels), so it is a description of the SAME model population that produced the published
cells. The booster-level part (HP landscape, feature importances, tree anatomy) needs the
capture runs and lives in a separate pass.

Reported per symbol, at the deployed policy and budget:

  A. SCORE AGREEMENT       pairwise Pearson / Spearman of the per-seed composite scores
                           on the test rows. Rank agreement is what matters: the policy
                           only ever uses the score's ORDER.
  B. SELECTION OVERLAP     Jaccard of the per-seed selected sets, and the share of the
                           ensemble's trades that a given seed would also have taken.
                           Two seeds can correlate at 0.9 and still overlap poorly in
                           the tail - the tail is all the policy trades.
  C. SIDE AGREEMENT        how often the per-seed B models pick the same direction, on
                           all rows and on the selected ones.
  D. 2x2 ATTRIBUTION       {ens|seed} score x {ens|seed} side. The ensemble touches two
                           channels - the ranking that decides WHICH decisions are taken
                           and the vote that decides the DIRECTION. This separates them.
  E. k-SEED CURVE          EV as a function of how many seeds are averaged (all subsets
                           if few, else a sample). Shows whether the gain is still
                           growing at the deployed seed count.
  F. DISPERSION PROBE      EV of selected trades bucketed by the cross-seed sd of the
                           score. If disagreement predicts worse trades, the ensemble is
                           not just averaging noise - it is avoiding a bad region, which
                           is the mechanism a 'harmony'-style filter exploits.

Env: SYM, SCORE_SUB, NSEED, K, POLICY (fixq|dyn), OUT.
"""
import io, itertools, json, os

import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
SCORE_SUB = os.environ.get("SCORE_SUB", "research_runs/maker_labels_tb3s_h150anch")
NSEED = int(os.environ.get("NSEED", "4"))
K = float(os.environ.get("K", "10"))
POLICY = os.environ.get("POLICY", "fixq")
OUT = os.environ.get("OUT", f"research_runs/strictfill_cells/ANATOMY_{SYM}.json")
KDAYS = 30
bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s):
    print(s, flush=True)


nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SCORE_SUB}/PERFOLD_S0_{SYM}_qm0_f")
         if b.name.endswith(".npz"))
folds = []
for f in range(nf):
    zs = [np.load(io.BytesIO(bk.blob(f"{SCORE_SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
          for s in range(NSEED)]
    z0 = zs[0]
    folds.append(dict(day_tr=z0["day_tr"].astype(int), day_te=z0["day_te"].astype(int),
                      fl=z0["fl"].astype(bool), fs=z0["fs"].astype(bool),
                      nl=z0["netl"].astype(np.float64), ns=z0["nets"].astype(np.float64),
                      STE=np.array([z["axb_te"].astype(np.float64) for z in zs]),
                      STR=np.array([z["axb_tr"].astype(np.float64) for z in zs]),
                      SIDE=np.array([z["side"].astype(bool) for z in zs])))
tot_days = sum(len(np.unique(p["day_te"])) for p in folds)
log(f"[{SYM}] folds={nf} seeds={NSEED} policy={POLICY} K={K} test_days={tot_days} "
    f"rows={sum(len(p['day_te']) for p in folds)}")


def sel_fixq(p, te, tr, k):
    trd = sorted(set(p["day_tr"].tolist()))[-KDAYS:]
    s = tr[np.isin(p["day_tr"], trd)]
    if not len(s):
        return np.array([], int)
    wpd = len(s) / max(len(trd), 1)
    return np.where(te >= float(np.quantile(s, max(0.0, 1.0 - k / max(wpd, 1.0)))))[0]


def sel_dyn(p, te, tr, k):
    days = sorted(set(p["day_te"].tolist())); wpd = len(te) / max(len(days), 1)
    q = max(0.0, 1.0 - k / max(wpd, 1.0))
    trd = sorted(set(p["day_tr"].tolist())); sd = np.isin(p["day_tr"], trd[-KDAYS:])
    buf = list(tr[sd]); cap = max(int(KDAYS * wpd), 1); out = []
    for d in days:
        i = np.where(p["day_te"] == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        out.extend(i[te[i] >= tau].tolist()); buf.extend(te[i].tolist()); buf = buf[-cap:]
    return np.array(out, int)


SEL = {"fixq": sel_fixq, "dyn": sel_dyn}[POLICY]


def cell(sels, sides):
    nets = []
    for p, sel, side in zip(folds, sels, sides):
        sd = side[sel]
        net = np.where(sd, p["nl"][sel], p["ns"][sel])
        fc = np.where(sd, p["fl"][sel], p["fs"][sel])
        ex = fc & np.isfinite(net)
        nets.append(net[ex])
    a = np.concatenate(nets) if nets else np.array([])
    if not len(a):
        return dict(ev=float("nan"), n=0, bpd=float("nan"))
    return dict(ev=float(a.mean()), n=int(len(a)), bpd=float(a.sum() / tot_days),
                hit=float((a > 0).mean()))


def rank(x):
    o = np.argsort(np.argsort(x))
    return o.astype(np.float64) / max(len(x) - 1, 1)


res = {"sym": SYM, "policy": POLICY, "K": K, "nseed": NSEED, "nfolds": nf, "tot_days": tot_days}

# ---- A. score agreement -----------------------------------------------------
P = np.zeros((NSEED, NSEED)); S = np.zeros((NSEED, NSEED)); wsum = 0
for p in folds:
    w = len(p["day_te"]); wsum += w
    R = np.array([rank(p["STE"][s]) for s in range(NSEED)])
    for i in range(NSEED):
        for j in range(NSEED):
            P[i, j] += w * np.corrcoef(p["STE"][i], p["STE"][j])[0, 1]
            S[i, j] += w * np.corrcoef(R[i], R[j])[0, 1]
P /= wsum; S /= wsum
off = ~np.eye(NSEED, dtype=bool)
res["score_pearson_mean_offdiag"] = float(P[off].mean())
res["score_spearman_mean_offdiag"] = float(S[off].mean())
res["score_spearman_matrix"] = S.round(4).tolist()
log(f"A. score agreement across seeds: Pearson {P[off].mean():.3f}, "
    f"Spearman {S[off].mean():.3f} (min pair {S[off].min():.3f}, max {S[off].max():.3f})")

# ---- selections -------------------------------------------------------------
seed_sel = [[SEL(p, p["STE"][s], p["STR"][s], K) for p in folds] for s in range(NSEED)]
ens_te = [p["STE"].mean(0) for p in folds]
ens_tr = [p["STR"].mean(0) for p in folds]
ens_sel = [SEL(p, te, tr, K) for p, te, tr in zip(folds, ens_te, ens_tr)]
ens_side = [p["SIDE"].sum(0) >= int(np.ceil(NSEED / 2)) for p in folds]

# ---- B. selection overlap ---------------------------------------------------
J = np.eye(NSEED)
for i in range(NSEED):
    for j in range(i + 1, NSEED):
        inter = un = 0
        for f in range(nf):
            a, b = set(seed_sel[i][f].tolist()), set(seed_sel[j][f].tolist())
            inter += len(a & b); un += len(a | b)
        J[i, j] = J[j, i] = inter / max(un, 1)
share_ens = []
for s in range(NSEED):
    inter = tot = 0
    for f in range(nf):
        a, b = set(ens_sel[f].tolist()), set(seed_sel[s][f].tolist())
        inter += len(a & b); tot += len(a)
    share_ens.append(inter / max(tot, 1))
res["selection_jaccard_mean"] = float(J[off].mean())
res["selection_jaccard_matrix"] = J.round(4).tolist()
res["seed_share_of_ensemble_trades"] = [round(x, 4) for x in share_ens]
log(f"B. selection overlap at t{K:.0f}: mean pairwise Jaccard {J[off].mean():.3f} "
    f"(min {J[off].min():.3f}) | a single seed would also take "
    f"{100*np.mean(share_ens):.1f}% of the ensemble's decisions")

# ---- C. side agreement ------------------------------------------------------
agr_all, agr_sel = [], []
for p, es in zip(folds, ens_sel):
    v = p["SIDE"].sum(0)
    agr_all.append(np.maximum(v, NSEED - v) / NSEED)
    if len(es):
        agr_sel.append((np.maximum(v, NSEED - v) / NSEED)[es])
res["side_agreement_all"] = float(np.concatenate(agr_all).mean())
res["side_agreement_selected"] = float(np.concatenate(agr_sel).mean()) if agr_sel else float("nan")
log(f"C. side agreement: all rows {100*res['side_agreement_all']:.1f}%, "
    f"selected rows {100*res['side_agreement_selected']:.1f}%")

# ---- D. 2x2 attribution -----------------------------------------------------
per_seed = [cell(seed_sel[s], [p["SIDE"][s] for p in folds]) for s in range(NSEED)]
mix_ensscore_seedside = [cell(ens_sel, [p["SIDE"][s] for p in folds]) for s in range(NSEED)]
mix_seedscore_ensside = [cell(seed_sel[s], ens_side) for s in range(NSEED)]
ens = cell(ens_sel, ens_side)
res["per_seed"] = per_seed
res["ens"] = ens
res["mix_ensscore_seedside"] = mix_ensscore_seedside
res["mix_seedscore_ensside"] = mix_seedscore_ensside
ms = float(np.mean([c["ev"] for c in per_seed])); sds = float(np.std([c["ev"] for c in per_seed], ddof=1))
m_es = float(np.mean([c["ev"] for c in mix_ensscore_seedside]))
m_se = float(np.mean([c["ev"] for c in mix_seedscore_ensside]))
res["attribution"] = dict(per_seed_mean=ms, per_seed_sd=sds, ens=ens["ev"],
                          ens_score_only=m_es, ens_side_only=m_se,
                          gain_total=ens["ev"] - ms,
                          gain_via_ranking=m_es - ms, gain_via_side=m_se - ms)
log(f"D. per-seed EV {ms:+.2f} +- {sds:.2f} -> ENSEMBLE {ens['ev']:+.2f} "
    f"(gain {ens['ev']-ms:+.2f}bp) | ens-score+seed-side {m_es:+.2f} ({m_es-ms:+.2f}) "
    f"| seed-score+ens-side {m_se:+.2f} ({m_se-ms:+.2f})")
log(f"   per-seed EVs: {[round(c['ev'],2) for c in per_seed]}  n: {[c['n'] for c in per_seed]}")

# ---- E. k-seed curve --------------------------------------------------------
curve = {}
for k in range(1, NSEED + 1):
    combos = list(itertools.combinations(range(NSEED), k))
    if len(combos) > 20:
        rng = np.random.default_rng(0)
        combos = [tuple(rng.choice(NSEED, k, replace=False)) for _ in range(20)]
    evs, ns = [], []
    for cb in combos:
        te = [p["STE"][list(cb)].mean(0) for p in folds]
        tr = [p["STR"][list(cb)].mean(0) for p in folds]
        sl = [SEL(p, t, r, K) for p, t, r in zip(folds, te, tr)]
        sd = [p["SIDE"][list(cb)].sum(0) >= int(np.ceil(len(cb) / 2)) for p in folds]
        c = cell(sl, sd)
        evs.append(c["ev"]); ns.append(c["n"])
    curve[k] = dict(ev_mean=float(np.nanmean(evs)), ev_sd=float(np.nanstd(evs, ddof=1)) if len(evs) > 1 else 0.0,
                    n_mean=float(np.mean(ns)), n_combos=len(combos))
    log(f"E. k={k}: EV {curve[k]['ev_mean']:+.2f} +- {curve[k]['ev_sd']:.2f} "
        f"(n~{curve[k]['n_mean']:.0f}, {len(combos)} subsets)")
res["k_seed_curve"] = curve

# ---- F. dispersion probe ----------------------------------------------------
disp, netv = [], []
for p, es, esd in zip(folds, ens_sel, ens_side):
    if not len(es):
        continue
    sd_ = p["STE"][:, es].std(0)
    side = esd[es]
    net = np.where(side, p["nl"][es], p["ns"][es])
    fc = np.where(side, p["fl"][es], p["fs"][es])
    ex = fc & np.isfinite(net)
    disp.append(sd_[ex]); netv.append(net[ex])
disp = np.concatenate(disp); netv = np.concatenate(netv)
qs = np.quantile(disp, [0, .25, .5, .75, 1.0])
buckets = []
for i in range(4):
    m = (disp >= qs[i]) & (disp <= qs[i + 1] if i == 3 else disp < qs[i + 1])
    buckets.append(dict(q=i + 1, lo=float(qs[i]), hi=float(qs[i + 1]), n=int(m.sum()),
                        ev=float(netv[m].mean()) if m.any() else float("nan")))
res["dispersion_buckets"] = buckets
res["dispersion_corr"] = float(np.corrcoef(disp, netv)[0, 1])
log("F. cross-seed score dispersion vs outcome (quartiles of sd, low->high): "
    + " | ".join(f"Q{b['q']} EV {b['ev']:+.2f}(n={b['n']})" for b in buckets)
    + f"  corr(sd,net)={res['dispersion_corr']:+.3f}")

bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
log(f"[saved] gs://{BUCKET}/{OUT}")
