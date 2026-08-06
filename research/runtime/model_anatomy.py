#!/usr/bin/env python3
"""Booster-level anatomy of the captured model population (MODELCAP rev1, part 2b).

Consumes what the capture campaign writes: MODELS_S{seed}_{SYM}_f{fold}_{A,Bg,Bf}.json,
the matching _hp.json, and the TRIALS_*_index.json search logs. Answers the half of the
user's question that the score artifacts cannot: what actually differs between two seeds
of the same symbol, and between symbols.

  1. HP LANDSCAPE   per (symbol, seed, fold): where the search landed, and how flat the
                    surface was around it (spread of the objective across the trials, and
                    how much worse the median trial is than the argmax). A flat surface
                    means the seed difference is NOT an HP-basin difference.
  2. CHOSEN HP      spread of the selected hyperparameters across seeds - if seeds pick
                    very different depth/eta/regularisation, the models are structurally
                    different, not just differently sampled.
  3. TREE ANATOMY   trees, mean depth, and the split-feature distribution per model.
  4. IMPORTANCE     gain / weight / cover per feature, then cross-seed and cross-symbol
                    agreement (Spearman over the feature ranking). This is the direct
                    'how different are these models' number.

Deliberately NOT here: prediction correlations, which need a scoring pass over the
dataset and belong with the score-level anatomy in ensemble_anatomy.py.

Env: SYMS ("DOGE,XRP,BTC,ETH"), SUBS (matching capture subdirs, comma-separated),
     NSEEDS (matching, comma-separated), OUT.
"""
import io, json, os
from collections import defaultdict

import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMS = os.environ.get("SYMS", "DOGE,XRP,BTC,ETH").split(",")
SUBS = os.environ.get("SUBS", "modelcap_h150anch,modelcap_h150anch,modelcap_h150d,modelcap_v2notod").split(",")
NSEEDS = [int(x) for x in os.environ.get("NSEEDS", "4,4,4,8").split(",")]
OUT = os.environ.get("OUT", "research_runs/strictfill_cells/MODEL_ANATOMY.json")
NAMES = [f"x{c}" for c in range(64)] + ["btc_ret5", "btc_ret30", "btc_ret60",
                                        "sin_h", "cos_h", "sin_f8", "cos_f8"]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s):
    print(s, flush=True)


def load_json(key):
    b = bk.blob(key)
    return json.loads(b.download_as_bytes()) if b.exists() else None


def tree_stats(model):
    """Walk the xgboost json dump: trees, depth, split-feature counts, gain per feature."""
    gb = model["learner"]["gradient_booster"]["model"]
    trees = gb["trees"]
    nfeat = int(model["learner"]["learner_model_param"]["num_feature"])
    gain = np.zeros(nfeat); cover = np.zeros(nfeat); cnt = np.zeros(nfeat)
    depths = []
    for t in trees:
        lc = t["left_children"]; rc = t["right_children"]
        sidx = t["split_indices"]; sc = t.get("loss_changes", []); sw = t.get("sum_hessian", [])
        # depth by BFS from the root
        d = {0: 0}; mx = 0
        stack = [0]
        while stack:
            i = stack.pop()
            mx = max(mx, d[i])
            if lc[i] != -1:
                d[lc[i]] = d[i] + 1; d[rc[i]] = d[i] + 1
                stack += [lc[i], rc[i]]
        depths.append(mx)
        for i in range(len(lc)):
            if lc[i] == -1:
                continue
            f = int(sidx[i])
            if f < nfeat:
                cnt[f] += 1
                if i < len(sc):
                    gain[f] += float(sc[i])
                if i < len(sw):
                    cover[f] += float(sw[i])
    return dict(n_trees=len(trees), mean_depth=float(np.mean(depths)) if depths else 0.0,
                max_depth=int(np.max(depths)) if depths else 0,
                n_splits=int(cnt.sum()), gain=gain, cover=cover, count=cnt, nfeat=nfeat)


def srank(x):
    o = np.argsort(np.argsort(-np.asarray(x, float)))
    return o.astype(float)


res = {"syms": SYMS, "subs": SUBS, "per_model": {}, "hp": {}, "landscape": {}}
IMP = defaultdict(dict)   # (sym, kind) -> {(seed,fold): gain vector}

for sym, sub, ns in zip(SYMS, SUBS, NSEEDS):
    nfold = 6
    for s in range(ns):
        for f in range(nfold):
            hp = load_json(f"research_runs/{sub}/MODELS_S{s}_{sym}_f{f}_hp.json")
            if hp is None:
                continue
            res["hp"][f"{sym}_S{s}_f{f}"] = hp
            for kind in ("A", "Bg", "Bf"):
                m = load_json(f"research_runs/{sub}/MODELS_S{s}_{sym}_f{f}_{kind}.json")
                if m is None:
                    continue
                st = tree_stats(m)
                IMP[(sym, kind)][(s, f)] = st["gain"]
                res["per_model"][f"{sym}_S{s}_f{f}_{kind}"] = dict(
                    n_trees=st["n_trees"], mean_depth=st["mean_depth"], max_depth=st["max_depth"],
                    n_splits=st["n_splits"], nfeat=st["nfeat"],
                    top_gain=[[NAMES[i] if i < len(NAMES) else f"c{i}", float(st["gain"][i])]
                              for i in np.argsort(-st["gain"])[:10]],
                    gain_share_top5=float(np.sort(st["gain"])[::-1][:5].sum() / max(st["gain"].sum(), 1e-9)))
            # search landscape from the trial index
            for study in ("A", "B"):
                ix = load_json(f"research_runs/{sub}/TRIALS_S{s}_{sym}_f{f}_{study}_index.json")
                if not ix:
                    continue
                key = "auc" if study == "A" else "ic"
                v = np.array([t[key] for t in ix if np.isfinite(t.get(key, np.nan))])
                if not len(v):
                    continue
                res["landscape"][f"{sym}_S{s}_f{f}_{study}"] = dict(
                    n_trials=len(v), best=float(v.max()), median=float(np.median(v)),
                    p10=float(np.quantile(v, .1)), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    best_minus_median=float(v.max() - np.median(v)),
                    argmax_trial=int(np.argmax([t[key] for t in ix])))

# cross-seed / cross-symbol importance agreement
agree = {}
for (sym, kind), d in IMP.items():
    seeds = sorted({s for s, _ in d})
    per_seed = {}
    for s in seeds:
        g = [d[(s, f)] for (ss, f) in d if ss == s]
        if g:
            per_seed[s] = np.mean(np.array(g), 0)
    rho = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = per_seed.get(seeds[i]), per_seed.get(seeds[j])
            if a is None or b is None or len(a) != len(b):
                continue
            rho.append(float(np.corrcoef(srank(a), srank(b))[0, 1]))
    if rho:
        agree[f"{sym}_{kind}_cross_seed_importance_spearman"] = dict(
            mean=float(np.mean(rho)), min=float(np.min(rho)), max=float(np.max(rho)), n_pairs=len(rho))
        log(f"{sym} {kind}: cross-seed importance agreement rho={np.mean(rho):.3f} "
            f"[{np.min(rho):.3f},{np.max(rho):.3f}]")
    res.setdefault("mean_gain", {})[f"{sym}_{kind}"] = {
        NAMES[i] if i < len(NAMES) else f"c{i}": float(v)
        for i, v in enumerate(np.mean([per_seed[s] for s in per_seed], 0))} if per_seed else {}
res["importance_agreement"] = agree

# cross-symbol agreement on the shared feature block
for kind in ("A", "Bg", "Bf"):
    vecs = {}
    for sym in SYMS:
        mg = res.get("mean_gain", {}).get(f"{sym}_{kind}")
        if mg:
            vecs[sym] = np.array([mg.get(n, 0.0) for n in NAMES[:67]])
    ks = list(vecs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            r = float(np.corrcoef(srank(vecs[ks[i]]), srank(vecs[ks[j]]))[0, 1])
            res.setdefault("cross_symbol_importance_spearman", {})[f"{ks[i]}_{ks[j]}_{kind}"] = r
            log(f"cross-symbol {kind}: {ks[i]} vs {ks[j]} rho={r:.3f}")

bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
log(f"[saved] gs://{BUCKET}/{OUT}  models={len(res['per_model'])} landscapes={len(res['landscape'])}")
