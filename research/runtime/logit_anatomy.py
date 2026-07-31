#!/usr/bin/env python3
"""Where does the cross-seed decorrelation actually live: in A, in B, or only in the mix?

MODELCAP rev1, part 2c. The composite the policy ranks on is
    score = cdf_A(pA) * cdf_Bg(|pBg - 0.5|)
with both CDFs taken against that seed's OWN training distribution. Because the rank-CDF
is monotone and per-seed, any difference in the LOGIT SCALE between seeds is normalised
away - so the measured composite disagreement is genuine rank disagreement, not scale.
This script tests where it comes from by scoring the captured boosters directly:

  * corr(pA) and corr(pBg) across seeds, raw probability and rank, on the test rows
  * the same for |pBg - 0.5|, which is what the composite actually consumes
  * a SWAP decomposition: rebuild the composite with A varying and Bg pinned to seed 0,
    and with Bg varying and A pinned to seed 0. Whichever swap reproduces the full
    cross-seed composite disagreement is the model that carries it.

Features are rebuilt exactly as the trainer does (DROP_COLS first, then the day-wise
vol-norm over the trailing KNORM days in float64, cast to float32), so the predictions
here are the same numbers the deployed scoring produced.

Env: SYM, DATA_SUB (the dataset the cell trained on), MODEL_SUB (capture dir), NSEED,
     DROP_COLS, OUT, NROWS (per-fold subsample for the correlation, 0 = all).
"""
import io, json, os

import numpy as np
import xgboost as xgb
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
DATA_SUB = os.environ.get("DATA_SUB", "research_runs/maker_labels_tb3s_h150anch")
MODEL_SUB = os.environ.get("MODEL_SUB", "modelcap_h150anch")
NSEED = int(os.environ.get("NSEED", "4"))
DROP = [int(x) for x in os.environ.get("DROP_COLS", "").split(",") if x != ""]
NROWS = int(os.environ.get("NROWS", "200000"))
OUT = os.environ.get("OUT", f"research_runs/strictfill_cells/LOGIT_ANATOMY_{SYM}.json")
W, T, EMB, KNORM = 200, 30, 2, 20
bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s):
    print(s, flush=True)


log(f"[load] {DATA_SUB}/{SYM}.npz")
d = np.load(io.BytesIO(bk.blob(f"{DATA_SUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int)
if DROP:
    keep = [i for i in range(F.shape[1]) if i not in DROP]
    F = F[:, keep]; log(f"  dropped {DROP} -> {F.shape[1]} cols")
nfeat = F.shape[1]
day_mean = np.zeros((ndays, nfeat)); day_var = np.zeros((ndays, nfeat))
for dd in range(ndays):
    mk = day == dd
    if mk.sum() > 1:
        day_mean[dd] = F[mk].mean(0); day_var[dd] = F[mk].var(0)
gstd = F.std(0); mu_ref = np.zeros((ndays, nfeat)); sd_ref = np.zeros((ndays, nfeat))
for dd in range(ndays):
    sl = slice(max(0, dd - KNORM), dd) if dd > 0 else slice(0, 1)
    mu_ref[dd] = day_mean[sl].mean(0); sd_ref[dd] = np.sqrt(np.maximum(day_var[sl].mean(0), 0))
sd_ref = np.maximum(sd_ref, 0.2 * gstd[None, :] + 1e-9)
Fn = ((F - mu_ref[day]) / sd_ref[day]).astype(np.float32)
del F
log(f"  Fn={Fn.shape}")

FOLDS = []; ts = W + EMB
while ts < ndays:
    te = min(ts + T, ndays); trn = (day >= ts - EMB - W) & (day < ts - EMB); tst = (day >= ts) & (day < te)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        FOLDS.append((trn, tst))
    ts += T
log(f"  folds={len(FOLDS)}")


def rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64) / max(len(x) - 1, 1)


def cdf_map(x, ref):
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def load_booster(s, f, nm):
    import tempfile
    p = tempfile.mktemp(suffix=".json")
    bk.blob(f"research_runs/{MODEL_SUB}/MODELS_S{s}_{SYM}_f{f}_{nm}.json").download_to_filename(p)
    b = xgb.Booster(); b.load_model(p); os.remove(p); return b


res = {"sym": SYM, "nseed": NSEED, "nfolds": len(FOLDS), "per_fold": []}
acc = {k: [] for k in ("pA_raw", "pA_rank", "pBg_raw", "pBg_rank", "absB_raw", "absB_rank",
                       "score", "score_swapA", "score_swapB")}
for fi, (trn, tst) in enumerate(FOLDS):
    tri = np.where(trn)[0]; tei = np.where(tst)[0]
    if NROWS and len(tei) > NROWS:
        tei = tei[np.linspace(0, len(tei) - 1, NROWS).astype(int)]
    Xtr = Fn[tri]; Xte = Fn[tei]
    pA, pB, sA, sBg = [], [], [], []
    for s in range(NSEED):
        A = load_booster(s, fi, "A"); Bg = load_booster(s, fi, "Bg")
        a_tr = A.inplace_predict(Xtr, validate_features=False)
        b_tr = Bg.inplace_predict(Xtr, validate_features=False)
        pA.append(A.inplace_predict(Xte, validate_features=False).astype(np.float64))
        pB.append(Bg.inplace_predict(Xte, validate_features=False).astype(np.float64))
        sA.append(np.sort(a_tr)); sBg.append(np.sort(np.abs(b_tr - 0.5)))
    pA = np.array(pA); pB = np.array(pB); absB = np.abs(pB - 0.5)
    sc = np.array([cdf_map(pA[s], sA[s]) * cdf_map(absB[s], sBg[s]) for s in range(NSEED)])
    # swap decomposition: vary one model, pin the other to seed 0
    sc_swapA = np.array([cdf_map(pA[s], sA[s]) * cdf_map(absB[0], sBg[0]) for s in range(NSEED)])
    sc_swapB = np.array([cdf_map(pA[0], sA[0]) * cdf_map(absB[s], sBg[s]) for s in range(NSEED)])

    def offdiag(M, rankit=False):
        X = np.array([rank(v) for v in M]) if rankit else M
        C = np.corrcoef(X)
        o = ~np.eye(NSEED, dtype=bool)
        return float(C[o].mean())

    row = dict(fold=fi, n=len(tei),
               pA_raw=offdiag(pA), pA_rank=offdiag(pA, True),
               pBg_raw=offdiag(pB), pBg_rank=offdiag(pB, True),
               absB_raw=offdiag(absB), absB_rank=offdiag(absB, True),
               score=offdiag(sc, True), score_swapA=offdiag(sc_swapA, True),
               score_swapB=offdiag(sc_swapB, True))
    for k in acc:
        acc[k].append(row[k])
    res["per_fold"].append(row)
    log(f"  fold{fi} n={len(tei)}: pA {row['pA_rank']:.3f} | pBg {row['pBg_rank']:.3f} | "
        f"|pB-.5| {row['absB_rank']:.3f} | score {row['score']:.3f} "
        f"(A-only {row['score_swapA']:.3f}, B-only {row['score_swapB']:.3f})")

res["mean"] = {k: float(np.mean(v)) for k, v in acc.items()}
log(f"\n[{SYM}] mean cross-seed agreement (Spearman, off-diagonal):")
log(f"  pA   raw {res['mean']['pA_raw']:.3f}  rank {res['mean']['pA_rank']:.3f}")
log(f"  pBg  raw {res['mean']['pBg_raw']:.3f}  rank {res['mean']['pBg_rank']:.3f}")
log(f"  |pBg-0.5| raw {res['mean']['absB_raw']:.3f}  rank {res['mean']['absB_rank']:.3f}")
log(f"  composite score      {res['mean']['score']:.3f}")
log(f"  composite, A varies  {res['mean']['score_swapA']:.3f}")
log(f"  composite, Bg varies {res['mean']['score_swapB']:.3f}")
bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
log(f"[saved] gs://{BUCKET}/{OUT}")
