#!/usr/bin/env python3
"""FACTUAL feature diagnostic per symbol (single rep split, vol-normed feats):
1. XGBoost B importance (gain) -> btc_lead share + top features (by name).
2. RAW 30s-direction predictivity: IC_raw = corr(pred(sign rH30)-0.5, rH30) + AUC_raw  -> 'are the
   features predictive on their own' (the user's claim that LTC is most predictive).
3. MAKER-side IC = corr(pred(better-side)-0.5, netl-nets)  -> the baseline target.
4. Univariate: feature with max |corr(feat, rH30)| on train.
Separates 'feature predictivity' (raw return) from 'maker-side IC' (the deploy target).
Usage: python3 subs60_xgb_featdiag.py SYM [nthread]
"""
import io, json, sys
import numpy as np
from google.cloud import storage
import xgboost as xgb

SYM = sys.argv[1]; NTHREAD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
W, EMB, TST, KNORM = 200, 2, 60, 20; CFGIDX, RHKEY = 1, "rH30"
bk = storage.Client(project=PROJ).bucket(BUCKET)
HP = {"objective": "binary:logistic", "tree_method": "hist", "nthread": NTHREAD, "seed": 0,
      "max_depth": 5, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}

d = np.load(io.BytesIO(bk.blob(f"research_runs/maker_labels_h/{SYM}.npz").download_as_bytes()), allow_pickle=True)
m = json.loads(str(d["meta"])); ndays = int(m["n_days"])
F = d["F"].astype(np.float64); day = d["day"].astype(int); rH = d[RHKEY].astype(np.float64)
netl = d["pnl_long"][CFGIDX, 0, :].astype(np.float64) * 100.0; nets = d["pnl_short"][CFGIDX, 0, :].astype(np.float64) * 100.0
fl = d["fill_long"].astype(bool)[0]; fs = d["fill_short"].astype(bool)[0]
fnames = [str(x) for x in d["feat_names"]] if "feat_names" in d else [f"f{i}" for i in range(F.shape[1])]
nfeat = F.shape[1]
btc_idx = [i for i, nm in enumerate(fnames) if "btc" in nm.lower()] or [64, 65, 66]

# vol-norm (blanket causal trailing z) -- same as baseline
day_mean = np.zeros((ndays, nfeat)); day_var = np.zeros((ndays, nfeat))
for dd in range(ndays):
    mk = day == dd
    if mk.sum() > 1:
        day_mean[dd] = F[mk].mean(0); day_var[dd] = F[mk].var(0)
gstd = F.std(0); mu = np.zeros((ndays, nfeat)); sd = np.zeros((ndays, nfeat))
for dd in range(ndays):
    sl = slice(max(0, dd - KNORM), dd) if dd > 0 else slice(0, 1)
    mu[dd] = day_mean[sl].mean(0); sd[dd] = np.sqrt(np.maximum(day_var[sl].mean(0), 0))
sd = np.maximum(sd, 0.2 * gstd[None, :] + 1e-9)
Fn = ((F - mu[day]) / sd[day]).astype(np.float32)

f0 = int(day.min()); trn = (day >= f0) & (day < f0 + W); tst = (day >= f0 + W + EMB) & (day < f0 + W + EMB + TST)


def ic(p, tgt):
    s = p - 0.5
    return float(np.corrcoef(s, tgt)[0, 1]) if s.std() > 0 and len(s) > 2 else float("nan")


# --- raw 30s-direction model ---
yR = (rH > 0).astype(int)
spwR = float((yR[trn] == 0).sum() / max((yR[trn] == 1).sum(), 1))
bR = xgb.train(dict(HP, scale_pos_weight=spwR), xgb.DMatrix(Fn[trn], label=yR[trn]), num_boost_round=300)
pR = bR.predict(xgb.DMatrix(Fn[tst]))
ic_raw = ic(pR, rH[tst])
order = np.argsort(pR); ranks = np.empty(len(pR)); ranks[order] = np.arange(1, len(pR) + 1)
n1 = (yR[tst] == 1).sum(); n0 = (yR[tst] == 0).sum()
auc_raw = float((ranks[yR[tst] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

# --- maker better-side model ---
yB = (np.where(fl, netl, -np.inf) > np.where(fs, nets, -np.inf)).astype(int); both = fl & fs
wq = np.where(both, np.abs(netl - nets), np.where(fl, np.abs(netl), np.where(fs, np.abs(nets), 0.0)))
tb = trn & (fl | fs); pos = wq[tb][wq[tb] > 0]
wc = np.clip(wq[tb], 0, np.quantile(pos, 0.99) if len(pos) else 1.0)
bB = xgb.train(HP, xgb.DMatrix(Fn[tb], label=yB[tb], weight=wc), num_boost_round=300)
tvb = tst & both
pB = bB.predict(xgb.DMatrix(Fn[tvb]))
ic_maker = ic(pB, (netl[tvb] - nets[tvb]))

# --- importance (gain) of the maker model ---
gain = bB.get_score(importance_type="gain"); tot = sum(gain.values()) or 1.0
per_idx = {}
for k, v in gain.items():
    per_idx[int(k[1:])] = per_idx.get(int(k[1:]), 0.0) + v
btc_share = sum(per_idx.get(i, 0.0) for i in btc_idx) / tot
top = sorted(per_idx.items(), key=lambda kv: -kv[1])[:6]
top_named = [(fnames[i] if i < len(fnames) else f"f{i}", round(g / tot, 3)) for i, g in top]

# --- univariate raw predictivity ---
uc = np.array([np.corrcoef(Fn[trn][:, j], rH[trn])[0, 1] if Fn[trn][:, j].std() > 0 else 0.0 for j in range(nfeat)])
uc = np.nan_to_num(uc); j = int(np.argmax(np.abs(uc)))
univ_best = (fnames[j] if j < len(fnames) else f"f{j}", round(float(uc[j]), 4))

res = {"sym": SYM, "n_train": int(trn.sum()), "n_test": int(tst.sum()),
       "ic_raw_dir": round(ic_raw, 4), "auc_raw_dir": round(auc_raw, 4), "ic_maker_side": round(ic_maker, 4),
       "btc_lead_gain_share": round(float(btc_share), 3), "top_feats_gain": top_named,
       "univ_best_feat_vs_rH30": univ_best, "btc_idx": btc_idx}
bk.blob(f"research_runs/maker_labels_h/featdiag/{SYM}.json").upload_from_string(json.dumps(res, default=float))
print(f"[{SYM}] IC_raw_dir={ic_raw:+.4f} AUC_raw={auc_raw:.3f} | IC_maker={ic_maker:+.4f} | "
      f"btc_lead_gain_share={btc_share:.2f} | univ_best={univ_best} | top_gain={top_named[:4]}", flush=True)
