#!/usr/bin/env python3
"""HD3 rev8: YEAR ensemble cell for SYM — anch_ens.py generalized (SYM arg + fold-count
discovery), methodology IDENTICAL to the DOGE rev7 cell: deployed scoring = mean of the 4
per-seed rank-scores, side = majority vote of per-seed sides (ties counted), causal t5,
base EV + per-fold + LOFO + score-jitter perturbation (sd 0.02/0.05 x 100 reps).
Also prints the per-seed AxB_t5 summary from the OPTUNA_IC SEED jsons.
Artifact subdir overridable via XSYM_SUB."""
import io
import json
import os
import sys
import numpy as np
from google.cloud import storage

SYM = sys.argv[1]
bk = storage.Client(project='project-0998ac51-36ba-445c-bc7').bucket('market-data-0998ac51')
SUB = 'research_runs/' + os.environ.get('XSYM_SUB', 'maker_labels_tb3s_h150anch')
KDAYS = 30

evs = []
for s in range(4):
    r = json.loads(bk.blob(f'{SUB}/OPTUNA_IC_{SYM}_qm0_SEED{s}.json').download_as_bytes())
    x = r['AxB_t5']
    evs.append(x['ev'])
    print(f'  seed{s} AxB_t5: EV {x["ev"]:+.2f}bp tpd {x["tpd"]:.1f} ann {x["ann"]:+.2f} '
          f'hit {100*x["hit"]:.1f}% perfold {x["perfold"]}', flush=True)
evs = np.array(evs)
print(f'=== {SYM} PER-SEED t5: {evs.mean():+.2f} +- {evs.std(ddof=1):.2f} bp '
      f'[{"/".join(f"{e:+.1f}" for e in evs)}] {int((evs > 0).sum())}/4 positive ===', flush=True)

nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f'{SUB}/PERFOLD_S0_{SYM}_qm0_f')
         if b.name.endswith('.npz'))
print(f'{SYM}: {nf} folds', flush=True)
folds = []
tie_n = 0; tot_n = 0
for f in range(nf):
    zs = [np.load(io.BytesIO(bk.blob(f'{SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz').download_as_bytes()))
          for s in range(4)]
    tr = np.mean([z['axb_tr'].astype(np.float64) for z in zs], 0)
    te = np.mean([z['axb_te'].astype(np.float64) for z in zs], 0)
    votes = np.sum([z['side'].astype(int) for z in zs], 0)
    side = votes >= 2
    tie_n += int((votes == 2).sum()); tot_n += len(votes)
    z0 = zs[0]
    folds.append(dict(tr=tr, te=te, day_tr=z0['day_tr'], day_te=z0['day_te'], side=side,
                      fl=z0['fl'], fs=z0['fs'], nl=z0['netl'].astype(np.float64),
                      ns=z0['nets'].astype(np.float64)))
print(f'side ties (2-2): {tie_n}/{tot_n} = {100*tie_n/max(tot_n,1):.1f}%', flush=True)


def causal(p, tgt=5.0, jit=0.0, rng=None):
    days = sorted(set(p['day_te'].tolist())); wpd = len(p['te']) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(p['day_tr'].tolist())); seed = np.isin(p['day_tr'], trd[-KDAYS:])
    te = p['te'] + (rng.normal(0, jit, len(p['te'])) if jit > 0 else 0.0)
    buf = list(p['tr'][seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(p['day_te'] == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[te[idx] >= tau].tolist()); buf.extend(te[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    sd_ = p['side'][sel]; net = np.where(sd_, p['nl'][sel], p['ns'][sel])
    fc = np.where(sd_, p['fl'][sel], p['fs'][sel])
    ex = fc & np.isfinite(net); return net[ex]


base = [causal(p) for p in folds]
tot = np.concatenate(base)
print(f'=== {SYM} ENSEMBLE year t5: EV {tot.mean():+.2f}bp ({len(tot)}tr, '
      f'hit {100*(tot>0).mean():.1f}%) ===', flush=True)
print('per-fold EV:', [f'{b.mean():+.1f}({len(b)})' if len(b) else 'n/a' for b in base], flush=True)
print('per-fold sum%:', [f'{b.sum()*0.01:+.1f}' for b in base], flush=True)
for f in range(nf):
    a = np.concatenate([base[g] for g in range(nf) if g != f])
    if len(a):
        print(f'  LOFO -fold{f}: {a.mean():+.2f}bp', flush=True)
for sd_j in (0.02, 0.05):
    rng = np.random.default_rng(0); r = []
    for rep in range(100):
        a = np.concatenate([causal(p, jit=sd_j, rng=rng) for p in folds])
        r.append(a.mean() if len(a) else np.nan)
    r = np.array(r)
    print(f'  jitter sd={sd_j}: p10/p50/p90 = {np.nanquantile(r,.1):+.2f}/{np.nanquantile(r,.5):+.2f}/'
          f'{np.nanquantile(r,.9):+.2f} bp | P(EV>0)={100*np.nanmean(r>0):.0f}%', flush=True)
res = dict(sym=SYM, nf=nf, per_seed_ev=[float(e) for e in evs],
           ens_ev=float(tot.mean()), ens_n=int(len(tot)), ens_hit=float((tot > 0).mean()),
           perfold_ev=[float(b.mean()) if len(b) else None for b in base],
           perfold_n=[int(len(b)) for b in base], ties_pct=100 * tie_n / max(tot_n, 1))
bk.blob(f'{SUB}/ENS_{SYM}_t5.json').upload_from_string(json.dumps(res))
print(f'[ENS {SYM} DONE]', flush=True)
