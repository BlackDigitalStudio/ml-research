#!/usr/bin/env python3
"""CAN THE MODEL GET US OUT FROM INSIDE THE TRADE, AND IS THE VOL GATE WHAT DRAGS THE BAD
FILLS IN?

User design (2026-08-04): the deployed composite answers two questions at once -- "will it
fill?" (A, the vol gate, which is close to a fill proxy) and "which way?" (Bg). Once we are
already filled the first question is settled, so a signal applied from INSIDE the position
is not contaminated by it. Two tests, both model-free of any new training:

PART A -- DOES THE SIGNAL STILL WORK FROM INSIDE THE POSITION?
Decisions sit on a ~3s grid, so a 150s hold spans ~50 of them. For a position opened at
row i, row i+k carries its own ensemble score and side. The forward move from i+k is
    m = (netl - nets)/2                (the 150s signed move at that row, in bp)
and signing it by OUR OPEN POSITION gives what the rest of the trade is worth. Split by
whether the model at i+k still AGREES with the position we are holding:

    agree     side[i+k] == side[i]     -> hypothesis: forward value positive, keep holding
    disagree  side[i+k] != side[i]     -> hypothesis: forward value negative, exit here

Reported separately on the ALONE population (the adversely-filled rows worth -4.4..-8.2bp)
because that is the population the design is meant to rescue.

APPROXIMATION, STATED: PERFOLD carries day but not ts, so "k rows later" is ~3k seconds
only. Rows are required to share a day; dedupe gaps make the spacing a lower bound. This is
a first-pass read on whether the structure exists, not a backtest of an exit rule.

PART B -- IS IT A THAT PULLS IN THE ADVERSE FILLS?
PERFOLD carries noa_te alongside axb_te: the same ensemble scored WITHOUT the A head. Side
is unaffected (it comes from mean pBg), so switching the ranking changes only WHICH rows a
given selectivity picks. At matched top-q the two rankings are compared on the share of
adversely-filled rows they select and on the EV they realise. If the vol gate is what drags
the bad fills in, the noA ranking selects a lower ALONE share at the same q.

noA IS NOT THE DEPLOYED POLICY (CLAUDE.md interpreting-records rule): it is used here only
as the A-ablated CONTRAST, and nothing about the deployed AxB cell is claimed from it.

Env: SYM, SCORE_SUB, NSEED, LOCAL_DIR.
"""
import io, json, os

import numpy as np

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
SCORE_SUB = os.environ.get("SCORE_SUB", "research_runs/maker_labels_tb3s_h150anch")
NSEED = int(os.environ.get("NSEED", "4"))
LOCAL_DIR = os.environ.get("LOCAL_DIR", "")
KS = [1, 2, 5, 10, 25, 50]                     # rows after entry ~ 3, 6, 15, 30, 75, 150 s
QS = [0.035, 0.1, 1.0, 10.0]                   # top-q selectivities, percent


def log(s):
    print(s, flush=True)


if LOCAL_DIR:
    def load(seed, fold):
        return np.load(f"{LOCAL_DIR}/PERFOLD_S{seed}_{SYM}_qm0_f{fold}.npz")
    nf = len([f for f in os.listdir(LOCAL_DIR)
              if f.startswith(f"PERFOLD_S0_{SYM}_qm0_f") and f.endswith(".npz")])
else:
    from google.cloud import storage
    bk = storage.Client(project=PROJ).bucket(BUCKET)

    def load(seed, fold):
        return np.load(io.BytesIO(
            bk.blob(f"{SCORE_SUB}/PERFOLD_S{seed}_{SYM}_qm0_f{fold}.npz").download_as_bytes()))
    nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SCORE_SUB}/PERFOLD_S0_{SYM}_qm0_f")
             if b.name.endswith(".npz"))

log(f"[{SYM}] folds={nf} seeds={NSEED}")

# ---------------------------------------------------------------- PART A, per fold
accA = {k: {"alone": [[], []], "both": [[], []], "all": [[], []]} for k in KS}
selrows = []          # (fold-local pct of axb, pct of noa, alone, ev, finite)

for f in range(nf):
    zs = [load(s, f) for s in range(NSEED)]
    z0 = zs[0]
    axb = np.mean([x["axb_te"].astype(np.float64) for x in zs], 0)
    noa = np.mean([x["noa_te"].astype(np.float64) for x in zs], 0)
    side = np.sum([x["side"].astype(int) for x in zs], 0) >= int(np.ceil(NSEED / 2))
    day = z0["day_te"].astype(int)
    fl = z0["fl"].astype(bool); fs = z0["fs"].astype(bool)
    netl = z0["netl"].astype(np.float64); nets = z0["nets"].astype(np.float64)
    n = len(axb)

    fil_ours = np.where(side, fl, fs)
    fil_other = np.where(side, fs, fl)
    net_ours = np.where(side, netl, nets)
    alone = fil_ours & ~fil_other
    both = fil_ours & fil_other
    m = (netl - nets) / 2.0                       # signed 150s move at each row

    for k in KS:
        i = np.arange(n - k)
        ok = (day[i] == day[i + k]) & fil_ours[i] & np.isfinite(m[i + k])
        i = i[ok]
        if not len(i):
            continue
        fwd = np.where(side[i], m[i + k], -m[i + k])       # forward move signed by OUR position
        agree = side[i + k] == side[i]
        for tag, msk in (("alone", alone[i]), ("both", both[i]), ("all", np.ones(len(i), bool))):
            accA[k][tag][0].append(fwd[msk & agree])
            accA[k][tag][1].append(fwd[msk & ~agree])

    # ---- PART B inputs: within-fold percentile of both rankings
    def pctile(v):
        o = np.argsort(v, kind="stable"); r = np.empty(len(v)); r[o] = np.arange(len(v))
        return 100.0 * r / max(len(v) - 1, 1)

    selrows.append((pctile(axb), pctile(noa), alone, net_ours,
                    fil_ours & np.isfinite(net_ours)))
    log(f"  fold{f}: n={n}")

log(f"\n=== {SYM} PART A — forward value from INSIDE the position, by rows-after-entry")
log(f"{'k rows':>7} {'~sec':>5} | {'pop':>6} {'n agree':>9} {'fwd|agree':>10} "
    f"{'n disagr':>9} {'fwd|disagree':>13} {'gap':>8}")
resA = {}
for k in KS:
    for tag in ("alone", "both"):
        a = np.concatenate(accA[k][tag][0]) if accA[k][tag][0] else np.array([])
        d = np.concatenate(accA[k][tag][1]) if accA[k][tag][1] else np.array([])
        if not len(a) or not len(d):
            continue
        resA[f"k{k}_{tag}"] = dict(n_agree=int(len(a)), fwd_agree=float(a.mean()),
                                   se_agree=float(a.std(ddof=1) / np.sqrt(len(a))),
                                   n_dis=int(len(d)), fwd_dis=float(d.mean()),
                                   se_dis=float(d.std(ddof=1) / np.sqrt(len(d))))
        log(f"{k:>7} {3*k:>5} | {tag:>6} {len(a):>9,} {a.mean():>+10.3f} "
            f"{len(d):>9,} {d.mean():>+13.3f} {a.mean()-d.mean():>+8.3f}")

log(f"\n=== {SYM} PART B — AxB ranking vs A-ablated (noA) ranking at MATCHED top-q")
log(f"{'top-q %':>8} | {'rank':>5} {'n sel':>8} {'n filled':>9} {'ALONE share':>12} {'EV':>8}")
resB = {}
for q in QS:
    for tag, idx in (("AxB", 0), ("noA", 1)):
        nsel = nfil = nalone = 0; evs = []
        for pa, pn, alone, netv, okv in selrows:
            p = pa if idx == 0 else pn
            sel = p >= (100.0 - q)
            nsel += int(sel.sum())
            m2 = sel & okv
            nfil += int(m2.sum()); nalone += int((m2 & alone).sum())
            evs.append(netv[m2])
        x = np.concatenate(evs)
        share = 100.0 * nalone / max(nfil, 1)
        resB[f"q{q}_{tag}"] = dict(n_sel=nsel, n_filled=nfil, alone_share=float(share),
                                   ev=float(x.mean()) if len(x) else 0.0)
        log(f"{q:>8.3f} | {tag:>5} {nsel:>8,} {nfil:>9,} {share:>11.2f}% "
            f"{x.mean() if len(x) else 0:>+8.3f}")

if LOCAL_DIR:
    with open(f"{LOCAL_DIR}/INSIDEPOS_{SYM}.json", "w") as fh:
        json.dump({"sym": SYM, "partA": resA, "partB": resB}, fh, indent=1, default=float)
    log(f"\nwrote {LOCAL_DIR}/INSIDEPOS_{SYM}.json")
