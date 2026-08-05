#!/usr/bin/env python3
"""WHERE THE BULK'S -1.8bp ACTUALLY LIVES — the fill-asymmetry split.

Follow-up to flip_tax_surface.py, which measured (DOGE, gate-validated against the
published bulk cell) that the side-symmetric round-trip tax is ~+0.06bp and the
directional edge is POSITIVE (+0.31bp) and monotone in the score — yet the same
population is worth -1.824bp. Those two facts only reconcile one way: the loss is not a
symmetric per-trade cost, it is a SELECTION EFFECT IN THE FILL.

Rows where BOTH sides would have filled (price came back through both levels) are a
different, benign population from rows where ONLY OUR side filled (price left and did not
come back). This script splits the our-side-filled population into exactly those two and
prices each:

    ALONE  fil(side) & ~fil(~side)   -- we got filled and the other side never traded
    BOTH   fil(side) &  fil(~side)   -- both levels traded

The accounting identity EV(all) = w_both*EV(both) + w_alone*EV(alone) is asserted, so the
split cannot silently drift from the gate-validated cell. Reading ALONE as "adverse
selection" is INTERPRETATION; the split itself is arithmetic.

Same frozen inputs and semantics as flip_tax_surface.py. Env: SYM, SCORE_SUB, NSEED,
LOCAL_DIR.
"""
import io, json, os

import numpy as np

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
SCORE_SUB = os.environ.get("SCORE_SUB", "research_runs/maker_labels_tb3s_h150anch")
NSEED = int(os.environ.get("NSEED", "4"))
LOCAL_DIR = os.environ.get("LOCAL_DIR", "")
BANDS = [(0, 10), (10, 50), (50, 90), (90, 99), (99, 99.9), (99.9, 99.99), (99.99, 100)]


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

PCT, SIDE, FL, FS, NL, NS = [], [], [], [], [], []
for f in range(nf):
    zs = [load(s, f) for s in range(NSEED)]
    z0 = zs[0]
    te = np.mean([x["axb_te"].astype(np.float64) for x in zs], 0)
    order = np.argsort(te, kind="stable")
    rank = np.empty(len(te), dtype=np.float64); rank[order] = np.arange(len(te), dtype=np.float64)
    PCT.append(100.0 * rank / max(len(te) - 1, 1))
    SIDE.append(np.sum([x["side"].astype(int) for x in zs], 0) >= int(np.ceil(NSEED / 2)))
    FL.append(z0["fl"].astype(bool)); FS.append(z0["fs"].astype(bool))
    NL.append(z0["netl"].astype(np.float64)); NS.append(z0["nets"].astype(np.float64))

pct = np.concatenate(PCT); side = np.concatenate(SIDE)
fl = np.concatenate(FL); fs = np.concatenate(FS)
netl = np.concatenate(NL); nets = np.concatenate(NS)

net = np.where(side, netl, nets)
fil_ours = np.where(side, fl, fs)
fil_other = np.where(side, fs, fl)
fin = np.isfinite(net)


def cell(m):
    x = net[m]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), ev=float(x.mean()),
                se=float(x.std(ddof=1) / np.sqrt(len(x))),
                hit=float(100.0 * (x > 0).mean()))


res = {"sym": SYM, "nseed": NSEED, "bands": []}
rows = []
for lo, hi in [(0, 100)] + BANDS:
    bm = np.ones(len(net), bool) if (lo, hi) == (0, 100) else (
        (pct >= lo) & (pct < hi) if hi < 100 else (pct >= lo))
    base = bm & fil_ours & fin
    a = cell(base & ~fil_other)      # ALONE
    b = cell(base & fil_other)       # BOTH
    t = cell(base)
    if not t.get("n"):
        continue
    w_alone = a.get("n", 0) / t["n"]
    ident = w_alone * a.get("ev", 0.0) + (1 - w_alone) * b.get("ev", 0.0)
    rows.append((lo, hi, t, a, b, w_alone, ident))
    res["bands"].append(dict(lo=lo, hi=hi, all=t, alone=a, both=b, w_alone=w_alone,
                             identity_check=float(ident)))

log(f"\n=== {SYM}  fill-asymmetry split of the our-side-filled population")
log(f"{'band':>14} {'n all':>9} {'EV all':>8} | {'n ALONE':>9} {'EV':>8} {'hit':>6} {'share':>6} "
    f"| {'n BOTH':>9} {'EV':>8} {'hit':>6} | {'ident':>7}")
for lo, hi, t, a, b, w, ident in rows:
    log(f"{lo:>6}-{hi:<7} {t['n']:>9} {t['ev']:>+8.3f} | {a.get('n',0):>9} {a.get('ev',0):>+8.3f} "
        f"{a.get('hit',0):>6.2f} {100*w:>5.1f}% | {b.get('n',0):>9} {b.get('ev',0):>+8.3f} "
        f"{b.get('hit',0):>6.2f} | {ident:>+7.3f}")

if LOCAL_DIR:
    with open(f"{LOCAL_DIR}/FILLSPLIT_{SYM}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    log(f"wrote {LOCAL_DIR}/FILLSPLIT_{SYM}.json")
