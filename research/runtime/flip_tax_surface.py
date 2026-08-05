#!/usr/bin/env python3
"""THE SIDE-FLIP / ROUND-TRIP-TAX DECOMPOSITION — what the bulk's -1.8bp is actually made of.

User question (2026-08-04): "the bulk sits at 40-47% hit — what if we just swap long and
short there?" The question reduces to one inequality. Write the bulk EV as

    EV(our side) = edge - tax          edge = directional component
    EV(flipped)  = -edge - tax         tax  = maker round-trip tax, side-symmetric

so EV(flip) = -EV(base) - 2*tax, and the flip pays iff tax < |EV(base)|/2. Both terms are
directly measurable because PERFOLD carries BOTH sides of every decision:

    netl, nets  net P&L in bp had we gone long / short
    fl,   fs    whether the maker entry would have filled on that side

For a symmetric move, long earns +m and short earns -m, so the SUM isolates the tax and the
DIFFERENCE isolates the move:

    tax = -mean(netl + nets)/2        m = (netl - nets)/2        edge = mean(sign(side)*m)

THE FLIP IS NOT A SIGN CHANGE. Quoting the other side fills at different moments, so the
flipped cell is computed with the OTHER side's fill flag (fs where we used fl), not by
negating a realised P&L. Both pairs are in the artifacts, so this is exact, not a proxy.

Three populations are reported, each by within-fold score percentile band:
  * OUR SIDE   ok = filled(side)          -- reproduces the published bulk cell
  * FLIPPED    ok = filled(~side)         -- the answer to the question
  * BOTH-FILLED  fl & fs                  -- the only rows where edge/tax split is exact,
    plus avg win / avg loss separately, which EV and hit (their product) hide.

VALIDATION GATE: the our-side full-population cell must reproduce
fillyear-20260731_untraded_population_value exactly (DOGE -1.824bp, hit 42.96%,
3,393,760 filled of 4,191,122). Printed as a gate, not assumed.

Ensemble score / side semantics transcribed VERBATIM from consensus_surface.py, which in
turn took them from strictfill_cells.py. No retraining, frozen labels.

Env: SYM, SCORE_SUB, NSEED, LOCAL_DIR (read npz from disk instead of GCS), OUT.
"""
import io, json, os

import numpy as np

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
SCORE_SUB = os.environ.get("SCORE_SUB", "research_runs/maker_labels_tb3s_h150anch")
NSEED = int(os.environ.get("NSEED", "4"))
LOCAL_DIR = os.environ.get("LOCAL_DIR", "")
OUT = os.environ.get("OUT", f"research_runs/objsel/FLIPTAX_{SYM}.json")

# within-fold score percentile bands, matching modelcap-20260801's marginal bands
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

log(f"[{SYM}] folds={nf} seeds={NSEED} src={'local' if LOCAL_DIR else 'gcs'}")

PCT, SIDE, FL, FS, NL, NS = [], [], [], [], [], []
for f in range(nf):
    zs = [load(s, f) for s in range(NSEED)]
    z0 = zs[0]
    te = np.mean([x["axb_te"].astype(np.float64) for x in zs], 0)
    side = np.sum([x["side"].astype(int) for x in zs], 0) >= int(np.ceil(NSEED / 2))
    # within-fold percentile of the ensemble score over ALL decisions of the fold
    order = np.argsort(te, kind="stable")
    rank = np.empty(len(te), dtype=np.float64)
    rank[order] = np.arange(len(te), dtype=np.float64)
    PCT.append(100.0 * rank / max(len(te) - 1, 1))
    SIDE.append(side)
    FL.append(z0["fl"].astype(bool)); FS.append(z0["fs"].astype(bool))
    NL.append(z0["netl"].astype(np.float64)); NS.append(z0["nets"].astype(np.float64))
    log(f"  fold{f}: n_te={len(te)}")

pct = np.concatenate(PCT); side = np.concatenate(SIDE)
fl = np.concatenate(FL); fs = np.concatenate(FS)
netl = np.concatenate(NL); nets = np.concatenate(NS)
del PCT, SIDE, FL, FS, NL, NS

net_base = np.where(side, netl, nets)          # P&L on the side the ensemble picked
fil_base = np.where(side, fl, fs)              # ...and whether THAT side would have filled
net_flip = np.where(side, nets, netl)          # P&L quoting the opposite side
fil_flip = np.where(side, fs, fl)              # ...with the opposite side's own fill flag


def pop(net, fil, mask):
    """EV / hit / avg win / avg loss over a filled population."""
    ok = mask & fil & np.isfinite(net)
    x = net[ok]
    if not len(x):
        return dict(n=0)
    w = x[x > 0]; l = x[x <= 0]
    return dict(n=int(len(x)), ev=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))),
                hit=float(100.0 * len(w) / len(x)),
                avg_win=float(w.mean()) if len(w) else 0.0,
                avg_loss=float(l.mean()) if len(l) else 0.0)


def split(mask):
    """edge / tax decomposition — only exact on rows where BOTH sides would have filled."""
    ok = mask & fl & fs & np.isfinite(netl) & np.isfinite(nets)
    if not ok.sum():
        return dict(n=0)
    a = netl[ok]; b = nets[ok]
    m = (a - b) / 2.0                                   # the signed move
    e = np.where(side[ok], m, -m)                       # move signed by OUR chosen side
    t = -(a + b) / 2.0                                  # the round-trip tax
    n = len(a)
    return dict(n=int(n), edge=float(e.mean()), edge_se=float(e.std(ddof=1) / np.sqrt(n)),
                tax=float(t.mean()), tax_se=float(t.std(ddof=1) / np.sqrt(n)),
                ev_long=float(a.mean()), ev_short=float(b.mean()),
                ev_ourside=float((e - t).mean()), ev_flip=float((-e - t).mean()),
                abs_move=float(np.abs(m).mean()))


res = {"sym": SYM, "nseed": NSEED, "score_sub": SCORE_SUB, "folds": int(nf),
       "n_decisions": int(len(netl)), "bands": []}

allm = np.ones(len(netl), dtype=bool)
res["ALL"] = {"our_side": pop(net_base, fil_base, allm),
              "flipped": pop(net_flip, fil_flip, allm),
              "split_both_filled": split(allm)}

for lo, hi in BANDS:
    m = (pct >= lo) & (pct < hi) if hi < 100 else (pct >= lo)
    res["bands"].append({"lo": lo, "hi": hi, "n_rows": int(m.sum()),
                         "our_side": pop(net_base, fil_base, m),
                         "flipped": pop(net_flip, fil_flip, m),
                         "split_both_filled": split(m)})

a = res["ALL"]
log(f"\n=== {SYM} FULL FILLED POPULATION (validation gate vs the published bulk cell)")
log(f"  our side : n={a['our_side']['n']:>9} EV={a['our_side']['ev']:+.3f}bp "
    f"hit={a['our_side']['hit']:.2f}%  win={a['our_side']['avg_win']:+.2f} "
    f"loss={a['our_side']['avg_loss']:+.2f}")
log(f"  FLIPPED  : n={a['flipped']['n']:>9} EV={a['flipped']['ev']:+.3f}bp "
    f"hit={a['flipped']['hit']:.2f}%  win={a['flipped']['avg_win']:+.2f} "
    f"loss={a['flipped']['avg_loss']:+.2f}")
s = a["split_both_filled"]
log(f"  both-filled n={s['n']}: edge={s['edge']:+.4f}+-{s['edge_se']:.4f}  "
    f"tax={s['tax']:+.4f}+-{s['tax_se']:.4f}  |move|={s['abs_move']:.2f}")
log(f"    -> EV(our)={s['ev_ourside']:+.3f}  EV(flip)={s['ev_flip']:+.3f}  "
    f"(long {s['ev_long']:+.3f} / short {s['ev_short']:+.3f})")

log(f"\n=== {SYM} BY WITHIN-FOLD SCORE PERCENTILE")
log(f"{'band':>14} {'n(our)':>9} {'EV our':>8} {'hit':>6} {'win':>7} {'loss':>7} "
    f"{'EV flip':>8} {'hit_f':>6} {'edge':>8} {'tax':>7}")
for b in res["bands"]:
    o = b["our_side"]; fp = b["flipped"]; sp = b["split_both_filled"]
    if not o.get("n"):
        continue
    log(f"{b['lo']:>6}-{b['hi']:<7} {o['n']:>9} {o['ev']:>+8.2f} {o['hit']:>6.2f} "
        f"{o['avg_win']:>+7.2f} {o['avg_loss']:>+7.2f} {fp.get('ev', 0):>+8.2f} "
        f"{fp.get('hit', 0):>6.2f} {sp.get('edge', 0):>+8.3f} {sp.get('tax', 0):>+7.3f}")

if LOCAL_DIR:
    with open(f"{LOCAL_DIR}/FLIPTAX_{SYM}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    log(f"\nwrote {LOCAL_DIR}/FLIPTAX_{SYM}.json")
else:
    bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
    log(f"\nwrote gs://{BUCKET}/{OUT}")
