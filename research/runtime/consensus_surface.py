#!/usr/bin/env python3
"""EV / win-rate / trade-count as a function of SEED CONSENSUS — the second selectivity
axis, the one the ETH "harmony" overlay uses and the one nobody has mapped.

The deployed policies select on ONE axis: the ensemble score against a causal tau. The
ETH SAFETY form adds a different axis entirely —

    cons = number of seeds whose OWN score clears its OWN frozen per-seed tau
    harmony blocks 2 <= cons <= NSEED-2, i.e. keeps only cons <= 1 or cons >= NSEED-1

— which is NOT a threshold on the score, so the traded set is not a prefix of the score
ranking. The ledger holds exactly one point on this axis (ETH DYN t5 +15.79/430 without
the overlay, +17.07/295 with it) and no surface. This script produces the surface:

  * the BASE RATE of cons over every filled decision (the overlay is nearly inert on the
    population — the question is what it does inside the traded set)
  * within the deployed ensemble selection: n, EV/tr and hit as a function of cons
  * the two cumulative readings — "require at least k votes" (cons >= k) and its mirror
    (cons <= k) — because the deployed rule keeps BOTH extremes and it has never been
    shown that both branches earn their place
  * the deployed harmony cell, reproduced, as the reference point

Everything comes from the PERFOLD artifacts alone: no retraining, no dataset download,
frozen labels as carried by PERFOLD itself. fixq_tau / sel_fixq / sel_dyn / the harmony
rule are transcribed VERBATIM from strictfill_cells.py so the selection semantics cannot
drift; the only new code is the tabulation.

Env: SYM, SCORE_SUB, NSEED, K, POLICY (fixq|dyn), CFGIDX unused, OUT.
"""
import io, json, os

import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "ETH")
SCORE_SUB = os.environ.get("SCORE_SUB", "research_runs/maker_labels_tb3s_h150danch_v2notod")
NSEED = int(os.environ.get("NSEED", "8"))
K = float(os.environ.get("K", "5"))
POLICY = os.environ.get("POLICY", "dyn")
OUT = os.environ.get("OUT", f"research_runs/objsel/CONSENSUS_{SYM}.json")
KDAYS = 30
bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s):
    print(s, flush=True)


# ---- transcribed VERBATIM from strictfill_cells.py (selection semantics, do not edit)
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

nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SCORE_SUB}/PERFOLD_S0_{SYM}_qm0_f")
         if b.name.endswith(".npz"))
log(f"[{SYM}] folds={nf} seeds={NSEED} policy={POLICY} K={K}")

rows_sel = []      # (cons, net, filled) for the ensemble-selected decisions
base_cons = np.zeros(NSEED + 1, dtype=np.int64)   # cons histogram over all filled rows
tot_days = 0

for f in range(nf):
    zs = [np.load(io.BytesIO(bk.blob(f"{SCORE_SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
          for s in range(NSEED)]
    z0 = zs[0]
    p = dict(day_tr=z0["day_tr"].astype(int), day_te=z0["day_te"].astype(int),
             tr=np.mean([x["axb_tr"].astype(np.float64) for x in zs], 0),
             te=np.mean([x["axb_te"].astype(np.float64) for x in zs], 0),
             side=np.sum([x["side"].astype(int) for x in zs], 0) >= int(np.ceil(NSEED / 2)),
             seed_tr=[x["axb_tr"].astype(np.float64) for x in zs],
             seed_te=[x["axb_te"].astype(np.float64) for x in zs])
    tot_days += len(np.unique(p["day_te"]))

    cons = np.zeros(len(p["te"]), int)
    for s in range(NSEED):
        cons += (p["seed_te"][s] >= fixq_tau(p["seed_tr"][s], p["day_tr"], K)).astype(int)

    netl = z0["netl"].astype(np.float64); nets = z0["nets"].astype(np.float64)
    fl = z0["fl"].astype(bool); fs = z0["fs"].astype(bool)
    net = np.where(p["side"], netl, nets)
    filled = np.where(p["side"], fl, fs)
    ok = filled & np.isfinite(net)
    base_cons += np.bincount(cons[ok], minlength=NSEED + 1)

    sel = SEL(p, p["te"], p["tr"], K)
    for i in sel:
        rows_sel.append((int(cons[i]), float(net[i]), bool(ok[i]), int(p["day_te"][i])))
    log(f"  fold{f}: n_te={len(p['te'])} selected={len(sel)} filled_rows={int(ok.sum())}")

C = np.array([r[0] for r in rows_sel]); N = np.array([r[1] for r in rows_sel])
OKm = np.array([r[2] for r in rows_sel]); DAY = np.array([r[3] for r in rows_sel])
res = {"sym": SYM, "nseed": NSEED, "policy": POLICY, "k": K, "tot_days": tot_days,
       "base_cons_all_filled": base_cons.tolist(), "by_cons": [], "cum_ge": [], "cum_le": []}


def cell(mask):
    """n / EV / hit / bp-day, WITH the standard error of EV - these cells run 25-200
    trades, so the SE is not optional decoration: a 40-80bp per-trade sd over n=50 puts
    a +-8bp band on every EV quoted here."""
    m = mask & OKm
    n = int(m.sum())
    if not n:
        return dict(n=0, ev=float("nan"), se=float("nan"), hit=float("nan"), bpd=float("nan"))
    x = N[m]
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(n=n, ev=float(x.mean()), se=se, sd=float(x.std(ddof=1)) if n > 1 else float("nan"),
                hit=float((x > 0).mean()), bpd=float(x.sum() / max(tot_days, 1)))


tot = base_cons.sum()
log(f"\n[{SYM}] BASE RATE of cons over ALL filled decisions (n={tot:,})")
for c in range(NSEED + 1):
    if base_cons[c]:
        log(f"   cons={c}: {base_cons[c]:>10,}  {100.0*base_cons[c]/tot:8.4f}%")

log(f"\n[{SYM}] INSIDE the deployed ensemble selection ({POLICY} t{int(K)}), by cons:")
log(f"   {'cons':>4} {'n':>5} {'EV/tr':>8} {'hit%':>7} {'bp/day':>8}")
for c in range(NSEED + 1):
    d = cell(C == c); res["by_cons"].append(dict(cons=c, **d))
    if d["n"]:
        log(f"   {c:>4} {d['n']:>5} {d['ev']:>+8.2f} {d['se']:>6.2f} {100*d['hit']:>6.1f} {d['bpd']:>+8.2f}")

log(f"\n[{SYM}] REQUIRE AT LEAST k votes (cons >= k):")
log(f"   {'k':>4} {'n':>5} {'EV/tr':>8} {'hit%':>7} {'bp/day':>8}")
for c in range(NSEED + 1):
    d = cell(C >= c); res["cum_ge"].append(dict(k=c, **d))
    if d["n"]:
        log(f"   {c:>4} {d['n']:>5} {d['ev']:>+8.2f} {d['se']:>6.2f} {100*d['hit']:>6.1f} {d['bpd']:>+8.2f}")

log(f"\n[{SYM}] the MIRROR branch (cons <= k), which the deployed rule also keeps:")
log(f"   {'k':>4} {'n':>5} {'EV/tr':>8} {'hit%':>7} {'bp/day':>8}")
for c in range(NSEED + 1):
    d = cell(C <= c); res["cum_le"].append(dict(k=c, **d))
    if d["n"]:
        log(f"   {c:>4} {d['n']:>5} {d['ev']:>+8.2f} {d['se']:>6.2f} {100*d['hit']:>6.1f} {d['bpd']:>+8.2f}")

full = cell(np.ones(len(C), bool))
harm = cell((C <= 1) | (C >= NSEED - 1))
res["cell_no_overlay"] = full
res["cell_harmony"] = harm
res["cell_harmony_low_branch"] = cell(C <= 1)
res["cell_harmony_high_branch"] = cell(C >= NSEED - 1)
log(f"\n[{SYM}] REFERENCE CELLS")
log(f"   no overlay          : n={full['n']:>4} EV {full['ev']:+7.3f}+-{full['se']:.2f} hit {100*full['hit']:.1f}%")
log(f"   harmony (deployed)  : n={harm['n']:>4} EV {harm['ev']:+7.3f}+-{harm['se']:.2f} hit {100*harm['hit']:.1f}%")
lo, hi = res["cell_harmony_low_branch"], res["cell_harmony_high_branch"]
log(f"     low branch cons<=1: n={lo['n']:>4} EV {lo['ev']:+7.3f}+-{lo['se']:.2f} hit {100*lo['hit']:.1f}%")
log(f"     high branch >=N-1 : n={hi['n']:>4} EV {hi['ev']:+7.3f}+-{hi['se']:.2f} hit {100*hi['hit']:.1f}%")

# ---------------------------------------------------------------- full economics
# metrics() transcribed VERBATIM from strictfill_economics.py (itself economics_3sym's,
# unchanged) so these numbers sit on the same methodology as the published 4.1 cells:
# equity *= 1 + FRAC*net per trade, daily returns over the FULL test span including
# zero-trade days, trade-level max drawdown, d0 = 202 = W+EMB = the first test day.
FRAC = {"DOGE": 0.5, "XRP": 0.5, "BTC": 0.5, "ETH": 1.25}.get(SYM, 0.5)
D1 = int(DAY.max())


def metrics(trades, frac=1.0, label="", d0=202, d1=None):
    if not trades:
        return None
    trades = sorted(trades, key=lambda t: t[0])
    days = np.array([t[0] for t in trades]); nets = np.array([t[1] for t in trades]) * 1e-4
    d1 = D1 if d1 is None else d1
    span = d1 - d0 + 1
    eq = 1.0; curve = []
    for d, r in zip(days, nets):
        eq *= (1.0 + frac * r); curve.append((d, eq))
    total = eq - 1.0
    daily = {}
    for d, r in zip(days, nets):
        daily[d] = daily.get(d, 1.0) * (1.0 + frac * r)
    dr = np.array([daily.get(d, 1.0) - 1.0 for d in range(d0, d1 + 1)])
    mu, sd = dr.mean(), dr.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(365.0) if sd > 0 else float("nan")
    eqs = np.array([c[1] for c in curve]); peak = np.maximum.accumulate(eqs)
    mdd = float(((eqs - peak) / peak).min())
    g = (1.0 + total) ** (1.0 / span)
    ann = g ** 365.0 - 1.0
    return dict(label=label, n=len(trades), tpd=len(trades) / span,
                ev_bp=float(np.array([t[1] for t in trades]).mean()),
                hit=float((nets > 0).mean()), monthly_roi=g ** 30.44 - 1.0, annual_roi=ann,
                sharpe=float(sharpe), max_dd=mdd,
                calmar=float(ann / abs(mdd)) if mdd < 0 else float("nan"),
                worst_day=float(dr.min()), zero_day_share=float((dr == 0).mean()))


VARIANTS = [("no_overlay", np.ones(len(C), bool)),
            ("harmony_deployed", (C <= 1) | (C >= NSEED - 1)),
            ("high_branch_ge_N-1", C >= NSEED - 1),
            ("UNANIMOUS_eq_N", C == NSEED)]
res["economics"] = {}
log("")
log(f"[{SYM}] FULL ECONOMICS (frac={FRAC}, span d202..d{D1}, same methodology as 4.1)")
log(f"   {'variant':>19} {'n':>5} {'EV':>7} {'hit%':>6} {'mo ROI':>8} {'Sharpe':>7} {'maxDD':>8} {'Calmar':>7}")
for nm, mk in VARIANTS:
    tr = [(int(DAY[i]), float(N[i])) for i in range(len(C)) if mk[i] and OKm[i]]
    m = metrics(tr, frac=FRAC, label=nm)
    res["economics"][nm] = m
    if m:
        log(f"   {nm:>19} {m['n']:>5} {m['ev_bp']:>+7.2f} {100*m['hit']:>5.1f} "
            f"{100*m['monthly_roi']:>+7.2f}% {m['sharpe']:>7.2f} {100*m['max_dd']:>+7.2f}% {m['calmar']:>7.2f}")

bk.blob(OUT).upload_from_string(json.dumps(res, default=float))
log(f"\n[saved] gs://{BUCKET}/{OUT}")
