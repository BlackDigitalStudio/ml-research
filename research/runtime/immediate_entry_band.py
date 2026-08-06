#!/usr/bin/env python3
"""WHAT IS A SCORE BAND WORTH IF WE ENTER IMMEDIATELY INSTEAD OF RESTING A MAKER QUOTE?

User question (2026-08-05): "take the moments where there IS edge - band 99.9-99.99 - and
naively assume we enter the trade the moment B says long/short. By what % would the
deployed cell's ROI change?"

WHY THE MAKER ARTIFACTS CANNOT ANSWER IT, stated first because it is the reason this
script exists. PERFOLD carries netl/nets only where that side's maker quote FILLED; on the
ALONE rows (our side filled, the other never traded) one leg is NaN, so the signed move
(netl-nets)/2 - the quantity every `edge` number in BULKEDGE rev1 is built from - is
computable ONLY on the both-filled rows. Those are exactly the rows selected FOR being
two-sided, so using their edge as the value of an immediate entry would import the fill
selection this tier is trying to escape.

THE INSTRUMENT THAT DOES ANSWER IT: the label archive carries rH15 / rH30 / rH60 -
TIME-based forward mid returns (+-5s tolerance, subs60_build_tb3s_labels.py) - defined for
EVERY decision regardless of any fill. Signing them by the ensemble's own side gives the
gross move an immediate entry would capture, with no fill selection anywhere in it.

WHAT THIS IS NOT, and the number is meaningless without it: rH is MID-TO-MID. It contains
NO execution cost - no half-spread, no taker fee, no slippage - while an immediate entry
pays all of them on BOTH legs. So the output is a GROSS move and a BREAK-EVEN COST, never
a net EV. The deployed +10.513bp is a NET maker number; the two are not comparable until a
cost is assumed, and this script refuses to assume one.

HORIZON, and it is the whole comparison (user correction 2026-08-05): the deployed cell
holds 150s FROM THE FILL, so pricing an immediate entry at 60s FROM THE DECISION compares
two different objects. The archive's longest leg is 60s, so the 150s move is CHAINED over
ts -- r150(i) = rH60(i) + rH60(j60) + rH30(j120), with j60/j120 the decision rows nearest
ts_i+60s / ts_i+120s (same day, within HOP_TOL). The joins leave a small unmeasured gap
whose move is mean-zero. Two guards, because the first pass got both wrong:
  * CHAIN GATE - chained rH30(i)+rH30(j30) must reproduce the DIRECT rH60(i).
  * PAIRED COMPARISON - r150 coverage FALLS WITH THE SCORE (0.92 in the bulk, 0.46 on the
    deployed set at +-5s), so an unpaired mean over covered rows is a biased subsample.
    Every immediate-vs-maker number is therefore computed on the SAME rows, and HOP_TOL is
    swept so the coverage/noise trade-off is visible instead of assumed.

GATES (all three must pass or the script exits):
  1. JOIN - the day-reconstructed archive index equals PERFOLD's day_te (verbatim from
     strictfill_cells.py).
  2. LABELS - archive fl/fs/pnl at that index equal PERFOLD's stored arrays.
  3. CELL - the deployed FIXQ t10 selection reproduces +10.513bp / n=1204 (DOGE).

Env: SYM, NSEED, LOCAL_DIR (PERFOLD dir), ARCHIVE (label .npz), BUDGET, CFGIDX.
"""
import json, os

import numpy as np

SYM = os.environ.get("SYM", "DOGE")
NSEED = int(os.environ.get("NSEED", "4"))
LOCAL_DIR = os.environ["LOCAL_DIR"]
ARCHIVE = os.environ["ARCHIVE"]
BUDGET = float(os.environ.get("BUDGET", "10"))
CFGIDX = int(os.environ.get("CFGIDX", "1"))          # 1 = the 150s hold, the deployed cell
KDAYS = 30
BANDS = [(0, 10), (10, 50), (50, 90), (90, 99), (99, 99.9), (99.9, 99.99), (99.99, 100)]


def log(s):
    print(s, flush=True)


def fixq_tau(tr, day_tr, k):
    trd = sorted(set(day_tr.tolist()))[-KDAYS:]
    s = tr[np.isin(day_tr, trd)]
    if not len(s):
        return float("inf")
    wpd = len(s) / max(len(trd), 1)
    return float(np.quantile(s, max(0.0, 1.0 - k / max(wpd, 1.0))))


z = np.load(ARCHIVE, allow_pickle=True)
day_all = z["day"].astype(int)
NL_A = z["pnl_long"][CFGIDX, 0, :].astype(np.float64) * 100.0
NS_A = z["pnl_short"][CFGIDX, 0, :].astype(np.float64) * 100.0
FL_A = z["fill_long"][0].astype(bool)
FS_A = z["fill_short"][0].astype(bool)
RH = {h: z[f"rH{h}"].astype(np.float64) for h in (15, 30, 60)}   # ALREADY bp (rh_time:
# "time-based fwd log-return over hs seconds ... (bp; NaN if no tick within +-5s)" -
# the *1e4 is applied inside the builder, so do NOT scale again (2026-08-05).
log(f"[archive] N={len(day_all):,} days={day_all.max()+1} cfgidx={CFGIDX}")

nf = len([f for f in os.listdir(LOCAL_DIR)
          if f.startswith(f"PERFOLD_S0_{SYM}_qm0_f") and f.endswith(".npz")])

PCT, SIDE, DEP, IDX, NET, FIL = [], [], [], [], [], []
for f in range(nf):
    zs = [np.load(f"{LOCAL_DIR}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz") for s in range(NSEED)]
    z0 = zs[0]
    day_te = z0["day_te"].astype(int)

    idx = np.where(np.isin(day_all, np.unique(day_te)))[0]
    if len(idx) != len(day_te) or not np.array_equal(day_all[idx], day_te):
        raise SystemExit(f"JOIN GATE 1 FAILED fold{f}: {len(idx)} vs {len(day_te)}")
    ok = (np.array_equal(FL_A[idx], z0["fl"].astype(bool))
          and np.array_equal(FS_A[idx], z0["fs"].astype(bool))
          and np.allclose(NL_A[idx], z0["netl"].astype(np.float64), rtol=0, atol=2e-3, equal_nan=True)
          and np.allclose(NS_A[idx], z0["nets"].astype(np.float64), rtol=0, atol=2e-3, equal_nan=True))
    if not ok:
        raise SystemExit(f"JOIN GATE 2 FAILED fold{f}: archive labels != PERFOLD labels")

    te = np.mean([x["axb_te"].astype(np.float64) for x in zs], 0)
    tr_ = np.mean([x["axb_tr"].astype(np.float64) for x in zs], 0)
    side = np.sum([x["side"].astype(int) for x in zs], 0) >= int(np.ceil(NSEED / 2))
    order = np.argsort(te, kind="stable")
    rank = np.empty(len(te)); rank[order] = np.arange(len(te))
    PCT.append(100.0 * rank / max(len(te) - 1, 1))
    SIDE.append(side); IDX.append(idx)
    DEP.append(te >= fixq_tau(tr_, z0["day_tr"].astype(int), BUDGET))
    netl = z0["netl"].astype(np.float64); nets = z0["nets"].astype(np.float64)
    NET.append(np.where(side, netl, nets))
    FIL.append(np.where(side, z0["fl"].astype(bool), z0["fs"].astype(bool)))
    log(f"  fold{f}: n={len(te):,}  gates 1-2 OK")

pct = np.concatenate(PCT); side = np.concatenate(SIDE); dep = np.concatenate(DEP)
idx = np.concatenate(IDX); net = np.concatenate(NET); fil = np.concatenate(FIL)
ndays = len(np.unique(day_all[idx]))

# signed forward mid move an immediate entry would capture, per horizon
FWD = {h: np.where(side, RH[h][idx], -RH[h][idx]) for h in (15, 30, 60)}

# ---------------------------------------------------------------- GATE 3: the cell
m = dep & fil & np.isfinite(net)
ev_dep, n_dep = float(net[m].mean()), int(m.sum())
log(f"\n[GATE 3] deployed FIXQ t{BUDGET:.0f}: n={n_dep} EV={ev_dep:+.3f}bp hit={100*(net[m]>0).mean():.2f}%"
    f"  (published DOGE cell: +10.513 / 1204)")
if SYM == "DOGE" and not (abs(ev_dep - 10.513) < 0.01 and n_dep == 1204):
    raise SystemExit("GATE 3 FAILED: the deployed cell does not reproduce")
bpd_dep = ev_dep * n_dep / ndays


def stat(mask, v):
    x = v[mask & np.isfinite(v)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), ev=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))),
                hit=float(100.0 * (x > 0).mean()))


res = {"sym": SYM, "budget": BUDGET, "cfgidx": CFGIDX, "n_days": int(ndays),
       "deployed": dict(n=n_dep, ev=ev_dep, bpd=bpd_dep), "bands": []}

log(f"\n=== {SYM} IMMEDIATE ENTRY: GROSS signed forward mid move by score band "
    f"(NO execution cost of any kind)")
log(f"{'band':>13} {'n rows':>8} {'per day':>8} | {'fwd15':>8} {'hit':>6} | {'fwd30':>8} {'hit':>6} "
    f"| {'fwd60':>8} {'hit':>6}")
for lo, hi in BANDS + [("DEP", "DEP")]:
    if lo == "DEP":
        bm = dep; label = f"FIXQ t{BUDGET:.0f}"
    else:
        bm = (pct >= lo) & (pct < hi) if hi < 100 else (pct >= lo)
        label = f"{lo}-{hi}"
    row = {h: stat(bm, FWD[h]) for h in (15, 30, 60)}
    res["bands"].append(dict(band=label, n_rows=int(bm.sum()), per_day=float(bm.sum() / ndays),
                             fwd={str(h): row[h] for h in (15, 30, 60)}))
    log(f"{label:>13} {int(bm.sum()):>8,} {bm.sum()/ndays:>8.1f} | "
        + " | ".join(f"{row[h].get('ev',0):>+8.3f} {row[h].get('hit',0):>6.2f}" for h in (15, 30, 60)))

log(f"\n=== {SYM} WHAT IT WOULD TAKE TO BEAT THE DEPLOYED CELL "
    f"(deployed bpd {bpd_dep:+.2f} = {ev_dep:+.3f}bp x {n_dep} trades / {ndays} days)")
log(f"{'band':>13} {'horizon':>8} {'n/day':>7} {'gross bp':>9} {'gross bpd':>10} "
    f"{'bpd ratio':>10} {'break-even cost':>16}")
for b in res["bands"]:
    for h in (15, 30, 60):
        s = b["fwd"][str(h)]
        if not s.get("n"):
            continue
        bpd = s["ev"] * b["per_day"]
        be = s["ev"] - bpd_dep / max(b["per_day"], 1e-9)     # cost at which it MATCHES deployed
        log(f"{b['band']:>13} {h:>7}s {b['per_day']:>7.1f} {s['ev']:>+9.3f} {bpd:>+10.2f} "
            f"{bpd/bpd_dep:>10.2f}x {be:>+16.3f}")

# ------------------------------------------------- the fill-selection identity, unconditional
# The same signed mid move, split by what the maker quote WOULD have done. This is the one
# table that shows the mechanism instead of its consequence: ALONE = filled because the
# price came at us, UNFILLED = the price left in OUR favour and the resting quote never
# traded. Their near-symmetry is what makes the realised (filled-only) sample negative
# while the unconditional move is ~0.
log(f"\n=== {SYM} THE FILL-SELECTION IDENTITY (signed mid move by band x what the quote did)")
log(f"{'band':>13} | {'ALONE n':>9} {'move':>8} | {'BOTH n':>9} {'move':>8} | "
    f"{'UNFILLED n':>10} {'move':>8} | {'ALL':>8}")
FILO = []
for f in range(nf):
    z0 = np.load(f"{LOCAL_DIR}/PERFOLD_S0_{SYM}_qm0_f{f}.npz")
    sd = SIDE[f]
    FILO.append(np.where(sd, z0["fs"].astype(bool), z0["fl"].astype(bool)))
filo = np.concatenate(FILO)
POPS = (("alone", fil & ~filo), ("both", fil & filo), ("unfilled", ~fil))
res["identity"] = []
for lo, hi in BANDS:
    bm = (pct >= lo) & (pct < hi) if hi < 100 else (pct >= lo)
    row = {tag: stat(bm & p, FWD[60]) for tag, p in POPS}
    row["all"] = stat(bm, FWD[60])
    res["identity"].append(dict(band=f"{lo}-{hi}", **{k: v for k, v in row.items()}))
    log(f"{f'{lo}-{hi}':>13} | " + " | ".join(
        f"{row[t].get('n',0):>9,} {row[t].get('ev',0):>+8.3f}" for t, _ in POPS)
        + f" | {row['all'].get('ev',0):>+8.3f}")

# ------------------------------------------------- THE 150s PAIRED COMPARISON (the answer)
TS = z["ts"].astype(np.int64)
if not np.all(np.diff(TS) >= 0):
    raise SystemExit("archive ts is not sorted -- the hop construction assumes it is")
NS = 1_000_000_000


def hop(sec, tol):
    tgt = TS + int(sec * NS)
    j = np.clip(np.searchsorted(TS, tgt, "left"), 0, len(TS) - 1)
    return j, (np.abs(TS[j] - tgt) <= tol * NS) & (day_all[j] == day_all)


j30, ok30 = hop(30, 5)
g = ok30 & np.isfinite(RH[30]) & np.isfinite(RH[30][j30]) & np.isfinite(RH[60])
ch = RH[30][g] + RH[30][j30][g]; di = RH[60][g]
log(f"\n[CHAIN GATE] chained(30+30) vs direct rH60 on n={int(g.sum()):,}: means {ch.mean():+.4f} / "
    f"{di.mean():+.4f}, corr {np.corrcoef(ch, di)[0, 1]:.4f}, sd(diff) {(ch-di).std():.3f}bp")
if abs(ch.mean() - di.mean()) > 0.05 or np.corrcoef(ch, di)[0, 1] < 0.95:
    raise SystemExit("CHAIN GATE FAILED: the leg composition does not reproduce the direct leg")

mk_ok = fil & np.isfinite(net)
res["paired150"] = []
log(f"\n=== {SYM} IMMEDIATE (gross mid, 150s from the decision) vs MAKER (net, 150s from the "
    f"fill), PAIRED on identical rows")
for tol in (5, 15, 30, 60):
    j60, ok60 = hop(60, tol); j120, ok120 = hop(120, tol)
    r150 = RH[60] + RH[60][j60] + RH[30][j120]
    v = (ok60 & ok120 & np.isfinite(r150))[idx]
    fwd150 = np.where(side, r150[idx], -r150[idx])
    log(f"--- hop tolerance +-{tol}s")
    log(f"{'band':>12} {'cov':>6} {'n paired':>9} | {'immediate':>10} {'hit':>6} | "
        f"{'maker':>8} {'hit':>6} | {'diff':>8} {'se':>6}")
    for lo, hi in BANDS + [("DEP", "DEP")]:
        if lo == "DEP":
            bm = dep; lab = f"FIXQ t{BUDGET:.0f}"
        else:
            bm = (pct >= lo) & (pct < hi) if hi < 100 else (pct >= lo)
            lab = f"{lo}-{hi}"
        p = bm & v & mk_ok
        if p.sum() < 30:
            continue
        a = fwd150[p]; b = net[p]; d = a - b
        rec = dict(tol=tol, band=lab, cov=float(v[bm].mean()), n=int(p.sum()),
                   immediate=float(a.mean()), maker=float(b.mean()), diff=float(d.mean()),
                   se=float(d.std(ddof=1) / np.sqrt(len(d))),
                   band_per_day=float(bm.sum() / ndays),
                   maker_trades_per_day=float((bm & mk_ok).sum() / ndays))
        res["paired150"].append(rec)
        log(f"{lab:>12} {rec['cov']:>6.3f} {rec['n']:>9,} | {a.mean():>+10.3f} "
            f"{100*(a>0).mean():>6.2f} | {b.mean():>+8.3f} {100*(b>0).mean():>6.2f} | "
            f"{d.mean():>+8.3f} {rec['se']:>6.3f}")

out = f"{LOCAL_DIR}/IMMEDIATE_{SYM}.json"
with open(out, "w") as fh:
    json.dump(res, fh, indent=1, default=float)
log(f"\nwrote {out}")
log("reading: 'break-even cost' is the per-trade round-trip execution cost at which that "
    "band/horizon delivers exactly the deployed cell's bp/day. Above it the switch loses.")
