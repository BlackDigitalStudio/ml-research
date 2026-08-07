#!/usr/bin/env python3
"""HBV1 rev4 analysis (non-frozen): UNION-of-seeds aggregation as an honest policy.

Policy: each seed s independently runs the frozen causal_rolling selection at
per-seed target T_s trades/day on ITS OWN scores; the union of selected decisions
is traded once each. Side = majority of the SELECTING seeds' sides (measured tie
rate reported; ties -> skip). Entry/exit/fill semantics unchanged (fold artifacts).

Battery (standing directive: NO jitter): EV/tr, tpd, hit; per-fold sums; 30-day
month buckets; day-block bootstrap (L=7, 1000 reps) CI90 + P(EV>0); consensus-k
decomposition. Sweep T_s in TGTS (union tpd reported per point).
Usage: union_policy.py [SYM]; env XSYM_SUB, TGTS (default "1.25,2.5,5").
"""
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
from google.cloud import storage

SYM = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
SUB = "research_runs/" + os.environ.get("XSYM_SUB", "maker_labels_tb3s_h150anch")
TGTS = [float(x) for x in os.environ.get("TGTS", "1.25,2.5,5").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3").split(",")]
# MEMBERS: cross-prefix member list "sub:seed,sub:seed,..." (overrides SUB/SEEDS) —
# lets one union mix protocols (v1+v2), since all prefixes share the same dataset,
# folds, days and labels (only the models differ).
_MEMBERS = os.environ.get("MEMBERS", "")
if _MEMBERS:
    MEMBERS = [(("research_runs/" + p.split(":")[0]), int(p.split(":")[1])) for p in _MEMBERS.split(",")]
else:
    MEMBERS = [(SUB, s) for s in SEEDS]
KDAYS = 30
bk = storage.Client(project="x").bucket("market-data-0998ac51")


def select(z, tgt):
    sc_tr = z["axb_tr"].astype(np.float64); sc_te = z["axb_te"].astype(np.float64)
    day_tr = z["day_tr"]; day_te = z["day_te"]
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    return set(sel)


_sub0 = MEMBERS[0][0]
nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{_sub0}/PERFOLD_S{MEMBERS[0][1]}_{SYM}_qm0_f")
         if b.name.endswith(".npz"))
Z = {m: [np.load(io.BytesIO(bk.blob(f"{m[0]}/PERFOLD_S{m[1]}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
         for f in range(nf)] for m in MEMBERS}
SEEDS = MEMBERS  # downstream indexing is by member key
print(f"{len(MEMBERS)} members {[(m[0].split('/')[-1], m[1]) for m in MEMBERS]} {SYM}: folds={nf} | "
      f"union policy (no jitter, standing directive)", flush=True)

out = {"nf": nf, "seeds": SEEDS}
for tgt in TGTS:
    fold_nets, fold_days_list = [], []
    kbucket = defaultdict(list)
    ties = 0; total = 0
    day_net = defaultdict(list)   # global day index -> nets
    all_days = []
    ordered_trades = []           # chronological nets for the compounded curve
    for f in range(nf):
        sets = {s: select(Z[s][f], tgt) for s in SEEDS}
        union = sorted(set().union(*sets.values()))
        z0 = Z[SEEDS[0]][f]
        day_te = z0["day_te"]
        nets_f = []
        for i in union:
            ks = [s for s in SEEDS if i in sets[s]]
            sides = [bool(Z[s][f]["side"][i]) for s in ks]
            nlong = sum(sides); n = len(sides)
            total += 1
            if nlong * 2 == n:
                ties += 1
                continue
            side = nlong * 2 > n
            z = Z[ks[0]][f]
            net = float(z["netl"][i]) if side else float(z["nets"][i])
            fill = bool(z["fl"][i]) if side else bool(z["fs"][i])
            if not (fill and np.isfinite(net)):
                continue
            nets_f.append(net)
            kbucket[len(ks)].append(net)
            day_net[int(day_te[i])].append(net)
        fold_nets.append(np.array(nets_f))
        ordered_trades.extend(nets_f)          # fold/decision order = chronological
        fdays = sorted(set(day_te.tolist()))
        fold_days_list.append(fdays)
        all_days.extend(fdays)
    a = np.concatenate([x for x in fold_nets if len(x)]) if fold_nets else np.array([])
    tot_days = len(set(all_days))
    ev = float(a.mean()); tpd = len(a) / max(tot_days, 1)
    print(f"\n== union T_s={tgt:g}: EV {ev:+.2f}bp n={len(a)} tpd {tpd:.2f} hit {100*(a>0).mean():.1f}% "
          f"ties {100*ties/max(total,1):.1f}%", flush=True)
    print("  per-fold EV(n): " + " ".join(f"{(x.mean() if len(x) else 0):+.1f}({len(x)})" for x in fold_nets), flush=True)
    print("  per-fold sum%: " + " ".join(f"{x.sum()*0.01:+.1f}" for x in fold_nets), flush=True)
    # VALIDITY GATE (user directive 2026-08-06): all folds positive + LOFO (worst = drop
    # the BEST fold) materially positive + BOOT floor > 0.
    lofo = []
    for f in range(nf):
        rest = np.concatenate([fold_nets[g] for g in range(nf) if g != f and len(fold_nets[g])]) \
            if nf > 1 else np.array([])
        lofo.append(float(rest.mean()) if len(rest) else float("nan"))
    print("  LOFO EV: " + " ".join(f"-f{f}:{v:+.1f}" for f, v in enumerate(lofo)), flush=True)
    n_neg_folds = sum(1 for x in fold_nets if len(x) and x.mean() < 0)
    print(f"  GATE: neg-folds={n_neg_folds} | LOFO-min {np.nanmin(lofo):+.2f}", flush=True)
    # 30-day month buckets over the global ordered day sequence
    days_sorted = sorted(set(all_days))
    mb = []
    for m0 in range(0, len(days_sorted), 30):
        dset = set(days_sorted[m0:m0 + 30])
        vals = [x for d in dset for x in day_net.get(int(d), [])]
        mb.append(round(float(np.sum(vals) * 0.01), 1) if vals else 0.0)
    print(f"  month buckets sum%: {mb}", flush=True)
    # ROI / maxDD (README trading_algorithm 4.1 methodology, @0.5 convention:
    # equity *= 1 + 0.5*net per trade, sequential; fair here — hold ~150s at
    # 3-10 tr/d means near-zero overlap, unlike the CL FIXQ burst stacks).
    FRAC = 0.5
    eq = np.cumprod(1.0 + FRAC * np.asarray(ordered_trades) * 1e-4)
    run_max = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    maxdd = float((1.0 - np.concatenate([[1.0], eq]) / run_max).max())
    span_days = max(len(days_sorted), 1)
    roi_month = float(eq[-1] ** (30.0 / span_days) - 1.0) if len(eq) else 0.0
    dret = []
    for d in days_sorted:
        vals = day_net.get(int(d), [])
        dret.append(float(np.prod([1.0 + FRAC * x * 1e-4 for x in vals]) - 1.0))
    dret = np.array(dret)
    worst_day = float(dret.min()) if len(dret) else 0.0
    sharpe = float(dret.mean() / dret.std() * np.sqrt(365.0)) if len(dret) and dret.std() > 0 else 0.0
    mroi = []
    for m0 in range(0, span_days, 30):
        mroi.append(round(100 * float(np.prod(1.0 + dret[m0:m0 + 30]) - 1.0), 1))
    print(f"  ROI@0.5: monthly {100*roi_month:+.1f}% | maxDD(trade-level) {-100*maxdd:.1f}% | "
          f"worst day {100*worst_day:+.1f}% | Sharpe(daily,ann) {sharpe:.2f}", flush=True)
    print(f"  month ROI%: {mroi}", flush=True)
    # day-block bootstrap L=7
    span = len(days_sorted)
    rngb = np.random.default_rng(1)
    b_ev, b_bpd = [], []
    for rep in range(1000):
        picked = []
        while len(picked) < span:
            i0 = rngb.integers(0, max(span - 7, 1))
            picked.extend(days_sorted[i0:i0 + 7])
        picked = picked[:span]
        tr = [x for d in picked for x in day_net.get(int(d), [])]
        if tr:
            b_ev.append(np.mean(tr)); b_bpd.append(np.sum(tr) / span)
    b_ev = np.array(b_ev); b_bpd = np.array(b_bpd)
    print(f"  BOOT L=7: EV CI90 [{np.quantile(b_ev,.05):+.2f}, {np.quantile(b_ev,.95):+.2f}] "
          f"P(EV>0)={100*np.mean(b_ev>0):.0f}% | bpd CI90 [{np.quantile(b_bpd,.05):+.2f}, {np.quantile(b_bpd,.95):+.2f}]", flush=True)
    ks = {k: {"n": len(v), "ev": float(np.mean(v))} for k, v in sorted(kbucket.items())}
    print("  consensus-k: " + " | ".join(f"k={k}: n={v['n']} EV {v['ev']:+.2f}" for k, v in ks.items()), flush=True)
    out[f"T{tgt:g}"] = dict(ev=ev, n=int(len(a)), tpd=tpd, hit=float((a > 0).mean()),
                            ties_pct=100 * ties / max(total, 1),
                            perfold_ev=[round(float(x.mean()), 2) if len(x) else None for x in fold_nets],
                            perfold_n=[int(len(x)) for x in fold_nets],
                            lofo=[round(v, 2) for v in lofo],
                            gate=dict(neg_folds=n_neg_folds, lofo_min=float(np.nanmin(lofo))),
                            roi=dict(frac=FRAC, monthly=roi_month, maxdd=maxdd,
                                     worst_day=worst_day, sharpe=sharpe, month_roi_pct=mroi),
                            perfold=[round(float(x.sum() * 0.01), 1) for x in fold_nets],
                            month=mb, consensus=ks,
                            boot=dict(ev_p5=float(np.quantile(b_ev, .05)), ev_p50=float(np.quantile(b_ev, .5)),
                                      ev_p95=float(np.quantile(b_ev, .95)), Ppos=float(100 * np.mean(b_ev > 0)),
                                      bpd_p5=float(np.quantile(b_bpd, .05)), bpd_p95=float(np.quantile(b_bpd, .95))))

_tag = os.environ.get("UTAG", "" if [m[1] for m in MEMBERS] == [0, 1, 2, 3] and not _MEMBERS else f"_s{len(MEMBERS)}")
bk.blob(f"{SUB}/HBV1_UNION_{SYM}{_tag}.json").upload_from_string(json.dumps(out, default=float))
print("\n[saved HBV1_UNION]", flush=True)
