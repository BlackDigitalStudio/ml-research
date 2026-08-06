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
SEEDS = [0, 1, 2, 3]
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


nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f"{SUB}/PERFOLD_S0_{SYM}_qm0_f") if b.name.endswith(".npz"))
Z = {s: [np.load(io.BytesIO(bk.blob(f"{SUB}/PERFOLD_S{s}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
         for f in range(nf)] for s in SEEDS}
print(f"{SUB.split('/')[-1]} {SYM}: folds={nf} | union-of-seeds policy (no jitter, standing directive)", flush=True)

out = {"nf": nf, "seeds": SEEDS}
for tgt in TGTS:
    fold_nets, fold_days_list = [], []
    kbucket = defaultdict(list)
    ties = 0; total = 0
    day_net = defaultdict(list)   # global day index -> nets
    all_days = []
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
        fdays = sorted(set(day_te.tolist()))
        fold_days_list.append(fdays)
        all_days.extend(fdays)
    a = np.concatenate([x for x in fold_nets if len(x)]) if fold_nets else np.array([])
    tot_days = len(set(all_days))
    ev = float(a.mean()); tpd = len(a) / max(tot_days, 1)
    print(f"\n== union T_s={tgt:g}: EV {ev:+.2f}bp n={len(a)} tpd {tpd:.2f} hit {100*(a>0).mean():.1f}% "
          f"ties {100*ties/max(total,1):.1f}%", flush=True)
    print("  per-fold sum%: " + " ".join(f"{x.sum()*0.01:+.1f}" for x in fold_nets), flush=True)
    # 30-day month buckets over the global ordered day sequence
    days_sorted = sorted(set(all_days))
    mb = []
    for m0 in range(0, len(days_sorted), 30):
        dset = set(days_sorted[m0:m0 + 30])
        vals = [x for d in dset for x in day_net.get(int(d), [])]
        mb.append(round(float(np.sum(vals) * 0.01), 1) if vals else 0.0)
    print(f"  month buckets sum%: {mb}", flush=True)
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
                            perfold=[round(float(x.sum() * 0.01), 1) for x in fold_nets],
                            month=mb, consensus=ks,
                            boot=dict(ev_p5=float(np.quantile(b_ev, .05)), ev_p50=float(np.quantile(b_ev, .5)),
                                      ev_p95=float(np.quantile(b_ev, .95)), Ppos=float(100 * np.mean(b_ev > 0)),
                                      bpd_p5=float(np.quantile(b_bpd, .05)), bpd_p95=float(np.quantile(b_bpd, .95))))

bk.blob(f"{SUB}/HBV1_UNION_{SYM}.json").upload_from_string(json.dumps(out, default=float))
print("\n[saved HBV1_UNION]", flush=True)
