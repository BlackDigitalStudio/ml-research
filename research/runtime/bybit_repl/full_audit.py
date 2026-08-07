#!/usr/bin/env python3
"""HBV1 consolidated audit: EVERY strategy form x the standard battery
(policy_metrics: FRAC=1.0, ROI/maxDD/worst-day/Sharpe, per-fold+LOFO gate, BOOT).

Forms per pool: mean-rank ensemble (mean of member rank scores, majority side)
at t targets; union-of-members at T_s targets; consensus K-of-N at (T_s, K)
grid (HBV1 rev14: decision traded iff >=K members independently pass their own
causal rolling threshold at per-member target T_s; side = majority of the
SELECTING members, ties skipped; K=1 == union). Pools are member lists
(sub:seed), so same-protocol, cross-protocol, no-OI and mixed-axis pools all go
through one code path. Nets/fills/days always come from members[0] (the EXEC
member) — in pools that mix label-horizon selectors (CFGIDX h90/h240), put an
h150-execution member first so every traded decision settles at deploy
semantics. Usage: full_audit.py  (env POOLS optional; defaults below;
CONS_T=""/CONS_KS gate the consensus grid — empty CONS_T keeps prior
invocations byte-identical in output set).
"""
import io
import json
import os
import sys

import numpy as np
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_metrics import battery, fmt  # noqa: E402

SYM = "DOGE"
KDAYS = 30
bk = storage.Client(project="x").bucket("market-data-0998ac51")

B = "maker_labels_tb3s_h150anch"
POOLS = {
    "v2-base": [(B, s) for s in range(4)],
    "v1": [(B + "_v1", s) for s in range(4)],
    "v2-nooi": [(B + "_v2_nooi", s) for s in range(4)],
    "xproto8(v1+v2)": [(B, s) for s in range(4)] + [(B + "_v1", s) for s in range(4)],
}
_env_pools = os.environ.get("POOLS", "")
if _env_pools:
    POOLS = {}
    for spec in _env_pools.split(";"):
        name, members = spec.split("=")
        POOLS[name] = [(("%s" % p.split(":")[0]), int(p.split(":")[1])) for p in members.split(",")]
ENS_T = [float(x) for x in os.environ.get("ENS_T", "1,2.5,5,10").split(",")]
UNION_T = [float(x) for x in os.environ.get("UNION_T", "0.625,1.25,2.5,5").split(",")]
CONS_T = [float(x) for x in os.environ.get("CONS_T", "").split(",") if x]
CONS_KS = [int(x) for x in os.environ.get("CONS_KS", "2,3,4,5,6,7,8").split(",") if x]
# FEE_BP: flat per-trade round-trip fee subtracted from every net (bp). Bybit linear
# non-VIP maker 0.02%/side -> 4 bp RT (both legs maker in the pegged cycle). The
# Bybit cells were simulated at 0 fee (CL DOGEUSDC promo convention) — this makes
# them honest for the venue that actually produced the data.
FEE_BP = float(os.environ.get("FEE_BP", "0"))

_cache = {}


def load(sub, seed, f):
    k = (sub, seed, f)
    if k not in _cache:
        _cache[k] = np.load(io.BytesIO(bk.blob(f"research_runs/{sub}/PERFOLD_S{seed}_{SYM}_qm0_f{f}.npz")
                                       .download_as_bytes()))
    return _cache[k]


def nfolds(sub, seed):
    return sum(1 for b in bk.client.list_blobs(bk, prefix=f"research_runs/{sub}/PERFOLD_S{seed}_{SYM}_qm0_f")
               if b.name.endswith(".npz"))


def causal_sel(sc_tr, sc_te, day_tr, day_te, tgt):
    days = sorted(set(day_te.tolist())); wpd = len(sc_te) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(day_tr.tolist())); seed = np.isin(day_tr, trd[-KDAYS:])
    buf = list(sc_tr[seed]); cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_te == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc_te[idx] >= tau].tolist()); buf.extend(sc_te[idx].tolist()); buf = buf[-cap:]
    return np.array(sel, dtype=int)


def run_pool(name, members):
    nf = nfolds(*members[0])
    Z = {m: [load(m[0], m[1], f) for f in range(nf)] for m in members}
    days_sorted = sorted({int(d) for f in range(nf) for d in np.unique(Z[members[0]][0 if False else f]["day_te"])})
    print(f"\n### pool {name}: {len(members)} members, {nf} folds, {len(days_sorted)} test days", flush=True)
    results = {}

    # ---- mean-rank ensemble forms
    for tgt in ENS_T:
        nets, tdays, fold_nets = [], [], []
        for f in range(nf):
            zs = [Z[m][f] for m in members]
            tr = np.mean([z["axb_tr"].astype(np.float64) for z in zs], 0)
            te = np.mean([z["axb_te"].astype(np.float64) for z in zs], 0)
            votes = np.sum([z["side"].astype(int) for z in zs], 0)
            side = votes >= len(members) / 2.0
            z0 = zs[0]
            sel = causal_sel(tr, te, z0["day_tr"], z0["day_te"], tgt)
            fn = []
            for i in sel:
                s_ = side[i]
                net = float(z0["netl"][i]) if s_ else float(z0["nets"][i])
                fill = bool(z0["fl"][i]) if s_ else bool(z0["fs"][i])
                if fill and np.isfinite(net):
                    net -= FEE_BP
                    fn.append(net); nets.append(net); tdays.append(int(z0["day_te"][i]))
            fold_nets.append(np.array(fn))
        m = battery(nets, tdays, fold_nets, days_sorted)
        results[f"ens_t{tgt:g}"] = m
        print(fmt(f"  ens t{tgt:g}", m), flush=True)

    # ---- union forms
    for tgt in UNION_T:
        nets, tdays, fold_nets = [], [], []
        for f in range(nf):
            sets = {m: set(causal_sel(Z[m][f]["axb_tr"].astype(np.float64), Z[m][f]["axb_te"].astype(np.float64),
                                      Z[m][f]["day_tr"], Z[m][f]["day_te"], tgt).tolist()) for m in members}
            z0 = Z[members[0]][f]
            fn = []
            for i in sorted(set().union(*sets.values())):
                ks = [m for m in members if i in sets[m]]
                sides = [bool(Z[m][f]["side"][i]) for m in ks]
                nl_ = sum(sides)
                if nl_ * 2 == len(sides):
                    continue
                s_ = nl_ * 2 > len(sides)
                net = float(z0["netl"][i]) if s_ else float(z0["nets"][i])
                fill = bool(z0["fl"][i]) if s_ else bool(z0["fs"][i])
                if fill and np.isfinite(net):
                    net -= FEE_BP
                    fn.append(net); nets.append(net); tdays.append(int(z0["day_te"][i]))
            fold_nets.append(np.array(fn))
        m = battery(nets, tdays, fold_nets, days_sorted)
        results[f"union_T{tgt:g}"] = m
        print(fmt(f"  union T{tgt:g}", m), flush=True)

    # ---- consensus K-of-N forms (rev14): one causal_sel pass per (T_s, fold),
    # then every K consumes the same selector sets. Ties (even split among the
    # selecting members) are skipped, as in the union form.
    for tgt in CONS_T:
        ks_here = sorted({k for k in CONS_KS if 2 <= k <= len(members)})
        if not ks_here:
            continue
        acc = {K: ([], [], []) for K in ks_here}   # nets, tdays, fold_nets
        for f in range(nf):
            sets = {m_: set(causal_sel(Z[m_][f]["axb_tr"].astype(np.float64),
                                       Z[m_][f]["axb_te"].astype(np.float64),
                                       Z[m_][f]["day_tr"], Z[m_][f]["day_te"], tgt).tolist())
                    for m_ in members}
            z0 = Z[members[0]][f]
            fn = {K: [] for K in ks_here}
            for i in sorted(set().union(*sets.values())):
                ks = [m_ for m_ in members if i in sets[m_]]
                if len(ks) < ks_here[0]:
                    continue
                sides = [bool(Z[m_][f]["side"][i]) for m_ in ks]
                nl_ = sum(sides)
                if nl_ * 2 == len(sides):
                    continue
                s_ = nl_ * 2 > len(sides)
                net = float(z0["netl"][i]) if s_ else float(z0["nets"][i])
                fill = bool(z0["fl"][i]) if s_ else bool(z0["fs"][i])
                if not (fill and np.isfinite(net)):
                    continue
                net -= FEE_BP
                for K in ks_here:
                    if len(ks) >= K:
                        fn[K].append((net, int(z0["day_te"][i])))
            for K in ks_here:
                nets_k, tdays_k, folds_k = acc[K]
                nets_k.extend(x[0] for x in fn[K])
                tdays_k.extend(x[1] for x in fn[K])
                folds_k.append(np.array([x[0] for x in fn[K]]))
        for K in ks_here:
            nets_k, tdays_k, folds_k = acc[K]
            m = battery(nets_k, tdays_k, folds_k, days_sorted)
            results[f"cons_T{tgt:g}_k{K}"] = m
            print(fmt(f"  cons T{tgt:g} k>={K}", m), flush=True)
    return results


if __name__ == "__main__":
    allres = {}
    tag = os.environ.get("OUT_TAG") or "_".join(POOLS)
    for name, members in POOLS.items():
        try:
            allres[name] = run_pool(name, members)
        except Exception as e:
            print(f"### pool {name}: FAILED {e}", flush=True)
        # incremental save after EVERY pool — preemption/timeout loses at most
        # the pool in flight (the 2026-08-07 rev14 4-pool run lost 60 min to a
        # preempt+restart and saved nothing)
        bk.blob(f"research_runs/HBV1_FULL_AUDIT_{tag}.json").upload_from_string(json.dumps(allres, default=float))
        print(f"[saved HBV1_FULL_AUDIT_{tag}.json through pool {name}]", flush=True)
