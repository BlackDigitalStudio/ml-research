#!/usr/bin/env python3
"""Quantify measured-policy (every above-tau grid decision = independent trade)
vs deployed-policy (single-position busy lock) on recorder replay artifacts.
Rows in D_day.npz are 3s-grid decisions in time order -> ts ~= 3*row_index s."""
import io, os, shutil, subprocess
import numpy as np

GCLOUD = shutil.which("gcloud") or "gcloud"
SCR = os.getcwd()  # caches under the run dir
CACHE = os.path.join(SCR, "recev_cache"); os.makedirs(CACHE, exist_ok=True)
STEP = 3.0

CASES = [
    # sym, prefix, mode, param
    ("DOGE", "_recev_h150anch2_DOGE", "frozen", 0.817010),   # deployed FIXQ t10
    ("XRP",  "_recev_h150anch2_XRP",  "frozen", 0.925442),   # deployed FIXQ t5
    ("BTC",  "_recev_h150d_BTC",      "dyn",    5.0),        # deployed DYN t5
    ("ETH",  "_recev_h150notod_ETH",  "dyn",    5.0),        # deployed DYN t5 (harmony ignored)
]
KDAYS = 30

def fetch(prefix):
    d = os.path.join(CACHE, prefix)
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True)
        subprocess.run([GCLOUD, "storage", "cp",
                        f"gs://market-data-0998ac51/research_runs/{prefix}/D_*.npz", d],
                       capture_output=True, text=True)
    fs = sorted(f for f in os.listdir(d) if f.startswith("D_") and f.endswith(".npz"))
    return [(f[2:10], np.load(os.path.join(d, f))) for f in fs]

def day_rows(z):
    sc = z["score"].astype(np.float64); side = z["side"].astype(bool)
    net = np.where(side, z["netl"].astype(np.float64), z["nets"].astype(np.float64))
    fill = np.where(side, z["FL"].astype(bool), z["FS"].astype(bool))
    return sc, net, fill

def select_frozen(days, tau):
    out = []
    for day, z in days:
        sc, net, fill = day_rows(z)
        sel = np.where(sc >= tau)[0]
        out.append((day, sel, net, fill))
    return out

def select_dyn(days, tgt):
    # causal rolling day-level tau, same as gate script
    nsc = sum(len(z["score"]) for _, z in days); wpd = nsc / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0)); buf = []; cap = max(int(KDAYS * wpd), 1)
    out = []
    for day, z in days:
        sc, net, fill = day_rows(z)
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel = np.where(sc >= tau)[0]
        out.append((day, sel, net, fill))
        buf.extend(sc.tolist()); buf = buf[-cap:]
    return out

def lock_sim(sel, fill, busy_fill, busy_nofill=60.0):
    taken = []
    free_at = -1e18
    for i in sel:
        t = i * STEP
        if t < free_at:
            continue
        taken.append(i)
        free_at = t + (busy_fill if fill[i] else busy_nofill)
    return np.array(taken, dtype=int)

def stats(net, fill, idx):
    ex = idx[fill[idx] & np.isfinite(net[idx])]
    if not len(ex):
        return 0, float("nan"), 0.0
    return len(ex), float(net[ex].mean()), float(net[ex].sum())

for sym, prefix, mode, prm in CASES:
    days = fetch(prefix)
    if not days:
        print(f"\n### {sym} {prefix}: NO DATA"); continue
    seldays = select_frozen(days, prm) if mode == "frozen" else select_dyn(days, prm)
    nd = len(seldays)
    print(f"\n### {sym} [{prefix}] {nd} days, policy={mode} {prm}")
    # measured policy
    N = E = 0; S = 0.0
    daily_meas = []
    conc_max = []
    first_ev, rest_ev = [], []
    for day, sel, net, fill in seldays:
        n, ev, s = stats(net, fill, sel)
        N += n; S += s; daily_meas.append(s)
        # concurrency of filled selected (hold interval ~ [t, t+240s])
        f = sel[fill[sel]]
        if len(f):
            t = f * STEP
            conc = [np.sum((t <= x) & (x < t + 240.0)) for x in t]
            conc_max.append(max(conc))
            # clusters by >210s gap on filled-selected
            cl = np.split(f, np.where(np.diff(t) > 210.0)[0] + 1)
            for c in cl:
                v = net[c][np.isfinite(net[c])]
                if len(v):
                    first_ev.append(v[0])
                    rest_ev.extend(v[1:].tolist())
        else:
            conc_max.append(0)
    evm = S / max(N, 1)
    print(f"  MEASURED (all signals): n={N} ({N/nd:.2f}/day) EV/tr={evm:+.2f}bp bpd={S/nd:+.1f}bp")
    print(f"  max concurrent positions: mean/day={np.mean(conc_max):.1f} max={max(conc_max)}")
    if first_ev:
        fe = np.array(first_ev); re = np.array(rest_ev)
        print(f"  cluster-first EV={fe.mean():+.2f}bp (n={len(fe)}) | cluster-rest EV={re.mean():+.2f}bp (n={len(re)})")
    # deployed policy variants
    for bf in (210.0, 330.0, 450.0):
        N2 = 0; S2 = 0.0
        for day, sel, net, fill in seldays:
            tk = lock_sim(sel, fill, bf)
            n, ev, s = stats(net, fill, tk)
            N2 += n; S2 += s
        ev2 = S2 / max(N2, 1)
        keep = 100.0 * S2 / S if S else float("nan")
        print(f"  LOCKED busy={bf:.0f}s: n={N2} ({N2/nd:.2f}/day) EV/tr={ev2:+.2f}bp bpd={S2/nd:+.1f}bp | bpd kept {keep:.0f}%")
