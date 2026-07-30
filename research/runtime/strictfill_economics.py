#!/usr/bin/env python3
"""README 4.1 economics under BOTH fill models, from the strictfill_cells.py trade lists.

OPS-FILLYEAR rev1. The metrics function is economics_3sym.py's `metrics()` verbatim
(equity *= 1 + FRAC*net per trade; daily returns over the FULL walk-forward test span
including zero-trade days; trade-level max drawdown; 30-day month buckets from d0=202;
portfolio aligned day-from-end because the datasets end on different days). Reproducing
the PUBLISHED numbers from the frozen side is the validation of this whole chain — if
the frozen column does not match ECONOMICS_4sym_20260717.json, nothing in the strict
column may be quoted.

Env: CELLS (comma-separated gs keys written by strictfill_cells.py), OUT.
"""
import io, json, os

import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
CELLS = os.environ.get("CELLS", "").split(",")
OUT = os.environ.get("OUT", "research_runs/strictfill_cells/ECONOMICS_strict.json")
# deployed sizing convention of README 4.1 (see the section's own caveat: these are a
# unit convention, not a statement about what an account can carry)
FRAC = {"DOGE": 0.5, "XRP": 0.5, "BTC": 0.5, "ETH": 1.25}
FRAC_REF = 0.5

bk = storage.Client(project=PROJ).bucket(BUCKET)


def metrics(trades, ndays_total, frac=1.0, label="", d0=None, d1=None, buckets=None):
    """economics_3sym.metrics, unchanged."""
    if not trades:
        return None
    trades = sorted(trades, key=lambda t: t[0])
    days = np.array([t[0] for t in trades]); nets = np.array([t[1] for t in trades]) * 1e-4
    if d0 is None:
        d0 = 202
    if d1 is None:
        d1 = ndays_total - 1
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
    g_daily = (1.0 + total) ** (1.0 / span)
    res = dict(label=label, n_trades=len(trades), span_days=span, tpd=len(trades) / span,
               ev_bp=float(np.array([t[1] for t in trades]).mean()),
               hit=float((nets > 0).mean()), total_return=total,
               monthly_roi=g_daily ** 30.44 - 1.0, annual_roi=g_daily ** 365.0 - 1.0,
               sharpe_daily_ann=float(sharpe), max_dd=mdd,
               best_day=float(dr.max()), worst_day=float(dr.min()),
               zero_day_share=float((dr == 0).mean()))
    if buckets:
        res["monthly_table"] = []
        for (b0, b1) in buckets:
            sel = (days >= b0) & (days <= b1)
            e = 1.0
            for r in nets[sel]:
                e *= (1.0 + frac * r)
            res["monthly_table"].append(dict(d0=int(b0), d1=int(b1), n=int(sel.sum()), ret=e - 1.0))
    print(f"[{label}] n={res['n_trades']} span={span}d tpd={res['tpd']:.2f} EV={res['ev_bp']:+.2f}bp "
          f"hit={100*res['hit']:.1f}% | monthly={100*res['monthly_roi']:+.2f}% "
          f"annual={100*res['annual_roi']:+.1f}% | Sharpe={sharpe:.2f} maxDD={100*mdd:.2f}% "
          f"worst={100*dr.min():+.2f}%", flush=True)
    return res


cells = {}
for key in [c for c in CELLS if c]:
    r = json.loads(bk.blob(key).download_as_bytes())
    cells[r["sym"]] = r
    print(f"[loaded] {key}: {r['sym']} {r['policy']} t{int(r['K'])} "
          f"harmony={r['harmony']} nseed={r['nseed']}", flush=True)

out = {"source_cells": {s: dict(policy=c["policy"], K=c["K"], nseed=c["nseed"],
                                harmony=c["harmony"], score_sub=c["score_sub"])
                        for s, c in cells.items()}}
ND = {}
for sym, c in cells.items():
    # n_days of the dataset = max test day + 1 is a lower bound; the published economics
    # used the dataset day count, which is the strict npz meta n_days
    ND[sym] = int(json.loads(str(np.load(io.BytesIO(
        bk.blob(f"{c['strict_sub']}/{sym}.npz").download_as_bytes()),
        allow_pickle=True)["meta"]))["n_days"])
    print(f"  {sym}: n_days={ND[sym]}", flush=True)

series = {"frozen": {}, "strict": {}}
for sym, c in cells.items():
    nd = ND[sym]
    buckets = [(202 + k * 30, min(202 + (k + 1) * 30 - 1, nd - 1)) for k in range((nd - 202) // 30 + 1)]
    for tag in ("frozen", "strict"):
        t = c["trades"].get(f"ens_{tag}")
        tr = sorted(zip([int(d) for d in t["day"]], [float(x) for x in t["net"]])) if t else []
        series[tag][sym] = tr
        print(f"--- {sym} {tag} ---", flush=True)
        out[f"{sym}_{tag}_deployed"] = metrics(tr, nd, FRAC[sym], f"{sym} {tag} @{FRAC[sym]}", buckets=buckets)
        if FRAC[sym] != FRAC_REF:
            out[f"{sym}_{tag}_ref05"] = metrics(tr, nd, FRAC_REF, f"{sym} {tag} @0.5 ref", buckets=buckets)

# portfolio, day-from-end aligned (datasets end on different days)
for tag in ("frozen", "strict"):
    for pname, syms, fr in (("PORT_deployed", [s for s in series[tag] if s != "BTC"], FRAC),
                            ("PORT_all05", list(series[tag]), {s: FRAC_REF for s in series[tag]})):
        port = []
        for sym in syms:
            off = ND[sym]
            port += [(d - off, n * fr[sym]) for d, n in series[tag][sym]]
        if not port:
            continue
        pd0 = min(202 - ND[s] for s in syms); pd1 = -1
        out[f"{pname}_{tag}"] = metrics(sorted(port), 365, 1.0, f"{pname} {tag}", d0=pd0, d1=pd1)

# daily-PnL correlation matrix per fill model
for tag in ("frozen", "strict"):
    dd = {}
    for sym in series[tag]:
        off = ND[sym]; m = {}
        for d, n in series[tag][sym]:
            m[d - off] = m.get(d - off, 0.0) + n
        dd[sym] = m
    corr = {}
    syms = list(dd)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a_, b_ = syms[i], syms[j]
            ks = [k for k in sorted(set(dd[a_]) | set(dd[b_]))
                  if k >= max(min(dd[a_], default=0), min(dd[b_], default=0))]
            if len(ks) > 3:
                va = np.array([dd[a_].get(k, 0.0) for k in ks])
                vb = np.array([dd[b_].get(k, 0.0) for k in ks])
                corr[f"{a_}_{b_}"] = float(np.corrcoef(va, vb)[0, 1])
    out[f"daily_pnl_corr_{tag}"] = corr
    print(f"daily PnL corr ({tag}):", {k: round(v, 3) for k, v in corr.items()}, flush=True)

bk.blob(OUT).upload_from_string(json.dumps(out, default=float))
print(f"[saved] gs://{BUCKET}/{OUT}", flush=True)
