#!/usr/bin/env python3
"""3-symbol compounded economics — extension of the frozen economics.py methodology
(trade reconstruction == the ens cells; equity *= 1 + FRAC*net per trade; daily
returns over the full test-window span incl. zero-trade days; day-from-end calendar
alignment, all datasets end 2026-06-02). Adds: BTC (honest h150d cell, DYN t5 = the
deployed policy) and the DEPLOYED-policy variants for DOGE (FIXQ t10) / XRP (FIXQ t5),
plus the 3-way portfolio at the live sizing (each instance 0.5 x shared equity).
BTC caveat: ~9 raw-gap days late May 2026 shift its mid-window day indices by up to
that amount in the alignment; end-anchored alignment is exact at the tail."""
import io
import json
import numpy as np
from google.cloud import storage

bk = storage.Client(project='project-0998ac51-36ba-445c-bc7').bucket('market-data-0998ac51')
KDAYS = 30
CFG = {
    'DOGE': dict(sub='research_runs/maker_labels_tb3s_h150anch', ndays=371),
    'XRP': dict(sub='research_runs/maker_labels_tb3s_h150anch', ndays=365),
    'BTC': dict(sub='research_runs/maker_labels_tb3s_h150d', ndays=370),
}


def causal_trades(p, tgt=5.0):
    days = sorted(set(p['day_te'].tolist())); wpd = len(p['te']) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    trd = sorted(set(p['day_tr'].tolist())); seed = np.isin(p['day_tr'], trd[-KDAYS:])
    buf = list(p['tr'][seed]); cap = max(int(KDAYS * wpd), 1); out = []
    for d in days:
        idx = np.where(p['day_te'] == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel = idx[p['te'][idx] >= tau]
        for i in sel:
            s = p['side'][i]; net = p['nl'][i] if s else p['ns'][i]
            fc = p['fl'][i] if s else p['fs'][i]
            if fc and np.isfinite(net):
                out.append((int(d), float(net)))
        buf.extend(p['te'][idx].tolist()); buf = buf[-cap:]
    return out


def fixq_trades(p, K):
    """HD5 FIXQ: tau = val-window quantile matched to K/day, frozen per fold."""
    trd = sorted(set(p['day_tr'].tolist()))[-KDAYS:]
    m = np.isin(p['day_tr'], trd)
    s = p['tr'][m]; nd = len(trd)
    if not len(s):
        return []
    wpd = len(s) / max(nd, 1)
    q = max(0.0, 1.0 - K / max(wpd, 1.0))
    tau = float(np.quantile(s, q))
    out = []
    for i in np.where(p['te'] >= tau)[0]:
        sd_ = p['side'][i]; net = p['nl'][i] if sd_ else p['ns'][i]
        fc = p['fl'][i] if sd_ else p['fs'][i]
        if fc and np.isfinite(net):
            out.append((int(p['day_te'][i]), float(net)))
    return out


def load_folds(sym):
    sub = CFG[sym]['sub']
    nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f'{sub}/PERFOLD_S0_{sym}_qm0_f')
             if b.name.endswith('.npz'))
    folds = []
    for f in range(nf):
        zs = [np.load(io.BytesIO(bk.blob(f'{sub}/PERFOLD_S{s}_{sym}_qm0_f{f}.npz').download_as_bytes()))
              for s in range(4)]
        folds.append(dict(
            tr=np.mean([z['axb_tr'].astype(np.float64) for z in zs], 0),
            te=np.mean([z['axb_te'].astype(np.float64) for z in zs], 0),
            day_tr=zs[0]['day_tr'], day_te=zs[0]['day_te'],
            side=(np.sum([z['side'].astype(int) for z in zs], 0) >= 2),
            fl=zs[0]['fl'], fs=zs[0]['fs'],
            nl=zs[0]['netl'].astype(np.float64), ns=zs[0]['nets'].astype(np.float64)))
    return folds


def metrics(trades, ndays_total, frac=1.0, label='', d0=None, d1=None, perfold_days=None):
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
    sharpe = (mu / sd) * np.sqrt(365.0) if sd > 0 else float('nan')
    eqs = np.array([c[1] for c in curve]); peak = np.maximum.accumulate(eqs)
    mdd = float(((eqs - peak) / peak).min())
    g_daily = (1.0 + total) ** (1.0 / span)
    monthly = g_daily ** 30.44 - 1.0; annual = g_daily ** 365.0 - 1.0
    lin_monthly = float(np.array([t[1] for t in trades]).mean() * (len(trades) / span) * 30.44 * 1e-4 * frac)
    res = dict(label=label, n_trades=len(trades), span_days=span, tpd=len(trades) / span,
               ev_bp=float((np.array([t[1] for t in trades])).mean()),
               hit=float((nets > 0).mean()), total_return=total,
               monthly_roi=monthly, monthly_roi_linear=lin_monthly, annual_roi=annual,
               sharpe_daily_ann=float(sharpe),
               max_dd=mdd, best_day=float(dr.max()), worst_day=float(dr.min()),
               daily_vol_bp=float(sd * 1e4),
               zero_day_share=float((dr == 0).mean()))
    # month-bucket table (30d buckets from d0)
    if perfold_days:
        mt = []
        for (b0, b1) in perfold_days:
            sel = (days >= b0) & (days <= b1)
            if sel.any():
                e = 1.0
                for r in nets[sel]:
                    e *= (1.0 + frac * r)
                mt.append(dict(d0=int(b0), d1=int(b1), n=int(sel.sum()), ret=e - 1.0))
            else:
                mt.append(dict(d0=int(b0), d1=int(b1), n=0, ret=0.0))
        res['monthly_table'] = mt
    print(f"[{label}] n={res['n_trades']} span={span}d tpd={res['tpd']:.2f} EV={res['ev_bp']:+.2f}bp "
          f"hit={100*res['hit']:.1f}% | total={100*total:+.2f}% monthly={100*monthly:+.2f}% "
          f"annual={100*annual:+.1f}% | Sharpe={sharpe:.2f} maxDD={100*mdd:.2f}% "
          f"worst_day={100*dr.min():+.2f}% zero_days={100*res['zero_day_share']:.0f}%", flush=True)
    if perfold_days:
        print('   monthly: ' + ' | '.join(f"{100*m['ret']:+.2f}%({m['n']})" for m in res['monthly_table']), flush=True)
    return res


out = {}
folds = {s: load_folds(s) for s in CFG}
DEPLOYED = {'DOGE': ('FIXQ', 10.0), 'XRP': ('FIXQ', 5.0), 'BTC': ('DYN', 5.0)}
tr_dep = {}; tr_dyn = {}
for sym in CFG:
    nd = CFG[sym]['ndays']
    buckets = [(202 + k * 30, min(202 + (k + 1) * 30 - 1, nd - 1)) for k in range((nd - 202) // 30 + 1)]
    pol, K = DEPLOYED[sym]
    tr_dyn[sym] = sorted([t for p in folds[sym] for t in causal_trades(p, 5.0)])
    if pol == 'FIXQ':
        tr_dep[sym] = sorted([t for p in folds[sym] for t in fixq_trades(p, K)])
    else:
        tr_dep[sym] = tr_dyn[sym] if K == 5.0 else sorted([t for p in folds[sym] for t in causal_trades(p, K)])
    print(f'--- {sym} (deployed: {pol} t{K:.0f}) ---', flush=True)
    out[f'{sym}_dyn5_frac1.0'] = metrics(tr_dyn[sym], nd, 1.0, f'{sym} DYN-t5 100%', perfold_days=buckets)
    out[f'{sym}_dyn5_frac0.5'] = metrics(tr_dyn[sym], nd, 0.5, f'{sym} DYN-t5 50%', perfold_days=buckets)
    if pol == 'FIXQ':
        out[f'{sym}_deployed_frac0.5'] = metrics(tr_dep[sym], nd, 0.5, f'{sym} {pol}-t{K:.0f} 50% (deployed)', perfold_days=buckets)

# 3-way portfolio: day-from-end alignment (all datasets end 2026-06-02)
for tag, series in (('deployed', tr_dep), ('alldyn5', tr_dyn)):
    port = []
    for sym in CFG:
        off = CFG[sym]['ndays']
        port += [(d - off, n * 0.5) for d, n in series[sym]]
    pd0 = min(202 - CFG[s]['ndays'] for s in CFG); pd1 = -1
    out[f'PORT3_{tag}_050505'] = metrics(sorted(port), 365, 1.0,
                                         f'PORTFOLIO 3x0.5 ({tag})', d0=pd0, d1=pd1)

# pairwise daily PnL correlations (deployed policies, day-from-end overlap)
dd = {}
for sym in CFG:
    off = CFG[sym]['ndays']; m = {}
    for d, n in tr_dep[sym]:
        k = d - off; m[k] = m.get(k, 0.0) + n
    dd[sym] = m
corr = {}
syms = list(CFG)
for i in range(len(syms)):
    for j in range(i + 1, len(syms)):
        a_, b_ = syms[i], syms[j]
        ks = [k for k in sorted(set(dd[a_]) | set(dd[b_])) if k >= max(min(dd[a_], default=0), min(dd[b_], default=0))]
        if len(ks) > 3:
            va = np.array([dd[a_].get(k, 0.0) for k in ks]); vb = np.array([dd[b_].get(k, 0.0) for k in ks])
            corr[f'{a_}_{b_}'] = float(np.corrcoef(va, vb)[0, 1])
out['daily_pnl_corr_deployed'] = corr
print('daily PnL correlations (deployed):', {k: round(v, 3) for k, v in corr.items()}, flush=True)

bk.blob('research_runs/ECONOMICS_3sym_20260716.json').upload_from_string(json.dumps(out, default=float))
print('[ECONOMICS3 DONE]', flush=True)
