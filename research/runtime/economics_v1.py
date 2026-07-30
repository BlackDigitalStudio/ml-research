#!/usr/bin/env python3
"""Compounded economics of the deployed year ensemble cells (DOGE, XRP) + the
2-instance portfolio. Trade sequences reconstructed EXACTLY as the ens cells
(mean 4-seed rank score, majority-vote side, causal t5, KDAYS=30) but returning
(day, net_bp) per trade. Sizing model: notional = FRAC x current equity, reinvested
after every trade (equity *= 1 + FRAC*net). Calendar alignment across symbols by
day-from-end (both datasets end 2026-06-02). Outputs JSON + prints."""
import io
import json
import numpy as np
from google.cloud import storage

bk = storage.Client(project='project-0998ac51-36ba-445c-bc7').bucket('market-data-0998ac51')
SUB = 'research_runs/maker_labels_tb3s_h150anch'
KDAYS = 30
NDAYS = {'DOGE': 371, 'XRP': 365}


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


def load_trades(sym):
    nf = sum(1 for b in bk.client.list_blobs(bk, prefix=f'{SUB}/PERFOLD_S0_{sym}_qm0_f')
             if b.name.endswith('.npz'))
    trades = []
    for f in range(nf):
        zs = [np.load(io.BytesIO(bk.blob(f'{SUB}/PERFOLD_S{s}_{sym}_qm0_f{f}.npz').download_as_bytes()))
              for s in range(4)]
        p = dict(tr=np.mean([z['axb_tr'].astype(np.float64) for z in zs], 0),
                 te=np.mean([z['axb_te'].astype(np.float64) for z in zs], 0),
                 day_tr=zs[0]['day_tr'], day_te=zs[0]['day_te'],
                 side=(np.sum([z['side'].astype(int) for z in zs], 0) >= 2),
                 fl=zs[0]['fl'], fs=zs[0]['fs'],
                 nl=zs[0]['netl'].astype(np.float64), ns=zs[0]['nets'].astype(np.float64))
        trades += causal_trades(p)
    trades.sort(key=lambda t: t[0])
    return trades


def metrics(trades, ndays_total, frac=1.0, label='', d0=None, d1=None):
    # d0/d1 = the FULL walk-forward test-window calendar bounds (not trade span!) —
    # sitting in the market on no-trade days counts. Default: fold start W+EMB=202.
    if not trades:
        return None
    days = np.array([t[0] for t in trades]); nets = np.array([t[1] for t in trades]) * 1e-4
    if d0 is None:
        d0 = 202
    if d1 is None:
        d1 = ndays_total - 1
    span = d1 - d0 + 1
    # equity curve, reinvest each trade at FRAC of current equity
    eq = 1.0; curve = []
    for d, r in zip(days, nets):
        eq *= (1.0 + frac * r); curve.append((d, eq))
    total = eq - 1.0
    # daily returns over the covered span (0 on no-trade days)
    daily = {}
    for d, r in zip(days, nets):
        daily[d] = daily.get(d, 1.0) * (1.0 + frac * r)
    dr = np.array([daily.get(d, 1.0) - 1.0 for d in range(d0, d1 + 1)])
    mu, sd = dr.mean(), dr.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(365.0) if sd > 0 else float('nan')
    # max drawdown on the trade-level curve
    eqs = np.array([c[1] for c in curve]); peak = np.maximum.accumulate(eqs)
    mdd = float(((eqs - peak) / peak).min())
    # geometric monthly / annual
    g_daily = (1.0 + total) ** (1.0 / span)
    monthly = g_daily ** 30.44 - 1.0; annual = g_daily ** 365.0 - 1.0
    lin_monthly = float(np.array([t[1] for t in trades]).mean() * (len(trades) / span) * 30.44 * 1e-4 * frac)
    res = dict(label=label, n_trades=len(trades), span_days=span, tpd=len(trades) / span,
               ev_bp=float((np.array([t[1] for t in trades])).mean()),
               hit=float((nets > 0).mean()), total_return=total,
               monthly_roi=monthly, monthly_roi_linear=lin_monthly, annual_roi=annual,
               sharpe_daily_ann=float(sharpe),
               max_dd=mdd, best_day=float(dr.max()), worst_day=float(dr.min()),
               daily_vol_bp=float(sd * 1e4))
    print(f"[{label}] n={res['n_trades']} span={span}d tpd={res['tpd']:.2f} EV={res['ev_bp']:+.2f}bp "
          f"hit={100*res['hit']:.1f}% | total={100*total:+.2f}% monthly={100*monthly:+.2f}% "
          f"(linear {100*lin_monthly:+.2f}%) annual={100*annual:+.1f}% | Sharpe={sharpe:.2f} "
          f"maxDD={100*mdd:.2f}% worst_day={100*dr.min():+.2f}%", flush=True)
    return res


out = {}
tr = {}
for sym in ('DOGE', 'XRP'):
    tr[sym] = load_trades(sym)
    print(f'{sym}: {len(tr[sym])} trades reconstructed', flush=True)
    out[sym + '_frac1.0'] = metrics(tr[sym], NDAYS[sym], 1.0, f'{sym} 100% capital')
    out[sym + '_frac0.5'] = metrics(tr[sym], NDAYS[sym], 0.5, f'{sym} 50% capital (as deployed)')

# portfolio: align by day-from-end (both end 2026-06-02); each instance trades 100% equity
port = []
for sym in ('DOGE', 'XRP'):
    off = NDAYS[sym]
    port += [(d - off, n) for d, n in tr[sym]]
port.sort(key=lambda t: t[0])
# portfolio window = union of test windows in from-end coords: DOGE 202-371 -> [-169,-1]
pd0 = min(202 - NDAYS[s] for s in NDAYS); pd1 = -1
out['PORT_100_100'] = metrics(port, 365, 1.0, 'PORTFOLIO DOGE100%+XRP100%', d0=pd0, d1=pd1)
port5 = [(d, n * 0.5) for d, n in port]
out['PORT_50_50'] = metrics(port5, 365, 1.0, 'PORTFOLIO 50%+50% (as deployed)', d0=pd0, d1=pd1)
out['trades'] = {s: tr[s] for s in tr}   # capture-everything: raw (day, net_bp) sequences

# daily correlation between symbols on overlapping day-from-end range
dd = {}
for sym in ('DOGE', 'XRP'):
    off = NDAYS[sym]; m = {}
    for d, n in tr[sym]:
        k = d - off; m[k] = m.get(k, 0.0) + n
    dd[sym] = m
ks = sorted(set(dd['DOGE']) | set(dd['XRP']))
ks = [k for k in ks if k >= max(min(dd['DOGE']), min(dd['XRP']))]
a = np.array([dd['DOGE'].get(k, 0.0) for k in ks]); b = np.array([dd['XRP'].get(k, 0.0) for k in ks])
corr = float(np.corrcoef(a, b)[0, 1]) if len(ks) > 3 else float('nan')
out['daily_pnl_corr'] = corr
print(f'daily PnL correlation DOGE vs XRP (overlap {len(ks)}d): {corr:+.3f}', flush=True)

bk.blob(f'{SUB}/ECONOMICS_year_cells.json').upload_from_string(json.dumps(out, default=float))
print('[ECONOMICS DONE]', flush=True)
