import io, os
import numpy as np
from google.cloud import storage
bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")
WPD = 28800

def load(sym):
    blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"research_runs/_recev_h150_{sym}/D_") if b.name.endswith(".npz"))
    days=[]
    for bn in blobs:
        z=np.load(io.BytesIO(bk.blob(bn).download_as_bytes()))
        net=np.where(z["side"].astype(bool),z["netl"].astype(np.float64),z["nets"].astype(np.float64))
        fc=np.where(z["side"].astype(bool),z["FL"].astype(bool),z["FS"].astype(bool))
        days.append(dict(day=bn.split("_")[-1][:-4], sc=z["score"].astype(np.float64), net=net, ok=fc&np.isfinite(net)))
    return days

def causal(days, tgt):
    q=1.0-tgt/WPD; buf=[]; perday={}; alltr=[]
    for d in days:
        if buf:
            tau=float(np.quantile(np.asarray(buf),q)); t=(d["sc"]>=tau)&d["ok"]
            pnl=d["net"][t]; perday[d["day"]]=pnl; alltr.append(pnl)
        else:
            perday[d["day"]]=np.array([])
        buf.extend(d["sc"].tolist())
    a=np.concatenate(alltr) if alltr else np.array([])
    return perday, a

res={}
for sym in ("DOGE","BTC"):
    days=load(sym); pd5,a5=causal(days,5); pd10,a10=causal(days,10)
    dbp={k:(v.sum() if len(v) else 0.0) for k,v in pd5.items()}  # daily summed bp at budget5
    ev=a5.mean() if len(a5) else float('nan')
    # daily account return series (notional=sub-deposit): sum(bp)/1e4 per day
    dser=np.array([dbp[k]/1e4 for k in sorted(dbp)])
    dsh=(dser.mean()/dser.std()*np.sqrt(365)) if dser.std()>0 else 0.0
    res[sym]=dict(ev5=ev, n5=len(a5), hit5=(100*(a5>0).mean() if len(a5) else 0), ev10=(a10.mean() if len(a10) else float('nan')),
                  dbp=dbp, dser=dser, dsh=dsh, days=sorted(dbp))
    print(f"{sym}: budget5 EV/tr {ev:+.2f}bp ({len(a5)}tr, {100*(a5>0).mean() if len(a5) else 0:.0f}% hit) | budget10 {res[sym]['ev10']:+.2f}bp | daily-Sharpe(ann) {dsh:+.2f} | daily bp {[round(dbp[k],0) for k in sorted(dbp)]}")

# portfolio: align shared days, 50/50 capital
common=sorted(set(res["DOGE"]["days"])&set(res["BTC"]["days"]))
dg=np.array([res["DOGE"]["dbp"][k]/1e4 for k in common]); bt=np.array([res["BTC"]["dbp"][k]/1e4 for k in common])
corr=np.corrcoef(dg,bt)[0,1] if len(dg)>2 else float('nan')
for w in ((0.5,0.5),(0.6,0.4),(0.4,0.6)):
    port=w[0]*dg+w[1]*bt
    psh=(port.mean()/port.std()*np.sqrt(365)) if port.std()>0 else 0.0
    ann=(np.prod(1+port)**(365/len(port))-1) if len(port) else 0.0
    print(f"portfolio DOGE:{w[0]} BTC:{w[1]} | daily-ret mean {100*port.mean():+.3f}% sd {100*port.std():.3f}% | Sharpe(ann) {psh:+.2f} | naive annual ROI {100*ann:+.0f}%")
print(f"DOGE<->BTC daily-return correlation: {corr:+.3f}  (over {len(common)} shared days)")
