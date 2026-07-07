import io, os
import numpy as np
from google.cloud import storage
bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")
SYM = os.environ.get("SYM","DOGE"); WPD_DEPLOY = 28800  # DECIDE_S=3 -> 86400/3
blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"research_runs/_recev_h150_{SYM}/D_") if b.name.endswith(".npz"))
days=[]
for bn in blobs:
    z=np.load(io.BytesIO(bk.blob(bn).download_as_bytes()))
    net=np.where(z["side"].astype(bool),z["netl"].astype(np.float64),z["nets"].astype(np.float64))
    fc=np.where(z["side"].astype(bool),z["FL"].astype(bool),z["FS"].astype(bool))
    days.append(dict(day=bn.split("_")[-1][:-4], sc=z["score"].astype(np.float64), net=net, ok=fc&np.isfinite(net)))
wpd=np.median([len(d["sc"]) for d in days])
print(f"{SYM}: {len(days)} days, ~{wpd:.0f} decisions/day (deploy WPD={WPD_DEPLOY})")
for tgt in (5,10,20):
    # CAUSAL rolling: tau for day i from buffer of days < i (day 0 = warmup, no trades). q uses deploy WPD.
    q=1.0-tgt/WPD_DEPLOY; buf=[]; taken=[]; perday=[]
    for d in days:
        if buf:
            tau=float(np.quantile(np.asarray(buf),q))
            sel=d["sc"]>=tau; t=sel&d["ok"]
            taken.append(d["net"][t]); perday.append(int(sel.sum()))
        else:
            perday.append(-1)  # warmup
        buf.extend(d["sc"].tolist())
    pnl=np.concatenate(taken) if taken else np.array([])
    ev=pnl.mean() if len(pnl) else float("nan"); n=len(pnl)
    nd_eval=len([p for p in perday if p>=0])
    print(f"  budget {tgt}/day CAUSAL: {n} trades over {nd_eval} eval-days ({n/max(nd_eval,1):.1f}/day) | "
          f"EV/tr {ev:+.2f}bp | hit {100*(pnl>0).mean() if n else 0:.1f}% | sum {pnl.sum() if n else 0:+.0f}bp | per-day sel {perday}")
