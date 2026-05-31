#!/usr/bin/env python3
"""Stability of the FINE-TUNED ETH B2 model: is the executed-hold net real edge or
half1(near-train)-loaded? Inference with B2_ETH_hold.best.pt on ETH test days ->
realized hold = sign(logit)*rH60, day-clustered t-stat + day-block bootstrap +
half-split (early=near-train vs late=near-today)."""
import io, sys
import numpy as np
import torch
sys.path.insert(0, "/tmp")
import mamba2_cascade as mc
from subs60_gru_gridsim import predict          # reuse batched LOB forward
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
CACHE = "hd2_sub60_cache"; SYM = "ETH-USDT-PERP"; SYM_ID = 1
MM = 4.0; MODEL = "/tmp/B2_ETH_hold.best.pt"
bk = storage.Client(project=PROJ).bucket(BUCKET)
rng = np.random.default_rng(0)

st = torch.load(MODEL, map_location="cpu"); F = st["F"]; cfg = st["cfg"]; sd = st["model"]
n_sym = sd["sym.weight"].shape[0] if "sym.weight" in sd else 0
model = mc.Cascade2Stream(F, cfg["cell"], cfg["d1"], cfg["n1"], cfg["d2"], cfg["n2"],
                          n_sym=n_sym, dropout=cfg.get("dropout", 0.1)); model.load_state_dict(sd); model.eval()
mu = [torch.tensor(st[k]) for k in ("lob_mu", "lob_sd", "ft_mu", "ft_sd")]
L = cfg["L"]; warmup = cfg["warmup"]; step = cfg["dec_stride_s"]

cb = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{CACHE}/{SYM}/") if b.name.endswith(".npz"))
nd = len(cb); te = cb[int(nd * 0.68):]                    # purged test = newest 32%
print(f"[{SYM}] {nd} days, test={len(te)} (newest 32%); model best_ep={st.get('best_ep')}")
P, R, DAY = [], [], []
for di, nm in enumerate(te):
    z = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
    lob = z["lob"].astype(np.float32); t0 = z["t0"].astype(np.int64); feat = z["feat"].astype(np.float32)
    rH = z["rH60"].astype(np.float64); y = z["y60"].astype(bool); v = z["v60"].astype(bool)
    keep = np.zeros(len(t0), bool); keep[::step] = True
    ctx = t0 - (t0 // L) * L
    keep &= (ctx >= min(warmup, L - 1)) & v & y           # match eval: stride/warmup/valid/non-flat
    dp = np.where(keep)[0]
    if len(dp) < 20:
        continue
    with torch.no_grad():
        lg = predict(model, cfg, mu[0], mu[1], mu[2], mu[3], lob, t0, feat, dp, SYM_ID)
    P.append(np.asarray(lg)); R.append(rH[dp]); DAY.append(np.full(len(dp), di, np.int64))
P = np.concatenate(P); R = np.concatenate(R); DAY = np.concatenate(DAY)
realized = np.sign(P) * R                                  # executed hold payoff (bp)


def stats(rl, day, tag):
    days = np.unique(day); per = [rl[day == d] for d in days]; nD = len(days)
    dmean = np.array([p.mean() for p in per])
    t = dmean.mean() / (dmean.std(ddof=1) / np.sqrt(nD)) if nD > 1 else 0.0
    boot = np.array([np.concatenate([per[i] for i in rng.integers(0, nD, nD)]).mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    h = nD // 2
    h1 = np.concatenate([per[i] for i in range(h)]).mean()
    h2 = np.concatenate([per[i] for i in range(h, nD)]).mean()
    g = rl.mean()
    print(f"  {tag:>10}: net_mm={g-MM:+6.2f} [CI {lo-MM:+6.2f},{hi-MM:+6.2f}] t_day={t:+5.2f} "
          f"nD={nD} n={len(rl)} | half1(near-train)={h1-MM:+6.2f} half2(near-today)={h2-MM:+6.2f}")


print("selectivity by |logit|  (net_mm bp, day-clustered):")
stats(realized, DAY, "all")
for q in (20.0, 10.0, 5.0, 2.0):
    k = max(20, int(len(P) * q / 100)); top = np.argsort(-np.abs(P))[:k]
    stats(realized[top], DAY[top], f"top{q}%")
