#!/usr/bin/env python3
"""Entry-POLICY sweep over the persisted maker arrays (ETH_gated_arrays.npz).
Reimplements simulate_maker_entry physics in numpy + adaptive policies, all offline.
Entry & exit at the near touch (bid/ask) -> spread captured; commission separate.
Policies: maker(offset×queue×window) | maker->taker fallback(X) | cancel-on-toxic(thr)
| pure taker. Each at conviction sub-gates (by A-logit). EV/trade over FILLED + WR + fill.
Done = all policy types swept + results saved to GCS.
"""
import io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
OUTP = "research_runs/gru_makergrid"; TO = 563                 # 60s @ ~106ms/tick
TIERS = {"mm": 4.0, "mt": 7.0, "tt": 10.0}                     # RT bp (maker-maker/maker-taker/taker-taker)
bk = storage.Client(project=PROJ).bucket(BUCKET)

z = np.load(io.BytesIO(bk.blob(f"{OUTP}/ETH_gated_arrays.npz").download_as_bytes()))
side = z["side"].astype(bool)                                  # True=long
alog = z["alog"].astype(np.float64)
EL = z["entry_long"].astype(np.float64); ES = z["entry_short"].astype(np.float64)
BID = z["book_paths"][:, :, 0].astype(np.float64); ASK = z["book_paths"][:, :, 1].astype(np.float64)
SELL = z["flow_paths"][:, :, 1].astype(np.float64); BUY = z["flow_paths"][:, :, 0].astype(np.float64)
EQ = z["entry_q"].astype(np.float64)
N, H = BID.shape; AR = np.arange(N)
print(f"[loaded] N={N} H={H} long_frac={side.mean():.2f}")


def first_true(mask, window):
    m = mask.copy()
    if window + 1 < m.shape[1]:
        m[:, window + 1:] = False
    has = m.any(1)
    return np.where(has, m.argmax(1), -1)


def maker_fill(is_long, off_bp, qm, window, toxic_thr=None):
    """Resting-limit fill (touch/queue/gap/MISS) for ALL samples under side=is_long."""
    if is_long:
        lvl = EL * (1 - off_bp / 1e4); nb = BID; cf = SELL; q = EQ[:, 0] * qm
        eps = lvl * 1e-7
        gap = nb < lvl[:, None] - eps[:, None]
        atl = (nb <= lvl[:, None] + eps[:, None]) & (nb >= lvl[:, None] - eps[:, None])
    else:
        lvl = ES * (1 + off_bp / 1e4); nb = ASK; cf = BUY; q = EQ[:, 1] * qm
        eps = lvl * 1e-7
        gap = nb > lvl[:, None] + eps[:, None]
        atl = (nb >= lvl[:, None] - eps[:, None]) & (nb <= lvl[:, None] + eps[:, None])
    consumed = np.cumsum(cf * atl, 1)
    fillmask = gap | (atl & (consumed >= q[:, None]))
    ft = first_true(fillmask, window)
    filled = ft >= 0
    if toxic_thr is not None:                                  # cancel if fill driven by big adverse taker burst
        ftk = np.clip(ft, 0, H - 1)
        big = cf[AR, ftk] > toxic_thr
        filled = filled & ~big
    return filled, ft, lvl                                     # fill_px = lvl


def combine(fl, fs):                                           # select predicted-side result
    filled = np.where(side, fl[0], fs[0])
    ft = np.where(side, fl[1], fs[1])
    fpx = np.where(side, fl[2], fs[2])
    return filled, ft, fpx


def pnl_bp(filled, ft, fpx):
    """Exit as a TAKER at the near touch at ft+TO (long->sell at bid, short->buy at ask)."""
    et = np.clip(ft + TO, 0, H - 1)
    exitpx = np.where(side, BID[AR, et], ASK[AR, et])
    p = np.where(side, (exitpx - fpx) / fpx, (fpx - exitpx) / fpx) * 1e4
    return p, filled & np.isfinite(p) & (ft >= 0)


def report(p, ok, label, fee_tier):
    out = {"label": label, "fee_tier": fee_tier, "cuts": {}}
    for cut, qpc in (("all", 0.0), ("top0.2pct", 80.0), ("top0.05pct", 95.0)):  # of the top-1% superset by A
        thr = np.quantile(alog, qpc / 100.0)
        m = ok & (alog >= thr)
        if m.sum() < 30:
            continue
        g = float(p[m].mean()); wr = float((p[m] > 0).mean()); fr = float(m.sum() / max((alog >= thr).sum(), 1))
        out["cuts"][cut] = {"gross_bp": g, "wr": wr, "fill_rate": fr, "n": int(m.sum()),
                            "net_bp": {t: g - TIERS[t] for t in TIERS}}
    c = out["cuts"].get("all", {})
    print(f"  {label:<34} [{fee_tier}] all: gross={c.get('gross_bp',float('nan')):+6.2f}bp "
          f"WR={c.get('wr',0):.2f} fill={c.get('fill_rate',0):.2f} n={c.get('n',0)} | "
          f"net_{fee_tier}={c.get('net_bp',{}).get(fee_tier,float('nan')):+6.2f}")
    return out


results = {"symbol": "ETH-USDT-PERP", "N": int(N), "TO_ticks": TO, "policies": []}

# --- E: pure taker (immediate cross) — entry & exit both taker => taker-taker ---
ft0 = np.zeros(N, int); fpx_t = np.where(side, ASK[:, 0], BID[:, 0])
p, ok = pnl_bp(np.ones(N, bool), ft0, fpx_t)
results["policies"].append(report(p, ok, "taker_immediate", "tt"))

# --- A: pure maker, offset × queue × window (cancel-after-X = window knob) ---
for off in (0.0, 1.0, 2.0):
    for qm in (0.0, 1.0):
        for win in (60, 120):
            f = combine(maker_fill(True, off, qm, win), maker_fill(False, off, qm, win))
            p, ok = pnl_bp(*f)
            results["policies"].append(report(p, ok, f"maker_off{off}_q{qm}_w{win}", "mt"))

# --- B: maker -> taker fallback after X ticks (unfilled by X => cross at X) ---
for X in (30, 60, 120):
    fl = list(maker_fill(True, 0.0, 0.0, X)); fs = list(maker_fill(False, 0.0, 0.0, X))
    # long fallback: unfilled -> buy ask[X]; short: -> sell bid[X]
    miss_l = ~fl[0]; fl[1] = np.where(miss_l, X, fl[1]); fl[2] = np.where(miss_l, ASK[:, X], fl[2]); fl[0] = np.ones(N, bool)
    miss_s = ~fs[0]; fs[1] = np.where(miss_s, X, fs[1]); fs[2] = np.where(miss_s, BID[:, X], fs[2]); fs[0] = np.ones(N, bool)
    p, ok = pnl_bp(*combine(tuple(fl), tuple(fs)))
    results["policies"].append(report(p, ok, f"maker_then_taker_X{X}", "mt"))

# --- D: cancel-on-toxic-flow (latency-sensitive CEILING; cancel adverse-burst fills) ---
thr_sell = np.quantile(SELL[SELL > 0], [0.90, 0.99]); thr_buy = np.quantile(BUY[BUY > 0], [0.90, 0.99])
for qi, qn in ((0, "p90"), (1, "p99")):
    f = combine(maker_fill(True, 0.0, 0.0, 120, toxic_thr=thr_sell[qi]),
                maker_fill(False, 0.0, 0.0, 120, toxic_thr=thr_buy[qi]))
    p, ok = pnl_bp(*f)
    results["policies"].append(report(p, ok, f"maker_cancel_toxic_{qn}", "mt(+latency)"))

bk.blob(f"{OUTP}/ETH_policy_sweep.json").upload_from_string(json.dumps(results, default=float))
print(f"\n[saved] {OUTP}/ETH_policy_sweep.json  ({len(results['policies'])} policies)")
