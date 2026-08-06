#!/usr/bin/env python3
"""Deploy-window ROI under the DEPLOYED execution policy and the HONEST fill model.

Per symbol, from the day its current configuration went live through the end of the
recorder window:
  * fills/PnL   = STRICT price-resolved model on the USDC book (OPS-EXEC rev16)
  * selection   = the deployed tau (frozen FIXQ, or dynamic causal warmed up on the days
                  BEFORE the deploy date — deployment-like, never an empty buffer)
  * execution   = exec v3 slot pool, MAX_CONC=50, one-way (opposite side skipped)

Two sizing scenarios, both applied to the SAME trade list (percentages only):
  A "showcase"  per-trade notional = FRAC_A x equity, FRAC from trading_algorithm
                README 4.1 (DOGE 0.5, XRP 0.5, ETH 1.25). BTC excluded per request.
  B "risky"     per-trade notional = 1.0 x equity for every symbol.
Daily returns are summed within a day and compounded ACROSS days (per-trade sequential
compounding is invalid under concurrency). Peak gross exposure is reported because these
conventions imply MULT x M of it and that is what makes them fundable or not.

Env: SYMS, MAXC(50), OUT(json path).
"""
import io, json, os
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; MKT = "market-data-0998ac51"; REC = "recorder-data-asia-0998ac51"
RB = "chronos/scalper-recorder/binance_futures"
W = 50; H = 6000; STEP_S = 3.0; NS = 1_000_000_000; KDAYS = 30
MAXC = int(os.environ.get("MAXC", "50"))
BUSY_FILL = 155.0; BUSY_NOFILL = 60.0
TD = os.environ.get("WORKDIR", "/home/delmi/roi_tmp"); os.makedirs(TD, exist_ok=True)

CFG = {   # symbol: (deploy day, tau or None, budget, score prefixes, showcase FRAC)
    "DOGE": ("20260715", 0.817010, 10, ["research_runs/_recev_dep_DOGE"], 0.5),
    "XRP":  ("20260715", 0.925442, 5,  ["research_runs/_recev_dep_XRP"], 0.5),
    "BTC":  ("20260716", None,     5,  ["research_runs/_recev_dep_BTC"], None),
    "ETH":  ("20260717", None,     5,  ["research_runs/_recev_dep_ETH"], 1.25),
}
SYMS = os.environ.get("SYMS", "DOGE,XRP,BTC,ETH").split(",")
cl = storage.Client(project=PROJ); mkt = cl.bucket(MKT); rec = cl.bucket(REC)


def usdt_slots(sym, day):
    ts = []
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{sym}USDT/depth_snapshot/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/{sym}x.parquet")
        t = pq.read_table(f"{TD}/{sym}x.parquet", columns=["exchange_event_ts_us"]).to_pandas()
        t = t[t["exchange_event_ts_us"].notna()]
        ts.append((t["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
    if not ts:
        return None
    bt = np.sort(np.concatenate(ts)); n = len(bt)
    if n < W + H + 100:
        return None
    mid0 = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()) * NS
    grid = np.arange(mid0, bt[-1], int(STEP_S * NS)); grid = grid[grid >= bt[0]]
    ends = np.unique(np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1))
    ends = ends[(ends >= W - 1) & (ends < n - H - 1)].astype(np.int64)
    if len(ends) < 50:
        return None
    return ((bt[ends] - mid0) // int(STEP_S * NS)).astype(np.int64)


def blob(prefix, day):
    try:
        return np.load(io.BytesIO(mkt.blob(f"{prefix}/D_{day}.npz").download_as_bytes()))
    except Exception:
        return None


def slots_of(zf, day):
    mid0 = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()) * 1000
    return (zf["sample_ts"].astype(np.int64) - mid0) // int(STEP_S * 1000)


def slot_sim(sel_idx, slot, side, fill, M):
    """exec v3: same-side stacking to M, opposite side skipped. slot = 3s grid index."""
    open_u = []; taken = []
    for i in sel_idx:
        t = float(slot[i]) * STEP_S
        open_u = [(tf, s2) for tf, s2 in open_u if tf > t]
        if any(s2 != side[i] for _, s2 in open_u):
            continue
        if len(open_u) >= M:
            continue
        taken.append(i)
        open_u.append((t + (BUSY_FILL if fill[i] else BUSY_NOFILL), side[i]))
    return np.array(taken, dtype=int)


results = {}
for sym in SYMS:
    dep, tau, budget, prefixes, frac_a = CFG[sym]
    fill_prefix = f"research_runs/_usdcfill_{sym}"
    days = sorted({b.name.split("/")[-1][2:10] for b in mkt.client.list_blobs(mkt, prefix=f"{fill_prefix}/D_")
                   if b.name.endswith(".npz")})
    per_day = []
    for day in days:
        zf = blob(fill_prefix, day)
        zs = None
        for p in prefixes:
            zs = blob(p, day)
            if zs is not None:
                break
        if zf is None or zs is None:
            continue
        su = usdt_slots(sym, day)
        if su is None or len(su) != len(zs["score"]):
            continue
        sf = slots_of(zf, day)
        common, iu, ic = np.intersect1d(su, sf, return_indices=True)
        if len(common) < 50:
            continue
        side = zs["side"][iu].astype(bool)
        per_day.append(dict(day=day, slot=common, score=zs["score"][iu].astype(np.float64), side=side,
                            # (c) strict price-resolved model on the USDC book
                            net=np.where(side, zf["netl_s"][ic], zf["nets_s"][ic]).astype(np.float64),
                            fill=np.where(side, zf["FL_s"][ic], zf["FS_s"][ic]).astype(bool),
                            # (a) the frozen model on the USDT book — what the cells used
                            net_a=np.where(side, zs["netl"][iu], zs["nets"][iu]).astype(np.float64),
                            fill_a=np.where(side, zs["FL"][iu], zs["FS"][iu]).astype(bool),
                            # (b) the frozen model on the USDC book — venue alone
                            net_b=np.where(side, zf["netl_f"][ic], zf["nets_f"][ic]).astype(np.float64),
                            fill_b=np.where(side, zf["FL_f"][ic], zf["FS_f"][ic]).astype(bool)))
    if not per_day:
        print(f"{sym}: no joined days"); continue
    warm = [d for d in per_day if d["day"] < dep]
    live = [d for d in per_day if d["day"] >= dep]
    if not live:
        print(f"{sym}: no days at/after deploy {dep}"); continue
    # selection
    if tau is not None:
        thr = {d["day"]: tau for d in live}
        pol = f"FIXQ tau={tau}"
    else:
        allsc = np.concatenate([d["score"] for d in per_day])
        wpd = len(allsc) / len(per_day); q = max(0.0, 1.0 - budget / max(wpd, 1.0))
        cap = max(int(KDAYS * wpd), 1)
        buf = list(np.concatenate([d["score"] for d in warm])) if warm else []
        buf = buf[-cap:]
        thr = {}
        for d in live:
            thr[d["day"]] = float(np.quantile(buf, q)) if buf else 0.0
            buf.extend(d["score"].tolist()); buf = buf[-cap:]
        pol = f"DYN budget={budget} (warmup {len(warm)}d pre-deploy)"
    daily_bp = {}; ntr = 0; peak = 0
    for d in live:
        sel = np.where(d["score"] >= thr[d["day"]])[0]
        tk = slot_sim(sel, d["slot"], d["side"], d["fill"], MAXC)
        ex = tk[d["fill"][tk] & np.isfinite(d["net"][tk])]
        if len(ex):
            t = d["slot"][ex] * STEP_S      # concurrent open trades at each entry instant
            peak = max(peak, max(int(np.sum((t <= x) & (t + BUSY_FILL > x))) for x in t))
        daily_bp[d["day"]] = float(d["net"][ex].sum()); ntr += len(ex)
    bp = np.array([daily_bp[d["day"]] for d in live])
    # same-window comparison of the three fill models (each gets its own slot sim,
    # because the busy duration depends on whether the entry filled)
    cells = {}
    for tag, nk, fk in (("a_frozen_USDT", "net_a", "fill_a"), ("b_frozen_USDC", "net_b", "fill_b"),
                        ("c_strict_USDC", "net", "fill")):
        s_bp = 0.0; s_n = 0
        for d in live:
            sel = np.where(d["score"] >= thr[d["day"]])[0]
            tk = slot_sim(sel, d["slot"], d["side"], d[fk], MAXC)
            ex = tk[d[fk][tk] & np.isfinite(d[nk][tk])]
            s_bp += float(d[nk][ex].sum()); s_n += len(ex)
        cells[tag] = dict(n=s_n, bpd=s_bp / len(live), ev=(s_bp / s_n) if s_n else float("nan"))
    results[sym] = dict(days=len(live), first=live[0]["day"], last=live[-1]["day"], trades=ntr,
                        pol=pol, bp=bp, by_day=daily_bp, frac_a=frac_a, peak=peak, cells=cells,
                        ev=float(np.nan if ntr == 0 else bp.sum() / ntr))
    print(f"{sym}: {len(live)}d {live[0]['day']}..{live[-1]['day']} | {pol} | trades {ntr} "
          f"| EV/tr {results[sym]['ev']:+.2f}bp | bpd {bp.mean():+.1f} | peak conc {peak}", flush=True)


def report(name, fracs, syms):
    print(f"\n===== SCENARIO {name} =====")
    print(f"{'sym':>5}{'days':>6}{'trades':>8}{'EV/tr':>9}{'frac':>7}{'ROI win':>10}{'ROI/mo':>10}{'worst d':>9}{'maxDD':>8}{'peak gross':>11}")
    # portfolio: align by CALENDAR DAY (deploy dates differ, so lengths differ)
    alldays = sorted({d for s in syms if s in results for d in results[s]["by_day"]})
    port = np.zeros(len(alldays))
    for s in syms:
        if s not in results:
            continue
        r = results[s]; f = fracs[s]
        port += np.array([r["by_day"].get(d, 0.0) for d in alldays]) * f / 1e4
        ret = r["bp"] * f / 1e4
        eq = np.cumprod(1 + ret); roi = 100 * (eq[-1] - 1)
        mo = 100 * (eq[-1] ** (30 / len(ret)) - 1)
        dd = 100 * (eq - np.maximum.accumulate(eq)).min() / max(np.maximum.accumulate(eq).max(), 1e-9)
        print(f"{s:>5}{r['days']:>6}{r['trades']:>8}{r['ev']:>+9.2f}{f:>7.2f}{roi:>+9.2f}%{mo:>+9.2f}%"
              f"{100*ret.min():>+8.2f}%{dd:>+7.2f}%{f*r['peak']:>10.1f}x")
    if len(syms) > 1:
        eq = np.cumprod(1 + port); roi = 100 * (eq[-1] - 1)
        mo = 100 * (eq[-1] ** (30 / len(port)) - 1)
        dd = 100 * (eq - np.maximum.accumulate(eq)).min() / max(np.maximum.accumulate(eq).max(), 1e-9)
        gross = sum(fracs[s] * results[s]["peak"] for s in syms if s in results)
        print(f"{'PORT':>5}{len(port):>6}{'':>8}{'':>9}{'':>7}{roi:>+9.2f}%{mo:>+9.2f}%"
              f"{100*port.min():>+8.2f}%{dd:>+7.2f}%{gross:>10.1f}x")

print("\n===== SAME-WINDOW FILL-MODEL COMPARISON (the legitimate read of this run) =====")
print("Not a validation cell: 12-14 days is statistically tiny. What IS comparable is the")
print("SAME decisions scored under three fill models over the SAME days.")
print(f"{'sym':>5}{'days':>6} | {'(a) frozen/USDT':>24} | {'(b) frozen/USDC':>24} | {'(c) STRICT/USDC':>24}")
print(f"{'':>5}{'':>6} | {'trades  EV/tr    bpd':>24} | {'trades  EV/tr    bpd':>24} | {'trades  EV/tr    bpd':>24}")
for s in ("DOGE", "XRP", "BTC", "ETH"):
    if s not in results:
        continue
    c = results[s]["cells"]
    row = f"{s:>5}{results[s]['days']:>6} |"
    for t in ("a_frozen_USDT", "b_frozen_USDC", "c_strict_USDC"):
        row += f" {c[t]['n']:>6}{c[t]['ev']:>+8.2f}{c[t]['bpd']:>+8.1f} |"
    print(row)

A_SYMS = [s for s in ("DOGE", "XRP", "ETH") if s in results]
report("A — showcase sizing (README 4.1), no BTC", {"DOGE": 0.5, "XRP": 0.5, "ETH": 1.25}, A_SYMS)
B_SYMS = [s for s in ("DOGE", "XRP", "BTC", "ETH") if s in results]
report("B — risky: 100% capital per symbol", {s: 1.0 for s in B_SYMS}, B_SYMS)
