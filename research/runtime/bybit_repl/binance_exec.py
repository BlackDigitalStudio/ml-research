#!/usr/bin/env python3
"""HBV1 rev12: conservative tape-only maker execution sim — Binance lower bound.

Two legs:
  validate  — run the tape-sim on BYBIT trades for the champion decisions and
              compare vs the frozen full-book labels (bias where truth is known).
  binance   — same sim on binance.vision DOGEUSDC aggTrades (0-fee venue).

Sim (pessimistic by construction): quotes inferred from aggressor prints
(sell-aggressor print at p -> bid<=p known exactly at p; buy-aggressor at p ->
ask=p). Entry: peg at inferred touch, 60s; FILL ONLY when a print goes STRICTLY
through the level (bid-side: print < level). Hold 150s from fill. Exit: peg at
inferred opposite touch, 300s chase, same strict rule; ran-out marked at touch.

Decisions: recomputed union selection (members/T_s via env) exactly as
full_audit; ts mapped through the anch dataset's fold structure (W200/T30/EMB2).
Env: MODE=validate|binance, MEMBERS, TGT (default 0.3125), LOCAL_GCS_ROOT.
"""
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
from google.cloud import storage

bk = storage.Client(project="x").bucket("market-data-0998ac51")
SYM = "DOGE"
KDAYS = 30
W, T, EMB = 200, 30, 2
TICK = 1e-5
MODE = os.environ.get("MODE", "validate")
TGT = float(os.environ.get("TGT", "0.3125"))
MEMBERS = [(("research_runs/" + p.split(":")[0]), int(p.split(":")[1]))
           for p in os.environ["MEMBERS"].split(",")]
ENTRY_S, HOLD_S, CHASE_S = 60.0, 150.0, 300.0
NS = 1_000_000_000


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


# ---- decisions: union selection + ts/day/side/frozen-net per selected decision
print("[loading dataset ts/day]", flush=True)
d = np.load(io.BytesIO(bk.blob("research_runs/maker_labels_tb3s_h150anch/DOGE.npz").download_as_bytes()),
            allow_pickle=True)
ts_all = d["ts"].astype(np.int64); day_all = d["day"].astype(int)
ndays = int(json.loads(str(d["meta"]))["n_days"])
folds = []
t0 = W + EMB
while t0 < ndays:
    te_end = min(t0 + T, ndays)
    tst = (day_all >= t0) & (day_all < te_end)
    trn = (day_all >= t0 - EMB - W) & (day_all < t0 - EMB)
    if tst.sum() >= 50 and trn.sum() >= 5000:
        folds.append(np.where(tst)[0])
    t0 += T
nf = len(folds)
print(f"[{nf} folds reconstructed]", flush=True)

Z = {m: [np.load(io.BytesIO(bk.blob(f"{m[0]}/PERFOLD_S{m[1]}_{SYM}_qm0_f{f}.npz").download_as_bytes()))
         for f in range(nf)] for m in MEMBERS}
for f in range(nf):
    assert len(Z[MEMBERS[0]][f]["axb_te"]) == len(folds[f]), f"fold {f} len mismatch"

decisions = []  # (ts_ns, day, side_long, frozen_net)
for f in range(nf):
    sets = {m: set(causal_sel(Z[m][f]["axb_tr"].astype(np.float64), Z[m][f]["axb_te"].astype(np.float64),
                              Z[m][f]["day_tr"], Z[m][f]["day_te"], TGT).tolist()) for m in MEMBERS}
    z0 = Z[MEMBERS[0]][f]
    for i in sorted(set().union(*sets.values())):
        ks = [m for m in MEMBERS if i in sets[m]]
        sides = [bool(Z[m][f]["side"][i]) for m in ks]
        nl_ = sum(sides)
        if nl_ * 2 == len(sides):
            continue
        side = nl_ * 2 > len(sides)
        net = float(z0["netl"][i]) if side else float(z0["nets"][i])
        fill = bool(z0["fl"][i]) if side else bool(z0["fs"][i])
        gidx = folds[f][i]
        decisions.append((int(ts_all[gidx]), int(day_all[gidx]), side,
                          net if (fill and np.isfinite(net)) else np.nan))
print(f"[{len(decisions)} union decisions @T{TGT:g}]", flush=True)

# map day index -> date string via the daily npz listing (sorted, same order as combine)
daily = sorted(b.name for b in bk.client.list_blobs(bk, prefix="research_runs/maker_labels_tb3s_h150/daily/DOGE_")
               if b.name.endswith(".npz"))
dates = [n.split("_")[-1][:-4] for n in daily]


def bybit_trades(date):
    t = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
        f"/vol/gcs/market-data-0998ac51/raw/trades/exchange=BINANCE_FUTURES/symbol=DOGE-USDT-PERP/dt={date}/1.snappy.parquet")
    ts = np.asarray(t["timestamp"], np.int64)
    px = np.asarray(t["price"], np.float64)
    is_sell = np.array([s == "sell" for s in np.asarray(t["side"])])  # taker sold -> print at bid
    return ts, px, is_sell


def binance_aggtrades(date):
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/DOGEUSDC/DOGEUSDC-aggTrades-{date}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    import csv
    with zipfile.ZipFile(io.BytesIO(raw)) as z, z.open(z.namelist()[0]) as fobj:
        rd = csv.reader(io.TextIOWrapper(fobj))
        first = next(rd)
        rows = [] if first and first[0][:1].isdigit() else None
        if rows is None:
            rows = []
        else:
            rows.append(first)
        for r_ in rd:
            rows.append(r_)
    # agg_trade_id, price, quantity, first_id, last_id, transact_time(ms), is_buyer_maker
    px = np.array([float(r_[1]) for r_ in rows])
    ts = np.array([int(r_[5]) for r_ in rows], np.int64) * 1_000_000
    is_sell = np.array([r_[6].strip().lower() == "true" for r_ in rows])  # buyer_maker -> sell aggressor
    return ts, px, is_sell


def sim_day(dec_rows, ts, px, is_sell):
    """Conservative tape sim for one day's decisions. Returns list of (net_bp or nan)."""
    out = []
    n = len(ts)
    for dts, side in dec_rows:
        i0 = np.searchsorted(ts, dts, "left")
        if i0 >= n or i0 == 0:
            out.append(np.nan); continue
        # inferred touch at decision: last sell-aggressor print = bid, last buy = ask
        j = i0 - 1
        bid = ask = np.nan
        for k in range(j, max(-1, j - 500), -1):
            if is_sell[k] and np.isnan(bid):
                bid = px[k]
            elif not is_sell[k] and np.isnan(ask):
                ask = px[k]
            if not np.isnan(bid) and not np.isnan(ask):
                break
        if np.isnan(bid) and not np.isnan(ask):
            bid = ask - TICK
        if np.isnan(ask) and not np.isnan(bid):
            ask = bid + TICK
        if np.isnan(bid):
            out.append(np.nan); continue
        if ask <= bid:
            ask = bid + TICK
        # ENTRY: peg at touch on our side, strict-through fill within ENTRY_S
        lvl = bid if side else ask
        end_e = dts + int(ENTRY_S * NS)
        i = i0; filled = -1
        while i < n and ts[i] <= end_e:
            p = px[i]
            if side:
                if is_sell[i] and p < lvl - 1e-12:
                    filled = i; break
                if is_sell[i]:
                    lvl = min(lvl, p)        # touch moved down -> repeg (peg-to-touch)
                elif not is_sell[i] and p > lvl:
                    pass
            else:
                if (not is_sell[i]) and p > lvl + 1e-12:
                    filled = i; break
                if not is_sell[i]:
                    lvl = max(lvl, p)
            i += 1
        if filled < 0:
            out.append(np.nan); continue   # no maker fill (frozen sim also marks unfilled)
        entry = lvl
        # EXIT: from fill+HOLD, peg opposite touch, strict-through within CHASE_S
        t_exit0 = ts[filled] + int(HOLD_S * NS)
        ix = np.searchsorted(ts, t_exit0, "left")
        end_x = t_exit0 + int(CHASE_S * NS)
        # opposite touch estimate at exit start
        xlvl = np.nan
        for k in range(min(ix, n) - 1, max(-1, ix - 500), -1):
            if side and (not is_sell[k]):
                xlvl = px[k]; break
            if (not side) and is_sell[k]:
                xlvl = px[k]; break
        if np.isnan(xlvl):
            xlvl = entry
        i = ix; xfill = -1
        last_opp = xlvl
        while i < n and ts[i] <= end_x:
            p = px[i]
            if side:
                if (not is_sell[i]) and p > xlvl + 1e-12:
                    xfill = i; break
                if not is_sell[i]:
                    xlvl = max(xlvl, p); last_opp = p
                else:
                    last_opp = last_opp
            else:
                if is_sell[i] and p < xlvl - 1e-12:
                    xfill = i; break
                if is_sell[i]:
                    xlvl = min(xlvl, p); last_opp = p
            i += 1
        exitp = xlvl if xfill >= 0 else last_opp   # ran-out: mark at (inferred) touch
        net = (exitp - entry) / entry if side else (entry - exitp) / entry
        out.append(net * 1e4)
    return out


by_day = {}
for dts, dayi, side, fnet in decisions:
    by_day.setdefault(dayi, []).append((dts, side, fnet))

res_sim, res_frozen = [], []
n_days_done = 0
for dayi in sorted(by_day):
    date = dates[dayi]
    rowsd = by_day[dayi]
    try:
        if MODE == "validate":
            ts, px, is_sell = bybit_trades(date)
        else:
            ts, px, is_sell = binance_aggtrades(date)
    except Exception as e:
        print(f"  {date}: data fail {e}", flush=True)
        continue
    sims = sim_day([(r[0], r[1]) for r in rowsd], ts, px, is_sell)
    for (dts, side, fnet), s_ in zip(rowsd, sims):
        res_sim.append(s_); res_frozen.append(fnet)
    n_days_done += 1
    if n_days_done % 30 == 0:
        print(f"  {n_days_done} days...", flush=True)

sim = np.array(res_sim); frz = np.array(res_frozen)
m_sim = np.isfinite(sim); m_frz = np.isfinite(frz)
both = m_sim & m_frz
print(f"\n=== MODE={MODE} T{TGT:g}: decisions {len(sim)} | tape-sim fills {m_sim.sum()} "
      f"({100*m_sim.mean():.1f}%) | frozen fills {m_frz.sum()} ({100*m_frz.mean():.1f}%)", flush=True)
print(f"  tape-sim EV (filled): {np.nanmean(sim):+.2f}bp", flush=True)
if MODE == "validate":
    print(f"  frozen EV (filled): {np.nanmean(frz):+.2f}bp | on BOTH-filled: tape {sim[both].mean():+.2f} "
          f"vs frozen {frz[both].mean():+.2f} -> bias {sim[both].mean()-frz[both].mean():+.2f}bp", flush=True)
out = dict(mode=MODE, tgt=TGT, n=int(len(sim)), fill_sim=float(m_sim.mean()),
           ev_sim=float(np.nanmean(sim)),
           ev_frozen=float(np.nanmean(frz)) if MODE == "validate" else None,
           bias_both=float(sim[both].mean() - frz[both].mean()) if (MODE == "validate" and both.any()) else None)
bk.blob(f"research_runs/HBV1_BEXEC_{MODE}_T{TGT:g}.json").upload_from_string(json.dumps(out))
print("[saved]", flush=True)
