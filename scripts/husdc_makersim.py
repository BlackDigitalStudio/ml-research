#!/usr/bin/env python3
"""HUSDC maker-fill / adverse-selection prototype (USDT first; the missing baseline).

The existing sims have NO real maker-fill model: live_sim assumes the entry is
filled (book version even enters taker), grid_sim uses an IID Bernoulli fill_prob
(adverse-blind), and rev28's deployable-maker number entered at MID. None capture
adverse selection: a resting buy-limit fills preferentially when flow is against
you, and the orders that DON'T fill are the favorable runaways you miss.

This measures it directly by simulating a resting limit order against the REALIZED
trade stream (aggTrades). For each signal-gated decision at t wanting LONG, we
rest a maker buy at L=bid(t) (bid proxy = last taker-SELL price asof t):
  - TOUCH fill : first taker-SELL at price <= L within entry_window (Q0=0).
  - QUEUE fill : after >= Q0 cumulative SELL volume at price <= L (Q0 = qmult x
                 median trade size); MISS if not reached in entry_window.
  - MISS       : price ran up (no sell <= L) -> we miss the (favorable) move.
Then mark-out(tau) = mid(fill_time+tau)/L - 1 over FILLED decisions, vs the
rev28 PHANTOM mid(t+tau)/mid(t) - 1 over ALL gated. The gap = spread captured
MINUS adverse selection. SHORT is the mirror (sell at ask = last BUY price).

Reuses husdc_run.{fetch_day,load_contract,mid_asof,daterange}. Runs on a VM (ADC)
-> JSON+txt to gs://{BUCKET}/research_runs/husdc/.
  python3 husdc_makersim.py --smoke
  python3 husdc_makersim.py --symbols BTC ETH SOL DOGE --start 2026-05-01 --end 2026-05-14
"""
import argparse, json, sys, time, traceback
import numpy as np

import husdc_run as HR   # fetch_day, load_contract, mid_asof, daterange, BUCKET, OUT_PREFIX

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARKOUT = [1, 2, 5, 10, 30, 45, 60]   # mark-out horizons (s)
MOM_W = 5                              # momentum signal window (s)
ENTRY_WINDOW = 30                     # max seconds to wait for a maker fill
QMULTS = [0.0, 10.0, 40.0]            # Q0 = qmult * median trade size (0 = touch)
USDT_MAKER_RT_BP = 11.0               # rev28 maker round-trip; USDC promo = 0
MIN_GATED = 500                       # min gated decisions per side (stats floor)
MIN_FILL = 200                        # min filled decisions to report a markout


def asof(Tsrc, Psrc, q_ms, max_stale_ms=30000):
    """last value of Psrc at/just before each q_ms; nan if older than max_stale."""
    j = np.searchsorted(Tsrc, q_ms, side="right") - 1
    ok = j >= 0
    jc = np.clip(j, 0, len(Tsrc) - 1)
    v = np.where(ok, Psrc[jc], np.nan)
    age = np.where(ok, q_ms - Tsrc[jc], np.inf)
    return np.where(age <= max_stale_ms, v, np.nan)


def side_sim(side, grid, mid_g, quote_g, gate, T_take, P_take, cumv_take, med_sz,
             log):
    """side='long': maker buy at bid, filled by taker SELLs (T_take=sells).
       side='short': maker sell at ask, filled by taker BUYs.
    Returns dict of per-tau markout means for phantom(all-gated), touch, queue."""
    sign = 1.0 if side == "long" else -1.0
    gi = np.where(gate)[0]
    if len(gi) < MIN_GATED:
        return None
    tg = grid[gi]                       # decision times (ms)
    Lg = quote_g[gi]                    # rest price (bid for long, ask for short)
    midg = mid_g[gi]
    valid = np.isfinite(Lg) & np.isfinite(midg) & (Lg > 0)
    gi, tg, Lg, midg = gi[valid], tg[valid], Lg[valid], midg[valid]
    n = len(tg)

    out = {"n_gated": int(n), "markout": {}}
    # PHANTOM (rev28): fill at mid(t), all gated.
    ph = {}
    for tau in MARKOUT:
        m2 = asof(grid, mid_g, tg + tau * 1000)   # mid(t+tau) via the 1s mid grid
        ph[tau] = float(np.nanmean(sign * (m2 / midg - 1.0)) * 1e4)  # bp
    out["phantom_bp"] = ph

    # TOUCH and QUEUE fills, looped per decision (price-conditioned).
    win_ms = ENTRY_WINDOW * 1000
    res = {f"q{qm:g}": {"fill_t": np.full(n, -1, np.int64),
                        "L": np.full(n, np.nan)} for qm in QMULTS}
    for k in range(n):
        t0 = tg[k]; L = Lg[k]
        lo = np.searchsorted(T_take, t0, side="right")
        hi = np.searchsorted(T_take, t0 + win_ms, side="right")
        if hi <= lo:
            continue
        pw = P_take[lo:hi]
        # price condition: taker prints at/through our level.
        cond = (pw <= L) if side == "long" else (pw >= L)
        if not cond.any():
            continue
        tw = T_take[lo:hi]
        vw = (cumv_take[lo + 1:hi + 1] - cumv_take[lo:hi])  # per-trade vol slice
        cvol = np.cumsum(np.where(cond, vw, 0.0))
        for qm in QMULTS:
            Q0 = qm * med_sz
            idx = np.searchsorted(cvol, Q0 if Q0 > 0 else np.nextafter(0, 1))
            # smallest index with cvol>=Q0 AND cond true at/after it
            while idx < len(cond) and not cond[idx]:
                idx += 1
            if idx < len(cond):
                res[f"q{qm:g}"]["fill_t"][k] = tw[idx]
                res[f"q{qm:g}"]["L"][k] = L

    for qm in QMULTS:
        key = f"q{qm:g}"
        ft = res[key]["fill_t"]; Lf = res[key]["L"]
        filled = ft >= 0
        fr = float(filled.mean())
        mk = {}
        if filled.sum() >= MIN_FILL:
            ftf = ft[filled].astype(np.int64); Lff = Lf[filled]
            for tau in MARKOUT:
                m2 = asof(grid, mid_g, ftf + tau * 1000)
                mk[tau] = float(np.nanmean(sign * (m2 / Lff - 1.0)) * 1e4)
        out["markout"][key] = {"fill_rate": fr, "n_fill": int(filled.sum()),
                               "markout_bp": mk}
    log(f"    [{side}] gated={n:,} fill_rate touch={out['markout']['q0']['fill_rate']:.2f}"
        f" q40={out['markout']['q40']['fill_rate']:.2f}")
    return out


def analyze_symbol(sym, days, workers, log, stride=5):
    cT = f"{sym}USDT"
    log(f"\n{'='*78}\n### {sym} ({cT})  days={len(days)} stride={stride}s")
    d = HR.load_contract(cT, days, workers)
    if d is None:
        log("  MISSING -> skip"); return None
    T, P, Q, S, nd = d
    log(f"  trades={len(T):,} ({nd}d) med_size={np.median(Q):.4g}")
    sells = S < 0; buys = S > 0
    Tsell, Psell, Qsell = T[sells], P[sells], Q[sells].astype(np.float64)
    Tbuy, Pbuy, Qbuy = T[buys], P[buys], Q[buys].astype(np.float64)
    cv_sell = np.concatenate([[0.0], np.cumsum(Qsell)])
    cv_buy = np.concatenate([[0.0], np.cumsum(Qbuy)])
    med_sz = float(np.median(Q))

    t0 = T[0]; t1 = T[-1]
    grid = np.arange((t0 // 1000 + 1) * 1000, t1 - 70000, stride * 1000, dtype=np.int64)
    mid_g = asof(T, P, grid)
    bid_g = asof(Tsell, Psell, grid)   # bid proxy = last taker-sell price
    ask_g = asof(Tbuy, Pbuy, grid)     # ask proxy = last taker-buy price
    spread_bp = np.nanmedian((ask_g - bid_g) / mid_g) * 1e4
    log(f"  grid={len(grid):,} median spread~{spread_bp:.2f}bp")

    # momentum signal (trailing mid return over MOM_W); long if up, short if dn.
    midw = asof(T, P, grid - MOM_W * 1000)
    mom = np.log(mid_g / midw)
    out = {"trades": int(len(T)), "days": int(nd), "median_spread_bp": float(spread_bp),
           "grid_pts": int(len(grid))}
    out["long"] = side_sim("long", grid, mid_g, bid_g, mom > 0,
                           Tsell, Psell, cv_sell, med_sz, log)
    out["short"] = side_sim("short", grid, mid_g, ask_g, mom < 0,
                            Tbuy, Pbuy, cv_buy, med_sz, log)

    # headline: adverse-selection haircut at the rev28 hold (45s & 60s), touch fill.
    for tau in [45, 60]:
        for side in ["long", "short"]:
            s = out[side]
            if not s:
                continue
            ph = s["phantom_bp"].get(tau)
            mk = s["markout"]["q0"]["markout_bp"].get(tau)
            if ph is not None and mk is not None:
                haircut = ph - mk   # phantom(mid) minus realized maker(filled)
                log(f"  [{side} H={tau}s] phantom(mid)={ph:+.2f}bp  maker_touch={mk:+.2f}bp"
                    f"  spread_capture_net_adverse={mk-ph:+.2f}bp")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTC", "ETH", "SOL", "DOGE"])
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-05-14")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="maker1")
    a = ap.parse_args()
    if a.smoke:
        a.symbols = a.symbols[:1]; a.start = "2026-05-12"; a.end = "2026-05-14"; a.tag = "makersmoke"
    days = HR.daterange(a.start, a.end)
    buf = []
    def log(s):
        print(s, flush=True); buf.append(str(s))
    log(f"HUSDC maker-sim tag={a.tag} syms={a.symbols} {a.start}..{a.end} "
        f"({len(days)}d) entry_window={ENTRY_WINDOW}s qmults={QMULTS}")
    t0 = time.time()
    results = {"params": vars(a), "markout_h": MARKOUT, "per_symbol": {}}

    def _bk():
        from google.cloud import storage
        return storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket(HR.BUCKET)

    for sym in a.symbols:
        try:
            r = analyze_symbol(sym, days, a.workers, log, a.stride)
            if r:
                results["per_symbol"][sym] = r
                results["elapsed_s"] = time.time() - t0
                try:
                    pj = f"husdc_{a.tag}_partial.json"
                    json.dump(results, open(pj, "w"), indent=2, default=str)
                    _bk().blob(f"{HR.OUT_PREFIX}/{pj}").upload_from_filename(pj)
                except Exception as ex:
                    log(f"  [partial-warn] {ex}")
        except Exception:
            log(f"  ERROR {sym}:\n{traceback.format_exc()}")
    results["elapsed_s"] = time.time() - t0
    log(f"\nDONE in {results['elapsed_s']:.0f}s ok={list(results['per_symbol'])}")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    jn = f"husdc_{a.tag}_{stamp}.json"; tn = f"husdc_{a.tag}_{stamp}.txt"
    json.dump(results, open(jn, "w"), indent=2, default=str)
    open(tn, "w", encoding="utf-8").write("\n".join(buf))
    try:
        bk = _bk()
        bk.blob(f"{HR.OUT_PREFIX}/{jn}").upload_from_filename(jn)
        bk.blob(f"{HR.OUT_PREFIX}/{tn}").upload_from_filename(tn)
        bk.blob(f"{HR.OUT_PREFIX}/HUSDC_{a.tag}_DONE.txt").upload_from_string(
            f"{stamp} ok={list(results['per_symbol'])}")
        log(f"[saved] gs://{HR.BUCKET}/{HR.OUT_PREFIX}/{jn}")
    except Exception as ex:
        log(f"[save-warn] {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
