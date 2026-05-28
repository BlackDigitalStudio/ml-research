#!/usr/bin/env python3
"""HUSDC first decisive run — USDT->USDC signal-mirror MECHANISM test.

Self-contained, aggTrades-only (data.binance.vision futures/um/daily). For each
base symbol S we pull the USDC perp (S+USDC) and USDT perp (S+USDT) aggTrades,
build aligned 1s mid/flow series, and compute:

  STEP A  coupling: contemporaneous corr/beta(USDC_ret, USDT_ret) {5,30,60}s;
          lead-lag corr(USDC_fwd_H, USDT_trail_H) vs corr(USDT_fwd_H, USDC_trail_H)
          {15,60}s (who leads); basis = log(USDC_mid/USDT_mid) mean/sd(bp)/AR1
          (mean-reversion) + decoupling fraction.
  STEP B  transfer coeff: USDT-computed predictors {trail_ret_5s, trail_ret_30s,
          taker_flow_imb_30s} scored by rank-IC vs USDT_fwd_H AND vs USDC_fwd_H
          {15,60}s. transfer = IC_USDC / IC_USDT (does the USDT signal carry to
          USDC's forward move).
  STEP C  first-order economics: Roll effective-spread (bp) USDC vs USDT from
          trades; fee_saving (USDC maker 0% vs USDT maker) minus USDC excess
          spread = net bp the 0%-maker reclaims of the rev28 -4..-5bp maker
          deficit. (top-of-book/queue fidelity deferred; this is the trades-only
          first read.)

Runs on a GCP VM (ADC) -> results JSON+txt to gs://{BUCKET}/{OUT_PREFIX}/.
  python3 husdc_run.py --smoke                       # 1 sym x 3 days
  python3 husdc_run.py --symbols BTC ETH SOL DOGE --start 2026-04-15 --end 2026-05-14
"""
import argparse, io, json, sys, time, zipfile, traceback
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BUCKET = "market-data-0998ac51"
OUT_PREFIX = "research_runs/husdc"
BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
UA = {"User-Agent": "Mozilla/5.0"}
NS = 1000  # ms per second helper not used; times are ms

CONTEMP = [5, 30, 60]        # contemporaneous return windows (s)
HS = [15, 60]                # forward/trailing horizons (s)
TRAIL_PRED = [5, 30]         # predictor trailing-return windows (s)
TFI_W = 30                   # taker-flow-imbalance window (s)
MAX_STALE_S = 30             # asof mid invalid if last trade older than this
# USDT VIP0 maker round-trip the rev28 deployable surface pays (the cost the
# USDC 0% promo removes). rev28 used maker 0.04/0.07; round-trip bp:
USDT_MAKER_RT_BP = 11.0      # 0.04% + 0.07% = 0.11% = 11 bp (rev28 convention)


def daterange(s, e):
    d0 = date.fromisoformat(s); d1 = date.fromisoformat(e)
    out = []
    while d0 <= d1:
        out.append(d0.isoformat()); d0 += timedelta(days=1)
    return out


def fetch_day(contract, d):
    url = f"{BASE}/{contract}/{contract}-aggTrades-{d}.zip"
    for _ in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            raw = z.read(z.namelist()[0])
            head = raw[:40].split(b",")[0].decode("ascii", "ignore")
            skip = 1 if any(c.isalpha() for c in head) else 0
            # cols: aggId,price,qty,firstId,lastId,transact_time,is_buyer_maker
            df = pd.read_csv(io.BytesIO(raw), header=None, skiprows=skip,
                             usecols=[1, 2, 5, 6], names=["p", "q", "t", "m"],
                             dtype={"p": "float64", "q": "float64", "t": "int64"})
            if len(df) == 0:
                return None
            P = df["p"].to_numpy(); Q = df["q"].to_numpy()
            T = df["t"].to_numpy()
            mk = df["m"]
            # is_buyer_maker True => aggressor is SELLER => taker sell (-)
            if mk.dtype == bool:
                sign = np.where(mk.to_numpy(), -1.0, 1.0)
            else:
                tk = mk.astype(str).str.lower().str.startswith("t").to_numpy()
                sign = np.where(tk, -1.0, 1.0)
            fin = np.isfinite(P) & (P > 0)
            # downcast to fit a 16GB box for high-volume USDT contracts
            return (T[fin].astype(np.int64), P[fin].astype(np.float64),
                    Q[fin].astype(np.float32), sign[fin].astype(np.int8))
        except Exception as ex:
            print(f"   [retry] {contract} {d}: {type(ex).__name__} {ex}",
                  flush=True)
            continue
    return None


def load_contract(contract, days, workers):
    Ts, Ps, Qs, Ss = [], [], [], []
    nfound = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(lambda d: (d, fetch_day(contract, d)), days):
            d, v = res
            if v is None:
                continue
            nfound += 1
            Ts.append(v[0]); Ps.append(v[1]); Qs.append(v[2]); Ss.append(v[3])
    if not Ts:
        return None
    T = np.concatenate(Ts); P = np.concatenate(Ps)
    Q = np.concatenate(Qs); S = np.concatenate(Ss)
    o = np.argsort(T, kind="stable")
    return T[o], P[o], Q[o], S[o], nfound


def mid_asof(T, P, grid_ms):
    """last trade price at/just before each grid ms; staleness->nan."""
    j = np.searchsorted(T, grid_ms, side="right") - 1
    valid = j >= 0
    j_c = np.clip(j, 0, len(T) - 1)
    mid = np.where(valid, P[j_c], np.nan)
    stale = np.where(valid, grid_ms - T[j_c], np.inf)
    mid = np.where(stale <= MAX_STALE_S * 1000, mid, np.nan)
    return mid


def rank(x):
    r = np.empty(len(x), dtype=np.float64)
    o = np.argsort(x, kind="stable")
    r[o] = np.arange(len(x))
    return r


def rank_ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 500:
        return np.nan, int(m.sum())
    rx = rank(x[m]); ry = rank(y[m])
    rx -= rx.mean(); ry -= ry.mean()
    sx = np.sqrt((rx * rx).sum()); sy = np.sqrt((ry * ry).sum())
    if sx == 0 or sy == 0:
        return np.nan, int(m.sum())
    return float((rx * ry).sum() / (sx * sy)), int(m.sum())


def pearson_beta(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 500:
        return np.nan, np.nan, int(m.sum())
    x = x[m] - x[m].mean(); y = y[m] - y[m].mean()
    sx = np.sqrt((x * x).sum()); sy = np.sqrt((y * y).sum())
    if sx == 0 or sy == 0:
        return np.nan, np.nan, int(m.sum())
    r = float((x * y).sum() / (sx * sy))
    beta = float((x * y).sum() / (x * x).sum())   # y ~ beta*x  (x=USDT)
    return r, beta, int(m.sum())


def roll_spread_bp(T, P):
    """Roll (1984) effective spread from trade-price serial covariance, in bp."""
    if len(P) < 1000:
        return np.nan
    dp = np.diff(P)
    c = np.cov(dp[1:], dp[:-1])[0, 1]
    if not np.isfinite(c) or c >= 0:
        return 0.0
    spread = 2.0 * np.sqrt(-c)
    return float(spread / np.median(P) * 1e4)


def taker_flow_imb(T, Q, S, grid_ms, W):
    """signed taker volume / total over trailing (t-W, t], via prefix sums."""
    cs = np.concatenate([[0.0], np.cumsum(S * Q)])
    ca = np.concatenate([[0.0], np.cumsum(Q)])
    hi = np.searchsorted(T, grid_ms, side="right")
    lo = np.searchsorted(T, grid_ms - W * 1000, side="right")
    num = cs[hi] - cs[lo]
    den = ca[hi] - ca[lo]
    return np.where(den > 0, num / den, np.nan)


def analyze_symbol(sym, days, workers, log):
    cU = f"{sym}USDC"; cT = f"{sym}USDT"
    log(f"\n{'='*78}\n### {sym}  ({cU} vs {cT})  days={len(days)}")
    dU = load_contract(cU, days, workers)
    dT = load_contract(cT, days, workers)
    if dU is None or dT is None:
        log(f"  MISSING data: USDC={'ok' if dU else 'NONE'} "
            f"USDT={'ok' if dT else 'NONE'} -> skip")
        return None
    TU, PU, QU, SU, nU = dU
    TT, PT, QT, ST, nT = dT
    log(f"  trades: USDC={len(TU):,} ({nU}d)  USDT={len(TT):,} ({nT}d)")
    # liquidity / sparsity
    spanU = (TU[-1] - TU[0]) / 1000.0
    tpdU = len(TU) / max(1, nU); tpdT = len(TT) / max(1, nT)
    med_gap_U = float(np.median(np.diff(TU)) / 1000.0)
    med_gap_T = float(np.median(np.diff(TT)) / 1000.0)
    log(f"  trades/day: USDC={tpdU:,.0f}  USDT={tpdT:,.0f}  | "
        f"median trade-gap: USDC={med_gap_U:.2f}s  USDT={med_gap_T:.3f}s")

    t0 = max(TU[0], TT[0]); t1 = min(TU[-1], TT[-1])
    grid = np.arange((t0 // 1000 + 1) * 1000, t1, 1000, dtype=np.int64)
    log(f"  common 1s grid: {len(grid):,} pts")
    midU = mid_asof(TU, PU, grid); midT = mid_asof(TT, PT, grid)

    def fwd(mid, H):
        f = mid_asof(TU if mid is midU else TT,
                     PU if mid is midU else PT, grid + H * 1000)
        return np.log(f / mid)

    def trail(mid, W):
        b = mid_asof(TU if mid is midU else TT,
                     PU if mid is midU else PT, grid - W * 1000)
        return np.log(mid / b)

    out = {"trades_usdc": int(len(TU)), "trades_usdt": int(len(TT)),
           "days_usdc": int(nU), "days_usdt": int(nT),
           "trades_per_day_usdc": tpdU, "trades_per_day_usdt": tpdT,
           "median_gap_s_usdc": med_gap_U, "median_gap_s_usdt": med_gap_T,
           "grid_pts": int(len(grid))}

    # ---- STEP A: coupling ----
    log("  [A] contemporaneous corr(USDC,USDT) & beta(USDC~USDT):")
    A = {"contemp": {}, "leadlag": {}, "basis": {}}
    for w in CONTEMP:
        rU = trail(midU, w); rT = trail(midT, w)
        r, beta, n = pearson_beta(rT, rU)
        A["contemp"][w] = {"corr": r, "beta": beta, "n": n}
        log(f"      {w:>3}s: corr={r:+.4f} beta={beta:+.3f} (n={n:,})")
    log("  [A] lead-lag (who leads): corr(USDC_fwd,USDT_trail) vs corr(USDT_fwd,USDC_trail):")
    for H in HS:
        fU = fwd(midU, H); tT = trail(midT, H)
        fT = fwd(midT, H); tU = trail(midU, H)
        r_usdt_leads, _, n1 = pearson_beta(tT, fU)   # USDT trailing -> USDC fwd
        r_usdc_leads, _, n2 = pearson_beta(tU, fT)   # USDC trailing -> USDT fwd
        A["leadlag"][H] = {"usdt_leads_usdc": r_usdt_leads,
                            "usdc_leads_usdt": r_usdc_leads, "n": min(n1, n2)}
        lead = ("USDT" if r_usdt_leads > r_usdc_leads else "USDC")
        log(f"      H={H}s: USDT->USDC={r_usdt_leads:+.4f}  USDC->USDT={r_usdc_leads:+.4f}"
            f"  => {lead} leads")
    # basis
    b = np.log(midU / midT) * 1e4   # bp
    bm = b[np.isfinite(b)]
    if len(bm) > 1000:
        ar1 = float(np.corrcoef(bm[1:], bm[:-1])[0, 1])
        decoup = float(np.mean(np.abs(bm - np.median(bm)) > 5 * np.std(bm)))
        A["basis"] = {"mean_bp": float(np.mean(bm)), "sd_bp": float(np.std(bm)),
                      "ar1": ar1, "decouple_frac_5sd": decoup, "n": int(len(bm))}
        log(f"  [A] basis log(USDC/USDT): mean={np.mean(bm):+.2f}bp sd={np.std(bm):.2f}bp "
            f"AR1={ar1:+.3f} decouple_frac(>5sd)={decoup:.4f}")
    out["stepA"] = A

    # ---- STEP B: transfer coefficient ----
    log("  [B] USDT-signal rank-IC vs USDT-fwd vs USDC-fwd (transfer = IC_USDC/IC_USDT):")
    preds = {}
    for w in TRAIL_PRED:
        preds[f"trail_ret_{w}s"] = trail(midT, w)
    preds[f"tfi_{TFI_W}s"] = taker_flow_imb(TT, QT, ST, grid, TFI_W)
    B = {}
    for H in HS:
        fU = fwd(midU, H); fT = fwd(midT, H)
        B[H] = {}
        for pname, pv in preds.items():
            icT, nT_ = rank_ic(pv, fT)
            icU, nU_ = rank_ic(pv, fU)
            trans = (icU / icT) if (icT is not None and abs(icT) > 1e-4) else np.nan
            B[H][pname] = {"ic_usdt": icT, "ic_usdc": icU, "transfer": trans,
                           "n": min(nT_, nU_)}
            log(f"      H={H}s {pname:>14}: IC_USDT={icT:+.4f} IC_USDC={icU:+.4f}"
                f" transfer={trans:+.3f} (n={min(nT_, nU_):,})")
    out["stepB"] = B

    # ---- STEP C: first-order economics ----
    rsU = roll_spread_bp(TU, PU); rsT = roll_spread_bp(TT, PT)
    excess = (rsU - rsT) if (np.isfinite(rsU) and np.isfinite(rsT)) else np.nan
    net_reclaim = (USDT_MAKER_RT_BP - excess) if np.isfinite(excess) else np.nan
    out["stepC"] = {"roll_spread_bp_usdc": rsU, "roll_spread_bp_usdt": rsT,
                    "usdc_excess_spread_bp": excess,
                    "usdt_maker_rt_bp": USDT_MAKER_RT_BP,
                    "net_reclaim_bp": net_reclaim}
    log(f"  [C] Roll spread: USDC={rsU:.2f}bp USDT={rsT:.2f}bp  excess={excess:+.2f}bp")
    log(f"  [C] 0%-maker reclaims USDT maker RT {USDT_MAKER_RT_BP:.1f}bp minus USDC excess "
        f"spread {excess:+.2f}bp => NET +{net_reclaim:.2f}bp toward the rev28 -4..-5bp deficit")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTC", "ETH", "SOL", "DOGE"])
    ap.add_argument("--start", default="2026-04-15")
    ap.add_argument("--end", default="2026-05-14")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="run1")
    a = ap.parse_args()
    if a.smoke:
        a.symbols = a.symbols[:1]; a.start = "2026-05-12"; a.end = "2026-05-14"
        a.tag = "smoke"
    days = daterange(a.start, a.end)
    buf = []
    def log(s):
        print(s, flush=True); buf.append(str(s))
    log(f"HUSDC run tag={a.tag} syms={a.symbols} {a.start}..{a.end} ({len(days)}d) "
        f"workers={a.workers}")
    t0 = time.time()
    results = {"params": vars(a), "days": len(days), "per_symbol": {}}

    def _bk():
        from google.cloud import storage
        return storage.Client(
            project="project-0998ac51-36ba-445c-bc7").bucket(BUCKET)

    def save_partial():
        # durable progress: an OOM/timeout on a later (heavier) symbol must
        # not lose the symbols already computed.
        try:
            pj = f"husdc_{a.tag}_partial.json"
            json.dump(results, open(pj, "w"), indent=2, default=str)
            _bk().blob(f"{OUT_PREFIX}/{pj}").upload_from_filename(pj)
            log(f"  [partial saved] {list(results['per_symbol'])}")
        except Exception as ex:
            log(f"  [partial-save-warn] {type(ex).__name__}: {ex}")

    for sym in a.symbols:
        try:
            r = analyze_symbol(sym, days, a.workers, log)
            if r is not None:
                results["per_symbol"][sym] = r
                results["elapsed_s"] = time.time() - t0
                save_partial()
        except Exception:
            log(f"  ERROR {sym}:\n{traceback.format_exc()}")
    results["elapsed_s"] = time.time() - t0
    log(f"\nDONE in {results['elapsed_s']:.0f}s; symbols ok="
        f"{list(results['per_symbol'])}")

    # upload to GCS (best-effort; also write local)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    jname = f"husdc_{a.tag}_{stamp}.json"; tname = f"husdc_{a.tag}_{stamp}.txt"
    json.dump(results, open(jname, "w"), indent=2, default=str)
    open(tname, "w", encoding="utf-8").write("\n".join(buf))
    try:
        from google.cloud import storage
        bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket(BUCKET)
        bk.blob(f"{OUT_PREFIX}/{jname}").upload_from_filename(jname)
        bk.blob(f"{OUT_PREFIX}/{tname}").upload_from_filename(tname)
        bk.blob(f"{OUT_PREFIX}/HUSDC_{a.tag}_DONE.txt").upload_from_string(
            f"{stamp} ok={list(results['per_symbol'])} elapsed={results['elapsed_s']:.0f}s")
        log(f"[saved] gs://{BUCKET}/{OUT_PREFIX}/{jname}")
    except Exception as ex:
        log(f"[save-warn] {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
