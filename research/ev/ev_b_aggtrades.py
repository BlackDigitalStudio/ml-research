"""EV-B sub-minute executability via Binance aggTrades daily dumps (data.binance.vision).
DECISIVE test of the strongest HDATA signal: does the EV-B listing-fade survive at SECOND
resolution with realistic entry latency? (REST aggTrades does not serve >~recent history.)

For each tradeable EV-B signed event: pull the day's aggTrades, reconstruct the second-res
price path in [ts-300s, ts+300s], measure:
 - PRE-ts run-up (signed by event sign): how much already moved before the official ts
   (= leak/rumor leg + the value of a faster feed).
 - SHORT fade net PnL at entry latency L in {0,1,5,10,30}s, to {5,30,60,300}s, minus 13bp.
 - decay of edge with entry latency L (executability in the 1-3ms regime).
"""
import io
import zipfile
import json
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COST_BP = 13.0
BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
UA = {"User-Agent": "Mozilla/5.0"}
LAT = [0, 1, 5, 10, 30]      # entry latency (s after announcement ts)
HS = [5, 30, 60, 300]        # forward horizon (s)
PRE = [10, 60, 300]          # pre-ts run-up windows (s)
WIN_MS = 300_000

E = pd.read_parquet("C:/Dev/sb-data-poc/ev_b_announcements.parquet")
E["ts"] = pd.to_datetime(E["ts"], utc=True, format="mixed")
E = E[E["sign"] != 0].copy()


def cohort_of(src, kind, sign):
    if src == "upbit" and kind == "list":
        return "UPBIT_list"
    if src == "upbit" and sign < 0:
        return "UPBIT_delist_caution"
    if src == "binance" and kind == "list":
        return "BINANCE_will_list"
    if src == "binance" and kind == "airdrop":
        return "BINANCE_airdrop"
    return None


E["cohort"] = [cohort_of(r.source, r.kind, r.sign) for r in E.itertuples()]
E = E[E["cohort"].notna()].reset_index(drop=True)
E["date"] = E["ts"].dt.strftime("%Y-%m-%d")
print(f"events in cohorts: {E['cohort'].value_counts().to_dict()} total={len(E)}", flush=True)

pairs = sorted(set(zip(E["sym"], E["date"])))
print(f"unique (symbol,date) files to fetch: {len(pairs)}", flush=True)


def fetch_day(p):
    sym, date = p
    url = f"{BASE}/{sym}/{sym}-aggTrades-{date}.zip"
    for _ in range(2):
        try:
            r = requests.get(url, headers=UA, timeout=90)
            if r.status_code != 200:
                return p, None
            z = zipfile.ZipFile(io.BytesIO(r.content))
            raw = z.read(z.namelist()[0])
            head = raw[:32].split(b",")[0].decode("ascii", "ignore")
            hdr = 0 if any(c.isalpha() for c in head) else None
            df = pd.read_csv(io.BytesIO(raw), header=hdr)
            T = pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy()   # transact_time ms
            P = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()   # price
            m = np.isfinite(T) & np.isfinite(P)
            T, P = T[m], P[m]
            o = np.argsort(T, kind="stable")
            return p, (T[o], P[o])
        except Exception:
            continue
    return p, None


cache = {}
done = 0
with ThreadPoolExecutor(max_workers=24) as ex:
    for p, v in ex.map(fetch_day, pairs):
        done += 1
        if v is not None:
            cache[p] = v
        if done % 40 == 0:
            print(f"  fetched {done}/{len(pairs)} (ok={len(cache)})", flush=True)
print(f"loaded {len(cache)}/{len(pairs)} aggTrades files", flush=True)


def price_at(T, P, tgt):
    j = np.searchsorted(T, tgt, "right") - 1
    return P[j] if j >= 0 else np.nan


rows = []
for e in E.itertuples():
    kk = cache.get((e.sym, e.date))
    if kk is None:
        continue
    T, P = kk
    t0 = int(e.ts.value // 1_000_000)
    if t0 < T[0] - WIN_MS or t0 > T[-1] + WIN_MS:
        continue
    p0 = price_at(T, P, t0)
    if not np.isfinite(p0) or p0 <= 0:
        continue
    rec = {"cohort": e.cohort, "sign": e.sign}
    for w in PRE:
        pp = price_at(T, P, t0 - w * 1000)
        rec[f"pre{w}"] = (np.log(p0 / pp) * e.sign) if (np.isfinite(pp) and pp > 0) else np.nan
    for L in LAT:
        pe = price_at(T, P, t0 + L * 1000)
        for h in HS:
            pf = price_at(T, P, t0 + (L + h) * 1000)
            if np.isfinite(pe) and np.isfinite(pf) and pe > 0 and pf > 0:
                rec[f"short_L{L}_h{h}"] = -np.log(pf / pe)   # SHORT gross (return units)
            else:
                rec[f"short_L{L}_h{h}"] = np.nan
    rows.append(rec)
R = pd.DataFrame(rows)
print(f"\nevents with aggTrades path = {len(R)} | by cohort: {R['cohort'].value_counts().to_dict()}", flush=True)

store = {"cost_bp": COST_BP, "n": len(R), "cohorts": {}}
for coh in ["UPBIT_list", "UPBIT_delist_caution", "BINANCE_will_list", "BINANCE_airdrop", "ALL"]:
    d = R if coh == "ALL" else R[R["cohort"] == coh]
    if len(d) < 5:
        continue
    print(f"\n=== {coh} (n={len(d)}) ===")
    pre = {w: round(float(np.nanmean(d[f'pre{w}']) * 1e4), 1) for w in PRE}
    print(f"  PRE-ts run-up (signed, bp): " + " ".join(f"-{w}s:{pre[w]:+.1f}" for w in PRE))
    cd = {"n": len(d), "pre_runup_bp": pre, "short_net_bp": {}, "short_win": {}}
    print("  SHORT fade NET bp (gross - 13bp) by entry-latency L x horizon h:")
    print("        " + "".join(f"  h={h}s" for h in HS))
    for L in LAT:
        nets, wins = [], []
        line = f"   L={L:>2}s"
        for h in HS:
            s = d[f"short_L{L}_h{h}"].to_numpy()
            s = s[np.isfinite(s)]
            if len(s) < 5:
                line += "    n/a"
                nets.append(None); wins.append(None); continue
            net = float(np.mean(s) * 1e4 - COST_BP)
            nets.append(round(net, 1)); wins.append(round(float(np.mean(s > 0)), 3))
            line += f" {net:+7.1f}"
        cd["short_net_bp"][L] = nets
        cd["short_win"][L] = wins
        print(line + "   win@60s=" + (str(wins[HS.index(60)]) if 60 in HS else "?"))
    store["cohorts"][coh] = cd

json.dump(store, open("C:/Dev/sb-data-poc/ev_b_aggtrades.json", "w"), indent=2)
print("\nEV_B_AGGTRADES_DONE", flush=True)
