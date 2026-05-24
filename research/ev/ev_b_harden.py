"""EV-B hardening: cluster-robust significance + net-of-cost fade economics + KRW split.
Strategy framing: SHORT the Binance perp at the first 1m close after the announcement
(listings/airdrops/delist/caution all drop short-term); net PnL = -r_H - cost(13bp round-trip).
Cluster-robust: collapse co-timestamp events (batch listings) to one obs per ts -> t-stat on
unique-ts means (conservative; removes intra-batch pseudo-replication). Binance will-list V
long-leg (buy +5m, sell +60m) reported separately. Reuses ev_b_announcements.parquet + klines."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJ = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
KLINES = "free_v1/klines_1m/"
COST_BP = 13.0
HS = [1, 5, 15, 60]
TOL_MS = 120_000
MIN_MS = 60_000

E = pd.read_parquet("C:/Dev/sb-data-poc/ev_b_announcements.parquet")
E["ts"] = pd.to_datetime(E["ts"], utc=True, format="mixed")
E["krw"] = E["title"].str.contains("KRW", case=False, na=False) | E["title"].str.contains("원화", na=False)
fs = pafs.GcsFileSystem()
universe = {b.name.split("/")[-1].replace(".parquet", "")
            for b in storage.Client(project=PROJ).list_blobs(BUCKET, prefix=KLINES) if b.name.endswith(".parquet")}
E["has_perp"] = E["sym"].isin(universe)
need = sorted(E.loc[E["has_perp"], "sym"].unique())
print(f"announcements={len(E)} matched_syms={len(need)}", flush=True)


def load_close(sym):
    try:
        pf = pq.ParquetFile(fs.open_input_file(f"{BUCKET}/{KLINES}{sym}.parquet"))
        t = pf.read(columns=["open_time", "close"])
        ot = pd.to_numeric(pd.Series(t.column("open_time").to_numpy()), errors="coerce").to_numpy()
        cl = pd.to_numeric(pd.Series(t.column("close").to_numpy()), errors="coerce").to_numpy()
    except Exception:
        return sym, None, None
    m = np.isfinite(ot) & np.isfinite(cl)
    ot, cl = ot[m], cl[m]
    if len(ot) < 2:
        return sym, None, None
    o = np.argsort(ot, kind="stable")
    ot, cl = ot[o], cl[o]
    uq = np.concatenate(([True], np.diff(ot) > 0))
    return sym, ot[uq], cl[uq]


K = {}
with ThreadPoolExecutor(max_workers=32) as ex:
    for sym, ot, cl in ex.map(load_close, need):
        if ot is not None:
            K[sym] = (ot, cl)
print(f"loaded {len(K)} symbols", flush=True)


def cl_at(ot, cl, tt):
    j = np.searchsorted(ot, tt, "left")
    if j < len(ot) and abs(ot[j] - tt) <= TOL_MS:
        return cl[j]
    if j > 0 and abs(ot[j - 1] - tt) <= TOL_MS:
        return cl[j - 1]
    return np.nan


recs = []
for e in E[E["has_perp"]].itertuples():
    kk = K.get(e.sym)
    if kk is None:
        continue
    ot, cl = kk
    t0 = int(e.ts.value // 1_000_000)
    if t0 < ot[0] or t0 > ot[-1]:
        continue
    c0 = cl_at(ot, cl, t0)
    if not np.isfinite(c0):
        continue
    row = {"source": e.source, "kind": e.kind, "sign": e.sign, "sym": e.sym,
           "krw": bool(e.krw), "ts_min": t0 // MIN_MS}
    for h in HS:
        ch = cl_at(ot, cl, t0 + h * MIN_MS)
        row[f"r{h}"] = np.log(ch / c0) if np.isfinite(ch) else np.nan
    recs.append(row)
R = pd.DataFrame(recs)
print(f"tradeable events={len(R)}\n", flush=True)


def tstat_clustered(short_ret, ts_min):
    """collapse to one obs per ts (mean of co-listed tickers), t-stat of the mean."""
    df = pd.DataFrame({"x": short_ret, "ts": ts_min}).dropna()
    g = df.groupby("ts")["x"].mean().values
    n = len(g)
    if n < 3:
        return np.nan, n, np.nan
    t = np.mean(g) / (np.std(g, ddof=1) / np.sqrt(n) + 1e-12)
    return float(t), n, float(np.mean(g > 0))


def report(df, label, store):
    if len(df) < 5:
        return
    n_ts = df["ts_min"].nunique()
    print(f"=== {label} (n_events={len(df)}, n_ts={n_ts}, batch={len(df)/max(n_ts,1):.2f}x) ===")
    print("  SHORT-hold net PnL  (gross = -r_H; net = gross - 13bp cost):")
    d = {"n_events": len(df), "n_ts": int(n_ts), "H": {}}
    for h in HS:
        r = df[f"r{h}"].values
        short = -r  # short PnL gross (return units)
        ok = np.isfinite(short)
        if ok.sum() < 5:
            continue
        gross_bp = float(np.mean(short[ok]) * 1e4)
        med_net = float(np.median(short[ok]) * 1e4) - COST_BP
        win_ev = float(np.mean(short[ok] > 0))
        t, nts, win_ts = tstat_clustered(short[ok], df["ts_min"].values[ok])
        row = {"gross_bp": round(gross_bp, 1), "net_bp": round(gross_bp - COST_BP, 1),
               "med_net_bp": round(med_net, 1), "win_ev": round(win_ev, 3),
               "t_clust": round(t, 2) if t == t else None, "n_ts": nts,
               "win_ts": round(win_ts, 3) if win_ts == win_ts else None}
        d["H"][h] = row
        print(f"   h={h:>3}m gross={row['gross_bp']:+8.1f} net={row['net_bp']:+8.1f}bp "
              f"t_clust={row['t_clust']} win_ev={row['win_ev']:.3f} win_ts={row['win_ts']} med_net={row['med_net_bp']:+.1f}")
    store[label] = d


store = {"cost_bp": COST_BP}
report(R[(R["source"] == "upbit") & (R["kind"] == "list")], "UPBIT list(+) ALL", store)
report(R[(R["source"] == "upbit") & (R["kind"] == "list") & (R["krw"])], "UPBIT list(+) KRW-market", store)
report(R[(R["source"] == "upbit") & (R["kind"] == "list") & (~R["krw"])], "UPBIT list(+) non-KRW", store)
report(R[(R["source"] == "upbit") & (R["sign"] < 0)], "UPBIT delist+caution(-)", store)
report(R[(R["source"] == "binance") & (R["kind"] == "list")], "BINANCE will-list(+)", store)
report(R[(R["source"] == "binance") & (R["kind"] == "airdrop")], "BINANCE airdrop(+)", store)

# Binance will-list V long-leg: buy +5m, sell +60m (r_5to60 = r60 - r5)
wl = R[(R["source"] == "binance") & (R["kind"] == "list")].copy()
v = (wl["r60"] - wl["r5"]).values
ok = np.isfinite(v)
if ok.sum() >= 5:
    gross_bp = float(np.mean(v[ok]) * 1e4)
    t, nts, win_ts = tstat_clustered(v[ok], wl["ts_min"].values[ok])
    print(f"\n=== BINANCE will-list V long-leg (buy+5m, sell+60m) n={int(ok.sum())} ===")
    print(f"   gross={gross_bp:+.1f} net={gross_bp - COST_BP:+.1f}bp t_clust={round(t,2)} "
          f"win_ev={np.mean(v[ok] > 0):.3f} win_ts={round(win_ts,3)} n_ts={nts}")
    store["BINANCE will-list V long-leg"] = {"gross_bp": round(gross_bp, 1), "net_bp": round(gross_bp - COST_BP, 1),
                                             "t_clust": round(t, 2), "n_ts": nts}

json.dump(store, open("C:/Dev/sb-data-poc/ev_b_harden.json", "w"), indent=2)
print("\nEV_B_HARDEN_DONE", flush=True)
