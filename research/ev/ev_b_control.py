"""EV-B firm-up: de-confound listing-event reactions from general alt drift.
(1) RANDOM-TIMESTAMP NULL: keep each cohort's (symbol, sign) set but replace the
    event time with a random time on the SAME symbol; build the null dir_ret
    distribution -> real cohort dir_ret z-score + percentile. Answers "is the
    event reaction beyond that symbol's own baseline drift?".
(2) BTC-RELATIVE: event_r - BTC_r over the same window -> cohort dir_ret on excess.
Reuses ev_b_announcements.parquet + free_v1/klines_1m (col projection)."""
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
COST = 0.0013
HS = [1, 5, 15, 60]
TOL_MS = 120_000
MIN_MS = 60_000
NULL_B = 1000
rng = np.random.default_rng(0)

E = pd.read_parquet("C:/Dev/sb-data-poc/ev_b_announcements.parquet")
E["ts"] = pd.to_datetime(E["ts"], utc=True, format="mixed")
fs = pafs.GcsFileSystem()
universe = {b.name.split("/")[-1].replace(".parquet", "")
            for b in storage.Client(project=PROJ).list_blobs(BUCKET, prefix=KLINES) if b.name.endswith(".parquet")}
E["has_perp"] = E["sym"].isin(universe)
need = sorted(set(E.loc[E["has_perp"], "sym"].unique()) | {"BTCUSDT"})
print(f"announcements={len(E)} matched_syms={len(need)-1}", flush=True)


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
done = 0
with ThreadPoolExecutor(max_workers=32) as ex:
    for sym, ot, cl in ex.map(load_close, need):
        done += 1
        if ot is not None:
            K[sym] = (ot, cl)
        if done % 80 == 0:
            print(f"  klines ...{done}/{len(need)}", flush=True)
print(f"loaded {len(K)} symbols (incl BTC={'BTCUSDT' in K})", flush=True)
btc_ot, btc_cl = K["BTCUSDT"]


def cl_at(ot, cl, tt):
    j = np.searchsorted(ot, tt, "left")
    if j < len(ot) and abs(ot[j] - tt) <= TOL_MS:
        return cl[j]
    if j > 0 and abs(ot[j - 1] - tt) <= TOL_MS:
        return cl[j - 1]
    return np.nan


# real events: raw r_h and BTC-relative rel_h
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
    b0 = cl_at(btc_ot, btc_cl, t0)
    if not (np.isfinite(c0) and np.isfinite(b0)):
        continue
    row = {"source": e.source, "kind": e.kind, "sign": e.sign, "sym": e.sym}
    for h in HS:
        ch = cl_at(ot, cl, t0 + h * MIN_MS)
        bh = cl_at(btc_ot, btc_cl, t0 + h * MIN_MS)
        row[f"r{h}"] = np.log(ch / c0) if np.isfinite(ch) else np.nan
        row[f"rel{h}"] = (np.log(ch / c0) - np.log(bh / b0)) if (np.isfinite(ch) and np.isfinite(bh)) else np.nan
    recs.append(row)
R = pd.DataFrame(recs)
print(f"tradeable events={len(R)}", flush=True)


def sample_rs(sym, h, B):
    ot, cl = K[sym]
    lo, hi = ot[0], ot[-1] - h * MIN_MS
    if hi <= lo + MIN_MS:
        return np.full(B, np.nan)
    tt = rng.integers(lo, hi, B)
    j0 = np.clip(np.searchsorted(ot, tt, "left"), 0, len(ot) - 1)
    jh = np.clip(np.searchsorted(ot, tt + h * MIN_MS, "left"), 0, len(ot) - 1)
    return np.log(cl[jh] / cl[j0])


def cohort(df, label, store):
    if len(df) < 5:
        return
    d = {"n": len(df), "h": {}}
    print(f"\n=== {label} (n={len(df)}) ===")
    for h in HS:
        sgn = df["sign"].values.astype(float)
        r = df[f"r{h}"].values
        rel = df[f"rel{h}"].values
        ok = np.isfinite(r) & (sgn != 0)
        if ok.sum() < 5:
            continue
        real = float(np.mean(sgn[ok] * r[ok]))
        real_rel = float(np.nanmean(sgn[ok] * rel[ok]))
        # random-timestamp null on same (symbol, sign)
        M = np.vstack([sample_rs(s, h, NULL_B) * g for s, g in zip(df["sym"].values[ok], sgn[ok])])
        nm = np.nanmean(M, axis=0)
        nmu, nsd = float(np.nanmean(nm)), float(np.nanstd(nm))
        z = (real - nmu) / (nsd + 1e-12)
        pct = float(np.mean(nm < real))
        row = {"n": int(ok.sum()), "dir_ret_bp": round(real * 1e4, 2),
               "null_mean_bp": round(nmu * 1e4, 2), "null_sd_bp": round(nsd * 1e4, 2),
               "z": round(z, 2), "pctile": round(pct, 3), "rel_dir_ret_bp": round(real_rel * 1e4, 2)}
        d["h"][h] = row
        print(f"  h={h:>3}m dir_ret={row['dir_ret_bp']:+8.2f}bp | null {row['null_mean_bp']:+7.2f}±{row['null_sd_bp']:.2f} "
              f"z={row['z']:+.2f} pctile={row['pctile']:.3f} | BTC-rel={row['rel_dir_ret_bp']:+8.2f}bp n={row['n']}")
    store[label] = d


store = {"n_tradeable": len(R), "null_B": NULL_B}
cohort(R[(R["source"] == "upbit") & (R["kind"] == "list")], "UPBIT list(+)", store)
cohort(R[(R["source"] == "upbit") & (R["sign"] < 0)], "UPBIT delist+caution(-)", store)
cohort(R[(R["source"] == "binance") & (R["kind"] == "list")], "BINANCE will-list(+)", store)
cohort(R[(R["source"] == "binance") & (R["kind"] == "airdrop")], "BINANCE airdrop(+)", store)
cohort(R[R["sign"] != 0], "ALL signed combined", store)
json.dump(store, open("C:/Dev/sb-data-poc/ev_b_control.json", "w"), indent=2)
print("\nEV_B_CONTROL_DONE", flush=True)
