"""EV-B (HDATA cell EV-B): exchange-listing announcement event-study.
Pulls Upbit + Binance listing/delisting announcements (precise publish ts),
matches the named ticker -> Binance USDT perp (524-sym universe), and measures
the forward reaction {1,5,15,60}m from the announcement ts (entry = first 1m
close >= ts; events where the perp did not yet exist are dropped).

Sign is KNOWN from the announcement (listing=+, delist/caution=-), so this is
reported EXP-7-style: per cohort (source x kind), dir_ret = sign*r_h, win-rate,
med|r|, clear_cost@0.13%. Raw announcements -> GCS + local parquet.
"""
import io
import re
import json
import time
import numpy as np
import pandas as pd
import requests
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJ = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
KLINES = "free_v1/klines_1m/"
OUT_GCS = "free_v1/orthogonal/events/listings/announcements.parquet"
OUT_LOCAL = "C:/Dev/sb-data-poc/ev_b_announcements.parquet"
COST = 0.0013
HS = [1, 5, 15, 60]
TOL_MS = 120_000
MIN_MS = 60_000
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}
TICK_RE = re.compile(r"\(([A-Z0-9]{2,15})\)")
STOP = {"SPOT", "PERPETUAL", "PERP", "MARGIN", "EARN", "CONVERT", "LOAN", "FUTURES",
        "HODLER", "AIRDROP", "VIP", "KRW", "USD", "USDT", "USDC", "API", "ETF", "NFT", "P2P"}


def upbit_sign(title):
    t = title
    low = title.lower()
    if "lifted" in low or "해제" in t:            # warning removed -> neutral
        return "other", 0
    if ("termination of trading support" in low or "end of trading support" in low
            or "delist" in low or "종료" in t or "폐지" in t):
        return "delist", -1
    if "investment warning" in low or "유의" in t:
        return "caution", -1
    if ("market support for" in low or "추가" in t or "디지털 자산" in t
            or "신규" in t or "개시" in t):
        return "list", 1
    return "other", 0


def bnc_sign(title):
    t = title.lower()
    if "delist" in t or "will remove" in t or "will cease" in t or "removal of" in t:
        return "delist", -1
    if "will list" in t:
        return "list", 1
    if "perpetual" in t or "futures will" in t:
        return "perp_list", 1
    if "introducing" in t and ("hodler" in t or "airdrop" in t or "launchpool" in t or "megadrop" in t):
        return "airdrop", 1
    if "will add" in t or "will support" in t:
        return "add", 1
    return "other", 0


def tickers(title):
    return [t for t in TICK_RE.findall(title) if t not in STOP and not t.isdigit()]


def pull_upbit():
    rows = []
    page = 1
    while page <= 40:
        try:
            r = requests.get("https://api-manager.upbit.com/api/v1/announcements",
                             params={"os": "web", "page": page, "per_page": 30, "category": "trade"},
                             headers=UA, timeout=25)
            notices = r.json()["data"]["notices"]
        except Exception as e:
            print("upbit page", page, "ERR", str(e)[:120]); break
        if not notices:
            break
        for n in notices:
            ts = n.get("listed_at") or n.get("first_listed_at")
            rows.append({"source": "upbit", "ts": ts, "title": n.get("title", ""), "ref": n.get("id")})
        page += 1
        time.sleep(0.25)
    print(f"upbit pulled={len(rows)} pages={page-1}", flush=True)
    return rows


def pull_binance():
    rows = []
    page = 1
    while page <= 60:
        arts = None
        for attempt in range(4):  # retry transient JSON/empty responses (rate-limit blips)
            try:
                r = requests.get("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query",
                                 params={"type": 1, "catalogId": 48, "pageNo": page, "pageSize": 50},
                                 headers={**UA, "clienttype": "web", "lang": "en"}, timeout=25)
                cats = r.json()["data"]["catalogs"]
                arts = cats[0]["articles"] if cats else []
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if arts is None:
            print(f"binance page {page} gave up after retries", flush=True); break
        if not arts:
            break
        for a in arts:
            rows.append({"source": "binance", "ts": a.get("releaseDate"),
                         "title": a.get("title", ""), "ref": a.get("code")})
        page += 1
        time.sleep(0.4)
    print(f"binance cat48 pulled_total={len(rows)} pages={page-1}", flush=True)
    return rows


# ---- pull announcements ----
raw = pull_upbit() + pull_binance()
ev = []
for r in raw:
    if r["ts"] is None:
        continue
    if r["source"] == "upbit":
        ts = pd.to_datetime(r["ts"], utc=True)
        kind, sign = upbit_sign(r["title"])
    else:
        ts = pd.to_datetime(r["ts"], unit="ms", utc=True)
        kind, sign = bnc_sign(r["title"])
    for tk in tickers(r["title"]):
        ev.append({"source": r["source"], "ts": ts, "ticker": tk, "kind": kind, "sign": sign,
                   "sym": tk + "USDT", "title": r["title"], "ref": r["ref"]})
E = pd.DataFrame(ev).drop_duplicates(["source", "ts", "ticker"]).reset_index(drop=True)
print(f"events(ticker-level)={len(E)} | by source: {E['source'].value_counts().to_dict()}", flush=True)
print(f"kind x sign:\n{E.groupby(['source','kind']).size()}", flush=True)

# save raw announcements -> local + GCS
Esave = E.assign(ts=E["ts"].astype(str), ref=E["ref"].astype(str))
Esave.to_parquet(OUT_LOCAL, index=False)
buf = io.BytesIO()
Esave.to_parquet(buf, index=False)
buf.seek(0)
storage.Client(project=PROJ).bucket(BUCKET).blob(OUT_GCS).upload_from_file(buf, content_type="application/octet-stream")
print(f"saved announcements -> {OUT_LOCAL} + gs://{BUCKET}/{OUT_GCS}", flush=True)

# ---- match to perp universe + load klines for matched symbols ----
fs = pafs.GcsFileSystem()
universe = {b.name.split("/")[-1].replace(".parquet", "")
            for b in storage.Client(project=PROJ).list_blobs(BUCKET, prefix=KLINES) if b.name.endswith(".parquet")}
E["has_perp"] = E["sym"].isin(universe)
need = sorted(E.loc[E["has_perp"], "sym"].unique())
print(f"matched symbols={len(need)} of universe {len(universe)}; events w/ perp={int(E['has_perp'].sum())}", flush=True)


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
        if done % 50 == 0:
            print(f"  klines ...{done}/{len(need)}", flush=True)
print(f"loaded klines for {len(K)} symbols", flush=True)


def close_at(ot, cl, tt):
    j = np.searchsorted(ot, tt, "left")
    if j < len(ot) and abs(ot[j] - tt) <= TOL_MS:
        return cl[j]
    if j > 0 and abs(ot[j - 1] - tt) <= TOL_MS:
        return cl[j - 1]
    return np.nan


# ---- build forward returns per event ----
recs = []
for _, e in E[E["has_perp"]].iterrows():
    ot, cl = K.get(e["sym"], (None, None))
    if ot is None:
        continue
    t0 = int(e["ts"].value // 1_000_000)  # ns -> ms
    if t0 < ot[0] or t0 > ot[-1]:
        continue  # perp did not exist at announcement (or after data end)
    c0 = close_at(ot, cl, t0)
    if not np.isfinite(c0):
        continue
    row = {"source": e["source"], "kind": e["kind"], "sign": e["sign"], "sym": e["sym"]}
    for h in HS:
        ch = close_at(ot, cl, t0 + h * MIN_MS)
        row[f"r{h}"] = np.log(ch / c0) if np.isfinite(ch) else np.nan
    recs.append(row)
R = pd.DataFrame(recs)
print(f"\ntradeable events (perp existed at ts)={len(R)} | by source/kind:\n{R.groupby(['source','kind']).size()}", flush=True)


def report(df, label, store):
    if len(df) < 5:
        return
    d = {"n": len(df), "sign_mix": df["sign"].value_counts().to_dict(), "h": {}}
    print(f"\n=== {label} (n={len(df)}, signs={d['sign_mix']}) ===")
    sgn = df["sign"].values
    for h in HS:
        r = df[f"r{h}"].values
        ok = np.isfinite(r) & (sgn != 0)
        if ok.sum() < 5:
            continue
        dr = sgn[ok] * r[ok]
        bull = r[ok][sgn[ok] > 0]
        bear = r[ok][sgn[ok] < 0]
        row = {"n": int(ok.sum()), "dir_ret_bp": float(np.mean(dr) * 1e4), "win": float(np.mean(dr > 0)),
               "med_abs_pct": float(np.median(np.abs(r[ok])) * 100), "clear_cost": float(np.mean(np.abs(r[ok]) > COST)),
               "bull_bp": float(np.mean(bull) * 1e4) if len(bull) else None,
               "bear_bp": float(np.mean(bear) * 1e4) if len(bear) else None}
        d["h"][h] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        print(f"  h={h:>3}m dir_ret={row['dir_ret_bp']:+.2f}bp win={row['win']:.3f} "
              f"med|r|={row['med_abs_pct']:.3f}% clear={row['clear_cost']:.3f} "
              f"bull={row['bull_bp']} bear={row['bear_bp']} n={row['n']}")
    store[label] = d


store = {"n_announcements": len(E), "n_tradeable": len(R),
         "by_source_kind": {f"{s}/{k}": int(v) for (s, k), v in R.groupby(["source", "kind"]).size().items()}}
report(R[R["source"] == "upbit"], "UPBIT all", store)
report(R[(R["source"] == "upbit") & (R["kind"] == "list")], "UPBIT list(+)", store)
report(R[(R["source"] == "upbit") & (R["sign"] < 0)], "UPBIT delist+caution(-)", store)
report(R[R["source"] == "binance"], "BINANCE all", store)
report(R[(R["source"] == "binance") & (R["kind"] == "list")], "BINANCE will-list(+)", store)
report(R[(R["source"] == "binance") & (R["kind"] == "add")], "BINANCE will-add(+)", store)
report(R[(R["source"] == "binance") & (R["kind"] == "airdrop")], "BINANCE airdrop(+)", store)
report(R, "ALL cohorts combined", store)
json.dump(store, open("C:/Dev/sb-data-poc/ev_b_listings.json", "w"), indent=2)
print("\nEV_B_LISTINGS_DONE", flush=True)
