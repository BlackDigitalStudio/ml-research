#!/usr/bin/env python3
"""Quantify whether the book-SAMPLING feature drift (cl-sparse vs chronos-dense) actually changes
the MODEL OUTPUT. Runs the full pipeline (feature_builder -> feat71 -> vol-norm -> A/Bg -> AxB score)
on 2026-06-05 two ways: cryptolake book/trades vs chronos-converted book/trades. feat71/norm/models
are IDENTICAL for both -> only the 64 X features differ (the sampling drift). Compares pA, pBg, AxB
score, and would-trade decisions. If the model output agrees -> the sampling drift is not critical.
Run on hd2-feats-003 (has /tmp/feature_builder).
"""
import os, subprocess, tempfile, io, json, datetime
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage
import xgboost as xgb
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
FB = os.environ.get("FB_BIN", "/tmp/feature_builder"); BUNDLE = os.environ.get("BUNDLE_DIR", "research_runs/deploy/DOGE"); NS = 1_000_000_000; GRID_S = 1.0; LV = 20; DAY = "2026-06-05"
SYMF = "DOGE-USDT-PERP"; BTC = "BTC-USDT-PERP"; FEATS = "feats_sub60"
bk = storage.Client(project=PROJ).bucket(BUCKET); TD = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")


def first(prefix, dst):
    n = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    return (bk.blob(n).download_to_filename(dst) or dst) if n else None


def bts_of(path):
    t = pq.read_table(path, columns=["timestamp"]); return np.sort(t["timestamp"].to_numpy().astype(np.int64))


def run_fb(book, trades, tag):
    bts = bts_of(book)
    grid = np.arange(bts[0] + 120 * NS, bts[-1] - 70 * NS, int(GRID_S * NS), dtype=np.int64)
    idx = np.unique(np.clip(np.searchsorted(bts, grid, "right") - 1, 0, len(bts) - 1)).astype(np.int64)
    np.save(f"{TD}/idx_{tag}.npy", idx); op = f"{TD}/f_{tag}.npy"
    r = subprocess.run([FB, "--depth", book, "--indices", f"{TD}/idx_{tag}.npy", "--out", op, "--trades", trades], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-200:]
    return bts[idx], np.load(op).astype(np.float64)


def feat71(dtd, X, bt, bm):
    nb = len(bt); i = np.clip(np.searchsorted(bt, dtd, "right") - 1, 0, nb - 1); bl = []
    for W in (5, 30, 60):
        j = np.clip(np.searchsorted(bt, dtd - int(W * NS), "right") - 1, 0, nb - 1)
        a = bm[j]; b = bm[i]
        bl.append(np.where((a > 0) & (b > 0), np.log(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0)), 0.0) * 1e4)
    h = ((dtd / NS) % 86400.0) / 3600.0; hf = h % 8.0
    tod = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * hf / 8), np.cos(2 * np.pi * hf / 8)]
    return np.concatenate([X, np.stack(bl, 1), np.stack(tod, 1)], axis=1).astype(np.float32)


# bundle
refs = np.load(io.BytesIO(bk.blob(f"{BUNDLE}/refs.npz").download_as_bytes()))
meta = json.loads(bk.blob(f"{BUNDLE}/meta.json").download_as_bytes()); KNORM = meta["KNORM"]
A = xgb.Booster(); Bg = xgb.Booster()
for nm, mdl in [("A", A), ("Bg", Bg)]:
    p = f"{TD}/{nm}.json"; bk.blob(f"{BUNDLE}/{nm}.json").download_to_filename(p); mdl.load_model(p)
gstd = refs["gstd"].astype(np.float64); sA = refs["sA"]; sBg = refs["sBg"]
day_mean = refs["day_mean"].astype(np.float64); day_var = refs["day_var"].astype(np.float64)
mu = day_mean[-KNORM:].mean(0); sd = np.maximum(np.sqrt(np.maximum(day_var[-KNORM:].mean(0), 0)), 0.2 * gstd + 1e-9)
# BTC mid (feat71)
tds, mds = [], []
for off in (-1, 0, 1):
    d = (datetime.date.fromisoformat(DAY) + datetime.timedelta(days=off)).isoformat()
    try:
        z = np.load(io.BytesIO(bk.blob(f"{FEATS}/{BTC}/{d}.npz").download_as_bytes())); tds.append(z["td"].astype(np.int64)); mds.append(z["mid"].astype(np.float64))
    except Exception: pass
bt = np.concatenate(tds) if tds else np.array([0], np.int64); bm = np.concatenate(mds) if mds else np.array([1.0])  # no BTC feats -> btc_lead=0 (identical for both paths, cancels in the cl-vs-chronos comparison)
o = np.argsort(bt); bt, bm = bt[o], bm[o]
cdf = lambda x, ref: np.searchsorted(ref, x, "right") / max(len(ref), 1)


def signal(td, X):
    F = feat71(td, X, bt, bm); Fn = ((F - mu) / sd).astype(np.float32)
    pA = A.predict(xgb.DMatrix(Fn)); pBg = Bg.predict(xgb.DMatrix(Fn))
    score = cdf(pA, sA) * cdf(np.abs(pBg - 0.5), sBg)
    return pA, pBg, score


# cryptolake path
book_cl = first(f"raw/book/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{TD}/book_cl.parquet")
trd_cl = first(f"raw/trades/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{TD}/trd_cl.parquet")
cl_lo, cl_hi = bts_of(book_cl)[[0, -1]]
td_cl, X_cl = run_fb(book_cl, trd_cl, "cl")
pA_cl, pBg_cl, sc_cl = signal(td_cl, X_cl)

# chronos path: convert depth+trades (in cl window + warmup) to cl format
cols = ["timestamp", "receipt_timestamp", "sequence_number"] + [f"{s}_{i}_{f}" for i in range(LV) for s in ("bid", "ask") for f in ("price", "size")]
acc = {c: [] for c in cols}
for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/depth_snapshot/"):
    if not b.name.endswith(".parquet"): continue
    bk.blob(b.name).download_to_filename(f"{TD}/d.parquet")
    df = pq.read_table(f"{TD}/d.parquet", columns=["exchange_event_ts_us", "local_ts_us", "bid_prices", "bid_qtys", "ask_prices", "ask_qtys"]).to_pandas()
    df = df[df["exchange_event_ts_us"].notna()]; ts = (df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy()
    m = (ts >= cl_lo - 3600 * NS) & (ts <= cl_hi + NS)
    if not m.any(): continue
    df = df[m]; acc["timestamp"].append(ts[m]); acc["receipt_timestamp"].append((df["local_ts_us"].astype("int64") * 1000).to_numpy()); acc["sequence_number"].append(np.zeros(int(m.sum()), np.int64))
    for i in range(LV):
        for s, col in (("bid", "bid_prices"), ("ask", "ask_prices")):
            acc[f"{s}_{i}_price"].append(df[col].apply(lambda x: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
        for s, col in (("bid", "bid_qtys"), ("ask", "ask_qtys")):
            acc[f"{s}_{i}_size"].append(df[col].apply(lambda x: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
ar = {c: np.concatenate(acc[c]) for c in cols}; od = np.argsort(ar["timestamp"], kind="stable"); ar = {c: ar[c][od] for c in cols}
if os.environ.get("SAMPLECL", "0") == "1":  # sample chronos at cryptolake's snapshot cadence (feed data like cryptolake)
    cl_ts = bts_of(book_cl); cht = ar["timestamp"]
    jj = np.clip(np.searchsorted(cht, cl_ts), 1, len(cht) - 1)
    keep = np.unique(np.where(np.abs(cht[jj - 1] - cl_ts) < np.abs(cht[jj] - cl_ts), jj - 1, jj))
    ar = {c: ar[c][keep] for c in cols}
    print(f"  SAMPLECL: chronos book sampled to cryptolake cadence -> {len(keep)} snapshots", flush=True)
if os.environ.get("DEDUP", "0") == "1":  # LIVE-REALISTIC rule: keep snapshot only when L0 (top-of-book price/size) changes
    bp = ar["bid_0_price"]; bs = ar["bid_0_size"]; ap = ar["ask_0_price"]; az = ar["ask_0_size"]
    chg = np.ones(len(bp), bool); chg[1:] = (bp[1:] != bp[:-1]) | (bs[1:] != bs[:-1]) | (ap[1:] != ap[:-1]) | (az[1:] != az[:-1])
    keep = np.where(chg)[0]; ar = {c: ar[c][keep] for c in cols}
    print(f"  DEDUP(L0-change): chronos book -> {len(keep)} snapshots ({len(keep)/4.5/3600:.2f}/s) [no cl timestamps used]", flush=True)
MININT = int(os.environ.get("MININT", "0"))
if MININT > 0:  # LIVE-REALISTIC: keep a snapshot only if >= MININT ms since last kept (-> matches cl DENSITY, not exact times)
    mi = MININT * 1_000_000; ts = ar["timestamp"]; keep = [0]; last = ts[0]
    for i in range(1, len(ts)):
        if ts[i] - last >= mi:
            keep.append(i); last = ts[i]
    keep = np.array(keep); ar = {c: ar[c][keep] for c in cols}
    print(f"  MININT={MININT}ms: chronos book -> {len(keep)} snapshots ({len(keep)/4.5/3600:.2f}/s) [no cl timestamps used]", flush=True)
if os.environ.get("TRADESAMPLE", "0") == "1":  # LIVE-REALISTIC: snapshot the book at each TRADE event (hypothesis: cl's rule)
    tts = []
    for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/trade/"):
        if not b.name.endswith(".parquet"): continue
        bk.blob(b.name).download_to_filename(f"{TD}/tt.parquet")
        e = pq.read_table(f"{TD}/tt.parquet", columns=["exchange_event_ts_us"]).to_pandas(); e = e[e["exchange_event_ts_us"].notna()]
        tts.append((e["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
    tts = np.sort(np.concatenate(tts)); cht = ar["timestamp"]
    jj = np.clip(np.searchsorted(cht, tts), 1, len(cht) - 1)
    keep = np.unique(np.where(np.abs(cht[jj - 1] - tts) < np.abs(cht[jj] - tts), jj - 1, jj))
    ar = {c: ar[c][keep] for c in cols}
    print(f"  TRADESAMPLE: book sampled at trade events -> {len(keep)} ({len(keep)/4.5/3600:.2f}/s) [replicable live]", flush=True)
ar["sequence_number"] = np.arange(len(ar["timestamp"]), dtype=np.int64)
pq.write_table(pa.table({c: ar[c] for c in cols}), f"{TD}/book_ch.parquet")
tc = ["side", "amount", "price", "id", "timestamp", "receipt_timestamp"]; ta = {c: [] for c in tc}
for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/trade/"):
    if not b.name.endswith(".parquet"): continue
    bk.blob(b.name).download_to_filename(f"{TD}/t.parquet")
    df = pq.read_table(f"{TD}/t.parquet", columns=["exchange_event_ts_us", "local_ts_us", "trade_id", "price", "qty", "is_buyer_maker"]).to_pandas()
    df = df[df["exchange_event_ts_us"].notna()]; ts = (df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy()
    m = (ts >= cl_lo - 3600 * NS) & (ts <= cl_hi + NS)
    if not m.any(): continue
    df = df[m]; ta["side"].append(np.where(df["is_buyer_maker"].to_numpy(), "sell", "buy")); ta["amount"].append(df["qty"].astype("float64").to_numpy())
    ta["price"].append(df["price"].astype("float64").to_numpy()); ta["id"].append(df["trade_id"].astype("int64").to_numpy()); ta["timestamp"].append(ts[m]); ta["receipt_timestamp"].append((df["local_ts_us"].astype("int64") * 1000).to_numpy())
tar = {c: np.concatenate(ta[c]) for c in tc}; to = np.argsort(tar["timestamp"], kind="stable"); tar = {c: tar[c][to] for c in tc}
pq.write_table(pa.table({c: tar[c] for c in tc}), f"{TD}/trd_ch.parquet")
td_ch, X_ch = run_fb(f"{TD}/book_ch.parquet", f"{TD}/trd_ch.parquet", "ch")
pA_ch, pBg_ch, sc_ch = signal(td_ch, X_ch)

# match decision points, compare MODEL OUTPUT
j = np.clip(np.searchsorted(td_ch, td_cl), 1, len(td_ch) - 1); dl = np.abs(td_ch[j - 1] - td_cl); dr = np.abs(td_ch[j] - td_cl)
jn = np.where(dl < dr, j - 1, j); ok = np.minimum(dl, dr) < 100_000_000
pa1, pa2 = pA_cl[ok], pA_ch[jn[ok]]; pb1, pb2 = pBg_cl[ok], pBg_ch[jn[ok]]; s1, s2 = sc_cl[ok], sc_ch[jn[ok]]
print(f"\n=== MODEL-OUTPUT IMPACT of sampling drift ({DAY}, {int(ok.sum())} matched decision pts) ===", flush=True)
print(f"  pA   : corr={np.corrcoef(pa1,pa2)[0,1]:.4f} | med|Δ|={np.median(np.abs(pa1-pa2)):.4f} | rank-corr~", flush=True)
print(f"  pBg  : corr={np.corrcoef(pb1,pb2)[0,1]:.4f} | med|Δ|={np.median(np.abs(pb1-pb2)):.4f}", flush=True)
print(f"  side agree (sign pBg-0.5): {100*np.mean((pb1>=0.5)==(pb2>=0.5)):.1f}%", flush=True)
print(f"  AxB score: corr={np.corrcoef(s1,s2)[0,1]:.4f} | med|Δ|={np.median(np.abs(s1-s2)):.4f}", flush=True)
for q in (0.95, 0.99):
    thr = np.quantile(sc_cl, q); d1 = s1 >= thr; d2 = s2 >= thr
    inter = (d1 & d2).sum(); union = (d1 | d2).sum()
    print(f"  would-trade @top-{int((1-q)*100)}%: cl={int(d1.sum())} ch={int(d2.sum())} | overlap(Jaccard)={inter/max(union,1):.2f}", flush=True)
