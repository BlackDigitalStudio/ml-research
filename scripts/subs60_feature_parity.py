#!/usr/bin/env python3
"""FULL feature-level cross-source parity: convert chronos (our recorder) depth+trades to the
cryptolake raw format, run feature_builder on BOTH (chronos-converted vs cryptolake) over the
overlap window, and compare the 64 X features. This settles whether the cryptolake-trained model
will get matching features live from our recorder -- the existential train/live gate, beyond
top-of-book. Note: cryptolake trades are aggTrades (~169k/day), chronos are raw trades (~2.2M/day);
this measures whether that (and the snapshot-density difference) moves the features.
Run on hd2-feats-003 (has /tmp/feature_builder). DAY default 2026-06-05.
"""
import os, subprocess, tempfile, sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-06-05"
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
FB = "/tmp/feature_builder"; NS = 1_000_000_000; GRID_S = 1.0; LV = 20
SYMF = "DOGE-USDT-PERP"
WARMUP_NS = int(os.environ.get("WARMUP", "3600")) * NS  # chronos history pad before cl window (lookback warmup)
bk = storage.Client(project=PROJ).bucket(BUCKET)
TD = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")


def dl_first(prefix, dst):
    n = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not n:
        return None
    bk.blob(n).download_to_filename(dst); return dst


def book_ts_mid(path):
    t = pq.read_table(path, columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts = t["timestamp"].to_numpy().astype(np.int64)
    mid = 0.5 * (t["bid_0_price"].to_numpy().astype(np.float64) + t["ask_0_price"].to_numpy().astype(np.float64))
    o = np.argsort(ts, kind="stable"); return ts[o], mid[o]


def run_fb(book, trades, tag):
    bts, mid = book_ts_mid(book)
    grid = np.arange(bts[0] + 120 * NS, bts[-1] - 70 * NS, int(GRID_S * NS), dtype=np.int64)
    idx = np.unique(np.clip(np.searchsorted(bts, grid, "right") - 1, 0, len(bts) - 1)).astype(np.int64)
    ip = f"{TD}/idx_{tag}.npy"; np.save(ip, idx); op = f"{TD}/feat_{tag}.npy"
    r = subprocess.run([FB, "--depth", book, "--indices", ip, "--out", op, "--trades", trades], capture_output=True, text=True)
    assert r.returncode == 0, f"FB {tag} failed: {r.stderr[-200:]}"
    return bts[idx], np.load(op).astype(np.float64)


# ---- cryptolake side (reference) ----
book_cl = dl_first(f"raw/book/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{TD}/book_cl.parquet")
trd_cl = dl_first(f"raw/trades/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{TD}/trd_cl.parquet")
assert book_cl and trd_cl, "missing cryptolake book/trades"
cl_lo, cl_hi = book_ts_mid(book_cl)[0][[0, -1]]
print(f"[parity {DAY}] cryptolake book span {(cl_hi-cl_lo)/3.6e12:.1f}h ({cl_lo}..{cl_hi})", flush=True)

# ---- convert chronos -> cryptolake format (restricted to cryptolake time window) ----
cols = ["timestamp", "receipt_timestamp", "sequence_number"]
for i in range(LV):
    cols += [f"bid_{i}_price", f"bid_{i}_size", f"ask_{i}_price", f"ask_{i}_size"]
acc = {c: [] for c in cols}
for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/depth_snapshot/"):
    if not b.name.endswith(".parquet"):
        continue
    bk.blob(b.name).download_to_filename(f"{TD}/d.parquet")
    df = pq.read_table(f"{TD}/d.parquet", columns=["exchange_event_ts_us", "local_ts_us", "bid_prices", "bid_qtys", "ask_prices", "ask_qtys"]).to_pandas()
    df = df[df["exchange_event_ts_us"].notna()]
    ts = (df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy()
    m = (ts >= cl_lo - WARMUP_NS) & (ts <= cl_hi + NS)  # cryptolake window + small pad
    if not m.any():
        continue
    df = df[m]; ts = ts[m]
    acc["timestamp"].append(ts); acc["receipt_timestamp"].append((df["local_ts_us"].astype("int64") * 1000).to_numpy())
    acc["sequence_number"].append(np.zeros(len(ts), np.int64))
    for i in range(LV):
        acc[f"bid_{i}_price"].append(df["bid_prices"].apply(lambda x: float(x[i]) if x is not None and len(x) > i and x[i] is not None else 0.0).to_numpy())
        acc[f"bid_{i}_size"].append(df["bid_qtys"].apply(lambda x: float(x[i]) if x is not None and len(x) > i and x[i] is not None else 0.0).to_numpy())
        acc[f"ask_{i}_price"].append(df["ask_prices"].apply(lambda x: float(x[i]) if x is not None and len(x) > i and x[i] is not None else 0.0).to_numpy())
        acc[f"ask_{i}_size"].append(df["ask_qtys"].apply(lambda x: float(x[i]) if x is not None and len(x) > i and x[i] is not None else 0.0).to_numpy())
arrs = {c: np.concatenate(acc[c]) for c in cols}
order = np.argsort(arrs["timestamp"], kind="stable"); arrs = {c: arrs[c][order] for c in cols}
if os.environ.get("DOWNSAMPLE", "0") == "1":  # match cryptolake snapshot density
    n = len(arrs["timestamp"]); k = max(1, round(n / 20000))
    keep = np.arange(0, n, k); arrs = {c: arrs[c][keep] for c in cols}
    print(f"  DOWNSAMPLE: chronos book {n} -> {len(keep)} (every {k}th, ~cryptolake density)", flush=True)
arrs["sequence_number"] = np.arange(len(arrs["timestamp"]), dtype=np.int64)
pq.write_table(pa.table({c: arrs[c] for c in cols}), f"{TD}/book_ch.parquet")
print(f"  chronos book converted: {len(arrs['timestamp'])} snapshots in window", flush=True)

# chronos trades -> cryptolake format
tcols = ["side", "amount", "price", "id", "timestamp", "receipt_timestamp"]
tacc = {c: [] for c in tcols}
for b in bk.client.list_blobs(bk, prefix="tmp_chronos_parity/DOGEUSDT/trade/"):
    if not b.name.endswith(".parquet"):
        continue
    bk.blob(b.name).download_to_filename(f"{TD}/t.parquet")
    df = pq.read_table(f"{TD}/t.parquet", columns=["exchange_event_ts_us", "local_ts_us", "trade_id", "price", "qty", "is_buyer_maker"]).to_pandas()
    df = df[df["exchange_event_ts_us"].notna()]
    ts = (df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy()
    m = (ts >= cl_lo - WARMUP_NS) & (ts <= cl_hi + NS)
    if not m.any():
        continue
    df = df[m]; ts = ts[m]
    tacc["side"].append(np.where(df["is_buyer_maker"].to_numpy(), "sell", "buy"))
    tacc["amount"].append(df["qty"].astype("float64").to_numpy()); tacc["price"].append(df["price"].astype("float64").to_numpy())
    tacc["id"].append(df["trade_id"].astype("int64").to_numpy()); tacc["timestamp"].append(ts)
    tacc["receipt_timestamp"].append((df["local_ts_us"].astype("int64") * 1000).to_numpy())
tarr = {c: np.concatenate(tacc[c]) for c in tcols}
to = np.argsort(tarr["timestamp"], kind="stable"); tarr = {c: tarr[c][to] for c in tcols}
AGG = os.environ.get("AGG", "0") == "1"
if AGG:  # approximate Binance aggTrade: merge consecutive same (side, price) raw trades -> sum qty
    sidev = tarr["side"]; pr = tarr["price"]
    newg = np.ones(len(pr), bool); newg[1:] = (sidev[1:] != sidev[:-1]) | (pr[1:] != pr[:-1])
    gid = np.cumsum(newg) - 1; nG = int(gid[-1]) + 1
    fi = np.searchsorted(gid, np.arange(nG))
    amt = np.zeros(nG); np.add.at(amt, gid, tarr["amount"])
    tarr = {"side": tarr["side"][fi], "amount": amt, "price": tarr["price"][fi], "id": tarr["id"][fi],
            "timestamp": tarr["timestamp"][fi], "receipt_timestamp": tarr["receipt_timestamp"][fi]}
pq.write_table(pa.table({c: tarr[c] for c in tcols}), f"{TD}/trd_ch.parquet")
print(f"  chronos trades converted: {len(tarr['timestamp'])} trades in window (AGG={AGG})", flush=True)

# ---- run feature_builder on both ----
td_cl, X_cl = run_fb(book_cl, trd_cl, "cl")
td_ch, X_ch = run_fb(f"{TD}/book_ch.parquet", f"{TD}/trd_ch.parquet", "ch")
print(f"\n  feature_builder: cryptolake X{X_cl.shape} (n={len(td_cl)}) | chronos X{X_ch.shape} (n={len(td_ch)})", flush=True)

# ---- match by timestamp (within 100ms) and compare ----
j = np.clip(np.searchsorted(td_ch, td_cl), 1, len(td_ch) - 1)
dl = np.abs(td_ch[j - 1] - td_cl); dr = np.abs(td_ch[j] - td_cl); jn = np.where(dl < dr, j - 1, j); dt = np.minimum(dl, dr)
ok = dt < 100_000_000  # 100ms
A = X_cl[ok]; B = X_ch[jn[ok]]
print(f"\n=== FULL FEATURE PARITY ({DAY}, {int(ok.sum())} matched decision pts) ===", flush=True)
sd = X_cl.std(0) + 1e-9
absd = np.abs(A - B); reld = absd / sd
print(f"  per-feature |ΔX|/std  : median={np.median(reld):.4f}  mean={reld.mean():.4f}  p95={np.percentile(reld,95):.4f}", flush=True)
print(f"  features with median |ΔX|/std < 0.02 : {int((np.median(reld,0)<0.02).sum())}/{X_cl.shape[1]}", flush=True)
worst = np.argsort(-np.median(reld, 0))[:8]
print(f"  worst-8 features (col: median |ΔX|/std): {[(int(c), round(float(np.median(reld[:,c])),3)) for c in worst]}", flush=True)
ex = (absd < 1e-4).mean(0)
print(f"  features ~exact (|ΔX|<1e-4) on >95% pts: {int((ex>0.95).sum())}/{X_cl.shape[1]}", flush=True)
good = float((np.median(reld, 0) < 0.05).mean())
print(f"  -> {'FEATURE PARITY OK' if good > 0.9 else 'PARTIAL -- '+str(int((np.median(reld,0)>=0.05).sum()))+' feats drift (likely aggTrade vs raw-trade / depth-density)'} ({100*good:.0f}% feats within 0.05 std)", flush=True)
