#!/usr/bin/env python3
"""Shadow parity gate: live axb_live decisions vs the frozen offline pipeline, same day.

For DAY (YYYYMMDD): (1) run the offline recorder-EV day machinery (recorder depth_snapshot +
agg_trade + liquidation -> CL format -> husdc build_samples + grid_sim_exitdbg + robust FB ->
feat71 -> frozen norm -> A/Bg -> score) saving per-sample TIMESTAMPS; (2) load the live shadow
decision log (research_runs/axb_shadow/DOGE/decisions/DAY.jsonl); (3) match live decisions to
offline samples by nearest book-ts (tolerance MATCH_TOL_S) and report:
  - score/pA/pBg correlation + mean|delta| at matched pairs (live @depth20 vs recorder diff-book);
  - side agreement; per-budget take-rate live vs offline;
  - EV of the LIVE-taken trades using the matched offline maker pegged-exit labels (filled only).
Gate: corr(score) >= 0.95, side agreement >= 0.95 on takes, live-take EV consistent with offline.
Run on hd2-feats-003 (needs /tmp/fb_target + /tmp/husdc_target binaries, see subs60_recorder_ev).
Usage: python3 axb_shadow_eval.py DAY [BUDGET...]
"""
import io, json, os, subprocess, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage
import xgboost as xgb

DAY = sys.argv[1]
BUDGETS = [float(x) for x in (sys.argv[2:] or ["5", "10", "20", "40"])]
PROJ = "project-0998ac51-36ba-445c-bc7"
MKT = "market-data-0998ac51"; REC = "recorder-data-asia-0998ac51"
RECP = "chronos/scalper-recorder/binance_futures/DOGEUSDT"
FB = os.environ.get("FB_BIN", "/tmp/fb_target/release/feature_builder")
BS = os.environ.get("BS_BIN", "/tmp/husdc_target/release/build_samples")
GRID = os.environ.get("GRID_BIN", "/tmp/husdc_target/release/grid_sim_exitdbg")
BUNDLE = "research_runs/deploy_robust2/DOGE"
SHADOW = "research_runs/axb_shadow/DOGE"
NS = 1_000_000_000; LV = 20; TARGET = 8000; H_TICKS = 700; ENTRY_WIN = 120; WINDOW = 50
CFGS = [{"tp": 50.0, "sl": 50.0, "to": 141, "par": False, "tr": False},
        {"tp": 50.0, "sl": 50.0, "to": 282, "par": False, "tr": False},
        {"tp": 50.0, "sl": 50.0, "to": 563, "par": False, "tr": False}]
CFGIDX, QM = 1, 1.0
MATCH_TOL_S = 5.5
TD = os.environ.get("WORKDIR", "/home/delmi/recev_work"); os.makedirs(TD, exist_ok=True)
cl = storage.Client(project=PROJ); mkt = cl.bucket(MKT); rec = cl.bucket(REC)

# ---- bundle (same as axb_live / recorder_ev) ----
refs = np.load(io.BytesIO(mkt.blob(f"{BUNDLE}/refs.npz").download_as_bytes()))
meta = json.loads(mkt.blob(f"{BUNDLE}/meta.json").download_as_bytes()); KNORM = meta["KNORM"]
A = xgb.Booster(); Bg = xgb.Booster()
for nm, m in [("A", A), ("Bg", Bg)]:
    p = f"{TD}/{nm}.json"; mkt.blob(f"{BUNDLE}/{nm}.json").download_to_filename(p); m.load_model(p)
gstd = refs["gstd"].astype(np.float64); sA = refs["sA"]; sBg = refs["sBg"]
mu = refs["day_mean"].astype(np.float64)[-KNORM:].mean(0)
sd = np.maximum(np.sqrt(np.maximum(refs["day_var"].astype(np.float64)[-KNORM:].mean(0), 0)), 0.2 * gstd + 1e-9)
cdf = lambda x, ref: np.searchsorted(ref, x, "right") / max(len(ref), 1)

# ---- offline day build (mirrors subs60_recorder_ev converters) ----
def dl_rec(stream, day, cols):
    rows = {c: [] for c in cols}
    for b in rec.client.list_blobs(rec, prefix=f"{RECP}/{stream}/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/r.parquet")
        df = pq.read_table(f"{TD}/r.parquet").to_pandas()
        for c in cols:
            rows[c].append(df[c].to_numpy() if c in df.columns else np.full(len(df), np.nan))
    return {c: (np.concatenate(rows[c]) if rows[c] else np.array([])) for c in cols}


def build_cl_book(day):
    cols = ["timestamp", "receipt_timestamp", "sequence_number"] + [f"{s}_{i}_{f}" for i in range(LV) for s in ("bid", "ask") for f in ("price", "size")]
    acc = {c: [] for c in cols}
    for b in rec.client.list_blobs(rec, prefix=f"{RECP}/depth_snapshot/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/d.parquet")
        df = pq.read_table(f"{TD}/d.parquet", columns=["exchange_event_ts_us", "local_ts_us", "bid_prices", "bid_qtys", "ask_prices", "ask_qtys"]).to_pandas()
        df = df[df["exchange_event_ts_us"].notna()]
        if not len(df):
            continue
        acc["timestamp"].append((df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
        acc["receipt_timestamp"].append((df["local_ts_us"].astype("int64") * 1000).to_numpy())
        acc["sequence_number"].append(np.zeros(len(df), np.int64))
        for i in range(LV):
            for s, pcol, qcol in (("bid", "bid_prices", "bid_qtys"), ("ask", "ask_prices", "ask_qtys")):
                acc[f"{s}_{i}_price"].append(df[pcol].apply(lambda x, i=i: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
                acc[f"{s}_{i}_size"].append(df[qcol].apply(lambda x, i=i: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
    if not acc["timestamp"]:
        return None
    ar = {c: np.concatenate(acc[c]) for c in cols}; o = np.argsort(ar["timestamp"], kind="stable"); ar = {c: ar[c][o] for c in cols}
    ar["sequence_number"] = np.arange(len(ar["timestamp"]), dtype=np.int64)
    pq.write_table(pa.table({c: ar[c] for c in cols}), f"{TD}/book.parquet")
    return len(ar["timestamp"])


def build_cl_trades(day):
    d = dl_rec("agg_trade", day, ["exchange_event_ts_us", "local_ts_us", "trade_id", "price", "qty", "is_buyer_maker"])
    if not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    t = pa.table({"side": np.where(d["is_buyer_maker"][m].astype(bool), "sell", "buy"),
                  "amount": d["qty"][m].astype(np.float64), "price": d["price"][m].astype(np.float64),
                  "id": d["trade_id"][m].astype(np.int64), "timestamp": (d["exchange_event_ts_us"][m].astype(np.int64) * 1000),
                  "receipt_timestamp": (d["local_ts_us"][m].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, f"{TD}/trades.parquet"); return t.num_rows


def build_cl_liq(day):
    d = dl_rec("liquidation", day, ["exchange_event_ts_us", "local_ts_us", "side", "original_qty", "price"])
    if not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    if not m.any():
        return 0
    n = int(m.sum())
    t = pa.table({"side": np.array([str(s).lower() for s in d["side"][m]]), "quantity": d["original_qty"][m].astype(np.float64),
                  "price": d["price"][m].astype(np.float64), "id": np.arange(n, dtype=np.int64),
                  "status": np.array(["filled"] * n), "timestamp": (d["exchange_event_ts_us"][m].astype(np.int64) * 1000),
                  "receipt_timestamp": (d["local_ts_us"][m].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, f"{TD}/liq.parquet"); return t.num_rows


def feat71(td, X):
    h = ((td / NS) % 86400.0) / 3600.0; hf = h % 8.0
    tod = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * hf / 8), np.cos(2 * np.pi * hf / 8)], 1)
    return np.concatenate([X, np.zeros((len(td), 3)), tod], axis=1).astype(np.float32)


print(f"== offline build {DAY} ==", flush=True)
nb = build_cl_book(DAY); assert nb, "no book"
ntr = build_cl_trades(DAY); nliq = build_cl_liq(DAY); assert ntr, "no trades"
step = max(1, -(-nb // TARGET))
od = f"{TD}/bs"; os.makedirs(od, exist_ok=True)
for f in os.listdir(od):
    os.remove(f"{od}/{f}")
json.dump(CFGS, open(f"{TD}/cfg.json", "w"))
r = subprocess.run([BS, "--depth", f"{TD}/book.parquet", "--trades", f"{TD}/trades.parquet", "--out-dir", od,
                    "--window", str(WINDOW), "--horizon", str(H_TICKS), "--step", str(step), "--max-samples", "2000000"],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-300:]
sts = np.load(f"{od}/sample_ts.npy").astype(np.int64) * 1_000_000
rr = subprocess.run([GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
                     "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
                     "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
                     "--entry-q", f"{od}/entry_q.npy", "--configs", f"{TD}/cfg.json", "--out-prefix", f"{TD}/g",
                     "--queue-mult", str(QM), "--entry-window-ticks", str(ENTRY_WIN), "--maker-offset-frac", "0",
                     "--commission-win-pct", "0", "--commission-loss-pct", "0"], capture_output=True, text=True)
assert rr.returncode == 0, rr.stderr[-300:]
PL = np.load(f"{TD}/g_pnl_long.npy"); PS = np.load(f"{TD}/g_pnl_short.npy")
FL = np.load(f"{TD}/g_filled_long.npy").astype(bool); FS = np.load(f"{TD}/g_filled_short.npy").astype(bool)
netl = PL[CFGIDX] * 100.0; nets = PS[CFGIDX] * 100.0
bt = pq.read_table(f"{TD}/book.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
idx = np.clip(np.searchsorted(bt, sts, "right") - 1, 0, len(bt) - 1).astype(np.int64)
np.save(f"{TD}/idx.npy", idx)
cmd = [FB, "--depth", f"{TD}/book.parquet", "--indices", f"{TD}/idx.npy", "--out", f"{TD}/f.npy", "--trades", f"{TD}/trades.parquet"]
if nliq:
    cmd += ["--liquidations", f"{TD}/liq.parquet"]
fr = subprocess.run(cmd, capture_output=True, text=True)
assert fr.returncode == 0, fr.stderr[-300:]
X = np.load(f"{TD}/f.npy").astype(np.float64)
Fn = ((feat71(bt[idx], X) - mu) / sd).astype(np.float32)
pA_o = A.predict(xgb.DMatrix(Fn)); pB_o = Bg.predict(xgb.DMatrix(Fn))
sc_o = cdf(pA_o, sA) * cdf(np.abs(pB_o - 0.5), sBg)
off_ts_us = (bt[idx] // 1000).astype(np.int64)
print(f"offline: {len(sts)} samples, fill {float((FL | FS).mean()):.2f}", flush=True)

# ---- live decisions ----
raw = mkt.blob(f"{SHADOW}/decisions/{DAY}.jsonl").download_as_bytes().decode()
live = [json.loads(x) for x in raw.splitlines() if x.strip()]
lv_ts = np.array([d["book_ts_us"] for d in live], np.int64)
lv_sc = np.array([d["score"] for d in live]); lv_pA = np.array([d["pA"] for d in live])
lv_pB = np.array([d["pBg"] for d in live]); lv_side = np.array([d["side"] == "long" for d in live])
print(f"live: {len(live)} decisions", flush=True)

# ---- match nearest offline sample per live decision ----
pos = np.searchsorted(off_ts_us, lv_ts)
cand = np.stack([np.clip(pos - 1, 0, len(off_ts_us) - 1), np.clip(pos, 0, len(off_ts_us) - 1)])
d0 = np.abs(off_ts_us[cand[0]] - lv_ts); d1 = np.abs(off_ts_us[cand[1]] - lv_ts)
mi = np.where(d1 < d0, cand[1], cand[0]); dt = np.minimum(d0, d1) / 1e6
ok = dt <= MATCH_TOL_S
print(f"matched {ok.sum()}/{len(live)} within {MATCH_TOL_S}s (median dt {np.median(dt[ok]):.2f}s)", flush=True)
m = mi[ok]
side_o = pB_o[m] >= 0.5
print("\n== PARITY ==", flush=True)
print(f"corr(score) {np.corrcoef(lv_sc[ok], sc_o[m])[0,1]:.4f} | mean|d| {np.mean(np.abs(lv_sc[ok]-sc_o[m])):.4f}", flush=True)
print(f"corr(pA)    {np.corrcoef(lv_pA[ok], pA_o[m])[0,1]:.4f} | corr(pBg) {np.corrcoef(lv_pB[ok], pB_o[m])[0,1]:.4f}", flush=True)
print(f"side agree  {float((lv_side[ok] == side_o).mean()):.4f}", flush=True)
for t in BUDGETS:
    k = f"take{int(t)}"
    tk = np.array([d.get(k, False) for d in live])[ok]
    if not tk.any():
        print(f"t{int(t)}: live takes 0", flush=True)
        continue
    sel = m[tk]; sl = lv_side[ok][tk]
    net = np.where(sl, netl[sel], nets[sel]); fc = np.where(sl, FL[sel], FS[sel])
    ex = fc & np.isfinite(net)
    ev = net[ex].mean() if ex.any() else float("nan")
    print(f"t{int(t)}: live takes {int(tk.sum())} | filled {int(ex.sum())} | EV(matched labels) {ev:+.2f} bp | "
          f"side-agree-on-takes {float((sl == side_o[tk]).mean()):.3f}", flush=True)
