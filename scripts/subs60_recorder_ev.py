#!/usr/bin/env python3
"""RECORDER-EV: maker-realistic economic transfer test. Apply the frozen deploy_robust2 model
(CL-trained; vol 1s-grid + trade-dedup) to LIVE recorder data and measure the MAKER pegged-exit EV
(the same metric the CL backtest +2.79 was in), to test economic transfer.

Per recorder day: recorder depth_snapshot+agg_trade -> cryptolake book/trades; husdc build_samples
(forward book/flow paths) -> grid_sim_exitdbg (pegged-exit, always-last) -> pnl_long/short + fills
(CFGIDX=1 = 30s hold, QMIDX=0 = qm1.0); robust feature_builder at the SAME sample points (+liq) ->
feat71 -> frozen vol-norm -> A/Bg -> AxB score. Across days: causal_rolling exactly as optuna_ic
(threshold seeded from bundle axb_seed), realize maker PnL on side=pBg>=0.5, count filled only.
Env: DAYS (default 6), TARGET (build samples/day), BUDGETS. Run on hd2-feats-003.
btc_lead/funding/OI defaulted to 0 (first cut; ~7-10% of features) -> a slightly conservative EV.
"""
import io, os, json, subprocess, tempfile
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"
MKT = "market-data-0998ac51"; REC = "recorder-data-asia-0998ac51"
RECP = "chronos/scalper-recorder/binance_futures/DOGEUSDT"
FB = os.environ.get("FB_BIN", "/tmp/fb_target/release/feature_builder")             # robust (vol 1s-grid)
BS = os.environ.get("BS_BIN", "/tmp/husdc_target/release/build_samples")           # husdc paths
GRID = os.environ.get("GRID_BIN", "/tmp/husdc_target/release/grid_sim_exitdbg")     # husdc pegged-exit
BUNDLE = os.environ.get("BUNDLE_DIR", "research_runs/deploy_robust2/DOGE")
NS = 1_000_000_000; LV = 20; NDAYS = int(os.environ.get("DAYS", "6"))
TARGET = int(os.environ.get("TARGET", "8000"))                 # build samples/day ~ CL feat density
H_TICKS = 700; ENTRY_WIN = 120; WINDOW = 50
CFGS = [{"tp": 50.0, "sl": 50.0, "to": 141, "par": False, "tr": False},
        {"tp": 50.0, "sl": 50.0, "to": 282, "par": False, "tr": False},   # CFGIDX=1 -> 30s hold
        {"tp": 50.0, "sl": 50.0, "to": 563, "par": False, "tr": False}]
CFGIDX, QMIDX = 1, 0; QM = 1.0; KDAYS = 30
BUDGETS = [float(x) for x in os.environ.get("BUDGETS", "5,10").split(",")]
cl = storage.Client(project=PROJ); mkt = cl.bucket(MKT); rec = cl.bucket(REC)
TD = os.environ.get("WORKDIR", "/home/delmi/recev_work"); os.makedirs(TD, exist_ok=True)
import xgboost as xgb

refs = np.load(io.BytesIO(mkt.blob(f"{BUNDLE}/refs.npz").download_as_bytes()))
meta = json.loads(mkt.blob(f"{BUNDLE}/meta.json").download_as_bytes()); KNORM = meta["KNORM"]
A = xgb.Booster(); Bg = xgb.Booster()
for nm, m in [("A", A), ("Bg", Bg)]:
    p = f"{TD}/{nm}.json"; mkt.blob(f"{BUNDLE}/{nm}.json").download_to_filename(p); m.load_model(p)
gstd = refs["gstd"].astype(np.float64); sA = refs["sA"]; sBg = refs["sBg"]
axb_seed = refs["axb_seed"].astype(np.float64) if "axb_seed" in refs.files else None
mu = refs["day_mean"].astype(np.float64)[-KNORM:].mean(0)
sd = np.maximum(np.sqrt(np.maximum(refs["day_var"].astype(np.float64)[-KNORM:].mean(0), 0)), 0.2 * gstd + 1e-9)
cdf = lambda x, ref: np.searchsorted(ref, x, "right") / max(len(ref), 1)
print(f"bundle {BUNDLE} KNORM {KNORM} axb_seed {None if axb_seed is None else len(axb_seed)} | CFGIDX {CFGIDX} QM {QM}", flush=True)


def rec_days():
    dss = sorted(set(b.name.split("/")[-1][:8] for b in rec.client.list_blobs(rec, prefix=f"{RECP}/depth_snapshot/") if b.name.endswith(".parquet")))
    return dss[-NDAYS:]


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
    return np.concatenate([X, np.zeros((len(td), 3)), tod], axis=1).astype(np.float32)  # btc_lead=0


def causal_rolling(sc, day_arr, side, netl, nets, fl, fs, tgt, seed):
    days = sorted(set(day_arr.tolist())); wpd = len(sc) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0))
    buf = list(seed) if seed is not None else []; cap = max(int(KDAYS * wpd), 1); sel = []
    for dd in days:
        idx = np.where(day_arr == dd)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc[idx] >= tau].tolist()); buf.extend(sc[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    s = side[sel]; net = np.where(s, netl[sel], nets[sel]); fc = np.where(s, fl[sel], fs[sel])
    ex = fc & np.isfinite(net); return net[ex]


TMP = "research_runs/_recev_tmp"  # per-day persistence -> resumable across SSH drops
COMBINE = os.environ.get("COMBINE", "") == "1"
json.dump(CFGS, open(f"{TD}/cfg.json", "w"))
done_blobs = {b.name.split("/")[-1][:-4].split("_")[-1] for b in mkt.client.list_blobs(mkt, prefix=f"{TMP}/DOGE_") if b.name.endswith(".npz")}
for di, day in enumerate([] if COMBINE else rec_days()):
    if day in done_blobs:
        print(f"{day}: already persisted, skip", flush=True); continue
    nb = build_cl_book(day)
    if not nb:
        print(f"{day}: no book", flush=True); continue
    ntr = build_cl_trades(day); nliq = build_cl_liq(day)
    if not ntr:
        print(f"{day}: no trades", flush=True); continue
    step = max(1, -(-nb // TARGET))
    od = f"{TD}/bs"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(f"{od}/{f}")
    r = subprocess.run([BS, "--depth", f"{TD}/book.parquet", "--trades", f"{TD}/trades.parquet", "--out-dir", od,
                        "--window", str(WINDOW), "--horizon", str(H_TICKS), "--step", str(step), "--max-samples", "2000000"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{day}: BS fail {r.stderr[-160:]}", flush=True); continue
    sts = np.load(f"{od}/sample_ts.npy").astype(np.int64) * 1_000_000   # ms->ns
    if len(sts) < 20:
        print(f"{day}: few samples {len(sts)}", flush=True); continue
    g = f"{TD}/g"
    rr = subprocess.run([GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
                         "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
                         "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
                         "--entry-q", f"{od}/entry_q.npy", "--configs", f"{TD}/cfg.json", "--out-prefix", g,
                         "--queue-mult", str(QM), "--entry-window-ticks", str(ENTRY_WIN), "--maker-offset-frac", "0",
                         "--commission-win-pct", "0", "--commission-loss-pct", "0"], capture_output=True, text=True)
    if rr.returncode != 0:
        print(f"{day}: GRID fail {rr.stderr[-200:]}", flush=True); continue
    PL = np.load(f"{g}_pnl_long.npy"); PS = np.load(f"{g}_pnl_short.npy")   # (NC,N)
    FL = np.load(f"{g}_filled_long.npy").astype(bool); FS = np.load(f"{g}_filled_short.npy").astype(bool)  # (N,)
    netl = PL[CFGIDX] * 100.0; nets = PS[CFGIDX] * 100.0                     # bp
    # features at the SAME sample points
    bt = pq.read_table(f"{TD}/book.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    idx = np.clip(np.searchsorted(bt, sts, "right") - 1, 0, len(bt) - 1).astype(np.int64)
    np.save(f"{TD}/idx.npy", idx)
    cmd = [FB, "--depth", f"{TD}/book.parquet", "--indices", f"{TD}/idx.npy", "--out", f"{TD}/f.npy", "--trades", f"{TD}/trades.parquet"]
    if nliq:
        cmd += ["--liquidations", f"{TD}/liq.parquet"]
    fr = subprocess.run(cmd, capture_output=True, text=True)
    if fr.returncode != 0:
        print(f"{day}: FB fail {fr.stderr[-160:]}", flush=True); continue
    X = np.load(f"{TD}/f.npy").astype(np.float64)
    Fn = ((feat71(bt[idx], X) - mu) / sd).astype(np.float32)
    pA = A.predict(xgb.DMatrix(Fn)); pBg = Bg.predict(xgb.DMatrix(Fn))
    score = cdf(pA, sA) * cdf(np.abs(pBg - 0.5), sBg)
    buf = io.BytesIO()
    np.savez(buf, score=score.astype(np.float32), side=(pBg >= 0.5), netl=netl.astype(np.float32),
             nets=nets.astype(np.float32), FL=FL, FS=FS)
    mkt.blob(f"{TMP}/DOGE_{day}.npz").upload_from_string(buf.getvalue())
    fillr = float((FL | FS).mean())
    print(f"{day}: book {nb} agg {ntr} liq {nliq} | samp {len(sts)} step {step} | fill {fillr:.2f} "
          f"| netl(filled) {np.nanmean(np.where(FL,netl,np.nan)):+.2f}bp -> persisted", flush=True)

# ---- aggregate from ALL persisted days (resumable) ----
blobs = sorted(b.name for b in mkt.client.list_blobs(mkt, prefix=f"{TMP}/DOGE_") if b.name.endswith(".npz"))
if not blobs:
    print("NO DAYS PERSISTED", flush=True); raise SystemExit
aS, aSide, aNL, aNS, aFL, aFS, aDay = [], [], [], [], [], [], []
for di, bn in enumerate(blobs):
    z = np.load(io.BytesIO(mkt.blob(bn).download_as_bytes()))
    aS.append(z["score"].astype(np.float64)); aSide.append(z["side"].astype(bool))
    aNL.append(z["netl"].astype(np.float64)); aNS.append(z["nets"].astype(np.float64))
    aFL.append(z["FL"].astype(bool)); aFS.append(z["FS"].astype(bool)); aDay.append(np.full(len(z["score"]), di, np.int32))
score = np.concatenate(aS); side = np.concatenate(aSide); day_arr = np.concatenate(aDay)
netl = np.concatenate(aNL); nets = np.concatenate(aNS); FL = np.concatenate(aFL); FS = np.concatenate(aFS)
print(f"\n=== RECORDER-EV (maker pegged-exit 30s, {len(blobs)} days, {len(score)} decisions) ===", flush=True)
print(f"    (CL backtest reference: AxB t5 +2.79 bp)", flush=True)
for tgt in BUDGETS:
    pnl = causal_rolling(score, day_arr, side, netl, nets, FL, FS, tgt, axb_seed)
    if len(pnl):
        print(f"  AxB t{int(tgt)}: {len(pnl)} filled trades | EV/trade {pnl.mean():+.2f} bp | "
              f"hit {100*(pnl>0).mean():.1f}% | net(after 4bp maker RT) {pnl.mean()-4:+.2f} bp", flush=True)
    else:
        print(f"  AxB t{int(tgt)}: 0 selected", flush=True)
