#!/usr/bin/env python3
"""RECORDER-EV for the h150 deploy candidate: apply the 4-seed deploy_h150 ensemble to LIVE
recorder days with the SAME honest config as training — entry 60s, hold 150s from fill, pegged
never-taker chase 300s, always-last, 0 fee, 3s decision grid, FULL features (funding/liq/OI/ETH/
btc real). Measures weight-specific venue transfer + captures the recorder score distribution to
RE-SEED tau for deploy (CL seed runs hot -> under-trades).

Per recorder day (SYM): recorder depth+trade+mark_price+liquidation+derivatives_poll -> CL parquet;
BTC depth -> btc mid; ETH agg_trade -> CL trades. build_samples time-mode (3s ends, H big, skip xlob)
-> grid_sim_exitdbg time-mode (entry-window-ms 60000, chase-ms 300000, cfg to_ms 150000, qm1) ->
150s pegged pnl + fills. FB full inputs at the same ticks -> feat71 -> per-seed vol-norm+CDF score,
ensemble = mean of 4 per-seed rank-scores. Across days: causal_rolling budget 5/day, realize EV.
Env: SYM(DOGE) DAYS(10) DEPLOY_DIR(deploy_h150) BUDG(5). Run on hd2 (has husdc+fb bins).
"""
import io, os, json, subprocess
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; MKT = "market-data-0998ac51"; REC = "recorder-data-asia-0998ac51"
RB = "chronos/scalper-recorder/binance_futures"
SYM = os.environ.get("SYM", "DOGE"); SYMR = SYM + "USDT"
DEPLOY = os.environ.get("DEPLOY_DIR", "deploy_h150")
NDAYS = int(os.environ.get("DAYS", "10")); BUDG = [float(x) for x in os.environ.get("BUDG", "5,10").split(",")]
FB = "/tmp/fb_target/release/feature_builder"; BS = "/tmp/husdc_target/release/build_samples"
GRID = "/tmp/husdc_target/release/grid_sim_exitdbg"
NS = 1_000_000_000; LV = 20; W = 50; H = 6000; STEP_S = 3.0
ENTRY_MS = 60_000; CHASE_MS = 300_000; HOLD_MS = 150_000; KNORM = 20; KDAYS = 30
TD = os.environ.get("WORKDIR", "/home/delmi/recev_h150"); os.makedirs(TD, exist_ok=True)
cl = storage.Client(project=PROJ); mkt = cl.bucket(MKT); rec = cl.bucket(REC)
import xgboost as xgb

# ---- load 4-seed ensemble bundles ----
SEEDS = [0, 1, 2, 3]; BUN = []
for s in SEEDS:
    base = f"research_runs/{DEPLOY}/{SYM}/seed{s}"
    refs = np.load(io.BytesIO(mkt.blob(f"{base}/refs.npz").download_as_bytes()))
    A = xgb.Booster(); Bg = xgb.Booster()
    for nm, m in (("A", A), ("Bg", Bg)):
        p = f"{TD}/{nm}{s}.json"; mkt.blob(f"{base}/{nm}.json").download_to_filename(p); m.load_model(p)
    gstd = refs["gstd"].astype(np.float64)
    mu = refs["day_mean"].astype(np.float64)[-KNORM:].mean(0)
    sd = np.maximum(np.sqrt(np.maximum(refs["day_var"].astype(np.float64)[-KNORM:].mean(0), 0)), 0.2 * gstd + 1e-9)
    BUN.append(dict(A=A, Bg=Bg, mu=mu, sd=sd, sA=refs["sA"], sBg=refs["sBg"]))
meta = json.loads(mkt.blob(f"research_runs/{DEPLOY}/{SYM}/seed0/meta.json").download_as_bytes())
print(f"[recev_h150 {SYM}] {len(BUN)} seed bundles | H={H} entry {ENTRY_MS}ms hold {HOLD_MS}ms chase {CHASE_MS}ms", flush=True)
cdf = lambda x, ref: np.searchsorted(ref, x, "right") / max(len(ref), 1)


def dl_rec(sym, stream, day, cols):
    rows = {c: [] for c in cols}; got = False
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{sym}/{stream}/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/r.parquet")
        df = pq.read_table(f"{TD}/r.parquet").to_pandas(); got = True
        for c in cols:
            rows[c].append(df[c].to_numpy() if c in df.columns else np.full(len(df), np.nan))
    if not got:
        return None
    return {c: (np.concatenate(rows[c]) if rows[c] else np.array([])) for c in cols}


def book_cl(sym, day, out):
    cols = ["timestamp", "receipt_timestamp", "sequence_number"] + [f"{s}_{i}_{f}" for i in range(LV) for s in ("bid", "ask") for f in ("price", "size")]
    acc = {c: [] for c in cols}
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{sym}/depth_snapshot/{day}"):
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
            for s, pc, qc in (("bid", "bid_prices", "bid_qtys"), ("ask", "ask_prices", "ask_qtys")):
                acc[f"{s}_{i}_price"].append(df[pc].apply(lambda x, i=i: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
                acc[f"{s}_{i}_size"].append(df[qc].apply(lambda x, i=i: float(x[i]) if x is not None and len(x) > i else 0.0).to_numpy())
    if not acc["timestamp"]:
        return 0
    ar = {c: np.concatenate(acc[c]) for c in cols}; o = np.argsort(ar["timestamp"], kind="stable"); ar = {c: ar[c][o] for c in cols}
    ar["sequence_number"] = np.arange(len(ar["timestamp"]), dtype=np.int64)
    pq.write_table(pa.table({c: ar[c] for c in cols}), out); return len(ar["timestamp"])


def trades_cl(sym, day, out, stream="agg_trade"):
    d = dl_rec(sym, stream, day, ["exchange_event_ts_us", "local_ts_us", "trade_id", "price", "qty", "is_buyer_maker"])
    if d is None or not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    tid = d["trade_id"][m]
    _, ui = np.unique(tid.astype(np.int64), return_index=True); ui = np.sort(ui)   # dedup
    t = pa.table({"side": np.where(d["is_buyer_maker"][m][ui].astype(bool), "sell", "buy"),
                  "amount": d["qty"][m][ui].astype(np.float64), "price": d["price"][m][ui].astype(np.float64),
                  "id": tid[ui].astype(np.int64), "timestamp": (d["exchange_event_ts_us"][m][ui].astype(np.int64) * 1000),
                  "receipt_timestamp": (d["local_ts_us"][m][ui].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


def liq_cl(sym, day, out):
    d = dl_rec(sym, "liquidation", day, ["exchange_event_ts_us", "local_ts_us", "side", "original_qty", "price"])
    if d is None or not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    if not m.any():
        return 0
    n = int(m.sum())
    t = pa.table({"side": np.array([str(s).lower() for s in d["side"][m]]), "quantity": d["original_qty"][m].astype(np.float64),
                  "price": d["price"][m].astype(np.float64), "id": np.arange(n, dtype=np.int64), "status": np.array(["filled"] * n),
                  "timestamp": (d["exchange_event_ts_us"][m].astype(np.int64) * 1000), "receipt_timestamp": (d["local_ts_us"][m].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


# FUNDING_MODE (2026-07-08 parity audit):
#   anchor (DEFAULT) — the DEPLOYED policy: fund.parquet is a SINGLE row (first mark_price
#     rate of the day, mark 0, ts=1ms) -> col13 frozen per day, col44=0. This reproduces the
#     accidental-but-validated +6.59bp cell (LOO-positive, jitter-robust) as an intentional
#     variance reduction; byte-matches live axb_live write_window_parquet.
#   true — training semantics: full stream in ms -> col13 latest rate(t), col44 real basis.
#     Measured -2.14bp t5 on 20260628-0707 (_recev_h150fix_DOGE).
FUNDING_MODE = os.environ.get("FUNDING_MODE", "anchor")


def funding_cl(sym, day, out):
    d = dl_rec(sym, "mark_price", day, ["exchange_event_ts_us", "local_ts_us", "funding_rate", "mark_price"])
    if d is None or not len(d["exchange_event_ts_us"]):
        return 0
    m = ~np.isnan(d["exchange_event_ts_us"].astype(float))
    if FUNDING_MODE == "anchor":
        ets = d["exchange_event_ts_us"][m].astype(np.int64)
        fr = np.nan_to_num(d["funding_rate"][m].astype(np.float64))
        i = int(np.argmin(ets))
        t = pa.table({"funding_rate": np.array([fr[i]], np.float64),
                      "mark_price": np.array([0.0], np.float64),
                      "timestamp": np.array([1], np.int64)})
    else:
        # ts in MILLISECONDS: FB's funding_rate+mark_price reader takes timestamps as-is.
        # The pre-2026-07-08 ns write pinned col13 to the day's first row and zeroed col44.
        t = pa.table({"funding_rate": np.nan_to_num(d["funding_rate"][m].astype(np.float64)), "mark_price": d["mark_price"][m].astype(np.float64),
                      "timestamp": (d["exchange_event_ts_us"][m].astype(np.int64) // 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


def oi_cl(sym, day, out):
    d = dl_rec(sym, "derivatives_poll", day, ["local_ts_us", "open_interest"])
    if d is None or not len(d["local_ts_us"]):
        return 0
    m = ~np.isnan(d["local_ts_us"].astype(float)) & ~np.isnan(d["open_interest"].astype(float))
    if not m.any():
        return 0
    t = pa.table({"open_interest": d["open_interest"][m].astype(np.float64),
                  "timestamp": (d["local_ts_us"][m].astype(np.int64) * 1000)}).sort_by("timestamp")
    pq.write_table(t, out); return t.num_rows


def btc_mid(day):
    """recorder BTCUSDT L1 mid (ts_ns, mid) for btc_lead features."""
    ts, mid = [], []
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/BTCUSDT/depth_snapshot/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/bt.parquet")
        df = pq.read_table(f"{TD}/bt.parquet", columns=["exchange_event_ts_us", "bid_prices", "ask_prices"]).to_pandas()
        df = df[df["exchange_event_ts_us"].notna()]
        b0 = df["bid_prices"].apply(lambda x: float(x[0]) if x is not None and len(x) else np.nan).to_numpy()
        a0 = df["ask_prices"].apply(lambda x: float(x[0]) if x is not None and len(x) else np.nan).to_numpy()
        ts.append((df["exchange_event_ts_us"].astype("int64") * 1000).to_numpy()); mid.append((b0 + a0) / 2)
    if not ts:
        return np.array([]), np.array([])
    ts = np.concatenate(ts); mid = np.concatenate(mid); o = np.argsort(ts)
    return ts[o], mid[o]


def feat71(dtd, X, bts, bm):
    nb = len(bts); i = np.clip(np.searchsorted(bts, dtd, "right") - 1, 0, nb - 1) if nb else np.zeros(len(dtd), int)
    bl = []
    for Wd in (5, 30, 60):
        if nb:
            j = np.clip(np.searchsorted(bts, dtd - int(Wd * NS), "right") - 1, 0, nb - 1)
            a = bm[j]; b = bm[i]
            bl.append(np.where((a > 0) & (b > 0), np.log(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0)), 0.0) * 1e4)
        else:
            bl.append(np.zeros(len(dtd)))
    h = ((dtd / NS) % 86400.0) / 3600.0; hf = h % 8.0
    tod = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * hf / 8), np.cos(2 * np.pi * hf / 8)]
    return np.concatenate([X, np.stack(bl, 1), np.stack(tod, 1)], axis=1).astype(np.float32)


def rec_days():
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    ds = sorted(set(b.name.split("/")[-1][:8] for b in rec.client.list_blobs(rec, prefix=f"{RB}/{SYMR}/liquidation/") if b.name.endswith(".parquet")))
    ds = [d for d in ds if d < today]        # exclude the in-progress UTC day
    return ds[-NDAYS:]


# RECEV_PREFIX: override for reruns (e.g. the 2026-07-08 funding-ms fix) so historical
# prefixes that seed the LIVE tau are never clobbered.
TMP = os.environ.get("RECEV_PREFIX", f"research_runs/_recev_h150_{SYM}")
cfg = [{"tp": 50.0, "sl": 50.0, "to": 282, "to_ms": float(HOLD_MS), "par": False, "tr": False}]
json.dump(cfg, open(f"{TD}/cfg.json", "w"))
done = {b.name.split("/")[-1][:-4].split("_")[-1] for b in mkt.client.list_blobs(mkt, prefix=f"{TMP}/D_") if b.name.endswith(".npz")}
for day in rec_days():
    if day in done:
        print(f"{day}: persisted, skip", flush=True); continue
    nb = book_cl(SYMR, day, f"{TD}/book.parquet")
    ntr = trades_cl(SYMR, day, f"{TD}/trades.parquet")
    if not nb or not ntr:
        print(f"{day}: book={nb} tr={ntr} skip", flush=True); continue
    nliq = liq_cl(SYMR, day, f"{TD}/liq.parquet"); nfd = funding_cl(SYMR, day, f"{TD}/fund.parquet")
    noi = oi_cl(SYMR, day, f"{TD}/oi.parquet"); neth = trades_cl("ETHUSDT", day, f"{TD}/eth.parquet")
    bt = pq.read_table(f"{TD}/book.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    n = len(bt)
    if n < W + H + 100:
        print(f"{day}: thin {n}", flush=True); continue
    # Grid anchored at CALENDAR UTC MIDNIGHT (== live engine), not bt[0]: the recorder's
    # hour-00 files carry a pre-midnight flush-buffer tail, so a bt[0] anchor has a
    # flush-phase-dependent offset that live cannot reproduce in real time.
    from datetime import datetime as _dt, timezone as _tz
    mid0 = int(_dt.strptime(day, "%Y%m%d").replace(tzinfo=_tz.utc).timestamp()) * NS
    grid = np.arange(mid0, bt[-1], int(STEP_S * NS))
    grid = grid[grid >= bt[0]]
    ends = np.unique(np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1))
    ends = ends[(ends >= W - 1) & (ends < n - H - 1)].astype(np.int64)
    if len(ends) < 50:
        print(f"{day}: few-ends {len(ends)}", flush=True); continue
    np.save(f"{TD}/ends.npy", ends)
    od = f"{TD}/bs"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(f"{od}/{f}")
    r = subprocess.run([BS, "--depth", f"{TD}/book.parquet", "--trades", f"{TD}/trades.parquet", "--out-dir", od,
                        "--window", str(W), "--horizon", str(H), "--sample-ends", f"{TD}/ends.npy", "--skip-xlob"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{day}: BS fail {r.stderr[-160:]}", flush=True); continue
    se = np.load(f"{od}/end_indices.npy").astype(np.int64)
    g = f"{TD}/g"
    rr = subprocess.run([GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
                         "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy", "--entry-book", f"{od}/entry_book.npy",
                         "--flow-paths", f"{od}/flow_paths.npy", "--entry-q", f"{od}/entry_q.npy", "--configs", f"{TD}/cfg.json", "--out-prefix", g,
                         "--queue-mult", "1.0", "--exit-queue-mult", "1.0", "--ts-paths", f"{od}/ts_paths.npy", "--sample-ts", f"{od}/sample_ts.npy",
                         "--entry-window-ms", str(ENTRY_MS), "--chase-ms", str(CHASE_MS), "--entry-window-ticks", "120",
                         "--maker-offset-frac", "0", "--commission-win-pct", "0", "--commission-loss-pct", "0"], capture_output=True, text=True)
    if rr.returncode != 0:
        print(f"{day}: GRID fail {rr.stderr[-200:]}", flush=True); continue
    PL = np.load(f"{g}_pnl_long.npy")[0] * 100.0; PS = np.load(f"{g}_pnl_short.npy")[0] * 100.0
    FL = np.load(f"{g}_filled_long.npy").astype(bool); FS = np.load(f"{g}_filled_short.npy").astype(bool)
    idx = np.clip(np.searchsorted(bt, bt[se], "right") - 1, 0, n - 1).astype(np.int64)
    np.save(f"{TD}/idx.npy", se.astype(np.int64))
    fcmd = [FB, "--depth", f"{TD}/book.parquet", "--indices", f"{TD}/idx.npy", "--out", f"{TD}/f.npy", "--trades", f"{TD}/trades.parquet"]
    if nfd:
        fcmd += ["--funding", f"{TD}/fund.parquet"]
    if nliq:
        fcmd += ["--liquidations", f"{TD}/liq.parquet"]
    if noi:
        fcmd += ["--open-interest", f"{TD}/oi.parquet"]
    if neth:
        fcmd += ["--eth", f"{TD}/eth.parquet"]
    fr = subprocess.run(fcmd, capture_output=True, text=True)
    if fr.returncode != 0:
        print(f"{day}: FB fail {fr.stderr[-160:]}", flush=True); continue
    X = np.load(f"{TD}/f.npy").astype(np.float64)
    bts, bm = btc_mid(day)
    F = feat71(bt[se], X, bts, bm)
    # ensemble rank-score
    sc = np.zeros(len(F)); pbg_mean = np.zeros(len(F))
    for bd in BUN:
        Fn = ((F - bd["mu"]) / bd["sd"]).astype(np.float32)
        pA = bd["A"].predict(xgb.DMatrix(Fn)); pBg = bd["Bg"].predict(xgb.DMatrix(Fn))
        sc += cdf(pA, bd["sA"]) * cdf(np.abs(pBg - 0.5), bd["sBg"]); pbg_mean += pBg
    sc /= len(BUN); pbg_mean /= len(BUN); side = pbg_mean >= 0.5
    buf = io.BytesIO()
    np.savez(buf, score=sc.astype(np.float32), side=side, netl=PL.astype(np.float32), nets=PS.astype(np.float32), FL=FL, FS=FS)
    mkt.blob(f"{TMP}/D_{day}.npz").upload_from_string(buf.getvalue())
    fillr = float((FL | FS).mean())
    print(f"{day}: book {nb} tr {ntr} liq {nliq} fd {nfd} oi {noi} eth {neth} | samp {len(se)} fill {fillr:.2f} "
          f"score[p50/p99]={np.quantile(sc,.5):.3f}/{np.quantile(sc,.99):.3f} netl150(fill)={np.nanmean(np.where(FL,PL,np.nan)):+.2f}bp -> saved", flush=True)

# ---- aggregate + causal_rolling budget ----
blobs = sorted(b.name for b in mkt.client.list_blobs(mkt, prefix=f"{TMP}/D_") if b.name.endswith(".npz"))
if not blobs:
    print("NO DAYS", flush=True); raise SystemExit
aS, aSd, aNL, aNS, aFL, aFS, aDay, allsc = [], [], [], [], [], [], [], []
for di, bn in enumerate(blobs):
    z = np.load(io.BytesIO(mkt.blob(bn).download_as_bytes()))
    aS.append(z["score"].astype(np.float64)); aSd.append(z["side"].astype(bool)); aNL.append(z["netl"].astype(np.float64))
    aNS.append(z["nets"].astype(np.float64)); aFL.append(z["FL"].astype(bool)); aFS.append(z["FS"].astype(bool))
    aDay.append(np.full(len(z["score"]), di, np.int32)); allsc.append(z["score"].astype(np.float64))
score = np.concatenate(aS); side = np.concatenate(aSd); day_arr = np.concatenate(aDay)
netl = np.concatenate(aNL); nets = np.concatenate(aNS); FL = np.concatenate(aFL); FS = np.concatenate(aFS)
allsc = np.concatenate(allsc)
print(f"\n=== RECORDER-EV h150 {SYM} ({len(blobs)} days, {len(score)} decisions) ===", flush=True)
print(f"    recorder score dist for tau-seed: p90={np.quantile(allsc,.9):.4f} p99={np.quantile(allsc,.99):.4f} "
      f"p99.9={np.quantile(allsc,.999):.4f} max={allsc.max():.4f}", flush=True)
# save the recorder score distribution for deploy tau-seed (inside TMP so reruns
# under RECEV_PREFIX never clobber the deployed bundle's copy)
sb = io.BytesIO(); np.savez(sb, scores=allsc.astype(np.float32))
mkt.blob(f"{TMP}/recorder_scores.npz").upload_from_string(sb.getvalue())


def causal(sc, day_a, side, nl, ns, fl, fs, tgt):
    days = sorted(set(day_a.tolist())); wpd = len(sc) / max(len(days), 1)
    q = max(0.0, 1.0 - tgt / max(wpd, 1.0)); buf = []; cap = max(int(KDAYS * wpd), 1); sel = []
    for d in days:
        idx = np.where(day_a == d)[0]
        tau = float(np.quantile(buf, q)) if buf else 0.0
        sel.extend(idx[sc[idx] >= tau].tolist()); buf.extend(sc[idx].tolist()); buf = buf[-cap:]
    sel = np.array(sel, dtype=int)
    if not len(sel):
        return np.array([])
    s = side[sel]; net = np.where(s, nl[sel], ns[sel]); fc = np.where(s, fl[sel], fs[sel])
    ex = fc & np.isfinite(net); return net[ex]


for tgt in BUDG:
    pnl = causal(score, day_arr, side, netl, nets, FL, FS, tgt)
    if len(pnl):
        print(f"  budget {int(tgt)}/day: {len(pnl)} filled trades ({len(pnl)/len(blobs):.1f}/day) | EV/tr {pnl.mean():+.2f}bp | "
              f"hit {100*(pnl>0).mean():.1f}% | sum {pnl.sum():+.1f}bp", flush=True)
    else:
        print(f"  budget {int(tgt)}/day: 0 selected", flush=True)
