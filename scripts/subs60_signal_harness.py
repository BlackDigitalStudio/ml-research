#!/usr/bin/env python3
"""STEP-2a live-signal harness (no orders). Proves the deploy bundle is usable end-to-end on REAL
data: (1) re-runs the Rust feature_builder on a day's RAW depth/trades (replicating subs60_orch's
build_cell) and NUMERICALLY compares X to the stored feats_sub60 -> feature reproducibility-from-raw
parity; (2) feat71 (+btc-lead +ToD) + blanket vol-norm (from the bundle state) + A/Bg -> the AxB
deploy signal (trade y/n, side, score) via the causal-rolling threshold seeded from the bundle.
What this proves: raw -> FB -> X (matches training) -> feat71 -> norm -> models -> signal, all from
the saved bundle. What it does NOT prove: tick-level real-time feature parity (a streaming port of
feature_builder), and live execution -- the remaining gates before real money.
Usage: python3 subs60_signal_harness.py [SYM] [DAY]   (DAY in feats_sub60 & raw; default DOGE 2026-05-08)
"""
import io, json, os, subprocess, tempfile, sys
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage
import xgboost as xgb

SYM = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
DAY = sys.argv[2] if len(sys.argv) > 2 else "2026-05-08"
SYMF = f"{SYM}-USDT-PERP"; ETH = "ETH-USDT-PERP"; BTC = "BTC-USDT-PERP"
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
FB = "/tmp/feature_builder"; NS = 1_000_000_000; GRID_S = 1.0; HS = [15, 30, 45, 60]
FEATS = "feats_sub60"; TARGET_TPD = 5
bk = storage.Client(project=PROJ).bucket(BUCKET)


def dl(prefix, dst):
    n = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not n:
        return None
    bk.blob(n).download_to_filename(dst); return dst


def book_ts_mid(bookpath):
    t = pq.read_table(bookpath, columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts = t["timestamp"].to_numpy().astype(np.int64)
    mid = 0.5 * (t["bid_0_price"].to_numpy().astype(np.float64) + t["ask_0_price"].to_numpy().astype(np.float64))
    o = np.argsort(ts, kind="stable"); return ts[o], mid[o]


def feat71(dtd, X, bt, bm):
    nb = len(bt); i = np.clip(np.searchsorted(bt, dtd, "right") - 1, 0, nb - 1); bl = []
    for W in (5, 30, 60):
        j = np.clip(np.searchsorted(bt, dtd - int(W * NS), "right") - 1, 0, nb - 1)
        a = bm[j]; b = bm[i]
        bl.append(np.where((a > 0) & (b > 0), np.log(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0)), 0.0) * 1e4)
    h = ((dtd / NS) % 86400.0) / 3600.0; hf = h % 8.0
    tod = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * hf / 8), np.cos(2 * np.pi * hf / 8)]
    return np.concatenate([X, np.stack(bl, 1), np.stack(tod, 1)], axis=1).astype(np.float32)


def load_npz(path):
    return np.load(io.BytesIO(bk.blob(path).download_as_bytes()), allow_pickle=True)


# ---------- load bundle ----------
refs = load_npz(f"research_runs/deploy/{SYM}/refs.npz")
meta = json.loads(bk.blob(f"research_runs/deploy/{SYM}/meta.json").download_as_bytes())
KNORM = meta["KNORM"]; feat_names = meta["feat_names"]
A = xgb.Booster(); Bg = xgb.Booster()
for nm, mdl in [("A", A), ("Bg", Bg)]:
    p = f"/tmp/_dep_{nm}.json"; bk.blob(f"research_runs/deploy/{SYM}/{nm}.json").download_to_filename(p); mdl.load_model(p)
gstd = refs["gstd"].astype(np.float64); sA = refs["sA"]; sBg = refs["sBg"]; axb_seed = refs["axb_seed"]
day_mean = refs["day_mean"].astype(np.float64); day_var = refs["day_var"].astype(np.float64)
print(f"[harness {SYM} {DAY}] bundle loaded: {len(feat_names)} feats, A+Bg, refs, norm-state {day_mean.shape}", flush=True)

# ---------- (1) FEATURE PARITY: re-run feature_builder on raw, compare to stored feats ----------
with tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else "/tmp") as td:
    book = dl(f"raw/book/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{td}/b.parquet")
    tr = dl(f"raw/trades/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{td}/t.parquet")
    eth = dl(f"raw/trades/exchange=BINANCE_FUTURES/symbol={ETH}/dt={DAY}/", f"{td}/eth.parquet")
    fund = dl(f"raw/funding/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{td}/f.parquet")
    liq = dl(f"raw/liquidations/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{td}/l.parquet")
    oi = dl(f"raw/open_interest/exchange=BINANCE_FUTURES/symbol={SYMF}/dt={DAY}/", f"{td}/o.parquet")
    assert book and tr, "missing raw book/trades"
    bts, mid = book_ts_mid(book)
    grid = np.arange(bts[0] + 120 * NS, bts[-1] - 70 * NS, int(GRID_S * NS), dtype=np.int64)
    idx = np.unique(np.clip(np.searchsorted(bts, grid, "right") - 1, 0, len(bts) - 1)).astype(np.int64)
    ipath = f"{td}/idx.npy"; np.save(ipath, idx); opath = f"{td}/feat.npy"
    cmd = [FB, "--depth", book, "--indices", ipath, "--out", opath, "--trades", tr]
    if fund: cmd += ["--funding", fund]
    if eth: cmd += ["--eth", eth]
    if liq: cmd += ["--liquidations", liq]
    if oi: cmd += ["--open-interest", oi]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"FB failed: {r.stderr[-200:]}"
    X_live = np.load(opath).astype(np.float32); td_ns = bts[idx]

st = load_npz(f"{FEATS}/{SYMF}/{DAY}.npz")
X_stored = st["X"].astype(np.float32); td_stored = st["td"].astype(np.int64)
nmatch = min(len(X_live), len(X_stored))
align = np.array_equal(td_ns[:nmatch], td_stored[:nmatch])
dmax = float(np.abs(X_live[:nmatch] - X_stored[:nmatch]).max()) if X_live.shape[1] == X_stored.shape[1] else float("nan")
print(f"\n=== (1) FEATURE PARITY (raw -> feature_builder vs stored feats_sub60) ===", flush=True)
print(f"  X_live {X_live.shape} | X_stored {X_stored.shape} | ts-grid identical={align} | max|ΔX|={dmax:.3e}", flush=True)
print(f"  -> {'PARITY OK (X reproduces from raw, exact)' if align and dmax < 1e-4 else 'MISMATCH -- investigate'}", flush=True)

# ---------- (2) full pipeline -> signal ----------
import datetime
def load_btc_span(day):  # BTC mid over day-1..day+1 (robust to one-day gaps; build's load_btc_mid concatenates all days)
    base = datetime.date.fromisoformat(day); tds, mds = [], []
    for off in (-1, 0, 1):
        d = (base + datetime.timedelta(days=off)).isoformat()
        try:
            z = load_npz(f"{FEATS}/{BTC}/{d}.npz"); tds.append(z["td"].astype(np.int64)); mds.append(z["mid"].astype(np.float64))
        except Exception:
            pass
    bt = np.concatenate(tds); bm = np.concatenate(mds); o = np.argsort(bt, kind="stable"); return bt[o], bm[o]
bt, bm = load_btc_span(DAY)
F = feat71(td_ns, X_live, bt, bm)
assert F.shape[1] == len(feat_names), f"feat count {F.shape[1]} != {len(feat_names)}"
# match the deploy decision cadence: feat-stride 8 + valid_60 (as in subs60_makerlabel_build)
valid60 = st["valid_60"].astype(bool)
selw = np.zeros(len(F), bool); selw[np.arange(0, len(F), 8)] = True; selw &= valid60[:len(F)]
F = F[selw]; td_ns = td_ns[selw]
print(f"  signal cadence: stride-8 + valid_60 -> {len(F)} decision windows (~{len(F)} /day)", flush=True)
mu_ref = day_mean[-KNORM:].mean(0); sd_ref = np.sqrt(np.maximum(day_var[-KNORM:].mean(0), 0))
sd_ref = np.maximum(sd_ref, 0.2 * gstd + 1e-9)
Fn = ((F - mu_ref) / sd_ref).astype(np.float32)
pA = A.predict(xgb.DMatrix(Fn)); pBg = Bg.predict(xgb.DMatrix(Fn))
cdf = lambda x, ref: np.searchsorted(ref, x, "right") / max(len(ref), 1)
score = cdf(pA, sA) * cdf(np.abs(pBg - 0.5), sBg)
wpd = len(score); q = max(0.0, 1.0 - TARGET_TPD / max(wpd / 1.0, 1.0))  # 1 day of windows
tau = float(np.quantile(axb_seed, 1.0 - TARGET_TPD / max(len(axb_seed) / 30.0, 1.0)))  # ~TPD/day vs seed (30d)
sel = score >= tau; side = np.where(pBg[sel] >= 0.5, "LONG", "SHORT")
print(f"\n=== (2) AxB DEPLOY SIGNAL (no orders) ===", flush=True)
print(f"  windows today={wpd} | pA[min/med/max]={pA.min():.2f}/{np.median(pA):.2f}/{pA.max():.2f} | "
      f"|pBg-0.5| med={np.median(np.abs(pBg-0.5)):.3f}", flush=True)
print(f"  rolling tau(from bundle seed, target {TARGET_TPD}/day)={tau:.3f} -> SELECTED={int(sel.sum())} "
      f"({100*sel.mean():.2f}%) | side LONG={int((side=='LONG').sum())} SHORT={int((side=='SHORT').sum())}", flush=True)
topk = np.argsort(score)[::-1][:5]  # top-5 candidates by AxB score (shown regardless of tau)
print(f"  top-5 candidates today (highest AxB score):", flush=True)
for k in topk:
    tsec = (td_ns[k] / NS) % 86400
    print(f"    @{int(tsec)//3600:02d}:{(int(tsec)%3600)//60:02d} side={'LONG' if pBg[k]>=0.5 else 'SHORT'} "
          f"score={score[k]:.3f} pA={pA[k]:.2f} pBg={pBg[k]:.3f} {'<= would TRADE' if score[k]>=tau else ''}", flush=True)
print(f"\n[harness done] parity {'OK' if align and dmax<1e-4 else 'CHECK'} ; signal generated end-to-end from bundle.", flush=True)
print(f"  REMAINING gates before real money: tick-level live feature parity (stream port of feature_builder) + execution engine.", flush=True)
