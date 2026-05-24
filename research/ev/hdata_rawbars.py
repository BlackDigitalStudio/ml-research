"""HDATA RAWBARS cell (XGBoost only): does feeding RAW lagged bars at resolution
{1m,5m,15m} to a pooled XGB produce short-horizon directional signal? Isolates the
bar-RESOLUTION/timescale axis vs the slow engineered rollups of EXP-2/3/4 (whose
long-window features biased signal to long horizons). Per-bar features are RAW and
SHORT: log-return, range, log-volume, and TAKER-FLOW imbalance (klines
taker_buy_base/volume) = a free 1m order-flow proxy, universe-wide, unused at 1m by
the free-panel.

Reads free_v1/klines_1m from GCS; resamples 5m & 15m from 1m. Honest GLOBAL temporal
70/30 split + embargo; per-symbol train-fit z-score on the 4 base series (pre-cutoff);
R1 objective (logloss, |fwd move|-weighted, clipped p99). rank_IC = AUC-0.5
(roc_auc_score). Reports pooled + per-symbol-median rank_IC per (resolution, horizon).
Env: N_SYMBOLS (default 120), OUT_TAG.
"""
import os
import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from google.cloud import storage
from sklearn.metrics import roc_auc_score
import xgboost as xgb

PROJ = "project-26a24ad0-1059-4f73-93b"
BUCKET = "blackdigital-scalper-data"
KL = "free_v1/klines_1m/"
N_SYMBOLS = int(os.environ.get("N_SYMBOLS", "120"))
OUT_TAG = os.environ.get("OUT_TAG", "")
LOOKBACK = 24
RES = {"1m": 1, "5m": 5, "15m": 15}      # minutes/bar
SUB = {"1m": 15, "5m": 3, "15m": 1}      # decide every ~15 min (in bars)
ABS_H = [15, 60, 240]                    # absolute forward horizons (minutes)
TRAIN_FRAC = 0.70
EMB_MS = 24 * 3600 * 1000                # embargo gap each side of split
MIN_MS = 60_000

fs = pafs.GcsFileSystem()
sc = storage.Client(project=PROJ)


def pick_symbols(n):
    """top-n longest-history symbols by 1m row count (cheap footer read)."""
    blobs = [b.name for b in sc.list_blobs(BUCKET, prefix=KL) if b.name.endswith(".parquet")]
    rows = []
    for nm in blobs:
        try:
            md = pq.ParquetFile(fs.open_input_file(f"{BUCKET}/{nm}")).metadata
            rows.append((nm.split("/")[-1].replace(".parquet", ""), md.num_rows))
        except Exception:
            pass
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def load_1m(sym):
    pf = pq.ParquetFile(fs.open_input_file(f"{BUCKET}/{KL}{sym}.parquet"))
    t = pf.read(columns=["open_time", "open", "high", "low", "close", "volume", "taker_buy_base"])
    d = t.to_pandas()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open_time", "close"]).sort_values("open_time").drop_duplicates("open_time")
    return d


def resample(d1, r):
    if r == 1:
        return d1.reset_index(drop=True)
    g = (d1["open_time"] // (r * MIN_MS)) * (r * MIN_MS)
    out = pd.DataFrame({
        "open_time": g.values,
        "open": d1["open"].values, "high": d1["high"].values, "low": d1["low"].values,
        "close": d1["close"].values, "volume": d1["volume"].values, "taker_buy_base": d1["taker_buy_base"].values,
    }).groupby("open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), taker_buy_base=("taker_buy_base", "sum"))
    return out


def base_feats(b):
    c = b["close"].to_numpy()
    ret = np.zeros(len(c)); ret[1:] = np.log(c[1:] / c[:-1])
    rng = ((b["high"] - b["low"]) / b["close"].replace(0, np.nan)).fillna(0).to_numpy()
    lv = np.log1p(b["volume"].clip(lower=0).to_numpy())
    vol = b["volume"].to_numpy()
    tib = np.where(vol > 0, b["taker_buy_base"].to_numpy() / vol - 0.5, 0.0)
    return np.column_stack([ret, rng, lv, tib])  # (T,4)


# ---- collect pooled decision rows per resolution ----
syms = pick_symbols(N_SYMBOLS)
print(f"symbols={len(syms)} N_SYMBOLS={N_SYMBOLS}", flush=True)
pool = {r: {"X": [], "t": [], "sym": [], "fwd": {h: [] for h in ABS_H}} for r in RES}

for si, sym in enumerate(syms):
    try:
        d1 = load_1m(sym)
    except Exception as e:
        print("load fail", sym, str(e)[:80], flush=True); continue
    if len(d1) < 5000:
        continue
    for r, rm in RES.items():
        b = resample(d1, rm)
        if len(b) < LOOKBACK + max(ABS_H) // rm + 50:
            continue
        F = base_feats(b)                      # (T,4) raw base feats
        t = b["open_time"].to_numpy()
        c = b["close"].to_numpy()
        T = len(b)
        step = SUB[r]
        Hbars = {h: h // rm for h in ABS_H}
        idx = np.arange(LOOKBACK, T - max(Hbars.values()) - 1, step)
        if len(idx) == 0:
            continue
        # raw lagged feature block (LOOKBACK x 4) flattened, per decision point
        win = np.stack([F[i - LOOKBACK:i].reshape(-1) for i in idx])   # (n, LOOKBACK*4)
        pool[r]["X"].append(win.astype(np.float32))
        pool[r]["t"].append(t[idx])
        pool[r]["sym"].append(np.full(len(idx), si))
        for h in ABS_H:
            hb = Hbars[h]
            fwd = np.log(c[idx + hb] / c[idx])
            pool[r]["fwd"][h].append(fwd.astype(np.float32))
    if (si + 1) % 20 == 0:
        print(f"  built {si+1}/{len(syms)}", flush=True)

results = {}
for r in RES:
    if not pool[r]["X"]:
        continue
    X = np.concatenate(pool[r]["X"]); t = np.concatenate(pool[r]["t"]); sym = np.concatenate(pool[r]["sym"])
    # per-symbol train-fit z-score (cols), using rows before global cutoff
    cutoff = t.min() + TRAIN_FRAC * (t.max() - t.min())
    for s in np.unique(sym):
        m = sym == s
        tr = m & (t < cutoff)
        if tr.sum() < 50:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1
        X[m] = (X[m] - mu) / sd
    print(f"\n=== resolution {r}: pooled rows={len(X)} feats={X.shape[1]} cutoff={pd.to_datetime(int(cutoff),unit='ms',utc=True)} ===", flush=True)
    for h in ABS_H:
        fwd = np.concatenate(pool[r]["fwd"][h])
        y = (fwd > 0).astype(int)
        # honest split + embargo gap around cutoff
        tr = t < (cutoff - EMB_MS)
        te = t > (cutoff + EMB_MS)
        if tr.sum() < 1000 or te.sum() < 1000:
            continue
        w = np.minimum(np.abs(fwd[tr]), np.quantile(np.abs(fwd[tr]), 0.99))
        dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w)
        dte = xgb.DMatrix(X[te])
        bst = xgb.train({"objective": "binary:logistic", "max_depth": 5, "eta": 0.05,
                         "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                         "nthread": 0, "eval_metric": "logloss"}, dtr, num_boost_round=250)
        p = bst.predict(dte)
        auc = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan
        # per-symbol median AUC on test
        psym = sym[te]; pe = p; ye = y[te]
        per = []
        for s in np.unique(psym):
            mm = psym == s
            if mm.sum() >= 100 and len(np.unique(ye[mm])) > 1:
                per.append(roc_auc_score(ye[mm], pe[mm]) - 0.5)
        topdec = float(np.median(np.abs(fwd[te][p >= np.quantile(p, 0.9)]))) * 100
        cell = {"rank_IC": round(float(auc - 0.5), 4), "per_sym_median": round(float(np.median(per)), 4) if per else None,
                "n_oos": int(te.sum()), "pos_frac": round(float(y[te].mean()), 3),
                "top_decile_absmove_pct": round(topdec, 3), "n_sym": int(len(per))}
        results[f"{r}/H{h}m"] = cell
        print(f"  H={h:>3}m rank_IC={cell['rank_IC']:+.4f} per_sym_med={cell['per_sym_median']} "
              f"n_oos={cell['n_oos']} pos={cell['pos_frac']} topdec|mv|={cell['top_decile_absmove_pct']}%", flush=True)

out = {"n_symbols": len(syms), "lookback": LOOKBACK, "abs_horizons_min": ABS_H, "results": results}
fn = f"hdata_rawbars{OUT_TAG}.json"
json.dump(out, open(f"/root/{fn}" if os.path.exists("/root") else f"C:/Dev/sb-data-poc/{fn}", "w"), indent=2)
try:
    sc.bucket(BUCKET).blob(f"research_runs/hdata_rawbars/{fn}").upload_from_string(json.dumps(out, indent=2))
    print(f"uploaded research_runs/hdata_rawbars/{fn}", flush=True)
except Exception as e:
    print("gcs upload skipped:", str(e)[:80], flush=True)
print("RAWBARS_DONE", flush=True)
