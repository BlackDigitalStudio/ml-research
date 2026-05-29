#!/usr/bin/env python3
"""Model-in-loop e2e: real cache -> LOB features -> light XGBoost -> maker grid_sim -> EV.

PLUMBING test (per CLAUDE.md: confirms the FULL chain runs + coherent numbers,
NOT a deploy/alpha verdict). For 14d SOL L2:
  per day: build_samples NATIVE on flat raw L2 -> X_lob (features) + book/flow/
           entry_q (maker arrays) + mid_paths (label) + entry prices.
  concat -> label y = sign(fwd mid return over horizon).
  honest 65/35 walk-forward split. Light XGB on flattened X_lob -> p_up.
  grid_sim MAKER (--flow-paths --entry-q) -> per-config maker pnl + fill masks.
  EV: on the holdout, pick pnl by the model's predicted side, gate by max_prob,
      over FILLED trades. Report XGB holdout AUC + maker EV/trade/fill-rate.
"""
import argparse, glob, json, os, subprocess, sys, time
from datetime import date, timedelta
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BUCKET = "market-data-0998ac51"
RAW = f"gs://{BUCKET}/raw"
OK, FAIL = [], []
buf = []
def log(s): print(s, flush=True); buf.append(str(s))
def chk(n, c, d=""):
    (OK if c else FAIL).append(n); log(f"  [{'OK' if c else 'FAIL'}] {n} {d}")


def daterange(s, e):
    d0, d1 = date.fromisoformat(s), date.fromisoformat(e); out = []
    while d0 <= d1:
        out.append(d0.isoformat()); d0 += timedelta(days=1)
    return out


def one_flat(dt_dir, out_path):
    files = sorted(glob.glob(os.path.join(dt_dir, "**", "*.parquet"), recursive=True))
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    import pyarrow.parquet as pq  # concat multiple flat parts (schema preserved)
    pq.write_table(pq.read_table(files), out_path)
    return out_path


def run_day(bins, symbol, day, work, window, horizon, maxsamp):
    wd = os.path.join(work, day); os.makedirs(wd, exist_ok=True)
    subprocess.run(["gsutil", "-m", "-q", "cp", "-r",
                    f"{RAW}/book/exchange=BINANCE_FUTURES/symbol={symbol}/dt={day}/", os.path.join(wd, "b")], check=False)
    subprocess.run(["gsutil", "-m", "-q", "cp", "-r",
                    f"{RAW}/trades/exchange=BINANCE_FUTURES/symbol={symbol}/dt={day}/", os.path.join(wd, "t")], check=False)
    bp = one_flat(os.path.join(wd, "b"), os.path.join(wd, "book.parquet"))
    tp = one_flat(os.path.join(wd, "t"), os.path.join(wd, "trades.parquet"))
    if not bp or not tp:
        log(f"  {day}: missing raw -> skip"); return None
    outdir = os.path.join(wd, "bs")
    r = subprocess.run([os.path.join(bins, "build_samples"), "--depth", bp, "--trades", tp,
                        "--out-dir", outdir, "--window", str(window), "--horizon", str(horizon),
                        "--step", "2", "--max-samples", str(maxsamp)], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  {day}: build_samples FAIL {r.stderr.strip()[-160:]}"); return None
    A = {k: np.load(os.path.join(outdir, f"{k}.npy")) for k in
         ["entry_long", "entry_short", "mid", "mid_paths", "book_paths", "flow_paths", "entry_q", "X_lob"]}
    log(f"  {day}: samples={A['entry_long'].shape[0]:,} X_lob{A['X_lob'].shape}")
    subprocess.run(["rm", "-rf", os.path.join(wd, "b"), os.path.join(wd, "t"),
                    os.path.join(wd, "book.parquet"), os.path.join(wd, "trades.parquet")], check=False)
    return A


def auc(y, p):
    y = y.astype(bool); npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="stable"); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--symbol", default="SOL-USDT-PERP")
    ap.add_argument("--start", default="2026-04-25")
    ap.add_argument("--end", default="2026-05-08")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--maxsamp", type=int, default=12000)
    ap.add_argument("--work", default="/tmp/xgb")
    a = ap.parse_args()
    t0 = time.time()
    log(f"XGB model-in-loop e2e {a.symbol} {a.start}..{a.end} win={a.window} hor={a.horizon}")
    days = daterange(a.start, a.end); os.makedirs(a.work, exist_ok=True)
    keys = ["entry_long", "entry_short", "mid", "mid_paths", "book_paths", "flow_paths", "entry_q", "X_lob"]
    parts = {k: [] for k in keys}; nday = 0
    for d in days:
        try:
            A = run_day(a.bins, a.symbol, d, a.work, a.window, a.horizon, a.maxsamp)
            if A:
                for k in keys: parts[k].append(A[k])
                nday += 1
        except Exception as e:
            import traceback; log(f"  {d}: ERR {e}\n{traceback.format_exc()[-300:]}")
    chk("days built >=10", nday >= 10, f"{nday} days")
    if nday == 0:
        _save(); return
    C = {k: np.concatenate(parts[k], 0) for k in keys}
    N = C["entry_long"].shape[0]
    log(f"CONCAT {nday}d, {N:,} samples, X_lob {C['X_lob'].shape}")

    # label: sign of forward mid return over the horizon (mid_paths last vs entry mid)
    fwd = C["mid_paths"][:, -1] / np.where(C["mid"] > 0, C["mid"], np.nan) - 1.0
    y = (fwd > 0).astype(int)
    X = C["X_lob"].reshape(N, -1).astype(np.float32)
    hs = int(N * 0.65)
    chk("X finite", np.isfinite(X).all(), "")
    log(f"  features={X.shape[1]} train={hs:,} holdout={N-hs:,} pos_rate={y.mean():.3f}")

    try:
        import xgboost as xgb
    except Exception as e:
        chk("xgboost available", False, str(e)); _save(); return
    # native xgboost API (xgb.train/DMatrix) — no sklearn dependency.
    mtr = np.isfinite(fwd[:hs])
    dtrain = xgb.DMatrix(X[:hs][mtr], label=y[:hs][mtr].astype(np.float32))
    dall = xgb.DMatrix(X)
    params = {"max_depth": 4, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.7,
              "tree_method": "hist", "objective": "binary:logistic",
              "eval_metric": "logloss", "nthread": 4}
    bst = xgb.train(params, dtrain, num_boost_round=80)
    p_up = bst.predict(dall)
    a_oos = auc(y[hs:][np.isfinite(fwd[hs:])], p_up[hs:][np.isfinite(fwd[hs:])])
    acc = float(((p_up[hs:] >= 0.5).astype(int) == y[hs:])[np.isfinite(fwd[hs:])].mean())
    chk("xgb trained + scored", True, f"holdout AUC={a_oos:.4f} acc={acc:.3f}")
    chk("xgb has SOME signal (AUC!=~0.5 sanity, not a verdict)", abs(a_oos - 0.5) > 0.005, f"AUC={a_oos:.4f}")

    pred = np.where(p_up >= 0.5, 0, 1).astype(np.int64)   # 0=UP,1=DN
    mprob = np.maximum(p_up, 1 - p_up).astype(np.float64)

    p = lambda f: os.path.join(a.work, f)
    np.save(p("el.npy"), C["entry_long"]); np.save(p("es.npy"), C["entry_short"])
    np.save(p("mid_p.npy"), C["mid_paths"]); np.save(p("book.npy"), C["book_paths"])
    np.save(p("flow.npy"), C["flow_paths"].astype(np.float32)); np.save(p("eq.npy"), C["entry_q"])
    np.save(p("eb.npy"), np.stack([C["entry_long"], C["entry_short"]], 1))
    cfgs = [{"tp": 0.13, "sl": 0.05, "to": 60, "par": False, "tr": False},
            {"tp": 0.20, "sl": 0.08, "to": 90, "par": False, "tr": False}]
    json.dump(cfgs, open(p("cfg.json"), "w"))
    r = subprocess.run([os.path.join(a.bins, "grid_sim"),
                        "--entry-long", p("el.npy"), "--entry-short", p("es.npy"), "--mid-paths", p("mid_p.npy"),
                        "--book-paths", p("book.npy"), "--entry-book", p("eb.npy"),
                        "--flow-paths", p("flow.npy"), "--entry-q", p("eq.npy"),
                        "--configs", p("cfg.json"), "--out-prefix", p("mk"),
                        "--commission-win-pct", "0", "--commission-loss-pct", "0",
                        "--entry-window-ticks", "60", "--queue-mult", "0"], capture_output=True, text=True)
    chk("grid_sim maker rc==0", r.returncode == 0, r.stderr.strip()[-200:])
    if r.returncode != 0:
        _save(); return
    pl = np.load(p("mk_pnl_long.npy")); ps = np.load(p("mk_pnl_short.npy"))
    fl = np.load(p("mk_filled_long.npy")); fsm = np.load(p("mk_filled_short.npy"))
    ndays_oos = nday * (1 - 0.65)
    log("\n  MODEL-GATED MAKER EV on holdout (XGB-predicted side, filled only):")
    for ci in range(pl.shape[0]):
        chosen = np.where(pred == 0, pl[ci], ps[ci])  # predicted-side maker pnl
        for thr in [0.50, 0.55]:
            sel = np.zeros(pl.shape[1], bool); sel[hs:] = True
            sel &= (mprob >= thr) & np.isfinite(chosen)
            n = int(sel.sum())
            if n < 50:
                log(f"   cfg{ci} thr{thr}: n={n} (too few)"); continue
            ev = float(np.mean(chosen[sel])); wr = float(np.mean(chosen[sel] > 0))
            log(f"   cfg{ci} tp{cfgs[ci]['tp']}/sl{cfgs[ci]['sl']} thr{thr}: "
                f"n={n} EV/tr={ev:+.4f}% WR={wr*100:.1f}% trd/d={n/ndays_oos:.0f}")
    log(f"  fill-rate long={fl.mean():.3f} short={fsm.mean():.3f}")
    chk("model-in-loop chain produced EV numbers", True, "")
    log(f"\nDONE in {time.time()-t0:.0f}s"); log("XGB_E2E_DONE")
    _save()


def _save():
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tn = f"husdc_xgb_e2e_{stamp}.txt"
    open(tn, "w", encoding="utf-8").write("\n".join(buf))
    res = "PASS" if (not FAIL and any("XGB_E2E_DONE" in b for b in buf)) else "FAIL"
    try:
        from google.cloud import storage
        bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket(BUCKET)
        bk.blob(f"research_runs/husdc/{tn}").upload_from_filename(tn)
        bk.blob("research_runs/husdc/HUSDC_XGB_STATUS.txt").upload_from_string(
            f"{res} {stamp} OK={len(OK)} FAIL={len(FAIL)} {FAIL}")
        print(f"[saved] {tn}", flush=True)
    except Exception as ex:
        print("[save-warn]", ex, flush=True)


if __name__ == "__main__":
    main()
