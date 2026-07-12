#!/usr/bin/env python3
"""HD4 rev1 — stage-1 screen: model-free 10s-ahead direction predictivity of the deployed
F71 feature signals (no ML, no fitting) on the tb3s 3s decision grid, CL year.

Motivation (hypothesis-2 of the BTC/ETH expansion): extend the h150 hold in 10s quanta
while a direction signal at t=fill+150s keeps pointing the position's way. Stage 1 asks
the prerequisite question only: is the NEXT-10s mid direction predictable AT ALL from the
deploy feature set, per symbol, and under what conditions (signal, horizon, strength)?

Per day:
  raw book (ts, bid0, ask0)  ->  mid series
  tb3s h150 daily npz (F 71 cols, ts)  ->  features at the same decision ticks
  forward mid log-returns at H in {5,10,15,20,30,60}s:
      j = last book tick <= t+H;  valid iff ts[j] >= t+H-2s (horizon in [H-2, H], no
      future-side slack -> no horizon stretching), both mids > 0.
  stats per (feature, horizon): daily rank-IC; dir-hit & signed capture mean(sign(x)*r)
      on |x| >= day-quantile q for q in {0.0, 0.5, 0.9, 0.99}; feature 71 = COMP, the
      preregistered fixed-sign rank composite over directional cols
      [0,1,12,26,27,28,62,63,64,65,66] (all prior sign +).

Artifacts (capture-everything):
  gs://.../research_runs/h2_dir10/daily/{SYM}_{day}.npz   ts, mid, bid0, ask0, R (6,N), RV (6,N)
  gs://.../research_runs/h2_dir10/{SYM}_dirstats.npz      per-day stat tensors + day list

Env: SYMF (DOGE-USDT-PERP), START, END, LABSUB (maker_labels_tb3s_h150), WORKDIR.
Run on an in-region VM (bucket EUROPE-WEST1) — zero egress.
"""
import io, os, time
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYMF = os.environ.get("SYMF", "DOGE-USDT-PERP"); SYMK = SYMF.split("-")[0]
RAWB = f"raw/book/exchange=BINANCE_FUTURES/symbol={SYMF}"
LABSUB = os.environ.get("LABSUB", "research_runs/maker_labels_tb3s_h150")
OUT = "research_runs/h2_dir10"
START = os.environ.get("START", "2025-05-09"); END = os.environ.get("END", "2026-06-02")
NS = 1_000_000_000
HORS = [5.0, 10.0, 15.0, 20.0, 30.0, 60.0]
TOL_S = 2.0                      # horizon realized in [H-TOL, H]
QS = [0.0, 0.5, 0.9, 0.99]       # |signal| strength cuts (per-day quantiles of |x|)
COMP_COLS = [0, 1, 12, 26, 27, 28, 62, 63, 64, 65, 66]  # preregistered, all prior sign +
NF = 72                          # 71 features + COMP at index 71
TD = os.environ.get("WORKDIR", os.path.expanduser("~/dir10")); os.makedirs(TD, exist_ok=True)

cl = storage.Client(project=PROJ); bk = cl.bucket(BUCKET)


def log(s):
    print(s, flush=True)


def dl_raw(prefix, dst):
    name = next((b.name for b in cl.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not name:
        return False
    bk.blob(name).download_to_filename(dst); return True


def rank01(x):
    """average-rank -> (0,1); ties get equal rank (dense enough for rank-IC)."""
    o = np.argsort(x, kind="stable"); r = np.empty(len(x), np.float64); r[o] = np.arange(len(x))
    # average ties
    xs = x[o]; b = np.r_[True, xs[1:] != xs[:-1]]; g = np.cumsum(b) - 1
    cnt = np.bincount(g); csum = np.bincount(g, weights=np.arange(len(x)))
    r[o] = (csum / cnt)[g]
    return (r + 0.5) / len(x)


def day_stats(F, R, RV):
    """per-day tensors: ric (NF,NH), hit (NF,NH,NQ), cap (NF,NH,NQ), n (NF,NH,NQ)."""
    nh = len(HORS)
    ric = np.full((NF, nh), np.nan, np.float32)
    hit = np.full((NF, nh, len(QS)), np.nan, np.float32)
    cap = np.full((NF, nh, len(QS)), np.nan, np.float32)
    cnt = np.zeros((NF, nh, len(QS)), np.int32)
    # composite = sum of centered ranks of COMP_COLS (computed on all rows of the day)
    comp = np.zeros(len(F), np.float64); alive = 0
    for c in COMP_COLS:
        x = F[:, c].astype(np.float64)
        if np.nanstd(x) > 0:
            comp += rank01(x) - 0.5; alive += 1
    Xall = np.concatenate([F.astype(np.float64), comp[:, None]], axis=1)
    for h in range(nh):
        v = RV[h]
        if v.sum() < 500:
            continue
        r = R[h][v].astype(np.float64)
        rr = rank01(r)
        for f in range(NF):
            x = Xall[v, f]
            sd = x.std()
            if not np.isfinite(sd) or sd == 0:
                continue
            rx = rank01(x)
            ric[f, h] = np.corrcoef(rx, rr)[0, 1]
            ax = np.abs(x - np.median(x)) if f == 71 else np.abs(x)
            # sign relative to a day-neutral center: raw sign for signed feats; rank-based
            # for COMP (already centered). Features that are strictly positive-valued get
            # sign from centered rank too (|IC| still meaningful; hit needs a center).
            sgn = np.sign(x) if (x < 0).any() and (x > 0).any() else np.sign(rx - 0.5)
            for qi, q in enumerate(QS):
                thr = np.quantile(ax, q) if q > 0 else -1.0
                m = ax >= thr
                if m.sum() < 20:
                    continue
                s = sgn[m]; rm = r[m]
                nz = s != 0
                if nz.sum() < 20:
                    continue
                hit[f, h, qi] = np.mean(np.sign(rm[nz]) == s[nz])
                cap[f, h, qi] = np.mean(s[nz] * rm[nz])
                cnt[f, h, qi] = int(nz.sum())
    return ric, hit, cap, cnt, alive


def process_day(day):
    bp = f"{TD}/b.parquet"
    if not dl_raw(f"{RAWB}/dt={day}/", bp):
        return None, "no-raw"
    try:
        z = np.load(io.BytesIO(bk.blob(f"{LABSUB}/daily/{SYMK}_{day}.npz").download_as_bytes()))
    except Exception:
        return None, "no-npz"
    F = z["F"]; dtd = z["ts"].astype(np.int64)
    tbl = pq.read_table(bp, columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts = tbl["timestamp"].to_numpy().astype(np.int64)
    b0 = tbl["bid_0_price"].to_numpy().astype(np.float64); a0 = tbl["ask_0_price"].to_numpy().astype(np.float64)
    o = np.argsort(ts, kind="stable"); ts, b0, a0 = ts[o], b0[o], a0[o]
    mid = np.where((b0 > 0) & (a0 > 0), 0.5 * (b0 + a0), 0.0)
    ii = np.clip(np.searchsorted(ts, dtd, "right") - 1, 0, len(ts) - 1)
    exact = float(np.mean(ts[ii] == dtd))
    m0 = mid[ii]
    R = np.full((len(HORS), len(dtd)), np.nan, np.float32)
    RV = np.zeros((len(HORS), len(dtd)), bool)
    for h, H in enumerate(HORS):
        tgt = dtd + int(H * NS)
        j = np.clip(np.searchsorted(ts, tgt, "right") - 1, 0, len(ts) - 1)
        ok = (ts[j] >= tgt - int(TOL_S * NS)) & (ts[j] <= tgt) & (mid[j] > 0) & (m0 > 0)
        R[h, ok] = (np.log(mid[j][ok] / m0[ok]) * 1e4).astype(np.float32)
        RV[h] = ok
    buf = io.BytesIO()
    np.savez_compressed(buf, ts=dtd, mid=m0.astype(np.float64), bid0=b0[ii].astype(np.float32),
                        ask0=a0[ii].astype(np.float32), R=R, RV=RV, hors=np.array(HORS))
    bk.blob(f"{OUT}/daily/{SYMK}_{day}.npz").upload_from_string(buf.getvalue())
    st = day_stats(F, R, RV)
    cov10 = float(RV[1].mean())
    return st, f"ok n={len(dtd)} exact={exact:.3f} cov10={cov10:.3f} compAlive={st[4]}"


def main():
    days = sorted(set(b.name.split("dt=")[1][:10] for b in cl.list_blobs(bk, prefix=RAWB + "/")
                      if "dt=" in b.name and b.name.endswith(".parquet")))
    days = [d for d in days if START <= d <= END]
    log(f"[dir10 {SYMK}] {len(days)} raw days {days[0]}..{days[-1]}")
    RIC, HIT, CAP, CNT, DAYS = [], [], [], [], []
    for d in days:
        t0 = time.time()
        try:
            st, msg = process_day(d)
        except Exception as e:
            st, msg = None, f"EXC {type(e).__name__}: {e}"
        log(f"  {d}: {msg} [{time.time()-t0:.0f}s]")
        if st is not None:
            RIC.append(st[0]); HIT.append(st[1]); CAP.append(st[2]); CNT.append(st[3]); DAYS.append(d)
    buf = io.BytesIO()
    np.savez_compressed(buf, ric=np.stack(RIC), hit=np.stack(HIT), cap=np.stack(CAP),
                        cnt=np.stack(CNT), days=np.array(DAYS), hors=np.array(HORS),
                        qs=np.array(QS), comp_cols=np.array(COMP_COLS))
    bk.blob(f"{OUT}/{SYMK}_dirstats.npz").upload_from_string(buf.getvalue())
    log(f"[dir10 {SYMK}] DONE {len(DAYS)} days -> {OUT}/{SYMK}_dirstats.npz")


if __name__ == "__main__":
    main()
