#!/usr/bin/env python3
"""STRICT ENTRY-FILL relabel of the year datasets, on the SAME (USDT/Cryptolake) book.

OPS-FILLYEAR rev1. Answers: what do the published year cells become when the ONLY
change is the entry-fill model (frozen unconditional gap-through -> strict
price-resolved queue rule, OPS-EXEC rev16)? rev17/rev20 measured that correction on
12-27 day recorder windows *on the USDC book*, where venue and model effects are
entangled. Here the venue is held fixed by construction, and the window is the full
walk-forward year.

WHAT IT DOES NOT DO: it does not rebuild features. F / rH15 / rH30 / rH60 / ts are
reused verbatim from the parent daily npz — the signal layer is not in question, and
reusing it makes the row join exact by construction. Only fills and PnL are recomputed.

Per day, driven off the PARENT dailies (so coverage matches the dataset exactly):
  1. download the same raw book+trades parquet, apply the same trade-id dedup;
  2. recompute the decision grid EXACTLY as scripts/subs60_build_tb3s_labels.py does;
  3. build_samples ... --emit-level-flow   (additive: the frozen aggregation path is
     byte-identical when the flag is off, so the paths are the same paths);
  4. grid_sim_exitdbg TWICE on those paths — once with the FROZEN flags (parity gate),
     once with --strict-entry-fill --level-flow-paths (the cell);
  5. PARITY GATE: rebuilt ts must equal the stored ts row-for-row, and the frozen
     rebuild must reproduce the stored pnl_long/pnl_short/FL/FS bit-exactly. A day that
     fails the gate is written with parity=False and is excluded from the combine.

COMBINE=1 concatenates in the parent's order, applying the parent's own
isfinite(rH30) row mask, so the output is row-aligned with the parent combined npz
(and therefore with every PERFOLD_S* score artifact derived from it).

Env:
  SYM        symbol key (DOGE/XRP/BTC/ETH)
  SRC_SUB    parent dataset subdir   (research_runs/maker_labels_tb3s_h150 | ..._h150d)
  OUT_SUB    output subdir           (default SRC_SUB + "strict")
  H_TICKS    MUST match the parent build or the decision grid shifts and nothing lines up.
             MEASURED per symbol (2026-07-30, exact ts match against the parent dailies on
             two sampled days each — do NOT assume the cross-symbol 1800):
                 DOGE 1500  (built by the DOGE-dedicated subs60_tb3s_h150_build.py)
                 XRP  1800  (cross-symbol h150 protocol)
                 BTC  5100  (h150d dense rebuild)   ETH 5100 (h150d)
  START END  day range (default the full year window)
  NWORK      parallel day workers (default 4)
  WORKDIR BINS
  COMBINE=1  combine step instead of the day loop
  LIMIT      process at most N days (smoke)
  DAYS       explicit comma-separated day list (smoke / re-run of failures)

Frozen-by-parent and NOT parameterised here: ENTRY_MS=60000, HOLDS_S=90,150,240,
CHASE_MS=300000, STEP_S=3, W=50, queue_mult=exit_queue_mult=1.0, commission 0/0.
"""
import io, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
SYMF = f"{SYM}-USDT-PERP"
RAWB = f"raw/book/exchange=BINANCE_FUTURES/symbol={SYMF}"
RAWT = f"raw/trades/exchange=BINANCE_FUTURES/symbol={SYMF}"
SRC_SUB = os.environ.get("SRC_SUB", "research_runs/maker_labels_tb3s_h150")
OUT_SUB = os.environ.get("OUT_SUB", SRC_SUB + "strict")
H_TICKS = int(os.environ.get("H_TICKS", "1800"))
START = os.environ.get("START", "2025-05-09"); END = os.environ.get("END", "2026-06-02")
NWORK = int(os.environ.get("NWORK", "4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
DAYS_ENV = os.environ.get("DAYS", "")
REDO_BAD = os.environ.get("REDO_BAD", "") == "1"
COMBINE = os.environ.get("COMBINE", "") == "1"
BINS = os.environ.get("BINS", "/home/delmi/research_bins")
BS = os.environ.get("BS_BIN", f"{BINS}/husdc_target/release/build_samples")
GRID = os.environ.get("GRID_BIN", f"{BINS}/husdc_target/release/grid_sim_exitdbg")
TD = os.environ.get("WORKDIR", f"/home/delmi/strictfill_{SYM}")

NS = 1_000_000_000
W = 50; STEP_S = 3.0
ENTRY_MS = 60_000; CHASE_MS = 300_000; HOLDS_S = [90.0, 150.0, 240.0]
QM = 1.0; EXIT_QM = 1.0

os.makedirs(TD, exist_ok=True)
cl = storage.Client(project=PROJ); bk = cl.bucket(BUCKET)
_pr = __import__("threading").Lock()


def log(s):
    with _pr:
        print(s, flush=True)


def dl_raw(prefix, dst):
    """Same first-match rule as the frozen builder."""
    name = next((b.name for b in cl.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not name:
        return False
    bk.blob(name).download_to_filename(dst); return True


def parent_days():
    out = []
    for b in cl.list_blobs(bk, prefix=f"{SRC_SUB}/daily/{SYM}_"):
        if not b.name.endswith(".npz"):
            continue
        d = b.name.split("_")[-1][:-4]
        if START <= d <= END:
            out.append(d)
    return sorted(set(out))


def run_day(day, _unused=None):
    """Returns a dict record for the day (parity flags + fill/EV summary).

    The workdir is PER DAY, not per worker slot: a `w{i % NWORK}` scheme keys off the
    task index, not the executing thread, so two concurrent days could share a dir and
    race on b.parquet / bs/*.npy (observed 2026-07-30 as BS-fail 'end of file' and a
    missing entry_q.npy). The parity gates caught it - no bad day could enter the
    combine - but the days were lost and had to be re-run.
    """
    wd = f"{TD}/{day}"
    os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    src = bk.blob(f"{SRC_SUB}/daily/{SYM}_{day}.npz")
    if not src.exists():
        return {"day": day, "status": "no-parent"}
    par = np.load(io.BytesIO(src.download_as_bytes()))
    bp, tp = f"{wd}/b.parquet", f"{wd}/t.parquet"
    if not (dl_raw(f"{RAWB}/dt={day}/", bp) and dl_raw(f"{RAWT}/dt={day}/", tp)):
        return {"day": day, "status": "no-raw"}
    # trade-id dedup, byte-identical to the frozen builder
    tt = pq.read_table(tp)
    ids = tt["id"].to_numpy()
    dup = len(ids) / max(len(np.unique(ids)), 1)
    if dup > 1.001:
        _, ui = np.unique(ids, return_index=True)
        pq.write_table(tt.take(np.sort(ui)), tp)
    tbl = pq.read_table(bp, columns=["timestamp", "bid_0_price", "ask_0_price"])
    ts_ns = tbl["timestamp"].to_numpy().astype(np.int64)
    n = len(ts_ns)
    if n < W + H_TICKS + 100:
        return {"day": day, "status": f"thin-book n={n}"}
    grid = np.arange(ts_ns[0], ts_ns[-1], int(STEP_S * NS))
    ends = np.unique(np.clip(np.searchsorted(ts_ns, grid, "right") - 1, 0, n - 1))
    ends = ends[(ends >= W - 1) & (ends < n - H_TICKS - 1)].astype(np.int64)
    if len(ends) < 100:
        return {"day": day, "status": f"few-ends {len(ends)}"}
    np.save(f"{wd}/ends.npy", ends)
    od = f"{wd}/bs"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(f"{od}/{f}")
    r = subprocess.run([BS, "--depth", bp, "--trades", tp, "--out-dir", od, "--window", str(W),
                        "--horizon", str(H_TICKS), "--sample-ends", f"{wd}/ends.npy", "--skip-xlob",
                        "--emit-level-flow"], capture_output=True, text=True)
    if r.returncode != 0:
        return {"day": day, "status": f"BS-fail:{r.stderr[-200:]}"}
    se = np.load(f"{od}/end_indices.npy").astype(np.int64)
    dtd = ts_ns[se]
    ts_ok = bool(len(dtd) == len(par["ts"]) and np.array_equal(dtd, par["ts"]))
    cfgs = [{"tp": 50.0, "sl": 50.0, "to": 282, "to_ms": hs * 1000.0, "par": False, "tr": False} for hs in HOLDS_S]
    json.dump(cfgs, open(f"{wd}/cfg.json", "w"))

    def grid_run(tag, extra):
        g = f"{wd}/g{tag}"
        cmd = [GRID, "--entry-long", f"{od}/entry_long.npy", "--entry-short", f"{od}/entry_short.npy",
               "--mid-paths", f"{od}/mid_paths.npy", "--book-paths", f"{od}/book_paths.npy",
               "--entry-book", f"{od}/entry_book.npy", "--flow-paths", f"{od}/flow_paths.npy",
               "--entry-q", f"{od}/entry_q.npy", "--configs", f"{wd}/cfg.json", "--out-prefix", g,
               "--queue-mult", str(QM), "--exit-queue-mult", str(EXIT_QM),
               "--entry-window-ticks", "120", "--maker-offset-frac", "0",
               "--commission-win-pct", "0", "--commission-loss-pct", "0",
               "--ts-paths", f"{od}/ts_paths.npy", "--sample-ts", f"{od}/sample_ts.npy",
               "--entry-window-ms", str(ENTRY_MS), "--chase-ms", str(CHASE_MS)] + extra
        rr = subprocess.run(cmd, capture_output=True, text=True)
        if rr.returncode != 0:
            raise RuntimeError(f"GRID-{tag}-fail:{rr.stderr[-250:]}")
        return (np.load(f"{g}_pnl_long.npy"), np.load(f"{g}_pnl_short.npy"),
                np.load(f"{g}_filled_long.npy"), np.load(f"{g}_filled_short.npy"))

    try:
        PLf, PSf, FLf, FSf = grid_run("F", [])
        PLs, PSs, FLs, FSs = grid_run("S", ["--strict-entry-fill",
                                            "--level-flow-paths", f"{od}/flow_lvl_paths.npy"])
    except RuntimeError as e:
        return {"day": day, "status": str(e)}

    # equal_nan: unfilled rows carry NaN PnL on BOTH sides, and NaN != NaN would make
    # a bit-identical rebuild look like a parity failure (it did, on the first smoke).
    lab_ok = bool(
        PLf.shape == par["pnl_long"].shape
        and np.array_equal(PLf.astype(np.float32), par["pnl_long"], equal_nan=True)
        and np.array_equal(PSf.astype(np.float32), par["pnl_short"], equal_nan=True)
        and np.array_equal(FLf.astype(np.uint8), par["FL"])
        and np.array_equal(FSf.astype(np.uint8), par["FS"]))
    parity = ts_ok and lab_ok

    buf = io.BytesIO()
    np.savez_compressed(buf, ts=dtd,
                        pnl_long_f=PLf.astype(np.float32), pnl_short_f=PSf.astype(np.float32),
                        FL_f=FLf.astype(np.uint8), FS_f=FSf.astype(np.uint8),
                        pnl_long_s=PLs.astype(np.float32), pnl_short_s=PSs.astype(np.float32),
                        FL_s=FLs.astype(np.uint8), FS_s=FSs.astype(np.uint8),
                        parity=np.array([ts_ok, lab_ok]))
    bk.blob(f"{OUT_SUB}/daily/{SYM}_{day}.npz").upload_from_string(buf.getvalue())
    __import__("shutil").rmtree(wd, ignore_errors=True)

    i150 = HOLDS_S.index(150.0)
    flf = FLf.astype(bool); fls = FLs.astype(bool)
    return {"day": day, "status": "ok", "parity": parity, "ts_ok": ts_ok, "lab_ok": lab_ok,
            "n": int(len(se)), "dup": round(float(dup), 3),
            "fill_f": round(float(flf.mean()), 4), "fill_s": round(float(fls.mean()), 4),
            "netl_f": round(float(np.nanmean(np.where(flf, PLf[i150] * 100.0, np.nan))), 3),
            "netl_s": round(float(np.nanmean(np.where(fls, PLs[i150] * 100.0, np.nan))), 3),
            "sec": round(time.time() - t0, 1)}


def combine():
    """Concatenate strict labels in the PARENT's order, with the parent's row mask."""
    pblobs = sorted(b.name for b in cl.list_blobs(bk, prefix=f"{SRC_SUB}/daily/{SYM}_")
                    if b.name.endswith(".npz"))
    log(f"[combine] parent dailies: {len(pblobs)}")
    aPLs, aPSs, aFLs, aFSs, aPLf, aPSf, aFLf, aFSf, aDay, aTs = [], [], [], [], [], [], [], [], [], []
    bad = []
    for di, nm in enumerate(pblobs):
        day = nm.split("_")[-1][:-4]
        par = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
        sb = bk.blob(f"{OUT_SUB}/daily/{SYM}_{day}.npz")
        if not sb.exists():
            bad.append((day, "missing")); continue
        z = np.load(io.BytesIO(sb.download_as_bytes()))
        if not bool(z["parity"].all()):
            bad.append((day, "parity")); continue
        m = np.isfinite(par["rH30"])
        aPLs.append(z["pnl_long_s"][:, m]); aPSs.append(z["pnl_short_s"][:, m])
        aFLs.append(z["FL_s"][m]); aFSs.append(z["FS_s"][m])
        aPLf.append(z["pnl_long_f"][:, m]); aPSf.append(z["pnl_short_f"][:, m])
        aFLf.append(z["FL_f"][m]); aFSf.append(z["FS_f"][m])
        aTs.append(z["ts"][m]); aDay.append(np.full(int(m.sum()), di, np.int32))
        if di % 40 == 0:
            log(f"  {di}/{len(pblobs)} {day}")
    if bad:
        log(f"[combine] EXCLUDED {len(bad)} days: {bad[:20]}")
        raise SystemExit("refusing to combine with missing/parity-failed days — "
                         "re-run them with DAYS=... first (the parent day index must stay dense)")
    meta = {"symbol": SYMF, "src_sub": SRC_SUB, "h_ticks": H_TICKS, "n_days": len(pblobs),
            "entry_ms": ENTRY_MS, "chase_ms": CHASE_MS, "holds_s": HOLDS_S,
            "note": "strict price-resolved entry fill (OPS-EXEC rev16) on the USDT book; "
                    "frozen-model labels carried alongside as the parity reference; "
                    "row order identical to the parent combined npz"}
    buf = io.BytesIO()
    np.savez_compressed(buf, day=np.concatenate(aDay), ts=np.concatenate(aTs),
                        pnl_long=np.concatenate(aPLs, 1)[:, None, :],
                        pnl_short=np.concatenate(aPSs, 1)[:, None, :],
                        fill_long=np.concatenate(aFLs)[None, :],
                        fill_short=np.concatenate(aFSs)[None, :],
                        pnl_long_frozen=np.concatenate(aPLf, 1)[:, None, :],
                        pnl_short_frozen=np.concatenate(aPSf, 1)[:, None, :],
                        fill_long_frozen=np.concatenate(aFLf)[None, :],
                        fill_short_frozen=np.concatenate(aFSf)[None, :],
                        meta=np.array(json.dumps(meta)))
    bk.blob(f"{OUT_SUB}/{SYM}.npz").upload_from_string(buf.getvalue())
    log(f"[saved] gs://{BUCKET}/{OUT_SUB}/{SYM}.npz N={len(np.concatenate(aDay))} "
        f"days={len(pblobs)} ({buf.tell()/1e6:.0f}MB)")


def main():
    if COMBINE:
        combine(); return
    days = [d for d in DAYS_ENV.split(",") if d] or parent_days()
    done = set()
    for b in cl.list_blobs(bk, prefix=f"{OUT_SUB}/daily/{SYM}_"):
        if not b.name.endswith(".npz"):
            continue
        d = b.name.split("_")[-1][:-4]
        if REDO_BAD:
            # a day that exists but failed a parity gate is NOT done - it would be
            # excluded from the combine, so re-run it rather than silently keep it
            z = np.load(io.BytesIO(b.download_as_bytes()))
            if not bool(z["parity"].all()):
                continue
        done.add(d)
    todo = [d for d in days if d not in done or DAYS_ENV]
    if LIMIT:
        todo = todo[:LIMIT]
    log(f"[strictfill {SYM}] src={SRC_SUB} out={OUT_SUB} H={H_TICKS} "
        f"days={len(days)} todo={len(todo)} workers={NWORK}")
    t0 = time.time(); recs = []
    with ThreadPoolExecutor(max_workers=NWORK) as ex:
        futs = [ex.submit(run_day, d) for d in todo]
        for i, f in enumerate(futs):
            try:
                r = f.result()
            except Exception as e:
                r = {"day": "?", "status": f"EXC {type(e).__name__}: {e}"}
            recs.append(r)
            log(f"  [{i+1}/{len(todo)} {(time.time()-t0)/60:.0f}m] {json.dumps(r)}")
    ok = [r for r in recs if r.get("status") == "ok"]
    npar = [r for r in ok if not r.get("parity")]
    log(f"[DONE] ok={len(ok)}/{len(recs)} parity-fail={len(npar)} "
        f"{[r['day'] for r in npar][:20]}")
    if ok:
        log(f"[SUMMARY] fill frozen={np.mean([r['fill_f'] for r in ok]):.4f} "
            f"strict={np.mean([r['fill_s'] for r in ok]):.4f} | "
            f"netl150(filled) frozen={np.nanmean([r['netl_f'] for r in ok]):+.3f}bp "
            f"strict={np.nanmean([r['netl_s'] for r in ok]):+.3f}bp")
    bk.blob(f"{OUT_SUB}/RUNLOG_{SYM}_{int(t0)}.json").upload_from_string(json.dumps(recs))


if __name__ == "__main__":
    main()
