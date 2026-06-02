"""Build TAKER labels (hold-60s) for training model B on taker movements (GRU stage-2 executed-payoff).

Taker entry = cross the spread at t0 (immediate fill at ask for long / bid for short), hold 60s, exit at
the timeout. These gross PL/PS are the executed-payoff sidecar for the B2 fine-tune
(L = -mean[sigma(z)*PL + (1-sigma(z))*PS]).  A (vol) and B-stage1 (IC on rH) are unchanged.

Implementation: grid_sim in TAKER mode (no --flow-paths => entry at entry_long/entry_short, immediate at
t0) on the saved maker_paths, hold config. Aligned to feats windows by ts; saved to taker_labels/{SYM}.npz
with pnl_long/pnl_short (gross, bp), ts. Reuses the validated Rust sim (same exit convention as maker).
"""
import argparse, io, json, os, subprocess, tempfile, shutil
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
PATHS = "research_runs/maker_paths"; OUT = "research_runs/taker_labels"
GRID = "/tmp/gridbuild/release/grid_sim"; TO_TICKS = 563
ARRS = ["entry_long", "entry_short", "mid_paths", "book_paths", "entry_book"]  # taker mode needs no flow/queue
bk = storage.Client(project=PROJ).bucket(BUCKET)


def run_taker(p, tmp):
    """grid_sim hold in TAKER mode (no flow-paths) on one day's path arrays -> (pnl_long, pnl_short) bp."""
    for k in ARRS:
        np.save(f"{tmp}/{k}.npy", p[k].astype(np.float64))
    json.dump([{"tp": 50.0, "sl": 50.0, "to": TO_TICKS, "par": False, "tr": False}], open(f"{tmp}/cfg.json", "w"))
    cmd = [GRID, "--entry-long", f"{tmp}/entry_long.npy", "--entry-short", f"{tmp}/entry_short.npy",
           "--mid-paths", f"{tmp}/mid_paths.npy", "--book-paths", f"{tmp}/book_paths.npy",
           "--entry-book", f"{tmp}/entry_book.npy", "--configs", f"{tmp}/cfg.json", "--out-prefix", f"{tmp}/g",
           "--commission-win-pct", "0", "--commission-loss-pct", "0"]   # no flow-paths => taker (immediate fill @t0)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-200:]
    pl = np.load(f"{tmp}/g_pnl_long.npy")[0] * 100.0    # % -> bp (gross)
    ps = np.load(f"{tmp}/g_pnl_short.npy")[0] * 100.0
    return (pl, ps), "ok"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--max-days", type=int, default=0); a = ap.parse_args()
    def log(s): print(s, flush=True)
    for sym in a.symbols:
        blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{PATHS}/{sym}/") if b.name.endswith(".npz"))
        if a.max_days:
            blobs = blobs[:a.max_days]
        log(f"=== {sym}: {len(blobs)} days ===")
        tmp = tempfile.mkdtemp(prefix="tk_", dir="/dev/shm")
        accPL, accPS, accTS = [], [], []; nday = 0
        try:
            for i, nm in enumerate(blobs):
                p = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
                (g, st) = run_taker(p, tmp) if "entry_long" in p.files else (None, "no-arrays")
                if g is None:
                    log(f"  {nm.split('/')[-1]}: {st}"); continue
                pl, ps = g; ts = p["ts"].astype(np.int64)
                n = min(len(pl), len(ts))
                accPL.append(pl[:n].astype(np.float32)); accPS.append(ps[:n].astype(np.float32)); accTS.append(ts[:n])
                nday += 1
                if i % 50 == 0:
                    log(f"  {i}/{len(blobs)} {nm.split('/')[-1]} n={n} taker pl_long_mean={np.nanmean(pl[:n]):+.3f}bp")
            if not accPL:
                log(f"  {sym}: nothing"); continue
            PL = np.concatenate(accPL); PS = np.concatenate(accPS); TS = np.concatenate(accTS)
            buf = io.BytesIO()
            np.savez_compressed(buf, pnl_long=PL, pnl_short=PS, ts=TS,
                                meta=np.array(json.dumps({"symbol": sym, "n": int(len(PL)), "n_days": nday,
                                "mode": "taker", "to_ticks": TO_TICKS, "unit": "bp_gross"})))
            bk.blob(f"{OUT}/{sym}.npz").upload_from_string(buf.getvalue())
            log(f"[saved] gs://{BUCKET}/{OUT}/{sym}.npz  N={len(PL)} days={nday} "
                f"(taker pl_long_mean={np.nanmean(PL):+.3f} pl_short_mean={np.nanmean(PS):+.3f} bp)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
