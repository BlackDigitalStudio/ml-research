#!/usr/bin/env python3
"""Save MAKER PATHS per (symbol, day) so grid_sim can sweep ANY config grid (e.g. 100k) on-demand
later (the §14 way: no pre-stored per-config pnl -> no PB blowup). For each feats decision point
(stride 8, valid_60) we save the matched build-sample's maker arrays (entry_long/short, mid_paths,
book_paths, flow_paths, entry_q, entry_book) keyed by feats ts. Streamed per-day to GCS (local disk
stays tiny). f32 to halve size.

Output: gs://.../research_runs/maker_paths/{SYM}/{DATE}.npz
Run (1 sym):  python3 subs60_save_maker_paths.py --symbols SOL-USDT-PERP --max-days 1 --probe
"""
import argparse, io, os, shutil, subprocess, tempfile, time
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
FEATS = "feats_sub60"; OUT = "research_runs/maker_paths"
RAWB = "raw/book/exchange=BINANCE_FUTURES"; RAWT = "raw/trades/exchange=BINANCE_FUTURES"
BS = "/tmp/husdc/rust_ingest/target/release/build_samples"
NS = 1_000_000_000; H_TICKS = 700; WINDOW = 50; TARGET = 40000; MAXS = 2_000_000; FEAT_STRIDE = 8; TOLMS = 2500.0
ARRS = ["entry_long", "entry_short", "mid_paths", "book_paths", "flow_paths", "entry_q", "entry_book"]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s): print(s, flush=True)


def dl_raw(prefix, dst):
    name = next((b.name for b in bk.client.list_blobs(bk, prefix=prefix) if b.name.endswith(".parquet")), None)
    if not name:
        return False
    bk.blob(name).download_to_filename(dst); return True


def build_day(tmp, sym, day):
    od = os.path.join(tmp, "bs"); os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    bp, tp = f"{tmp}/b.parquet", f"{tmp}/t.parquet"
    if not (dl_raw(f"{RAWB}/symbol={sym}/dt={day}/", bp) and dl_raw(f"{RAWT}/symbol={sym}/dt={day}/", tp)):
        return None, "no-raw"
    import pyarrow.parquet as pq
    step = max(1, -(-pq.ParquetFile(bp).metadata.num_rows // TARGET))
    r = subprocess.run([BS, "--depth", bp, "--trades", tp, "--out-dir", od, "--window", str(WINDOW),
                        "--horizon", str(H_TICKS), "--step", str(step), "--max-samples", str(MAXS)],
                       capture_output=True, text=True)
    for pf in (bp, tp):
        try: os.remove(pf)
        except OSError: pass
    return (od, "ok") if r.returncode == 0 else (None, f"BS-fail:{r.stderr[-160:]}")


def process_symbol(sym, max_days, probe):
    symk = sym.split("-")[0]
    blobs = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"{FEATS}/{sym}/") if b.name.endswith(".npz"))
    if max_days:
        blobs = blobs[:max_days]
    scratch = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
    tmp = tempfile.mkdtemp(prefix="mp_", dir=scratch)
    nday = 0; tot_mb = 0.0; t0 = time.time()
    try:
        for nm in blobs:
            day = nm.split("/")[-1].replace(".npz", "")
            d = np.load(io.BytesIO(bk.blob(nm).download_as_bytes()))
            td = d["td"].astype(np.int64); n = len(td)
            if n < 200:
                continue
            sel = np.arange(0, n, FEAT_STRIDE); v = d["valid_60"].astype(bool)[sel]; sel = sel[v]
            if len(sel) < 20:
                continue
            ftd = td[sel]
            od, st = build_day(tmp, sym, day)
            if od is None:
                log(f"  {day}: {st}"); continue
            sts = np.load(f"{od}/sample_ts.npy").astype(np.int64) * 1_000_000  # ms->ns
            if len(sts) < 5:
                continue
            pos = np.clip(np.searchsorted(sts, ftd), 0, len(sts) - 1)
            sf, sb = [], []
            for fi, (gd, p) in enumerate(zip(ftd, pos)):
                cand = [c for c in (p - 1, p) if 0 <= c < len(sts)]
                best = min(cand, key=lambda c: abs(sts[c] - gd))
                if abs(sts[best] - gd) <= TOLMS * 1_000_000:
                    sf.append(fi); sb.append(best)
            if not sf:
                log(f"  {day}: 0 matched"); continue
            sb = np.array(sb); sf = np.array(sf)
            out = {"ts": ftd[sf].astype(np.int64)}
            for k in ARRS:
                a = np.load(f"{od}/{k}.npy")[sb]
                out[k] = a.astype(np.float32) if a.dtype.kind == "f" else a
            buf = io.BytesIO(); np.savez_compressed(buf, **out)
            mb = buf.tell() / 1e6; tot_mb += mb
            if not probe:
                bk.blob(f"{OUT}/{symk}/{day}.npz").upload_from_string(buf.getvalue())
            for f in os.listdir(od):
                os.remove(os.path.join(od, f))
            nday += 1
            if probe or nday % 25 == 0:
                log(f"  {day}: matched={len(sf)} day_npz={mb:.1f}MB cum={tot_mb/1000:.2f}GB [{time.time()-t0:.0f}s]")
        log(f"[{symk}] days={nday} total={tot_mb/1000:.2f}GB ({tot_mb/max(nday,1):.1f}MB/day) in {time.time()-t0:.0f}s")
        return tot_mb / 1000, nday
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--probe", action="store_true")   # don't upload; report size
    a = ap.parse_args()
    for sym in a.symbols:
        process_symbol(sym, a.max_days, a.probe)


if __name__ == "__main__":
    main()
