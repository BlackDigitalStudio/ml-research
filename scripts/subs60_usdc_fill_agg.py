#!/usr/bin/env python3
"""Aggregate the USDC fill-layer run into the 3-way decomposition (OPS-EXEC rev17).

  (a) frozen fill model on the USDT book  — the existing _recev_* artifacts (what every
      historical cell used; netl/nets/FL/FS live there together with the scores)
  (b) frozen model on the USDC book       — venue error alone
  (c) strict model on the USDC book       — venue + model error

Row alignment: neither side stores a shared key, but both grids are calendar-anchored
3s grids, so each row maps to slot = floor((sample_ts - midnight)/3s) and the two sides
are inner-joined on that slot. The USDT sample timestamps are rebuilt from the USDT depth
timestamp column with the SAME rule the recev runner used (grid from midnight, ends =
unique(searchsorted(bt, grid)-1), clipped to [W-1, n-H-1]).

Env: SYM, SCORE_PREFIXES (comma-separated), FILL_PREFIX, TAU (frozen) or BUDGET (dynamic).
"""
import io, os
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; MKT = "market-data-0998ac51"; REC = "recorder-data-asia-0998ac51"
RB = "chronos/scalper-recorder/binance_futures"
SYM = os.environ.get("SYM", "DOGE")
SCORE_PREFIXES = os.environ.get("SCORE_PREFIXES", f"research_runs/_recev_h150anch2_{SYM}").split(",")
FILL_PREFIX = os.environ.get("FILL_PREFIX", f"research_runs/_usdcfill_{SYM}")
TAU = float(os.environ.get("TAU", "0"))
BUDGET = float(os.environ.get("BUDGET", "10"))
KDAYS = 30
W = 50; H = 6000; STEP_S = 3.0; NS = 1_000_000_000
TD = os.environ.get("WORKDIR", f"/home/delmi/agg_{SYM}"); os.makedirs(TD, exist_ok=True)
cl = storage.Client(project=PROJ); mkt = cl.bucket(MKT); rec = cl.bucket(REC)


def usdt_slots(day):
    """Rebuild the USDT decision-grid sample timestamps -> slot ids, as recev did."""
    ts = []
    for b in rec.client.list_blobs(rec, prefix=f"{RB}/{SYM}USDT/depth_snapshot/{day}"):
        if not b.name.endswith(".parquet"):
            continue
        rec.blob(b.name).download_to_filename(f"{TD}/x.parquet")
        t = pq.read_table(f"{TD}/x.parquet", columns=["exchange_event_ts_us"]).to_pandas()
        t = t[t["exchange_event_ts_us"].notna()]
        ts.append((t["exchange_event_ts_us"].astype("int64") * 1000).to_numpy())
    if not ts:
        return None
    bt = np.sort(np.concatenate(ts)); n = len(bt)
    if n < W + H + 100:
        return None
    mid0 = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()) * NS
    grid = np.arange(mid0, bt[-1], int(STEP_S * NS)); grid = grid[grid >= bt[0]]
    ends = np.unique(np.clip(np.searchsorted(bt, grid, "right") - 1, 0, n - 1))
    ends = ends[(ends >= W - 1) & (ends < n - H - 1)].astype(np.int64)
    if len(ends) < 50:
        return None
    return ((bt[ends] - mid0) // int(STEP_S * NS)).astype(np.int64)


def load(prefix, day):
    try:
        return np.load(io.BytesIO(mkt.blob(f"{prefix}/D_{day}.npz").download_as_bytes()))
    except Exception:
        return None


days = sorted({b.name.split("/")[-1][2:10] for b in mkt.client.list_blobs(mkt, prefix=f"{FILL_PREFIX}/D_")
               if b.name.endswith(".npz")})
rows = []
for day in days:
    zf = load(FILL_PREFIX, day)
    zs = None
    for p in SCORE_PREFIXES:
        zs = load(p.strip(), day)
        if zs is not None:
            break
    if zf is None or zs is None:
        continue
    su = usdt_slots(day)
    if su is None or len(su) != len(zs["score"]):
        print(f"{day}: slot rebuild {0 if su is None else len(su)} != scores {len(zs['score'])} — SKIP", flush=True)
        continue
    mid0 = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()) * 1000
    sc_slot = ((zf["sample_ts"].astype(np.int64) - mid0) // int(STEP_S * 1000))
    common, iu, ic = np.intersect1d(su, sc_slot, return_indices=True)
    if len(common) < 50:
        print(f"{day}: join {len(common)} — SKIP", flush=True); continue
    side = zs["side"][iu].astype(bool)
    rows.append(dict(
        day=day, score=zs["score"][iu].astype(np.float64), side=side,
        net_a=np.where(side, zs["netl"][iu], zs["nets"][iu]).astype(np.float64),
        fil_a=np.where(side, zs["FL"][iu], zs["FS"][iu]).astype(bool),
        net_b=np.where(side, zf["netl_f"][ic], zf["nets_f"][ic]).astype(np.float64),
        fil_b=np.where(side, zf["FL_f"][ic], zf["FS_f"][ic]).astype(bool),
        net_c=np.where(side, zf["netl_s"][ic], zf["nets_s"][ic]).astype(np.float64),
        fil_c=np.where(side, zf["FL_s"][ic], zf["FS_s"][ic]).astype(bool)))
    print(f"{day}: joined {len(common)} rows (usdt {len(su)}, usdc {len(sc_slot)})", flush=True)

if not rows:
    print("NO JOINED DAYS"); raise SystemExit
score = np.concatenate([r["score"] for r in rows])
dayi = np.concatenate([np.full(len(r["score"]), i) for i, r in enumerate(rows)])
nd = len(rows)
if TAU > 0:
    sel = np.where(score >= TAU)[0]
    pol = f"FIXQ tau={TAU}"
else:
    # DAY-0 WARMUP is mandatory: with an empty buffer day 1 selects ~everything and the
    # cell degenerates into an unconditional average (KNOWN_PITFALLS "Raw recorder-EV
    # output is NOT the measurement cell"). Seed the buffer with day 0 and score from day 1.
    wpd = len(score) / nd; q = max(0.0, 1.0 - BUDGET / max(wpd, 1.0))
    cap = max(int(KDAYS * wpd), 1)
    buf = list(score[dayi == 0]); buf = buf[-cap:]; s = []
    for i in range(1, nd):
        idx = np.where(dayi == i)[0]
        t = float(np.quantile(buf, q))
        s.extend(idx[score[idx] >= t].tolist()); buf.extend(score[idx].tolist()); buf = buf[-cap:]
    sel = np.array(s, dtype=int); nd = nd - 1   # day 0 is warmup, not an evaluation day
    pol = f"DYN budget={BUDGET} (day-0 warmup)"
print(f"\n=== {SYM} — {nd} joined days, {len(score)} decisions, {pol}, selected {len(sel)}")
print(f"{'cell':<34}{'fills':>7}{'fill%':>8}{'EV/tr':>9}{'bpd':>9}{'vs (a)':>9}")
out = {}
for tag, key, name in (("a", "a", "(a) frozen model / USDT book  [cells]"),
                       ("b", "b", "(b) frozen model / USDC book  [venue]"),
                       ("c", "c", "(c) STRICT model / USDC book  [honest]")):
    net = np.concatenate([r[f"net_{key}"] for r in rows])[sel]
    fil = np.concatenate([r[f"fil_{key}"] for r in rows])[sel]
    ex = fil & np.isfinite(net)
    n = int(ex.sum()); ev = float(net[ex].mean()) if n else float("nan")
    bpd = float(net[ex].sum()) / nd if n else float("nan")
    out[tag] = bpd
    rel = "" if tag == "a" else f"{100*bpd/out['a']:>8.0f}%"
    print(f"{name:<34}{n:>7}{100*ex.mean():>7.1f}%{ev:>+9.2f}{bpd:>+9.1f}{rel:>9}")
