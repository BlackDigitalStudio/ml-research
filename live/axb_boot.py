#!/usr/bin/env python3
"""axb_boot — one-shot bootstrap for the Rust axb_engine (systemd ExecStartPre).

Produces WORKDIR/boot:
  A{0..3}.json Bg{0..3}.json          — deploy bundle models (GCS)
  mu{s}.npy sd{s}.npy                 — f64 vol-norm (refs day_mean/day_var, KNORM tail)
  sA{s}.npy sBg{s}.npy                — f64 rank-CDF references
  base_A{s}.npy base_Bg{s}.npy        — EXACT xgboost base-margin bits, solved from a
                                        one-tree prediction (the float ProbToMargin
                                        formula can be 1 ulp off — measured)
  tau_seed.npy                        — anchored recorder score distribution (RECEV_DIR)
  anchor.json                         — funding day-anchor {day, rate}: recorder local
                                        hour-00 mark_price file, REST fallback (logged)
"""
from __future__ import annotations

import io
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"
MKT = "market-data-0998ac51"
SYMK = os.environ.get("SYMK", "DOGE")
SYM = os.environ.get("SIGNAL_SYM", "dogeusdt")
WORK = os.environ.get("WORKDIR", "/home/delmi/axb_h150")
BUNDLE = os.environ.get("BUNDLE_DIR", f"research_runs/deploy_h150/{SYMK}")
RECEV = os.environ.get("RECEV_DIR", f"research_runs/_recev_h150anch_{SYMK}")
RECDATA = "/home/scalper/crypto-market-recorder/data/binance_futures"
KNORM = 20
CAP = 30 * 28800
BOOT = f"{WORK}/boot"
os.makedirs(BOOT, exist_ok=True)

cl = storage.Client(project=PROJ)
mkt = cl.bucket(MKT)

# ---- models + refs -> npys ----
probe = xgb.DMatrix(np.zeros((1, 71), dtype=np.float32))
for s in range(4):
    base = f"{BUNDLE}/seed{s}"
    refs = np.load(io.BytesIO(mkt.blob(f"{base}/refs.npz").download_as_bytes()))
    gstd = refs["gstd"].astype(np.float64)
    mu = refs["day_mean"].astype(np.float64)[-KNORM:].mean(0)
    sd = np.maximum(np.sqrt(np.maximum(refs["day_var"].astype(np.float64)[-KNORM:].mean(0), 0)),
                    0.2 * gstd + 1e-9)
    np.save(f"{BOOT}/mu{s}.npy", mu)
    np.save(f"{BOOT}/sd{s}.npy", sd)
    np.save(f"{BOOT}/sA{s}.npy", refs["sA"].astype(np.float64))
    np.save(f"{BOOT}/sBg{s}.npy", refs["sBg"].astype(np.float64))
    for nm in ("A", "Bg"):
        p = f"{BOOT}/{nm}{s}.json"
        mkt.blob(f"{base}/{nm}.json").download_to_filename(p)
        # solve the exact f32 base-margin bits: unique b with f32(leaf_tree0(0)+b) == margin_1tree(0)
        m = json.load(open(p))
        bs = m["learner"]["learner_model_param"]["base_score"]
        t = m["learner"]["gradient_booster"]["model"]["trees"][0]
        lc, rc = t["left_children"], t["right_children"]
        scd = np.array(t["split_conditions"], dtype=np.float32)
        n = 0
        while lc[n] != -1:
            n = lc[n] if np.float32(0.0) < scd[n] else rc[n]
        leaf = scd[n]
        b = xgb.Booster()
        b.load_model(p)
        m1 = b.predict(probe, output_margin=True, iteration_range=(0, 1)).astype(np.float32)[0]
        pval = float(bs.strip("[]"))
        f32 = np.float32
        u0 = int(f32(-np.log(f32(1.0) / f32(pval) - f32(1.0))).view(np.uint32))
        sol = [(u0 + du) & 0xFFFFFFFF for du in range(-8, 9)
               if f32(leaf + np.uint32((u0 + du) & 0xFFFFFFFF).view(np.float32)) == m1]
        assert sol, f"base bits unsolved for {nm}{s}"
        np.save(f"{BOOT}/base_{nm}{s}.npy", np.array([sol[0]], dtype=np.uint32).view(np.float32))
        print(f"boot: {nm}{s} base bits {hex(sol[0])}")

# ---- tau seed ----
buf = []
for bl in sorted(b.name for b in cl.list_blobs(mkt, prefix=f"{RECEV}/D_") if b.name.endswith(".npz")):
    z = np.load(io.BytesIO(mkt.blob(bl).download_as_bytes()))
    buf.extend(z["score"].astype(np.float64).tolist())
buf = buf[-CAP:]
np.save(f"{BOOT}/tau_seed.npy", np.asarray(buf, dtype=np.float64))
print(f"boot: tau seed {len(buf)} scores from {RECEV}")

# ---- funding day-anchor ----
day = datetime.now(timezone.utc).strftime("%Y%m%d")
anchor = None
# anchor sources in order: recorder LOCAL file (same-VM deploys), recorder GCS bucket
# (split-VM deploys; hour-00 lands after ~01:00 UTC), REST approximation.
src = f"{RECDATA}/{SYM.upper()}/mark_price/{day}_00.parquet"
r = subprocess.run(["sudo", "-n", "cp", src, f"{BOOT}/anchor_mark.parquet"], capture_output=True)
if r.returncode != 0:
    try:
        rec = cl.bucket("recorder-data-asia-0998ac51")
        rec.blob(f"chronos/scalper-recorder/binance_futures/{SYM.upper()}/mark_price/{day}_00.parquet")            .download_to_filename(f"{BOOT}/anchor_mark.parquet")
        r = subprocess.CompletedProcess([], 0)
        print("boot: anchor file fetched from recorder GCS bucket")
    except Exception as ex:
        print(f"boot: recorder GCS anchor unavailable: {ex}")
if r.returncode == 0:
    subprocess.run(["sudo", "-n", "chmod", "644", f"{BOOT}/anchor_mark.parquet"], capture_output=True)
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(f"{BOOT}/anchor_mark.parquet",
                          columns=["exchange_event_ts_us", "funding_rate"])
        ets = t["exchange_event_ts_us"].to_numpy().astype(float)
        fr = t["funding_rate"].to_numpy().astype(float)
        m = ~np.isnan(ets) & ~np.isnan(fr)
        anchor = float(fr[m][int(np.argmin(ets[m]))])
        print(f"boot: funding anchor from recorder {day} = {anchor}")
    except Exception as ex:
        print(f"boot: recorder anchor read failed: {ex}")
if anchor is None:
    import urllib.request
    with urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={SYM.upper()}",
                                timeout=8) as resp:
        anchor = float(json.loads(resp.read())["lastFundingRate"])
    print(f"boot: funding anchor APPROXIMATED from REST = {anchor} (corrected at next day roll)")
json.dump({"day": day, "rate": anchor}, open(f"{BOOT}/anchor.json", "w"))
print("boot: done")
