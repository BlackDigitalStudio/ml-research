#!/usr/bin/env python3
"""Live drop audit: from engine decision logs (GCS), per symbol/day count
budget-flag signals vs executed trades; cluster structure; inter-executed gaps."""
import json, os, shutil, subprocess, sys
import numpy as np

GCLOUD = shutil.which("gcloud") or shutil.which("gcloud.cmd") or "gcloud"

SCR = os.getcwd()  # caches under the run dir
CACHE = os.path.join(SCR, "dec_cache"); os.makedirs(CACHE, exist_ok=True)
BASE = "gs://market-data-0998ac51/research_runs/axb_shadow_h150"

# symbol -> (budget flag, first FULL live day under current policy, harmony?)
CFG = {
    "DOGE": ("take10", "20260716", False),
    "XRP":  ("take5",  "20260716", False),
    "ETH":  ("take5",  "20260718", True),
}
LAST = "20260725"

def get(sym, day):
    p = os.path.join(CACHE, f"{sym}_{day}.jsonl")
    if not os.path.exists(p):
        r = subprocess.run([GCLOUD, "storage", "cp", f"{BASE}/{sym}/decisions/{day}.jsonl", p],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None
    return p

def days_range(a, b):
    from datetime import datetime, timedelta
    d0 = datetime.strptime(a, "%Y%m%d"); d1 = datetime.strptime(b, "%Y%m%d")
    out = []
    while d0 <= d1:
        out.append(d0.strftime("%Y%m%d")); d0 += timedelta(days=1)
    return out

for sym, (flag, day0, harm) in CFG.items():
    tot_dec = 0; tot_take = 0; tot_exec = 0; tot_hblk = 0
    take_ts_all = []; exec_ts_all = []
    per_day = []
    for day in days_range(day0, LAST):
        p = get(sym, day)
        if p is None:
            per_day.append((day, None)); continue
        n = t = e = hb = 0
        tts = []; ets = []
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                n += 1
                if not d.get(flag):
                    continue
                if harm and d.get("hblock"):
                    hb += 1
                    continue
                t += 1
                ts = d.get("grid_ts_us", 0) / 1e6
                tts.append(ts)
                if d.get("executed"):
                    e += 1; ets.append(ts)
        tot_dec += n; tot_take += t; tot_exec += e; tot_hblk += hb
        take_ts_all.extend(tts); exec_ts_all.extend(ets)
        per_day.append((day, (n, t, e, hb)))
    print(f"\n=== {sym} (flag={flag}{', harmony' if harm else ''}) {day0}..{LAST} ===")
    for day, r in per_day:
        if r is None:
            print(f"  {day}: NO LOG"); continue
        n, t, e, hb = r
        print(f"  {day}: dec={n:6d} signals={t:3d} executed={e:2d} dropped={t-e:3d}" +
              (f" hblock={hb}" if harm else ""))
    dr = 100.0 * (tot_take - tot_exec) / max(tot_take, 1)
    print(f"  TOTAL: dec={tot_dec} signals={tot_take} executed={tot_exec} -> DROPPED {tot_take-tot_exec} ({dr:.1f}%)")
    ndays = sum(1 for _, r in per_day if r is not None)
    print(f"  rate: signals/day={tot_take/max(ndays,1):.2f} executed/day={tot_exec/max(ndays,1):.2f}")
    # cluster structure of signals (gap<=6s -> same cluster; also 60s threshold)
    ts = np.array(sorted(take_ts_all))
    if len(ts) > 1:
        gaps = np.diff(ts)
        for g in (6, 60, 210, 330):
            ncl = 1 + int((gaps > g).sum())
            print(f"  signal clusters at gap>{g}s: {ncl} ({tot_take/max(ncl,1):.2f} signals/cluster)")
    ets = np.array(sorted(exec_ts_all))
    if len(ets) > 1:
        eg = np.diff(ets)
        eg = eg[eg < 3600]
        if len(eg):
            print(f"  inter-executed gaps <1h: n={len(eg)} min={eg.min():.0f}s p25={np.percentile(eg,25):.0f}s median={np.median(eg):.0f}s")
