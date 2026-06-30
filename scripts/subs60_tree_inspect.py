#!/usr/bin/env python3
"""Inspect the deployed XGBoost trees (A vol-gate, Bg direction): which features actually drive the
models, by total gain, and whether they are sampling-ROBUST (match cl-sparse <-> dense recorder) or
sampling-FRAGILE (drift with book sampling = partly cryptolake collection-timing artifact). If the
gain is concentrated in robust features -> the edge is real and transfers to our recorder live.
"""
import io, json
import numpy as np
import xgboost as xgb
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
bk = storage.Client(project=PROJ).bucket(BUCKET)

# sampling-FRAGILE base feature cols (drift under book downsample; subs60_feat_sampling_test K=4)
FRAGILE = {0, 2, 10, 12, 21, 22, 23, 25, 26, 27, 28, 29, 31, 34, 35, 36, 37, 38, 39, 40, 41, 46, 49}
meta = json.loads(bk.blob("research_runs/deploy/DOGE/meta.json").download_as_bytes())
fn = meta["feat_names"]
print(f"feat_names[64:71] (the feat71 extras): {fn[64:71]}", flush=True)


def idx_of(f):
    if f.startswith("f") and f[1:].isdigit():
        return int(f[1:])
    return fn.index(f) if f in fn else -1


def tag(idx):
    if idx >= 67:
        return "ToD(robust)"
    if idx >= 64:
        return "btc_lead(robust)"
    return "FRAGILE" if idx in FRAGILE else "robust"


def report(name):
    p = f"/tmp/{name}.json"; bk.blob(f"research_runs/deploy/DOGE/{name}.json").download_to_filename(p)
    m = xgb.Booster(); m.load_model(p)
    gain = m.get_score(importance_type="gain"); weight = m.get_score(importance_type="weight")
    total = sum(gain.values()); ntree = len(m.get_dump())
    items = sorted(gain.items(), key=lambda kv: -kv[1])
    print(f"\n=== {name}: {ntree} trees, {len(gain)}/{len(fn)} features used ===", flush=True)
    print(f"  {'feat':>16} {'gain%':>7} {'splits':>7}  class", flush=True)
    for f, g in items[:14]:
        i = idx_of(f)
        print(f"  {fn[i]:>16} {100*g/total:>6.1f}% {int(weight.get(f,0)):>7}  {tag(i)}", flush=True)
    frag = sum(g for f, g in gain.items() if idx_of(f) in FRAGILE)
    btc = sum(g for f, g in gain.items() if 64 <= idx_of(f) < 67)
    tod = sum(g for f, g in gain.items() if idx_of(f) >= 67)
    rob = total - frag
    print(f"  -- GAIN SPLIT: robust {100*rob/total:.0f}%  (of which btc_lead {100*btc/total:.0f}%, ToD {100*tod/total:.0f}%) | FRAGILE {100*frag/total:.0f}%", flush=True)
    return 100 * frag / total


fa = report("A"); fb = report("Bg")
print(f"\n=== VERDICT ===", flush=True)
print(f"  A  (vol-gate):  {fa:.0f}% of gain from sampling-fragile features", flush=True)
print(f"  Bg (direction): {fb:.0f}% of gain from sampling-fragile features", flush=True)
print(f"  -> {'edge leans on cl-sampling artifacts (fragile-heavy)' if max(fa,fb)>50 else 'edge mostly on robust features -> transfers to recorder'}", flush=True)
