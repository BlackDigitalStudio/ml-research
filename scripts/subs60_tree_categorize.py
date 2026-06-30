#!/usr/bin/env python3
"""Categorize every feature the deployed A (vol-gate) and Bg (direction) models use, by the CL-vs-live
mismatch type (from features.rs catalog + the parallel-agent audit), and sum gain% per bucket. Answers
how transferable each model is, and what fraction needs which fix.
"""
import json
import xgboost as xgb
from google.cloud import storage
bk = storage.Client(project="project-0998ac51-36ba-445c-bc7").bucket("market-data-0998ac51")
fn = json.loads(bk.blob("research_runs/deploy/DOGE/meta.json").download_as_bytes())["feat_names"]

# buckets: feature-index -> mismatch category (from catalog [0..55] + feat71 [64..70]; [56..63] unknown)
BUCKET = {}
def put(cat, ids):
    for i in ids: BUCKET[i] = cat
put("TRANSFERS (robust book: spread/imbalance/depth/microprice/OBI-ladder)", [1, 3, 4, 5, 20, 24, 32, 45, 61, 62, 63])
put("TRANSFERS (btc_lead)", [64, 65, 66])
put("TRANSFERS (ToD)", [67, 68, 69, 70])
put("FIX vol->1s-grid (book-sampling)", [10, 21, 23, 37, 38, 39])
put("FIX momentum->time-window", [2, 12, 34, 35, 36])
put("FIX EMA/VWAP->time", [11, 31, 33])
put("OFI sums (proto: DON'T fix -> drop)", [0, 26, 27, 28, 29, 40, 41, 46])
put("TRADE (x3 dup + aggTrade: dedup+reconcile)", [6, 7, 8, 9, 22, 42, 47, 48])
put("DROP hard (cancel: cl under-observes)", [25, 49])
put("ABSENT live (funding+LIQUIDATIONS dead, no REST -> DROP)", [19, 43, 44, 56, 57, 58])
put("cross-exch/ETH/OI (recorder has via REST, verify cadence)", [14, 15, 16, 17, 18, 30, 50, 51, 52, 53, 54, 55, 59, 60])


def report(name):
    p = f"/tmp/{name}.json"; bk.blob(f"research_runs/deploy/DOGE/{name}.json").download_to_filename(p)
    m = xgb.Booster(); m.load_model(p); gain = m.get_score(importance_type="gain"); total = sum(gain.values())
    agg = {}
    for f, g in gain.items():
        i = int(f[1:]) if f[1:].isdigit() else fn.index(f)
        agg[BUCKET.get(i, "UNCATEGORIZED")] = agg.get(BUCKET.get(i, "UNCATEGORIZED"), 0) + g
    print(f"\n=== {name} — gain% by mismatch bucket ===", flush=True)
    for cat, g in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {100*g/total:5.1f}%  {cat}", flush=True)
    transfers = sum(g for c, g in agg.items() if c.startswith("TRANSFERS"))
    fixable = sum(g for c, g in agg.items() if c.startswith("FIX"))
    print(f"  ---- transfers as-is {100*transfers/total:.0f}% | fixable {100*fixable/total:.0f}% | needs trade/absent/unknown {100*(total-transfers-fixable-sum(g for c,g in agg.items() if c.startswith('OFI')))/total:.0f}%", flush=True)


report("A"); report("Bg")
