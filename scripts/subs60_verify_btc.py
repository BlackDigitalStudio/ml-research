#!/usr/bin/env python3
"""Verify whether BTC-lead columns in hd2_sub60_cache are REAL or placeholder-zero."""
import io
import numpy as np
from google.cloud import storage
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
bk = storage.Client(project=PROJ).bucket(BUCKET)

# 1) BTC source feats: do td/mid exist and span the data?
bl = sorted(b.name for b in bk.client.list_blobs(bk, prefix="feats_sub60/BTC-USDT-PERP/") if b.name.endswith(".npz"))
print(f"[BTC feats] {len(bl)} day-files")
if bl:
    z = np.load(io.BytesIO(bk.blob(bl[0]).download_as_bytes()))
    print(f"  keys={list(z.keys())}")
    if "mid" in z:
        m = z["mid"].astype(float); print(f"  mid: n={len(m)} min={m.min():.4f} max={m.max():.4f} std={m.std():.4f} frac0={np.mean(m==0):.3f}")
    if "td" in z:
        t = z["td"].astype(np.int64); print(f"  td: n={len(t)} span_days={(t.max()-t.min())/1e9/86400:.1f}")

# 2) hd2_sub60_cache feat: inspect cols 60..71 (X tail 60-63 | BTC-lead 64-66 | ToD 67-70)
for sym in ["DOGE-USDT-PERP", "ETH-USDT-PERP"]:
    cb = sorted(b.name for b in bk.client.list_blobs(bk, prefix=f"hd2_sub60_cache/{sym}/") if b.name.endswith(".npz"))
    if not cb:
        print(f"[{sym}] NO cache"); continue
    z = np.load(io.BytesIO(bk.blob(cb[len(cb)//2]).download_as_bytes()))   # mid-period day
    feat = z["feat"].astype(np.float64); F = feat.shape[1]
    print(f"\n[{sym}] {cb[len(cb)//2].split('/')[-1]} feat shape={feat.shape}")
    for c in range(60, F):
        col = feat[:, c]
        tag = "BTC-lead" if 64 <= c <= 66 else ("ToD" if c >= 67 else "X-tail")
        print(f"  col{c:>2} {tag:>9}: std={col.std():.5f} mean={col.mean():+.5f} frac0={np.mean(col==0):.3f} min={col.min():+.4f} max={col.max():+.4f}")
