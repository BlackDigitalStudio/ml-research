#!/usr/bin/env python3
"""Assemble a trainable dataset that pairs the PUBLISHED features with the STRICT labels.

OPS-FILLYEAR rev1 / cell A2 input. A1 froze the models and only re-accounted the PnL;
A2 asks the different question - what does this protocol learn when the labels it is
trained on carry honest fills? That needs a dataset in the trainer's own schema, so this
script takes the feature side from the published dataset (whatever variant the cell used:
anchored funding for DOGE/XRP/ETH, true funding for BTC) and swaps in the strict
pnl_long / pnl_short / fill_long / fill_short.

Row alignment is asserted, not assumed: `ts` and `day` must match element-for-element
between the feature source and the strict labels, and the FROZEN labels carried in the
strict npz must match the feature source's own labels bit-exactly. If either fails, the
two files are not the same rows and nothing downstream would be interpretable.

Env: SYM, FEAT_SUB (published dataset), LABEL_SUB (strict labels), OUT_SUB.
"""
import io, json, os

import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SYM = os.environ.get("SYM", "DOGE")
FEAT_SUB = os.environ.get("FEAT_SUB", "research_runs/maker_labels_tb3s_h150anch")
LABEL_SUB = os.environ.get("LABEL_SUB", "research_runs/maker_labels_tb3s_h150strict")
OUT_SUB = os.environ.get("OUT_SUB", FEAT_SUB + "_strict")
bk = storage.Client(project=PROJ).bucket(BUCKET)


def log(s):
    print(s, flush=True)


log(f"[load features] {FEAT_SUB}/{SYM}.npz")
F = np.load(io.BytesIO(bk.blob(f"{FEAT_SUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)
log(f"[load labels]   {LABEL_SUB}/{SYM}.npz")
S = np.load(io.BytesIO(bk.blob(f"{LABEL_SUB}/{SYM}.npz").download_as_bytes()), allow_pickle=True)

n = len(F["day"])
log(f"  features N={n} cols={F['F'].shape[1]} | labels N={len(S['day'])}")
assert len(S["day"]) == n, f"row count mismatch {len(S['day'])} vs {n}"
assert np.array_equal(F["day"].astype(int), S["day"].astype(int)), "day arrays differ"
assert np.array_equal(F["ts"].astype(np.int64), S["ts"].astype(np.int64)), "ts arrays differ"
for a, b in (("pnl_long", "pnl_long_frozen"), ("pnl_short", "pnl_short_frozen")):
    assert np.array_equal(F[a], S[b], equal_nan=True), f"{a}: feature-source labels != carried frozen labels"
for a, b in (("fill_long", "fill_long_frozen"), ("fill_short", "fill_short_frozen")):
    assert np.array_equal(F[a].astype(bool), S[b].astype(bool)), f"{a}: fills differ"
log("[gates] row alignment + frozen-label identity PASS")

meta = json.loads(str(F["meta"]))
meta["strict_entry_fill"] = True
meta["label_source"] = LABEL_SUB
meta["feature_source"] = FEAT_SUB
meta["note"] = ("features from the published dataset, entry fills and PnL from the strict "
                "price-resolved queue model (OPS-EXEC rev16). " + meta.get("note", ""))

fill_l = S["fill_long"]
fill_s = S["fill_short"]
log(f"[fills] frozen {F['fill_long'].mean():.4f}/{F['fill_short'].mean():.4f} -> "
    f"strict {fill_l.mean():.4f}/{fill_s.mean():.4f}")

buf = io.BytesIO()
np.savez_compressed(buf, F=F["F"], rH30=F["rH30"], rH15=F["rH15"], rH60=F["rH60"],
                    day=F["day"], ts=F["ts"],
                    pnl_long=S["pnl_long"], pnl_short=S["pnl_short"],
                    fill_long=fill_l, fill_short=fill_s,
                    feat_names=F["feat_names"], meta=np.array(json.dumps(meta)))
bk.blob(f"{OUT_SUB}/{SYM}.npz").upload_from_string(buf.getvalue())
log(f"[saved] gs://{BUCKET}/{OUT_SUB}/{SYM}.npz ({buf.tell()/1e6:.0f}MB)")
