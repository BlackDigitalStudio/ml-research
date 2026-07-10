#!/usr/bin/env python3
"""HD3 rev8: anchored-semantics dataset for SYM — prep_anch_year.py generalized to a SYM arg,
transform IDENTICAL (dataset-only intervention): F col13 := the day's FIRST row value
(day-frozen funding), col44 := 0 (basis zeroed). Labels/folds/days/rH/fills byte-identical.
maker_labels_tb3s_h150/{SYM}.npz -> maker_labels_tb3s_h150anch/{SYM}.npz."""
import io
import sys
import numpy as np
from google.cloud import storage

SYM = sys.argv[1]
PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
SRC = f"research_runs/maker_labels_tb3s_h150/{SYM}.npz"
DST = f"research_runs/maker_labels_tb3s_h150anch/{SYM}.npz"
bk = storage.Client(project=PROJ).bucket(BUCKET)

d = dict(np.load(io.BytesIO(bk.blob(SRC).download_as_bytes()), allow_pickle=True))
F = d["F"]; day = d["day"].astype(int)
print("F", F.shape, F.dtype, "| days", day.min(), "..", day.max(), flush=True)
for dd in np.unique(day):
    mk = day == dd
    F[mk, 13] = F[np.where(mk)[0][0], 13]
nz44 = float((F[:, 44] != 0).mean())
F[:, 44] = 0.0
d["F"] = F
print(f"anchored: {len(np.unique(day))} days | col44 was nonzero {100*nz44:.1f}% -> 0", flush=True)
buf = io.BytesIO()
np.savez_compressed(buf, **d)
bk.blob(DST).upload_from_string(buf.getvalue())
print(f"uploaded {DST} ({buf.getbuffer().nbytes/1e6:.0f} MB)", flush=True)
