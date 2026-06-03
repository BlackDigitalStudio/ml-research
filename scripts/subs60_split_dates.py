#!/usr/bin/env python3
"""Recover the EXACT calendar train/val/test intervals behind the maker apred +3.00 result.

Reads only day/ts/meta from each maker_labels_rr/{SYM}.npz (lazy via GCS range requests,
no full download) and applies the IDENTICAL split() used by subs60_xgb_b2.py
(SPLIT=(0.65,0.68,0.85)) to map day-index boundaries -> UTC calendar dates.
"""
import io, json, sys
from datetime import datetime, timezone
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"
SYMS = ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
SPLIT = (0.65, 0.68, 0.85)
bk = storage.Client(project=PROJ).bucket(BUCKET)


def split(day, ndays):
    cut = int(ndays * SPLIT[0]); emb = int(ndays * SPLIT[1]); tr = day < cut
    td = sorted(set(day[tr].tolist())); vcut = td[int(len(td) * SPLIT[2])] if td else cut
    return (tr & (day < vcut)), (tr & (day >= vcut)), (day >= emb), cut, emb, vcut


def d2s(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


def lazy_npz(symk):
    """Open the blob as a seekable file -> NpzFile lazy-reads only accessed members."""
    try:
        f = bk.blob(f"{RR}/{symk}.npz").open("rb")
        return np.load(f, allow_pickle=True)
    except Exception as e:
        print(f"  [stream failed: {e}; downloading full blob]", flush=True)
        return np.load(io.BytesIO(bk.blob(f"{RR}/{symk}.npz").download_as_bytes()), allow_pickle=True)


def main():
    syms = sys.argv[1:] or SYMS
    print(f"{'SYM':5s} {'ndays':>5s} {'Nrows':>9s} | "
          f"{'TRAIN (idx0..vcut-1)':>34s} | {'VAL (vcut..cut-1)':>27s} | "
          f"{'embargo gap':>23s} | {'TEST (>=emb)':>27s}")
    for symk in syms:
        d = lazy_npz(symk)
        meta = json.loads(str(d["meta"])); ndays = int(meta["n_days"])
        day = d["day"]; ts = d["ts"].astype(np.int64)
        trn, val, te, cut, emb, vcut = split(day, ndays)

        def span(mask):
            if not mask.any():
                return "(empty)", None, None, 0
            t = ts[mask]
            return f"{d2s(t.min())}->{d2s(t.max())}", int(day[mask].min()), int(day[mask].max()), int(mask.sum())

        gap = (day >= cut) & (day < emb)
        (str_tr, dt0, dt1, ntr) = span(trn)
        (str_v, dv0, dv1, nv) = span(val)
        (str_g, dg0, dg1, ng) = span(gap)
        (str_te, de0, de1, nte) = span(te)
        full = f"{d2s(ts.min())}->{d2s(ts.max())}"
        print(f"{symk:5s} {ndays:5d} {len(day):9d} | "
              f"[{dt0:>3}-{dt1:>3}] {str_tr} | [{dv0:>3}-{dv1:>3}] {str_v} | "
              f"[{dg0:>3}-{dg1:>3}] {str_g} | [{de0:>3}-{de1:>3}] {str_te}")
        print(f"      full-span {full}  | rows tr/val/test = {ntr}/{nv}/{nte}  "
              f"| split idx: vcut={vcut} cut={cut} emb={emb}")
    print(f"\n[split rule] SPLIT={SPLIT}: train=day<vcut(={SPLIT[0]}*{SPLIT[2]}~0.5525*ndays), "
          f"val=[vcut,cut), embargo=[cut,emb) DROPPED, test=day>=emb(={SPLIT[1]}*ndays). "
          f"day = 0-based INDEX into each symbol's available days (gaps compress the calendar).")


if __name__ == "__main__":
    main()
