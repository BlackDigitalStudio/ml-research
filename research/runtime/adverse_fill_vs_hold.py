#!/usr/bin/env python3
"""DOES THE ADVERSE-FILL LOSS ACCRUE OVER THE HOLD, OR IS IT ALREADY THERE AT THE FILL?

User observation (2026-08-04): if the loss is a property of the ENTRY fill, it cannot be
attacked once we are already in the position -- but if it accrues along the path, then the
decision "exit now or hold n more seconds" attacks it directly, and that decision is free
of entry fill-selection by construction.

fill_selection_split.py measured (4 symbols, gate-validated) that the whole of the bulk's
-1.7bp sits in the rows where ONLY OUR SIDE filled (-4.65..-6.25bp) and is absent where
both sides filled (-0.42..+0.25). That was one hold length. This script prices the same
two populations at EVERY hold the label archive carries.

MODEL-FREE BY CONSTRUCTION: no score, no seeds, no selection. For each side s the
populations are defined by the fill flags alone --

    ALONE(s)  fill_s & ~fill_other      one-sided market: price left and did not come back
    BOTH(s)   fill_s &  fill_other      two-sided market: price came back through both

-- and both sides are pooled, so the result cannot depend on which side a model would have
picked. The fill flags are per-decision and carry NO hold index (verified in the archive
schema), so the population is held FIXED across holds and only the exit moves. That is
what makes the comparison a clean read on the path.

Reads the compact label archive directly (day/ts/pnl_{long,short}[_frozen]/fill_*).
FROZEN arrays by default = the same measurement layer as the PERFOLD cells; STRICT=1
switches to the honest-fill relabel.

Env: SYM, ARCHIVE (local .npz path), STRICT.
"""
import json, os

import numpy as np

ARCHIVE = os.environ["ARCHIVE"]
SYM = os.environ.get("SYM", "DOGE")

z = np.load(ARCHIVE, allow_pickle=True)
# Which array set does THIS archive carry? A strictfill_year.py relabel holds both the
# strict arrays (unsuffixed) and the parity-gate frozen ones (`_frozen`); the parent
# h150anch archive holds ONLY the frozen labels, unsuffixed. Detecting instead of
# assuming keeps the printed provenance honest -- reading the parent under STRICT=1 would
# otherwise report frozen numbers as strict ones (2026-08-05).
STRICT_WANTED = os.environ.get("STRICT") == "1"
HAS_FROZEN = "pnl_long_frozen" in z.files
SUF = "" if (STRICT_WANTED or not HAS_FROZEN) else "_frozen"
ARRAYS = ("strict" if STRICT_WANTED else "frozen") if HAS_FROZEN else "frozen (parent archive: only one label set, unsuffixed)"
if STRICT_WANTED and not HAS_FROZEN:
    raise SystemExit("STRICT=1 but the archive carries no `_frozen` companion arrays -- "
                     "this is the PARENT (frozen-label) archive, not a strictfill relabel. "
                     "Point ARCHIVE at a strictfill_year.py output or drop STRICT.")
meta = str(z["meta"]) if "meta" in z.files else ""
print(f"[{SYM}] archive={os.path.basename(ARCHIVE)} arrays={ARRAYS}")
print(f"[meta] {meta[:360]}")

PL = z[f"pnl_long{SUF}"].astype(np.float64) * 100.0     # (NC, 1, N) pct -> bp
PS = z[f"pnl_short{SUF}"].astype(np.float64) * 100.0
FL = z[f"fill_long{SUF}"][0].astype(bool)
FS = z[f"fill_short{SUF}"][0].astype(bool)
NC = PL.shape[0]
N = len(FL)
print(f"[dims] decisions={N:,} hold-configs={NC}  fill_long={FL.mean():.4f} fill_short={FS.mean():.4f}")

alone_l = FL & ~FS; both_l = FL & FS
alone_s = FS & ~FL; both_s = FS & FL


def pooled(mask_l, mask_s, ci):
    """EV/hit over both sides pooled: each side contributes its own filled rows."""
    a = PL[ci, 0, :][mask_l & np.isfinite(PL[ci, 0, :])]
    b = PS[ci, 0, :][mask_s & np.isfinite(PS[ci, 0, :])]
    x = np.concatenate([a, b])
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), ev=float(x.mean()),
                se=float(x.std(ddof=1) / np.sqrt(len(x))),
                hit=float(100.0 * (x > 0).mean()),
                avg_win=float(x[x > 0].mean()), avg_loss=float(x[x <= 0].mean()))


res = {"sym": SYM, "arrays": ARRAYS, "n_decisions": int(N),
       "fill_long": float(FL.mean()), "fill_short": float(FS.mean()), "holds": []}

print(f"\n=== {SYM}  ALONE vs BOTH at every hold  (model-free; population fixed, only the exit moves)")
print(f"{'cfg':>4} | {'n ALONE':>10} {'EV':>8} {'hit':>6} {'win':>7} {'loss':>8} "
      f"| {'n BOTH':>10} {'EV':>8} {'hit':>6} | {'gap':>8}")
for ci in range(NC):
    a = pooled(alone_l, alone_s, ci)
    b = pooled(both_l, both_s, ci)
    gap = a["ev"] - b["ev"]
    res["holds"].append(dict(cfg=ci, alone=a, both=b, gap=float(gap)))
    print(f"{ci:>4} | {a['n']:>10,} {a['ev']:>+8.3f} {a['hit']:>6.2f} {a['avg_win']:>+7.2f} "
          f"{a['avg_loss']:>+8.2f} | {b['n']:>10,} {b['ev']:>+8.3f} {b['hit']:>6.2f} | {gap:>+8.3f}")

print("\nreading: if |EV(ALONE)| GROWS with the hold index the loss accrues along the path and "
      "an exit rule attacks it; if it is flat the damage is at the fill and the exit cannot.")

out = os.path.join(os.path.dirname(ARCHIVE), f"ADVHOLD_{SYM}{'_strict' if STRICT_WANTED else ''}.json")
with open(out, "w") as fh:
    json.dump(res, fh, indent=1, default=float)
print(f"wrote {out}")
