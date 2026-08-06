#!/usr/bin/env python3
"""Build the fixed-HP specs for the OBJSEL rev2 refit arms, from the rev1 artifacts.

Four arms per (symbol, seed), each a per-fold hyperparameter assignment that the
patched trainer consumes instead of running its own search:

  INC    the trial bulk IC actually picked        -> the PARITY ANCHOR: this arm must
                                                     reproduce the stored PERFOLD cell
  MAXEV  the trial with the highest test EV       -> upper extreme of the rev1 spread
  MINEV  the trial with the lowest test EV        -> lower extreme
  EVBUD  the trial ev_budget would have picked    -> the head-to-head vs INC

hpA/biA are taken from the captured MODELS_*_hp.json and are IDENTICAL across arms -
only the B hyperparameters move, which is what rev1 measured and what the refit arm
is meant to isolate.

CONSISTENCY CHECK, run on every cell before anything is written: the INC arm's hpB/biB
rebuilt from the trial index must equal the hpB/biB the trainer stored. A mismatch means
the trial index and the chosen-HP file disagree about which trial won, and the whole
construction is invalid - so this aborts rather than writes.

Env: OBJSEL_DIR (rev1 jsons), HP_DIR (MODELS_*_hp.json), OUT_DIR, SEEDS, SYMS.
"""
import json, os, sys

OBJSEL_DIR = os.environ.get("OBJSEL_DIR", "objsel")
HP_DIR = os.environ.get("HP_DIR", "hp")
OUT_DIR = os.environ.get("OUT_DIR", "refit_spec")
SYMS = [s for s in os.environ.get("SYMS", "DOGE,XRP,BTC,ETH").split(",") if s]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3").split(",") if x != ""]
ARMS = ["INC", "MAXEV", "MINEV", "EVBUD"]
os.makedirs(OUT_DIR, exist_ok=True)


def split_params(p):
    """trial params -> (hpB, biB), exactly as tune_B_ic returns them."""
    hp = dict(p); nr = hp.pop("num_boost_round")
    return hp, int(nr)


def finite(x):
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


bad = []; written = 0; rows = []
for sym in SYMS:
    D = json.load(open(os.path.join(OBJSEL_DIR, f"OBJSEL_{sym}.json"), encoding="utf-8"))
    cells = {(c["seed"], c["fold"]): c for c in D["cells"]}
    for seed in SEEDS:
        spec = {a: {} for a in ARMS}
        for fold in range(6):
            c = cells.get((seed, fold))
            if c is None:
                bad.append(f"{sym} S{seed} f{fold}: missing rev1 cell"); continue
            hpf = os.path.join(HP_DIR, f"MODELS_S{seed}_{sym}_f{fold}_hp.json")
            H = json.load(open(hpf, encoding="utf-8"))
            tr = c["trials"]
            ev = [t["ev_test"] for t in tr]
            pick = {
                "INC":   c["incumbent_trial"],
                "MAXEV": max(range(len(tr)), key=lambda i: ev[i] if finite(ev[i]) else -1e18),
                "MINEV": min(range(len(tr)), key=lambda i: ev[i] if finite(ev[i]) else +1e18),
                "EVBUD": c["summary"]["ev_budget"]["pick"],
            }
            # --- consistency: the INC arm must reconstruct the trainer's own choice
            hpB_inc, biB_inc = split_params(tr[pick["INC"]]["params"])
            if hpB_inc != H["hpB"] or biB_inc != int(H["biB"]):
                bad.append(f"{sym} S{seed} f{fold}: INC hpB mismatch\n"
                           f"     rebuilt {hpB_inc} bi={biB_inc}\n"
                           f"     stored  {H['hpB']} bi={H['biB']}")
                continue
            for a in ARMS:
                hpB, biB = split_params(tr[pick[a]]["params"])
                spec[a][f"f{fold}"] = {"hpA": H["hpA"], "biA": int(H["biA"]),
                                       "hpB": hpB, "biB": biB,
                                       "src_trial": pick[a], "ev_test_rev1": ev[pick[a]]}
            rows.append((sym, seed, fold, pick["INC"], pick["MAXEV"], pick["MINEV"], pick["EVBUD"],
                         ev[pick["INC"]], ev[pick["MAXEV"]], ev[pick["MINEV"]], ev[pick["EVBUD"]]))
        for a in ARMS:
            if len(spec[a]) == 6:
                with open(os.path.join(OUT_DIR, f"HPSPEC_{a}_{sym}_S{seed}.json"), "w", encoding="utf-8") as f:
                    json.dump(spec[a], f, indent=1)
                written += 1

if bad:
    print(f"*** CONSISTENCY FAILURES: {len(bad)}", file=sys.stderr)
    for b in bad[:10]:
        print("   " + b, file=sys.stderr)
    raise SystemExit(1)

print(f"[ok] {len(rows)} cells passed the INC consistency check; wrote {written} spec files -> {OUT_DIR}")
print(f"\n{'sym':>5} {'S':>2} {'f':>2} | {'INC':>4} {'MAX':>4} {'MIN':>4} {'EVB':>4} | "
      f"{'ev_INC':>7} {'ev_MAX':>7} {'ev_MIN':>7} {'ev_EVB':>7}")
for r in rows[:8]:
    print(f"{r[0]:>5} {r[1]:>2} {r[2]:>2} | {r[3]:>4} {r[4]:>4} {r[5]:>4} {r[6]:>4} | "
          f"{r[7]:>+7.2f} {r[8]:>+7.2f} {r[9]:>+7.2f} {r[10]:>+7.2f}")
n_same = sum(1 for r in rows if r[3] == r[6])
print(f"\nEVBUD picks the same trial as INC in {n_same}/{len(rows)} cells "
      f"(those contribute an exact 0 to the head-to-head by construction)")
