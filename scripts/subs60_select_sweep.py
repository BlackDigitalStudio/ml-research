#!/usr/bin/env python3
"""Phase-1 selection-signal sweep (reanalysis from saved caches; NO re-download/infer).
Judge = realized hold-to-60s EV = mean(sign(Blog)*rH60) on selected windows.
Compares 4 inference-legal rankers x selectivity, per symbol. rH_full covers ALL
windows so B-/joint-/EV-based selection is valid across the full population (hold).
"""
import io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
OUTP = "research_runs/gru_gridsim"
SYMS = ["DOGE", "ETH", "LINK"]
TIERS = {"tt": 4.0 + 6.0, "mt": 4.0 + 3.0, "mm": 4.0}   # RT bp: taker-taker .10%/maker-taker .07%/maker-maker .04%
# (express as bp: tt=10, mt=7, mm=4)
TIERS = {"tt": 10.0, "mt": 7.0, "mm": 4.0}
QS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]             # selectivity (% of scored universe)
bk = storage.Client(project=PROJ).bucket(BUCKET)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def rankpct(x):                                          # 0..1 percentile rank
    o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def main():
    allres = {}
    for sym in SYMS:
        cz = np.load(io.BytesIO(bk.blob(f"{OUTP}/cache/{sym}_cache.npz").download_as_bytes()))
        A = cz["Alog_full"].astype(np.float64); B = cz["Blog_full"].astype(np.float64)
        rH = cz["rH_full"].astype(np.float64)            # signed bp @60s (label)
        n = len(A); nted = int(cz["nted"])
        side = np.sign(B); side[side == 0] = 1
        realized = side * rH                              # bp hold-to-60s per window
        signals = {
            "A":     A,                                   # vol-conviction (current gate)
            "|B|":   np.abs(B),                           # direction-conviction
            "joint": rankpct(A) + rankpct(np.abs(B)),     # both-confident (rank-based, calib-free)
            "EV":    sig(A) * np.abs(2 * sig(B) - 1),      # P(big move) x |dir-edge|  (proxy)
        }
        print(f"\n================  {sym}  (n_scored={n}, test_days={nted})  ================")
        print(f"{'q%':>6} {'trd/d':>6} | " + " | ".join(f"{s:>22}" for s in signals))
        print(f"{'':>6} {'':>6} | " + " | ".join(f"{'gross  net_mm   WR':>22}" for _ in signals))
        symres = {}
        for q in QS:
            k = max(int(round(n * q / 100.0)), 20)
            tpd = (n * q / 100.0) / nted
            row = []
            for sname, sv in signals.items():
                sel = np.argpartition(-sv, k)[:k]
                g = float(realized[sel].mean())           # gross bp
                wr = float((realized[sel] > 0).mean())
                row.append((sname, g, g - TIERS["mm"], wr))
                symres.setdefault(sname, {})[q] = {"gross_bp": g, "wr": wr, "trd_day": tpd,
                                                   "net": {t: g - rt for t, rt in TIERS.items()}}
            cells = " | ".join(f"{g:+6.2f} {g-TIERS['mm']:+6.2f} {wr:5.2f}" for _, g, _, wr in row)
            print(f"{q:>6} {tpd:>6.1f} | {cells}")
        # best (signal,q) by net_mm and by net_mt
        best = {}
        for tier in ["mm", "mt", "tt"]:
            bs = max(((s, q, d["net"][tier], d["gross_bp"], d["wr"], d["trd_day"])
                      for s, qd in symres.items() for q, d in qd.items()),
                     key=lambda z: z[2])
            best[tier] = {"signal": bs[0], "q": bs[1], "net_bp": bs[2], "gross_bp": bs[3],
                          "wr": bs[4], "trd_day": bs[5]}
        print(f"  BEST hold-EV: " + " | ".join(
            f"{t}: {b['signal']}@{b['q']}% net={b['net_bp']:+.2f} (gross={b['gross_bp']:+.2f} "
            f"WR={b['wr']:.2f} {b['trd_day']:.1f}/d)" for t, b in best.items()))
        allres[sym] = {"by_signal": symres, "best": best}
    bk.blob(f"{OUTP}/select_sweep.json").upload_from_string(json.dumps(allres, default=float))
    print(f"\n[saved] {OUTP}/select_sweep.json")


if __name__ == "__main__":
    main()
