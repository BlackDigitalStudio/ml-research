#!/usr/bin/env python3
"""OFFLINE alpha-surface from saved XGBoost predictions (NO retraining).

Reads research_runs/xgb_maker/preds_{SYM}.npz (subs60_xgb_makerlabel.py):
  per TEST sample -> ts, day, sid, rH60, yA, pA, pB, pnl_long/short (NC,QM,n) %, fill_*(QM,n),
  meta{cfgs, queue_mults, fee_bp, thr}.

Computes (per CLAUDE.md: surface + argmax, NOT a verdict):
  (A) Model A vol-gate: per-symbol AUC, prec@{1,0.5,0.2}%, vol_thr, nf rate.
  (B) Model B direction skill, ORACLE-gated (realized non-flat yA==1): executed net maker
      EV(bp) @ B-conviction top-{50,20,10}% + dir_acc + fill  [B's intrinsic skill].
  (C) HONEST CASCADE: gate by Model A PREDICTION pA (top-g%), take B side, executed net maker
      EV(bp) @ A-selectivity g in {5,2,1,0.5,0.2}% + dir_acc + fill + trades/day  [deployable].
  Over ALL maker configs (hold-60s / RR6 / RR2) x queue-mult (touch / queue). Reports the
  argmax cell per symbol. Saves full grid -> research_runs/xgb_maker/SURFACE.json.
"""
import argparse, io, json
import numpy as np
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"; OUT = "research_runs/xgb_maker"
SYMS = ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def auc(score, lab):
    o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


def load_preds(symk):
    try:
        d = np.load(io.BytesIO(bk.blob(f"{OUT}/preds_{symk}.npz").download_as_bytes()), allow_pickle=True)
    except Exception:
        return None
    m = json.loads(str(d["meta"]))
    return {"ts": d["ts"], "day": d["day"], "rH60": d["rH60"], "yA": d["yA"].astype(bool),
            "pA": d["pA"], "pB": d["pB"], "pnl_long": d["pnl_long"], "pnl_short": d["pnl_short"],
            "fill_long": d["fill_long"].astype(bool), "fill_short": d["fill_short"].astype(bool), "meta": m}


def sides(P, c, q):
    fee = P["meta"]["fee_bp"]
    nl = P["pnl_long"][c, q].astype(np.float64) * 100.0 - fee
    ns = P["pnl_short"][c, q].astype(np.float64) * 100.0 - fee
    return nl, ns, P["fill_long"][q], P["fill_short"][q]


def exec_ev(pB, nl, ns, fl, fs, idx):
    """Take B side on idx; executed net EV over filled-chosen + dir_acc + fill + n."""
    pl = pB[idx] >= 0.5
    cn = np.where(pl, nl[idx], ns[idx]); cf = np.where(pl, fl[idx], fs[idx])
    ex = cf & np.isfinite(cn)
    nle = np.where(fl[idx], nl[idx], -np.inf); nse = np.where(fs[idx], ns[idx], -np.inf)
    one = (fl[idx] | fs[idx]); better = (nle > nse)
    return {"execEV_bp": float(cn[ex].mean()) if ex.any() else float("nan"),
            "dir_acc": float((pl[one] == better[one]).mean()) if one.any() else float("nan"),
            "fill_chosen": float(cf.mean()), "n_exec": int(ex.sum()), "n_sel": int(len(idx))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=SYMS); a = ap.parse_args()
    P = {s: load_preds(s) for s in a.symbols}; P = {k: v for k, v in P.items() if v}
    if not P:
        print("no preds found"); return
    cfgs = P[list(P)[0]]["meta"]["cfgs"]; qms = P[list(P)[0]]["meta"]["queue_mults"]
    NC, QM = len(cfgs), len(qms)
    cfg_lab = [f"{('hold' if c['tp']>=1 else 'RR'+str(round(c['tp']/c['sl'])))}" for c in cfgs]
    qm_lab = [("touch" if q == 0 else f"queue{int(q)}") for q in qms]
    A_SEL = [5.0, 2.0, 1.0, 0.5, 0.2]; B_CONV = [50.0, 20.0, 10.0]
    surface = {"cfg_labels": cfg_lab, "qm_labels": qm_lab, "A": {}, "B_oracle": {}, "cascade": {}}

    # (A) Model A
    print("=== (A) MODEL A vol-gate (per-symbol, vol-adaptive thr) ===")
    print(f"{'SYM':5s} {'days':>5s} {'n_test':>8s} {'thr_bp':>6s} {'nf%':>5s} {'AUC':>6s} {'p@1%':>6s} {'p@.5%':>6s} {'p@.2%':>6s}")
    for s, p in P.items():
        nd = len(set(p["day"].tolist())); pa, ya = p["pA"], p["yA"]; order = np.argsort(-pa)
        prec = {q: float(ya[order[:max(20, int(len(pa)*q/100))]].mean()) for q in (1.0, 0.5, 0.2)}
        surface["A"][s] = {"days": nd, "n_test": int(len(pa)), "vol_thr_bp": p["meta"]["thr"],
                           "nf_test": float(ya.mean()), "auc": auc(pa, ya), "prec": prec}
        print(f"{s:5s} {nd:5d} {len(pa):8d} {p['meta']['thr']:6.1f} {ya.mean()*100:4.1f}% {auc(pa,ya):6.3f} "
              f"{prec[1.0]:6.2f} {prec[0.5]:6.2f} {prec[0.2]:6.2f}")

    # (B) oracle-gated B skill + (C) honest cascade — chosen cfg/qm headline, full grid saved
    ci0 = P[list(P)[0]]["meta"]["cfg_idx"]; qi0 = list(qms).index(P[list(P)[0]]["meta"]["qm"])
    print(f"\n=== (B) MODEL B oracle-gated direction skill [cfg={cfg_lab[ci0]} {qm_lab[qi0]}] ===")
    print(f"{'SYM':5s}  execEV@convTop 50/20/10%      dir_acc@10  fill@10")
    for s, p in P.items():
        nl, ns, fl, fs = sides(p, ci0, qi0); nf = np.where(p["yA"])[0]
        conv = np.abs(p["pB"][nf] - 0.5); o = nf[np.argsort(-conv)]
        row = {}
        for q in B_CONV:
            k = max(20, int(len(nf) * q / 100)); row[f"q{q}"] = exec_ev(p["pB"], nl, ns, fl, fs, o[:k])
        surface["B_oracle"].setdefault(s, {})[f"{cfg_lab[ci0]}_{qm_lab[qi0]}"] = row
        print(f"{s:5s}  {row['q50.0']['execEV_bp']:+6.2f}/{row['q20.0']['execEV_bp']:+6.2f}/{row['q10.0']['execEV_bp']:+6.2f}bp"
              f"     {row['q10.0']['dir_acc']:.3f}    {row['q10.0']['fill_chosen']:.2f}")

    print(f"\n=== (C) HONEST CASCADE: A-pred gate (top-g%) x B side [cfg={cfg_lab[ci0]} {qm_lab[qi0]}] ===")
    print(f"{'SYM':5s}  execEV @ A-top 5/2/1/0.5/0.2%                    trd/day@1%  dir@1%")
    for s, p in P.items():
        nl, ns, fl, fs = sides(p, ci0, qi0); nd = len(set(p["day"].tolist())); o = np.argsort(-p["pA"])
        row = {}
        for g in A_SEL:
            k = max(20, int(len(p["pA"]) * g / 100)); row[f"g{g}"] = exec_ev(p["pB"], nl, ns, fl, fs, o[:k])
        surface["cascade"].setdefault(s, {})[f"{cfg_lab[ci0]}_{qm_lab[qi0]}"] = row
        tpd = row["g1.0"]["n_exec"] / max(nd, 1)
        print(f"{s:5s}  {row['g5.0']['execEV_bp']:+6.2f}/{row['g2.0']['execEV_bp']:+6.2f}/{row['g1.0']['execEV_bp']:+6.2f}/"
              f"{row['g0.5']['execEV_bp']:+6.2f}/{row['g0.2']['execEV_bp']:+6.2f}bp   {tpd:7.1f}    {row['g1.0']['dir_acc']:.3f}")

    # full grid over ALL cfg x qm (saved, not all printed) + argmax cell per symbol
    print(f"\n=== ARGMAX maker cell per symbol (cascade, best execEV over cfg x qm x A-selectivity) ===")
    for s, p in P.items():
        best = None
        for c in range(NC):
            for q in range(QM):
                nl, ns, fl, fs = sides(p, c, q); o = np.argsort(-p["pA"])
                for g in A_SEL:
                    k = max(20, int(len(p["pA"]) * g / 100)); r = exec_ev(p["pB"], nl, ns, fl, fs, o[:k])
                    surface["cascade"].setdefault(s, {}).setdefault(f"{cfg_lab[c]}_{qm_lab[q]}", {})[f"g{g}"] = r
                    if r["n_exec"] >= 50 and (best is None or (np.isfinite(r["execEV_bp"]) and r["execEV_bp"] > best[0])):
                        best = (r["execEV_bp"], cfg_lab[c], qm_lab[q], g, r["dir_acc"], r["fill_chosen"], r["n_exec"])
        if best:
            surface["cascade"][s]["argmax"] = {"execEV_bp": best[0], "cfg": best[1], "qm": best[2],
                                               "A_sel_pct": best[3], "dir_acc": best[4], "fill": best[5], "n_exec": best[6]}
            print(f"{s:5s}  best execEV={best[0]:+6.2f}bp @ {best[1]}/{best[2]} A-top{best[3]}%  "
                  f"dir_acc={best[4]:.3f} fill={best[5]:.2f} n={best[6]}")

    bk.blob(f"{OUT}/SURFACE.json").upload_from_string(json.dumps(surface, default=float))
    print(f"\n[saved] gs://{BUCKET}/{OUT}/SURFACE.json")


if __name__ == "__main__":
    main()
