#!/usr/bin/env python3
"""B2-v2: per-symbol R:R via ON-DEMAND grid_sim over a FINE (e.g. ~100k) TP/SL grid from saved maker
paths (the §14 way -- no pre-stored per-config pnl). Sequence (matches GRU s14):
  Stage-1 B (hold) -> GRID: grid_sim sweeps n_tp x n_sl configs on the VAL daily-budget(budget/day)
  windows (model-sided) -> c* = argmax val daily-budget EV -> Stage-2 B trained on c* better-side ->
  eval daily-budget on TEST: c* vs hold-60s baseline.
Paths from research_runs/maker_paths/{SYM}/{DATE}.npz (subs60_save_maker_paths.py). B-train gate =
A-OOF-predicted non-flat top-gate_pct% (apred). budget trades/symbol/day.
Saves -> research_runs/maker_labels_rr/B2GRID_RESULT.json.
Run: python3 subs60_xgb_b2_grid.py --symbols BNB BTC DOGE ETH LINK LTC SOL XRP --n-tp 317 --n-sl 317 --budget 10
"""
import argparse, io, json, os, shutil, subprocess, tempfile
import numpy as np
from google.cloud import storage
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def auc(score, lab):
    o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


def tune_b(Xtr, ytr, wtr, Xiv, yiv, n_trials, seed=0):
    """Optuna-tune B XGB HPs maximizing inner-val AUC of the better-side label."""
    base = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": seed, "eval_metric": "auc"}
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr); div = xgb.DMatrix(Xiv, label=yiv)
    def obj(t):
        p = dict(base, max_depth=t.suggest_int("max_depth", 3, 9),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 200, log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-4, 5.0, log=True))
        b = xgb.train(p, dtr, num_boost_round=400, evals=[(div, "iv")], early_stopping_rounds=25, verbose_eval=False)
        return auc(b.predict(div, iteration_range=(0, b.best_iteration + 1)), yiv)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = dict(base, **st.best_params)
    bfin = xgb.train(bp, dtr, num_boost_round=400, evals=[(div, "iv")], early_stopping_rounds=25, verbose_eval=False)
    return bp, max(1, bfin.best_iteration + 1)

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; PATHS = "research_runs/maker_paths"; MAIN = "research_runs/xgb_maker"
FEATS = "feats_sub60"; GRID = "/tmp/husdc/rust_ingest/target/release/grid_sim"
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05; TO_TICKS = 563; ENTRY_WIN = 120
SYMS = ["BNB", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
PATH_ARRS = ["entry_long", "entry_short", "mid_paths", "book_paths", "flow_paths", "entry_q", "entry_book"]
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def load_rr(symk):
    d = np.load(io.BytesIO(bk.blob(f"{RR}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "ts": d["ts"].astype(np.int64), "pnl_long": d["pnl_long"].astype(np.float64),
            "pnl_short": d["pnl_short"].astype(np.float64), "fill_long": d["fill_long"].astype(bool),
            "fill_short": d["fill_short"].astype(bool), "ndays": m["n_days"], "fee": m["maker_rt_fee_pct"] * 100.0}


def sym_dates(sym):
    return sorted(b.name.split("/")[-1].replace(".npz", "")
                  for b in bk.client.list_blobs(bk, prefix=f"{FEATS}/{sym}/") if b.name.endswith(".npz"))


def split(day, ndays):
    cut = int(ndays * SPLIT[0]); emb = int(ndays * SPLIT[1]); tr = day < cut
    td = sorted(set(day[tr].tolist())); vcut = td[int(len(td) * SPLIT[2])] if td else cut
    return (tr & (day < vcut)), (tr & (day >= vcut)), (day >= emb)


def pct_rank(x):
    o = np.argsort(np.argsort(x)); return o / max(len(x) - 1, 1)


def daily_pick(day, score, n):
    order = np.lexsort((-score, day)); ds = day[order]
    st = np.zeros(len(order), bool); st[0] = True; st[1:] = ds[1:] != ds[:-1]
    si = np.where(st)[0]; within = np.arange(len(order)) - np.repeat(si, np.diff(np.append(si, len(order))))
    return order[within < n]


def oof_pA(F, yA, trn, day, hpA, k=5):
    tdays = sorted(set(day[trn].tolist())); fold = {d: i % k for i, d in enumerate(tdays)}
    fday = np.array([fold.get(int(d), -1) for d in day]); oof = np.full(len(F), np.nan)
    for kk in range(k):
        trk = trn & (fday != kk); vak = trn & (fday == kk)
        if vak.sum() < 50 or (yA[trk] == 1).sum() < 20:
            continue
        spw = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
        b = xgb.train(dict(base_p("auc", spw), **hpA["best_params"]), xgb.DMatrix(F[trk], label=yA[trk]),
                      num_boost_round=max(1, hpA["best_iter"] + 1))
        oof[np.where(vak)[0]] = b.predict(xgb.DMatrix(F[vak]))
    return oof


def base_p(metric, spw=None):
    p = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0, "eval_metric": metric}
    if spw is not None:
        p["scale_pos_weight"] = spw
    return p


def load_paths(symk, dates, want_ts):
    """Load maker-path npz for the given dates, keeping ONLY rows whose ts is in want_ts (memory-bounded
    by |want_ts|, not by total rows), returned ordered by want_ts. keep = mask of want_ts found."""
    want = set(int(t) for t in np.asarray(want_ts).tolist())
    acc = {k: [] for k in PATH_ARRS}; tsacc = []
    for dt in dates:
        try:
            d = np.load(io.BytesIO(bk.blob(f"{PATHS}/{symk}/{dt}.npz").download_as_bytes()))
        except Exception:
            continue
        dts = d["ts"].astype(np.int64)
        m = np.fromiter((int(t) in want for t in dts), bool, len(dts))
        if not m.any():
            continue
        tsacc.append(dts[m])
        for k in PATH_ARRS:
            acc[k].append(d[k][m])
    if not tsacc:
        return None
    gts = np.concatenate(tsacc); arrs = {k: np.concatenate(acc[k], 0) for k in PATH_ARRS}
    pos = {int(t): i for i, t in enumerate(gts)}
    idx = np.array([pos[int(t)] for t in want_ts if int(t) in pos])
    keep = np.array([int(t) in pos for t in want_ts])
    return {k: arrs[k][idx] for k in PATH_ARRS}, keep


def make_grid(n_tp, n_sl):
    tps = np.linspace(0.05, 1.5, n_tp); sls = np.linspace(0.03, 0.40, n_sl)
    cfgs = [{"tp": 50.0, "sl": 50.0, "to": TO_TICKS, "par": False, "tr": False}]  # hold (index 0)
    for sl in sls:
        for tp in tps:
            if 0.5 <= tp / sl <= 30.0:
                cfgs.append({"tp": round(float(tp), 4), "sl": round(float(sl), 4), "to": TO_TICKS, "par": False, "tr": False})
    return cfgs


def run_grid(tmp, arrs, cfgs):
    g = os.path.join(tmp, "g")
    dt = {"flow_paths": np.float32}   # grid_sim expects f64 for all maker arrays except flow (f32)
    for k in PATH_ARRS:
        np.save(f"{tmp}/{k}.npy", arrs[k].astype(dt.get(k, np.float64)))
    json.dump(cfgs, open(f"{tmp}/cfg.json", "w"))
    cmd = [GRID, "--entry-long", f"{tmp}/entry_long.npy", "--entry-short", f"{tmp}/entry_short.npy",
           "--mid-paths", f"{tmp}/mid_paths.npy", "--book-paths", f"{tmp}/book_paths.npy",
           "--entry-book", f"{tmp}/entry_book.npy", "--flow-paths", f"{tmp}/flow_paths.npy",
           "--entry-q", f"{tmp}/entry_q.npy", "--configs", f"{tmp}/cfg.json", "--out-prefix", g,
           "--queue-mult", "0", "--entry-window-ticks", str(ENTRY_WIN), "--maker-offset-frac", "0",
           "--commission-win-pct", "0", "--commission-loss-pct", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"grid_sim fail: {r.stderr[-300:]}")
    pl = np.load(f"{g}_pnl_long.npy"); ps = np.load(f"{g}_pnl_short.npy")
    fl = np.load(f"{g}_filled_long.npy").astype(bool); fs = np.load(f"{g}_filled_short.npy").astype(bool)
    for f in os.listdir(tmp):
        if f.endswith(".npy"):
            os.remove(f"{tmp}/{f}")
    return pl, ps, fl, fs   # pl/ps (NC,n) %, fl/fs (n,)


def net_side(pl_row, ps_row, fl, fs, fee, side_long):
    nl = pl_row * 100.0 - fee; ns = ps_row * 100.0 - fee
    net = np.where(side_long, nl, ns); fillc = np.where(side_long, fl, fs)
    return net, fillc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=SYMS)
    ap.add_argument("--n-tp", type=int, default=317); ap.add_argument("--n-sl", type=int, default=317)
    ap.add_argument("--budget", type=int, default=10); ap.add_argument("--gate-pct", type=float, default=5.0)
    ap.add_argument("--optuna-b", action="store_true")   # per-symbol Optuna-tune B (else reuse pooled B HP)
    ap.add_argument("--b-trials", type=int, default=25)
    a = ap.parse_args()
    def log(s): print(s, flush=True)
    cfgs = make_grid(a.n_tp, a.n_sl); NC = len(cfgs)
    log(f"GRID {NC} configs (hold + {NC-1} TP/SL) | budget={a.budget}/day gate_pct={a.gate_pct}")
    hpB = jload(f"{MAIN}/B_pool.json")
    tmp = tempfile.mkdtemp(prefix="b2g_", dir="/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
    res = {"n_configs": NC, "budget": a.budget, "per_symbol": {}}
    log(f"{'SYM':5s} {'c*_RR':>18s}  valEV*/hold  | test 1bud hold->c*  | top25 hold->c*")
    try:
        for symk in a.symbols:
            d = load_rr(symk); hpA = jload(f"{MAIN}/A_{symk}.json"); dates = sym_dates(symk + "-USDT-PERP")
            F = d["F"]; rH = d["rH"]; day = d["day"]; ts = d["ts"]; fee = d["fee"]
            trn, val, te = split(day, d["ndays"])
            thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
            vi = np.where(val)[0]; ti = np.where(te)[0]
            spw = float((yA[trn] == 0).sum() / max((yA[trn] == 1).sum(), 1))
            bstA = xgb.train(dict(base_p("auc", spw), **hpA["best_params"]), xgb.DMatrix(F[trn], label=yA[trn]),
                             num_boost_round=max(1, hpA["best_iter"] + 1))
            pA_v = bstA.predict(xgb.DMatrix(F[vi])); pA_t = bstA.predict(xgb.DMatrix(F[ti]))
            # apred gate (train windows B sees)
            oof = oof_pA(F, yA, trn, day, hpA); valid = trn & np.isfinite(oof)
            gthr = float(np.nanquantile(oof[valid], 1 - a.gate_pct / 100.0)); gate_train = valid & (oof >= gthr)
            # hold pnl from maker_labels_rr (config 0 = hold)
            holdl = d["pnl_long"][0, 0]; holds = d["pnl_short"][0, 0]; fl = d["fill_long"][0]; fs = d["fill_short"][0]

            def trainB(nl, ns, mask, idx):
                yB = (np.where(fl, nl, -np.inf) > np.where(fs, ns, -np.inf)).astype(int)
                both = fl & fs
                w = np.where(both, np.abs(nl - ns), np.where(fl, np.abs(nl), np.where(fs, np.abs(ns), 0.0)))
                clip = lambda m: np.clip(w[m], 0, np.quantile(w[m][w[m] > 0], 0.99) if (w[m] > 0).any() else 1.0)
                params, nr = dict(base_p("logloss"), **hpB["best_params"]), max(1, hpB["best_iter"] + 1)
                if a.optuna_b:                                  # per-symbol tune on inner split of gate_train (main val untouched)
                    md = sorted(set(day[mask].tolist())); vcut = md[int(len(md) * 0.85)] if md else 0
                    innr = mask & (day < vcut); innv = mask & (day >= vcut)
                    if innr.sum() > 300 and (yB[innr] == 1).sum() > 10 and (yB[innv] == 1).sum() > 10 and (yB[innv] == 0).sum() > 10:
                        params, nr = tune_b(F[innr], yB[innr], clip(innr), F[innv], yB[innv], a.b_trials)
                b = xgb.train(params, xgb.DMatrix(F[mask], label=yB[mask], weight=clip(mask)), num_boost_round=nr)
                return b.predict(xgb.DMatrix(F[idx]))

            # STAGE-1 B (hold) -> predict val + test
            keep = gate_train & (fl | fs)
            pB1_v = trainB(holdl * 100 - fee, holds * 100 - fee, keep, vi)
            pB1_t = trainB(holdl * 100 - fee, holds * 100 - fee, keep, ti)
            # VAL daily-budget selection (model-sided)
            selv = daily_pick(day[vi], pct_rank(pA_v) * pct_rank(np.abs(pB1_v - 0.5)), a.budget)
            valsel_ts = ts[vi][selv]; side1 = (pB1_v[selv] >= 0.5)
            vdates = [dates[i] for i in sorted(set(day[vi][selv].tolist())) if i < len(dates)]
            lp = load_paths(symk, vdates, valsel_ts)
            if lp is None or lp[1].sum() < 20:
                log(f"{symk}: val paths missing"); continue
            arrs, kmask = lp; side1 = side1[kmask]
            # GRID: grid_sim NC configs on val-selected windows -> c* by val daily-budget EV (B1-sided)
            pl, ps, gfl, gfs = run_grid(tmp, arrs, cfgs)
            evc = np.full(NC, -1e9)
            for c in range(NC):
                net, fc = net_side(pl[c], ps[c], gfl, gfs, fee, side1)
                ex = fc & np.isfinite(net)
                if ex.sum() >= 10:
                    evc[c] = net[ex].mean()
            cstar = int(np.argmax(evc)); cfg_star = cfgs[cstar]
            rr = "hold" if cfg_star["tp"] >= 1 else f"RR{cfg_star['tp']/cfg_star['sl']:.1f}({cfg_star['tp']}/{cfg_star['sl']})"
            # STAGE-2 B on c*: need c* pnl on gate_train windows -> grid_sim c* on train-gated paths
            gi = np.where(keep)[0]; gt_ts = ts[gi]
            gdates = [dates[i] for i in sorted(set(day[keep].tolist())) if i < len(dates)]
            lpt = load_paths(symk, gdates, gt_ts)
            if lpt is None:
                log(f"{symk}: train paths missing"); continue
            tarrs, tkeep = lpt
            plc, psc, tfl, tfs = run_grid(tmp, tarrs, [cfgs[0], cfg_star])  # idx1 = c*
            # build full-length c* nets on gate_train rows (others stay hold to keep array shape simple)
            nl_cs = holdl * 100 - fee; ns_cs = holds * 100 - fee   # default hold elsewhere
            nl_cs = nl_cs.copy(); ns_cs = ns_cs.copy()
            gi_keep = gi[tkeep]
            nl_cs[gi_keep] = plc[1] * 100 - fee; ns_cs[gi_keep] = psc[1] * 100 - fee
            # NOTE: fill for c* on train uses grid fill; approximate with maker fl/fs (entry-fill is config-independent)
            pB2_t = trainB(nl_cs, ns_cs, keep, ti)
            # EVAL on TEST daily-budget: hold (pB1_t) vs c* (pB2_t)
            def test_ev(pB_t, cfg_idx_in_pair, use_hold):
                selt = daily_pick(day[ti], pct_rank(pA_t) * pct_rank(np.abs(pB_t - 0.5)), a.budget)
                tsel_ts = ts[ti][selt]; sideb = (pB_t[selt] >= 0.5)
                tdates = [dates[i] for i in sorted(set(day[ti][selt].tolist())) if i < len(dates)]
                lpe = load_paths(symk, tdates, tsel_ts)
                if lpe is None:
                    return float("nan"), float("nan"), 0
                earrs, ekeep = lpe; sideb = sideb[ekeep]
                pl_e, ps_e, efl, efs = run_grid(tmp, earrs, [cfgs[0]] if use_hold else [cfgs[0], cfg_star])
                row = 0 if use_hold else 1
                net, fc = net_side(pl_e[row], ps_e[row], efl, efs, fee, sideb)
                ex = fc & np.isfinite(net); ev = float(net[ex].mean()) if ex.any() else float("nan")
                conv = (pct_rank(pA_t) * pct_rank(np.abs(pB_t - 0.5)))[selt][ekeep]
                top = np.argsort(-conv)[:max(5, ekeep.sum() // 4)]
                ex2 = fc[top] & np.isfinite(net[top]); ev25 = float(net[top][ex2].mean()) if ex2.any() else float("nan")
                return ev, ev25, int(ex.sum())
            evh, evh25, nh = test_ev(pB1_t, 0, True)
            evs, evs25, nsx = test_ev(pB2_t, 1, cstar == 0)
            res["per_symbol"][symk] = {"c_star": rr, "c_star_idx": cstar, "valEV_cstar": float(evc[cstar]),
                                       "valEV_hold": float(evc[0]), "test_hold_EV": evh, "test_cstar_EV": evs,
                                       "test_hold_top25": evh25, "test_cstar_top25": evs25, "n_test": nh}
            log(f"{symk:5s} {rr:>18s}  {evc[cstar]:+6.2f}/{evc[0]:+6.2f}  | {evh:+6.2f}->{evs:+6.2f}  | {evh25:+6.2f}->{evs25:+6.2f}")
        for k, lab in [("test_hold_EV", "hold"), ("test_cstar_EV", "c*")]:
            v = [res["per_symbol"][s][k] for s in res["per_symbol"] if np.isfinite(res["per_symbol"][s][k])]
            log(f"  POOLED test {lab}: {np.mean(v):+.2f}bp" if v else f"  {lab}: n/a")
        tag = "B2GRID_RESULT_optb" if a.optuna_b else "B2GRID_RESULT"
        bk.blob(f"{RR}/{tag}.json").upload_from_string(json.dumps(res, default=float))
        log(f"[saved] gs://{BUCKET}/{RR}/{tag}.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
