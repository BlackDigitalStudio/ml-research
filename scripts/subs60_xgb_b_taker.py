"""Model B for TAKER entry (catch the runaways the maker misses) — GRU 2-stage pipeline in XGBoost.

Pipeline (canonical, per user): Stage-1 B (direction on taker-hold) -> grid_sim TAKER over TP/SL/HOLD on
VAL (model-sided, daily-budget) -> c* per symbol -> Stage-2 B fine-tune with the EXECUTED-PAYOFF objective
L=-E[sigma(z)*PL+(1-sigma)*PS] on c*'s TAKER payoffs -> eval test: A^B 1/day, taker entry, net taker fee.

Taker entry = cross at t0 (immediate fill), so NO adverse selection on entry and NO miss: B is free to
target the favorable side of the big move. A (vol) reused; taker-hold labels from research_runs/taker_labels.
Reuses maker_paths + the recompiled grid_sim (taker mode = no --flow-paths). Saves all weights+preds.
"""
import argparse, io, json, os, subprocess, tempfile, shutil
import numpy as np, xgboost as xgb
from google.cloud import storage

PROJ = "project-0998ac51-36ba-445c-bc7"; BUCKET = "market-data-0998ac51"
RR = "research_runs/maker_labels_rr"; TK = "research_runs/taker_labels"; MAIN = "research_runs/xgb_maker"
BU = "research_runs/b_universe"; PATHS = "research_runs/maker_paths"; SAVE = "research_runs/b_taker"
FEATS = "feats_sub60"; GRID = "/tmp/gridbuild/release/grid_sim"
SPLIT = (0.65, 0.68, 0.85); NF_RATE = 0.05; TO_TICKS = 563
PA_T = ["entry_long", "entry_short", "mid_paths", "book_paths", "entry_book"]   # taker needs these 5
bk = storage.Client(project=PROJ).bucket(BUCKET)


def jload(p): return json.loads(bk.blob(p).download_as_bytes())


def load_booster(path):
    b = xgb.Booster()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    bk.blob(path).download_to_filename(tmp); b.load_model(tmp); os.remove(tmp); return b


def save_booster(b, name):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    b.save_model(tmp); bk.blob(f"{SAVE}/{name}").upload_from_filename(tmp); os.remove(tmp)


def load_data(symk):
    d = np.load(io.BytesIO(bk.blob(f"{RR}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    tkd = np.load(io.BytesIO(bk.blob(f"{TK}/{symk}.npz").download_as_bytes()), allow_pickle=True)
    # align taker-hold pnl to maker_labels_rr windows by ts
    pos = {int(t): i for i, t in enumerate(tkd["ts"].astype(np.int64))}
    ts = d["ts"].astype(np.int64)
    idx = np.array([pos.get(int(t), -1) for t in ts])
    ok = idx >= 0
    tkL = np.full(len(ts), np.nan); tkS = np.full(len(ts), np.nan)
    tkL[ok] = tkd["pnl_long"][idx[ok]]; tkS[ok] = tkd["pnl_short"][idx[ok]]   # bp gross (taker hold)
    return {"F": d["F"].astype(np.float32), "rH": d["rH60"].astype(np.float64), "day": d["day"],
            "ts": ts, "tkL": tkL, "tkS": tkS, "ndays": m["n_days"], "tk_ok": ok}


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


def make_grid(n_tp, n_sl):
    tps = np.linspace(0.05, 1.5, n_tp); sls = np.linspace(0.03, 0.40, n_sl)
    cfgs = [{"tp": 50.0, "sl": 50.0, "to": TO_TICKS, "par": False, "tr": False}]   # hold = idx 0
    for sl in sls:
        for tp in tps:
            if 0.5 <= tp / sl <= 30.0:
                cfgs.append({"tp": round(float(tp), 4), "sl": round(float(sl), 4), "to": TO_TICKS, "par": False, "tr": False})
    return cfgs


def load_paths(symk, dates, want_ts):
    want = set(int(t) for t in np.asarray(want_ts).tolist())
    acc = {k: [] for k in PA_T}; tsacc = []
    for dt in dates:
        try:
            d = np.load(io.BytesIO(bk.blob(f"{PATHS}/{symk}/{dt}.npz").download_as_bytes()))
        except Exception:
            continue
        dts = d["ts"].astype(np.int64); m = np.fromiter((int(t) in want for t in dts), bool, len(dts))
        if not m.any():
            continue
        tsacc.append(dts[m])
        for k in PA_T:
            acc[k].append(d[k][m])
    if not tsacc:
        return None
    gts = np.concatenate(tsacc); arrs = {k: np.concatenate(acc[k], 0) for k in PA_T}
    pos = {int(t): i for i, t in enumerate(gts)}
    idx = np.array([pos[int(t)] for t in want_ts if int(t) in pos])
    keep = np.array([int(t) in pos for t in want_ts])
    return {k: arrs[k][idx] for k in PA_T}, keep


def run_grid_taker(tmp, arrs, cfgs):
    """grid_sim TAKER (no --flow-paths => immediate cross-fill at entry_long/short at t0). Returns pl,ps (NC,n) %."""
    for k in PA_T:
        np.save(f"{tmp}/{k}.npy", arrs[k].astype(np.float64))
    json.dump(cfgs, open(f"{tmp}/cfg.json", "w"))
    cmd = [GRID, "--entry-long", f"{tmp}/entry_long.npy", "--entry-short", f"{tmp}/entry_short.npy",
           "--mid-paths", f"{tmp}/mid_paths.npy", "--book-paths", f"{tmp}/book_paths.npy",
           "--entry-book", f"{tmp}/entry_book.npy", "--configs", f"{tmp}/cfg.json", "--out-prefix", f"{tmp}/g",
           "--commission-win-pct", "0", "--commission-loss-pct", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"grid_sim taker fail: {r.stderr[-300:]}")
    pl = np.load(f"{tmp}/g_pnl_long.npy"); ps = np.load(f"{tmp}/g_pnl_short.npy")
    for f in os.listdir(tmp):
        if f.endswith(".npy"):
            os.remove(f"{tmp}/{f}")
    return pl * 100.0, ps * 100.0    # bp gross


def fit_logistic(F, y, w, hp, nr):
    p = {"objective": "binary:logistic", "tree_method": "hist", "nthread": 8, "seed": 0}
    return xgb.train(dict(p, **hp), xgb.DMatrix(F, label=y, weight=w), num_boost_round=max(1, nr))


def fit_payoff(F, PL, PS, hp, nr):
    """Stage-2: maximize E[sigma(z)*PL + (1-sigma)*PS]  (executed-payoff, GRU B2). PL/PS gross bp."""
    diff = (PL - PS).astype(np.float64)
    def obj(preds, dtrain):
        s = 1.0 / (1.0 + np.exp(-preds))
        return -s * (1 - s) * diff, s * (1 - s) * np.abs(diff) + 1e-6
    p = {"tree_method": "hist", "nthread": 8, "seed": 0, "base_score": 0.0,
         "max_depth": hp.get("max_depth", 5), "eta": hp.get("learning_rate", hp.get("eta", 0.05)),
         "subsample": hp.get("subsample", 0.8), "colsample_bytree": hp.get("colsample_bytree", 0.7),
         "min_child_weight": hp.get("min_child_weight", 20), "reg_lambda": hp.get("reg_lambda", 1.0)}
    return xgb.train(p, xgb.DMatrix(F, label=np.zeros(len(F))), num_boost_round=max(1, nr), obj=obj)


def fit_ic(F, rHt, hp, nr):
    """Stage-1: REGRESS rH -> predict the MOVE (sign=direction, |.|=magnitude). XGBoost-native realization of
    the GRU IC objective's intent (a true global-correlation custom-obj underfits XGBoost's greedy trees).
    Predicting the move (not the binary better-side) is what lets a high-TP/SL config win on big moves."""
    r = rHt.astype(np.float64)
    lo, hi = np.quantile(r, 0.005), np.quantile(r, 0.995); r = np.clip(r, lo, hi)   # robust to fat tails
    p = {"objective": "reg:squarederror", "tree_method": "hist", "nthread": 8, "seed": 0, **hp}
    return xgb.train(p, xgb.DMatrix(F, label=r), num_boost_round=max(1, int(nr)))


def auc(score, lab):
    lab = np.asarray(lab).astype(int); o = np.argsort(score); rk = np.empty(len(score)); rk[o] = np.arange(len(score))
    n1 = int(lab.sum()); n0 = len(lab) - n1
    return float((rk[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 > 20 and n0 > 20 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", nargs="+", default=["BTC", "LINK"])
    ap.add_argument("--n-tp", type=int, default=25); ap.add_argument("--n-sl", type=int, default=25)
    ap.add_argument("--budget", type=int, default=1); ap.add_argument("--gate-pct", type=float, default=5.0)
    ap.add_argument("--taker-fee", type=float, default=7.0)
    ap.add_argument("--cstar-k", type=int, default=3000)   # c* selected on top-K conviction val windows (stable for low-WR high-RR)
    a = ap.parse_args()
    def log(s): print(s, flush=True)
    hpA_g = None; hpB = jload(f"{MAIN}/B_pool.json")["best_params"]
    cfgs = make_grid(a.n_tp, a.n_sl); NC = len(cfgs)
    log(f"[B-taker] GRU pipeline | {NC} cfgs | budget={a.budget}/day | taker_fee={a.taker_fee}bp | gate={a.gate_pct}%")
    log(f"{'SYM':5s} {'c*':>20s} {'B_dirAUC':>8s} | {'maker+3.00':>10s} {'taker_net':>9s} {'taker_gross':>11s}")
    res = {}; tmp = tempfile.mkdtemp(prefix="bt_", dir="/dev/shm")
    try:
        for symk in a.symbols:
            d = load_data(symk); hpA = jload(f"{MAIN}/A_{symk}.json")
            F = d["F"]; rH = d["rH"]; day = d["day"]; ts = d["ts"]; tkL = d["tkL"]; tkS = d["tkS"]
            trn, val, te = split(day, d["ndays"]); vi = np.where(val)[0]; ti = np.where(te)[0]
            thr = float(np.quantile(np.abs(rH[trn]), 1 - NF_RATE)); yA = (np.abs(rH) >= thr).astype(int)
            dates = sym_dates(symk + "-USDT-PERP")
            # A: reuse saved; OOF gate (apred 5%)
            bstA = load_booster(f"{BU}/A_{symk}.xgb.json"); pA_t = bstA.predict(xgb.DMatrix(F[ti])); pA_v = bstA.predict(xgb.DMatrix(F[vi]))
            from_oof = None
            tdays = sorted(set(day[trn].tolist())); fold = {dd: i % 5 for i, dd in enumerate(tdays)}
            fday = np.array([fold.get(int(dd), -1) for dd in day]); oof = np.full(len(F), np.nan)
            for k in range(5):
                trk = trn & (fday != k); vak = trn & (fday == k)
                if vak.sum() < 50 or (yA[trk] == 1).sum() < 20:
                    continue
                spw = float((yA[trk] == 0).sum() / max((yA[trk] == 1).sum(), 1))
                bk_ = fit_logistic(F[trk], yA[trk], None, dict(hpA["best_params"], scale_pos_weight=spw), hpA["best_iter"] + 1)
                oof[np.where(vak)[0]] = bk_.predict(xgb.DMatrix(F[vak]))
            valid = trn & np.isfinite(oof) & d["tk_ok"]
            thrG = float(np.nanquantile(oof[valid], 1 - a.gate_pct / 100.0)); gate = valid & (oof >= thrG)
            # STAGE-1 B: IC objective on rH (GRU formula) -> predicts the MOVE (sign=dir, |.|=magnitude)
            b1 = fit_ic(F[gate], rH[gate], hpB, jload(f"{MAIN}/B_pool.json")["best_iter"] + 1)
            pB1_v = b1.predict(xgb.DMatrix(F[vi])); pB1_t = b1.predict(xgb.DMatrix(F[ti]))
            okt = np.isfinite(rH[ti]); b1auc = auc(pB1_t[okt], (rH[ti] > 0).astype(int)[okt])
            # c* SELECTION: on the VAL A-5% pool's TOP-K most-confident windows (captured-alpha-at-conviction,
            # like GRU) -> stable estimate for low-WR high-RR configs (mean over ~115 1/day windows was too noisy).
            amask_v = pA_v >= thrG; pool_v = vi[amask_v]; confv = np.abs(pB1_v[amask_v])
            order_v = np.argsort(-confv)[:min(len(pool_v), a.cstar_k)]
            gi = pool_v[order_v]; side_v = pB1_v[amask_v][order_v] > 0
            vdates = [dates[i] for i in sorted(set(day[gi].tolist())) if i < len(dates)]
            lp = load_paths(symk, vdates, ts[gi])
            if lp is None:
                log(f"{symk}: val paths missing"); continue
            arrs, kmask = lp; side_vk = side_v[kmask]
            plg, psg = run_grid_taker(tmp, arrs, cfgs)             # (NC, n) taker gross
            evc = np.full(NC, -1e9)
            for c in range(NC):
                net = np.where(side_vk, plg[c], psg[c]) - a.taker_fee
                evc[c] = net.mean() if len(net) >= 50 else -1e9
            cstar = int(np.argmax(evc)); cfg_star = cfgs[cstar]
            rr = "hold" if cfg_star["tp"] >= 1 else f"RR{cfg_star['tp']/cfg_star['sl']:.1f}({cfg_star['tp']}/{cfg_star['sl']})"
            # STAGE-2 B: executed-payoff on c*'s TAKER payoffs over gate windows
            ggi = np.where(gate)[0]; gdates = [dates[i] for i in sorted(set(day[gate].tolist())) if i < len(dates)]
            lpt = load_paths(symk, gdates, ts[ggi])
            if lpt is None:
                log(f"{symk}: train paths missing"); continue
            tarrs, tkeep = lpt; plc, psc = run_grid_taker(tmp, tarrs, [cfgs[0], cfg_star])   # idx1 = c*
            gk = ggi[tkeep]
            b2 = fit_payoff(F[gk], plc[1], psc[1], hpB, 200)
            pB2_t = b2.predict(xgb.DMatrix(F[ti]))
            # EVAL test CASCADE: (1) hard A-top-5% pool by pA -> (2) within it, top-N/day by B2-confidence -> taker c*
            BUDS = [1, 2, 5, 10]
            amask_t = pA_t >= thrG                               # A filter: top-5% test windows by pA (same gate threshold)
            def taker_ev(pB):
                pidx = ti[amask_t]; side = pB[amask_t] > 0; conf = np.abs(pB[amask_t])   # Stage-2 margin (centered 0)
                pdates = [dates[i] for i in sorted(set(day[pidx].tolist())) if i < len(dates)]
                lpe = load_paths(symk, pdates, ts[pidx])
                if lpe is None:
                    return {b: (float("nan"), float("nan"), 0) for b in BUDS}
                earrs, ek = lpe; side = side[ek]; conf = conf[ek]; dpool = day[pidx][ek]
                pe, qe = run_grid_taker(tmp, earrs, [cfg_star])
                gross = np.where(side, pe[0], qe[0]); out = {}
                for b in BUDS:                                   # B filter: top-b/day by B2-confidence WITHIN the A-pool
                    sel = daily_pick(dpool, conf, b)
                    g = float(gross[sel].mean()); out[b] = (g - a.taker_fee, g, int(len(sel)))
                return out
            sweep = taker_ev(pB2_t)
            dir_auc = auc(pB2_t[np.isfinite(rH[ti])], (rH[ti] > 0).astype(int)[np.isfinite(rH[ti])])  # B2 vs RAW direction
            mk = {"BNB": -2.46, "BTC": 2.81, "DOGE": 5.73, "ETH": 4.53, "LINK": 8.48, "LTC": 1.49, "SOL": 1.11, "XRP": 2.31}.get(symk, float("nan"))
            res[symk] = {"c_star": rr, "b1_dirAUC": b1auc, "b2_dirAUC": dir_auc, "maker_3p00": mk,
                         "net": {str(b): sweep[b][0] for b in BUDS}, "gross": {str(b): sweep[b][1] for b in BUDS},
                         "n": {str(b): sweep[b][2] for b in BUDS}, "val_ev_cstar": float(evc[cstar])}
            log(f"{symk:5s} c*={rr:>18s} S1auc{b1auc:.3f} S2auc{dir_auc:.3f} mk{mk:+.1f} | net@1/2/5/10d: " +
                "/".join(f"{sweep[b][0]:+.1f}" for b in BUDS) + " | gross@1d=" + f"{sweep[1][1]:+.1f}(n{sweep[1][2]})")
            save_booster(b1, f"B1_{symk}.xgb.json"); save_booster(b2, f"B2_{symk}.xgb.json")
            buf = io.BytesIO(); np.savez_compressed(buf, ti=ti.astype(np.int64), pA=pA_t.astype(np.float32),
                pB1=pB1_t.astype(np.float32), pB2=pB2_t.astype(np.float32), rH=rH[ti].astype(np.float32), day=day[ti].astype(np.int32),
                meta=np.array(json.dumps({"c_star": rr, "net": {str(b): sweep[b][0] for b in BUDS},
                    "gross": {str(b): sweep[b][1] for b in BUDS}, "taker_fee": a.taker_fee})))
            bk.blob(f"{SAVE}/preds_{symk}.npz").upload_from_string(buf.getvalue())
        log("--- POOLED (maker+3.00 vs taker: A-5% pool -> top-N/day by B-confidence) ---")
        mkv = [res[s]["maker_3p00"] for s in a.symbols if s in res and np.isfinite(res[s]["maker_3p00"])]
        if mkv:
            log(f"  maker baseline (1/day) = {np.mean(mkv):+.2f}")
        for b in [1, 2, 5, 10]:
            netv = [res[s]["net"][str(b)] for s in a.symbols if s in res and np.isfinite(res[s]["net"][str(b)])]
            grv = [res[s]["gross"][str(b)] for s in a.symbols if s in res and np.isfinite(res[s]["gross"][str(b)])]
            if netv:
                log(f"  {b:2d}/day: taker_net {np.mean(netv):+.2f} ({sum(1 for x in netv if x>0)}/{len(netv)}+) | taker_gross {np.mean(grv):+.2f}")
        bk.blob(f"{SAVE}/B_TAKER_RESULT.json").upload_from_string(json.dumps(res, default=float))
        log(f"[saved] gs://{BUCKET}/{SAVE}/B_TAKER_RESULT.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
