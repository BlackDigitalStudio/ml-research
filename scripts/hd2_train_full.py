#!/usr/bin/env python3
"""HD2 full-tier trainer: multi-day, multitask-H, streaming-stateful.

One training unit = (symbol, L, seed); it trains ONE Mamba-2 with 3 horizon
heads (multitask H in {180,600,1800}) over the symbol's 500-day stream, so the
shared LOB forward is computed once per epoch instead of 3x. Memory-bounded:
per-day metadata (t0/globals/labels) is held in RAM (~tens of MB); the big
fp16 stream is loaded per-day on demand and freed.

Days are independent reset segments (overnight gap => no cross-day state carry);
within a day the state resets every L ticks (the context knob). The global
honest 70/30 split + train/val + R1 weights + metrics are the FROZEN
hd1_seq_core. Readout fires at STRIDE=1 decision points past the warmup floor.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hd1_seq_core as core  # noqa: E402
import hd2_targets as TGT  # noqa: E402
import hd2_losses as LOSS  # noqa: E402
from hd2_mamba_stream import (HD2Mamba2, _periods, WARMUP, DEEP_CTX_FRAC,  # noqa
                              _rank_ic)

core.STRIDE = 1
HS = (180, 600, 1800)


@dataclass
class FullCfg:
    symbol: str = "SOL-USDT-PERP"
    L: int = 6000
    seed: int = 0
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 128
    batch_periods: int = 16
    epochs: int = 15
    patience: int = 3
    lr: float = 1e-3
    wd: float = 1e-3
    dropout: float = 0.1
    grad_ckpt_min_L: int = 20000
    ckpt_path: str = ""
    Hs: tuple = HS
    target_spec: dict = None     # None -> cached +-0.13 FP labels (rev2 default)
    target_name: str = "fp_0.13"  # for logging/ledger when target_spec is set
    loss_spec: dict = None       # None -> R1 (BCE x |move|); see hd2_losses (rev7)
    loss_name: str = "R1"
    eval_only: bool = False      # load ckpt, skip training, just re-score OOS metrics
    dt_min: float = 0.001        # SSM timescale range (round 4 sweep); mamba defaults
    dt_max: float = 0.1
    a_init_low: float = 1.0      # A_init_range = (a_init_low, a_init_high); mamba default (1,16)
    a_init_high: float = 16.0
    d_conv: int = 4              # local causal-conv width (round 4b sweep); mamba default 4
    pooled: bool = False         # data-scale: train on train_symbols pool, eval on eval_symbol
    train_symbols: tuple = ()    # pooled train set (e.g. all 8); empty -> single-symbol
    eval_symbol: str = "LTC-USDT-PERP"   # pooled eval (user: validate on LTC)
    split_date: str = "2025-12-10"       # global temporal cutoff: train < date, eval >= date
    train_day_stride: int = 1            # COMPUTE side-task: train on every Nth fit-day (eval full)
    dump_rl: bool = False                # RL side-experiment: dump 10s (mid,logits) OOS series
    profit3: bool = False                # LEGACY REPRO: 3-class UP/DN/FL profitability target + prec_NF
    p3_H: int = 130                      # profit3 forward window (seconds) — old SIM_HORIZON ~130s
    p3_tp: float = 0.20                  # profit3 TP %% (old vol-aware median)
    p3_sl: float = 0.10                  # profit3 SL %% (R:R 2:1)


def _day_paths(cache_dir, symbol):
    d = Path(cache_dir) / "hd2" / symbol
    return sorted(str(p) for p in d.glob("*.npz"))


def load_meta(day_paths, Hs, target_spec=None):
    """Per-day metadata (NO stream) + global concatenated decision-point arrays
    in chronological order. day[i] = {path, n_ticks, n, g_lo, g_hi}.

    target_spec=None -> cached +-0.13 first-passage labels (rev2 default).
    target_spec set  -> relabel at train time from /cache/midts/{sym}/{day}.npz
    (per-tick mid+ts) via hd2_targets.make_target (rev5 barrier/target sweep)."""
    days, g_t0, g_glob, g_day = [], [], [], []
    g_y = {H: [] for H in Hs}; g_rH = {H: [] for H in Hs}; g_rc = {H: [] for H in Hs}
    off = 0
    for di, p in enumerate(day_paths):
        d = np.load(p, allow_pickle=True)
        t0 = d["t0"].astype(np.int64)
        n = len(t0)
        meta = json.loads(str(d["meta"]))
        days.append({"path": p, "n_ticks": int(meta["n_ticks"]), "n": n,
                     "g_lo": off, "g_hi": off + n,
                     "date": Path(p).stem, "symbol": Path(p).parent.name})
        g_t0.append(t0); g_glob.append(d["globals"].astype(np.float32))
        g_day.append(np.full(n, di, np.int64))
        if target_spec is None:
            for H in Hs:
                g_y[H].append(d[f"y_{H}"].astype(np.int64))
                g_rH[H].append(d[f"rH_{H}"].astype(np.float32))
                g_rc[H].append(d[f"rc_{H}"].astype(bool))
        elif target_spec.get("form") == "terminal":
            # FAST terminal: the cached rH_{H} (labels_for_H: log(mid[jH]/m0)) IS
            # the terminal forward return == make_target terminal rH. So derive
            # y=sign(rH), rc=finite directly -> skip per-day midts read + make_target
            # (pooled meta-load ~1h40 -> minutes; bit-identical to the relabel path).
            for H in Hs:
                rHv = d[f"rH_{H}"].astype(np.float32)
                g_rH[H].append(rHv)
                g_y[H].append(np.sign(rHv).astype(np.int64))
                g_rc[H].append(np.isfinite(rHv) & (rHv != 0.0))
        else:
            pp = Path(p)
            mp = pp.parent.parent.parent / "midts" / pp.parent.name / pp.name
            md = np.load(str(mp))
            mid = md["mid"].astype(np.float64); ts = md["ts"].astype(np.int64)
            md.close()
            for H in Hs:
                y, rH, rc, _u = TGT.make_target(mid, ts, t0, H, target_spec)
                g_y[H].append(y.astype(np.int64))
                g_rH[H].append(rH.astype(np.float32))
                g_rc[H].append(rc)
        d.close()
        off += n
    G = {"t0": np.concatenate(g_t0), "globals": np.concatenate(g_glob),
         "day": np.concatenate(g_day), "n": off,
         "y": {H: np.concatenate(g_y[H]) for H in Hs},
         "rH": {H: np.concatenate(g_rH[H]) for H in Hs},
         "rc": {H: np.concatenate(g_rc[H]) for H in Hs}}
    return days, G


def load_profit3_labels(days, G, cache_dir, p3_H, p3_tp, p3_sl):
    """LEGACY REPRO: per-day 3-class profitability labels (0=UP,1=DN,2=FL)
    aligned with G['t0']. Reads /cache/midts/{sym}/{date}.npz and calls
    TGT.make_target_profit3 (TP/SL first-passage over p3_H seconds, net of taker
    commissions). FL is a TRAINED class (no mask) — unlike the binary fp path."""
    y3 = np.full(G["n"], 2, np.int8)
    for dd in days:
        mp = f"{cache_dir}/midts/{dd['symbol']}/{dd['date']}.npz"
        md = np.load(mp)
        mid = md["mid"].astype(np.float64); ts = md["ts"].astype(np.int64)
        md.close()
        lo, hi = dd["g_lo"], dd["g_hi"]
        y3[lo:hi] = TGT.make_target_profit3(mid, ts, G["t0"][lo:hi], p3_H, p3_tp, p3_sl)
    return y3


def lob_stats(day_paths, fit_day_ids):
    """Per-channel mean/std over FIT-day stream ticks (one pass, streamed)."""
    s = np.zeros(80, np.float64); ss = np.zeros(80, np.float64); cnt = 0
    for di in fit_day_ids:
        st = np.load(day_paths[di])["stream"].astype(np.float32)
        s += st.sum(0); ss += (st.astype(np.float64) ** 2).sum(0); cnt += st.shape[0]
    mu = (s / cnt).astype(np.float32)
    var = np.maximum(ss / cnt - (s / cnt) ** 2, 1e-12)
    return mu, np.sqrt(var).astype(np.float32)


def train_cell(cache_dir, cfg: FullCfg, log=print):
    import torch
    import torch.nn as nn
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Hs = cfg.Hs
    if cfg.pooled:
        # DATA-SCALE: pool all train_symbols; GLOBAL TEMPORAL split (no leak) ->
        # train = all symbols' days BEFORE split_date; eval = eval_symbol's days
        # ON/AFTER split_date. Days are independent reset segments and H<<1day, so
        # a date-level holdout has no intraday leakage. Stream is mid-relative +
        # log-size => scale-invariant across symbols, so pooled norm is valid.
        day_paths = []
        for s in cfg.train_symbols:
            day_paths += _day_paths(cache_dir, s)
        assert day_paths, f"no pooled cache {cfg.train_symbols} in {cache_dir}"
        days, G = load_meta(day_paths, Hs, cfg.target_spec)
        n = G["n"]
        day_date = np.array([d["date"] for d in days])
        day_sym = np.array([d["symbol"] for d in days])
        dp_date = day_date[G["day"]]; dp_sym = day_sym[G["day"]]
        before = dp_date < cfg.split_date
        tr = before
        te = (~before) & (dp_sym == cfg.eval_symbol)
        fit, val = core.train_val_split(tr)
        n_tr = int(tr.sum())
        log(f"[pooled] n={n} train={int(tr.sum())} eval({cfg.eval_symbol}>={cfg.split_date})"
            f"={int(te.sum())} syms={cfg.train_symbols}")
    else:
        day_paths = _day_paths(cache_dir, cfg.symbol)
        assert day_paths, f"no cache for {cfg.symbol} in {cache_dir}"
        days, G = load_meta(day_paths, Hs, cfg.target_spec)
        n = G["n"]
        tr, te, n_tr = core.honest_split(n)
        fit, val = core.train_val_split(tr)
    fit_day_ids = sorted(set(G["day"][fit].tolist()))
    if cfg.train_day_stride > 1:                       # COMPUTE side-task: every Nth fit-day
        keep = sorted(fit_day_ids[::cfg.train_day_stride])
        fit = fit & np.isin(G["day"], np.array(keep))
        fit_day_ids = keep
        log(f"[day-stride {cfg.train_day_stride}] train days {len(fit_day_ids)} "
            f"fit_dp {int(fit.sum())}")
    test_day_ids = sorted(set(G["day"][te].tolist()))

    s_mu, s_sd = lob_stats(day_paths, fit_day_ids)
    g_mu, g_sd = (G["globals"][fit].mean(0), G["globals"][fit].std(0))
    g_sd = np.where(g_sd > 1e-8, g_sd, 1.0)
    glob_std = (G["globals"] - g_mu) / g_sd

    # R1 weights + up labels per H (global)
    if cfg.profit3:
        G["y3"] = load_profit3_labels(days, G, cache_dir, cfg.p3_H, cfg.p3_tp, cfg.p3_sl)
        _cnt = np.array([(G["y3"][fit] == c).sum() for c in (0, 1, 2)], np.float64)
        _cw = 1.0 / np.sqrt(np.maximum(_cnt, 1.0)); _cw = _cw / _cw.sum() * 3.0
        cls_w_t = torch.tensor(_cw, dtype=torch.float32, device=dev)
        log(f"[profit3] H={cfg.p3_H}s TP{cfg.p3_tp}/SL{cfg.p3_sl} fit base "
            f"UP/DN/FL={(G['y3'][fit]==0).mean()*100:.1f}/{(G['y3'][fit]==1).mean()*100:.1f}/"
            f"{(G['y3'][fit]==2).mean()*100:.1f}%  cls_w={_cw.round(3).tolist()}")
        w1 = up = None
    else:
        w1 = {H: core.r1_weights(G["rH"][H], fit & G["rc"][H]) for H in Hs}
        up = {H: (G["y"][H] == 1).astype(np.float32) for H in Hs}

    # warmup-floor / deep-context per global decision point
    warm = min(cfg.L, WARMUP)
    ctx = G["t0"] - (G["t0"] // cfg.L) * cfg.L
    scored = ctx >= warm
    deep = ctx >= int(DEEP_CTX_FRAC * cfg.L)

    model = HD2Mamba2(cell_kind="mamba2", d_model=cfg.d_model,
                      n_layers=cfg.n_layers, d_state=cfg.d_state,
                      dropout=cfg.dropout, n_out=(3 if cfg.profit3 else len(Hs)),
                      dt_min=cfg.dt_min, dt_max=cfg.dt_max,
                      A_init_range=(cfg.a_init_low, cfg.a_init_high),
                      d_conv=cfg.d_conv).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    use_ckpt = cfg.L >= cfg.grad_ckpt_min_L
    H_idx = {H: k for k, H in enumerate(Hs)}

    start_ep, best_val, best_state, since = 0, -1e9, None, 0
    epoch_hist = []   # per-epoch learning curve (train_loss/val_ricsum) -> persisted in out
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        sd = torch.load(cfg.ckpt_path, map_location=dev)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        start_ep = sd["epoch"] + 1; best_val = sd["best_val"]; since = sd["since"]
        torch.set_rng_state(sd["rng"].cpu())
        log(f"[resume] epoch {start_ep} best_val={best_val:.5f}")
    if cfg.eval_only:
        assert os.path.exists(cfg.ckpt_path), "eval_only requires an existing ckpt"
        log("[eval-only] re-scoring OOS metrics from ckpt (no training)")

    def _save(ep):
        if not cfg.ckpt_path:
            return
        tmp = cfg.ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best_val": best_val, "since": since,
                    "rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, cfg.ckpt_path)

    def _day_stream(di):
        st = np.load(day_paths[di])["stream"].astype(np.float32)
        return (st - s_mu) / s_sd

    def _day_periods(di, mask):
        """reset-period batches for day di restricted to global mask."""
        lo, hi = days[di]["g_lo"], days[di]["g_hi"]
        loc = np.where(mask[lo:hi] & scored[lo:hi])[0]   # local dp indices
        if len(loc) == 0:
            return []
        t0loc = G["t0"][lo:hi]
        per = (t0loc[loc] // cfg.L)
        groups = {}
        for li, p in zip(loc, per):
            groups.setdefault(int(p), []).append(li)
        nt = days[di]["n_ticks"]
        out = []
        for p, lis in groups.items():
            a, b = p * cfg.L, min((p + 1) * cfg.L, nt)
            out.append((a, b, np.asarray(lis), lo))   # lo to map local->global
        return out

    def _forward(stream, batch, dev):
        # batch: list of (a,b,local_dp_idx,lo) all same length (==L) or solo tail
        T = batch[0][1] - batch[0][0]
        same = all((b - a) == T for (a, b, _, _) in batch)
        if not same:
            batch = [batch_i for grp in [[x] for x in batch] for batch_i in grp]
        xs = np.stack([stream[a:b] for (a, b, _, _) in batch])
        x = torch.from_numpy(xs).to(dev)
        hidden = model.encode_period(x, use_ckpt=use_ckpt)   # (k,T,d)
        rows, poss, gdp = [], [], []
        for j, (a, b, lis, lo) in enumerate(batch):
            rows.append(np.full(len(lis), j, np.int64))
            poss.append((G["t0"][lo + lis] - a).astype(np.int64))
            gdp.append(lo + lis)
        rows = np.concatenate(rows); poss = np.concatenate(poss)
        gdp = np.concatenate(gdp)
        hsel = hidden[torch.from_numpy(rows).to(dev), torch.from_numpy(poss).to(dev)]
        g = torch.from_numpy(glob_std[gdp]).to(dev)
        return model.readout(hsel, g), gdp        # logits (k, n_H), gdp global idx

    def _chunk(seq, k):
        for i in range(0, len(seq), k):
            yield seq[i:i + k]

    def _batches(di, mask):
        """Same-length period groups for day di: full (==L) periods packed
        cfg.batch_periods-wide; the short tail period (if any) solo. Used by
        BOTH train and eval so np.stack never mixes lengths."""
        pers = _day_periods(di, mask)
        full = [p for p in pers if (p[1] - p[0]) == cfg.L]
        rest = [p for p in pers if (p[1] - p[0]) != cfg.L]
        return list(_chunk(full, cfg.batch_periods)) + [[r] for r in rest]

    t_start = time.time()
    for ep in ([] if cfg.eval_only else range(start_ep, cfg.epochs)):
        model.train()
        rng = np.random.default_rng(cfg.seed * 1000 + ep)
        order = rng.permutation(fit_day_ids)
        ep_loss, ep_n, n_skip, dbg = 0.0, 0, 0, (ep == start_ep)
        for di in order:
            stream = _day_stream(di)
            for grp in _batches(di, fit):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, gdp = _forward(stream, grp, dev)
                    if dbg:
                        log(f"[dbg] first-batch gdp={len(gdp)} "
                            f"rc_sums={ {H: int(G['rc'][H][gdp].sum()) for H in Hs} }")
                        dbg = False
                    if cfg.profit3:
                        tg3 = torch.from_numpy(G["y3"][gdp].astype(np.int64)).to(dev)
                        loss = nn.functional.cross_entropy(
                            logits.float(), tg3, weight=cls_w_t)
                    else:
                        terms = []
                        for H in Hs:
                            rc = G["rc"][H][gdp]
                            if rc.sum() == 0:
                                continue
                            sel = torch.from_numpy(rc).to(dev)
                            lg = logits[:, H_idx[H]].float()[sel]
                            tg = torch.from_numpy(up[H][gdp][rc]).to(dev)
                            wv = torch.from_numpy(w1[H][gdp][rc].astype(np.float32)).to(dev)
                            rHv = torch.from_numpy(
                                G["rH"][H][gdp][rc].astype(np.float32)).to(dev)
                            terms.append(LOSS.compute_loss(lg, tg, rHv, wv, cfg.loss_spec))
                        loss = sum(terms) if terms else None
                if loss is None:
                    n_skip += 1
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss); ep_n += 1
            del stream

        # ---- validation metric (sum-rank_IC, or prec_NF for profit3) ----
        if cfg.profit3:
            vscore = _eval_prec_nf(model, G, val & scored, cfg, dev, _day_stream,
                                   _batches, _forward)["prec_nf"]
        else:
            vscore = _eval_score(model, G, val & scored, cfg, dev, _day_stream,
                                 _batches, _forward, H_idx)
        improved = vscore > best_val + 1e-5
        if improved:
            best_val = vscore; since = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        _save(ep)
        log(f"[ep {ep}] loss={ep_loss/max(1,ep_n):.4f} n_batch={ep_n} "
            f"n_skip={n_skip} val_ricsum={vscore:.5f} best={best_val:.5f} "
            f"since={since} elapsed={time.time()-t_start:.0f}s")
        epoch_hist.append({"ep": int(ep), "train_loss": ep_loss / max(1, ep_n),
                           "val_ricsum": float(vscore), "best_val": float(best_val),
                           "since": int(since)})
        if since >= cfg.patience:
            log("[early-stop]"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    if cfg.profit3:
        m = _eval_prec_nf(model, G, te & scored, cfg, dev, _day_stream,
                          _batches, _forward)
        out = {"symbol": cfg.symbol, "L": cfg.L, "seed": cfg.seed, "n_dp": int(n),
               "n_tr": int(n_tr), "profit3": True, "p3_H": cfg.p3_H,
               "p3_tp": cfg.p3_tp, "p3_sl": cfg.p3_sl,
               "best_val_prec_nf": float(best_val), "elapsed_s": time.time() - t_start,
               "epoch_hist": epoch_hist, **m}
        log(f"[profit3 RESULT] prec_NF={m['prec_nf']:.4f} vs base "
            f"UP/DN={m['base_up']:.3f}/{m['base_dn']:.3f} "
            f"coverage={m['coverage']:.3f} n_nf={m['n_nf']} n_eval={m['n_eval']}")
        return out, None
    # ---- final OOS metrics per H (all + deep) + persist preds ----
    preds = _eval_preds(model, G, te & scored, cfg, dev, _day_stream,
                        _batches, _forward)
    if cfg.dump_rl:
        # RL side-experiment: build a 10s-grid OOS series (mid + forward-filled
        # Mamba logits) per eval day. midts is tick-aligned with the stream (same
        # parquet), so ts_t[t0]=decision-point time. Frictionless price path for RL.
        ts_all = []; mid_all = []; log_all = []; day_all = []
        Hs_l = list(Hs)
        for di in sorted(set(G["day"][te].tolist())):
            md = np.load(f"{cache_dir}/midts/{days[di]['symbol']}/{days[di]['date']}.npz")
            ts_t = md["ts"].astype(np.int64); mid_t = md["mid"].astype(np.float64)
            dpg = np.where((G["day"] == di) & te & scored)[0]   # scored => valid logits
            if len(dpg) == 0:
                continue
            t0t = np.clip(G["t0"][dpg], 0, len(ts_t) - 1)
            order = np.argsort(ts_t[t0t], kind="stable")
            dp_ts = ts_t[t0t][order]
            dp_log = np.stack([preds[H][dpg] for H in Hs_l], axis=1)[order]
            b = ts_t // (10 * 10**9)                       # 10-second bins
            last = np.append(np.where(np.diff(b) != 0)[0], len(b) - 1)
            g_ts = ts_t[last]; g_mid = mid_t[last]
            pos = np.searchsorted(dp_ts, g_ts, side="right") - 1   # ffill logits
            keep = pos >= 0
            ts_all.append(g_ts[keep]); mid_all.append(g_mid[keep])
            log_all.append(dp_log[pos[keep]])
            day_all.append(np.full(int(keep.sum()), di, np.int64))
        rlpath = (cfg.ckpt_path[:-5] if cfg.ckpt_path.endswith(".ckpt")
                  else cfg.ckpt_path) + ".rlseries.npz"
        np.savez(rlpath, ts=np.concatenate(ts_all), mid=np.concatenate(mid_all),
                 logits=np.concatenate(log_all, axis=0), day=np.concatenate(day_all),
                 Hs=np.array(Hs_l))
        log(f"[dump_rl] {sum(len(x) for x in ts_all)} 10s-steps over "
            f"{len(ts_all)} days -> {rlpath}")
    out = {"symbol": cfg.symbol, "L": cfg.L, "seed": cfg.seed, "n_dp": int(n),
           "n_tr": int(n_tr), "best_val_ricsum": float(best_val),
           "target_name": cfg.target_name, "loss_name": cfg.loss_name,
           "elapsed_s": time.time() - t_start, "epoch_hist": epoch_hist, "by_H": {}}
    for H in Hs:
        out["by_H"][H] = {"all": _H_metrics(preds, G, te & scored, deep=None, H=H),
                          "deep": _H_metrics(preds, G, te & scored, deep=deep, H=H)}
    return out, preds


def _eval_predmap(model, G, mask, cfg, dev, _day_stream, _batches, _forward):
    """Per-H OOS prediction map over `mask` (shared by val-score + final)."""
    import torch
    H_idx = {H: k for k, H in enumerate(cfg.Hs)}
    was_training = model.training
    model.eval()
    pr = {H: np.full(G["n"], np.nan, np.float32) for H in cfg.Hs}
    with torch.no_grad():
        for di in sorted(set(G["day"][mask].tolist())):
            stream = _day_stream(di)
            for grp in _batches(di, mask):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, gdp = _forward(stream, grp, dev)
                for H in cfg.Hs:
                    pr[H][gdp] = logits[:, H_idx[H]].float().cpu().numpy()
            del stream
    if was_training:
        model.train()
    return pr


def _eval_prec_nf(model, G, mask, cfg, dev, _day_stream, _batches, _forward):
    """LEGACY REPRO metric: 3-class argmax over the profit3 head. prec_NF =
    precision on predicted non-FL (UP/DN); coverage = frac predicted non-FL.
    Coin baseline ~ base_up/base_dn (~0.20). Reuses the period machinery."""
    import torch
    was = model.training
    model.eval()
    pred = np.full(G["n"], -1, np.int64)
    pdir = np.full(G["n"], -1, np.int64)      # direction from UP/DN logits only (ignore FL)
    with torch.no_grad():
        for di in sorted(set(G["day"][mask].tolist())):
            stream = _day_stream(di)
            for grp in _batches(di, mask):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, gdp = _forward(stream, grp, dev)
                lg = logits.float().cpu().numpy()
                pred[gdp] = lg.argmax(-1)
                pdir[gdp] = lg[:, :2].argmax(-1)   # 0=UP,1=DN among the two dir logits
            del stream
    if was:
        model.train()
    sel = mask & (pred >= 0)
    y = G["y3"][sel]; p = pred[sel]; pd = pdir[sel]
    nf = p != 2
    n_nf = int(nf.sum()); n_ev = int(sel.sum())
    prec_nf = float((p[nf] == y[nf]).mean()) if n_nf else 0.0
    # DECONFOUND prec_NF = (volatility selection) x (direction):
    #  q_commit = among COMMITTED, frac truly non-FL (vs base_nonfl) -> volatility skill
    #  dir_acc_all = among TRULY directional (true in {UP,DN}), is the UP/DN-logit
    #    argmax right -> PURE direction skill, decoupled from FLAT detect. 0.5 = coin.
    base_nonfl = float((y != 2).mean()) if n_ev else 0.0
    q_commit = float((y[nf] != 2).mean()) if n_nf else 0.0
    td = (y != 2)                              # truly directional
    dir_acc_all = float((pd[td] == y[td]).mean()) if int(td.sum()) else 0.0
    cd = nf & (y != 2)                         # committed AND truly directional
    dir_acc_committed = float((p[cd] == y[cd]).mean()) if int(cd.sum()) else 0.0
    return {"prec_nf": prec_nf, "coverage": n_nf / max(1, n_ev), "n_nf": n_nf,
            "n_eval": n_ev, "base_up": float((y == 0).mean()) if n_ev else 0.0,
            "base_dn": float((y == 1).mean()) if n_ev else 0.0,
            "base_fl": float((y == 2).mean()) if n_ev else 0.0,
            "base_nonfl": base_nonfl, "q_commit_nonfl": q_commit,
            "dir_acc_all": dir_acc_all, "dir_acc_committed": dir_acc_committed,
            "n_dir": int(td.sum())}


def _eval_score(model, G, mask, cfg, dev, _day_stream, _batches, _forward, H_idx):
    pr = _eval_predmap(model, G, mask, cfg, dev, _day_stream, _batches, _forward)
    s = 0.0
    for H in cfg.Hs:
        m = mask & G["rc"][H] & np.isfinite(pr[H])
        if m.sum() >= 50:
            s += _rank_ic(pr[H][m], G["rH"][H][m])
    return s


def _eval_preds(model, G, mask, cfg, dev, _day_stream, _batches, _forward):
    return _eval_predmap(model, G, mask, cfg, dev, _day_stream, _batches, _forward)


def _H_metrics(preds, G, base_mask, deep, H):
    m = base_mask & G["rc"][H] & np.isfinite(preds[H])
    if deep is not None:
        m = m & deep
    if m.sum() < 50:
        return {"n": int(m.sum())}
    p = preds[H][m]; rH = G["rH"][H][m]; upm = (G["y"][H][m] == 1).astype(int)
    blk = core.block_size(H)
    # economic annotation (HA1-style): mean |move| in the top-confidence decile
    # vs the 0.13% cost floor -> is the alpha economically harvestable, not just
    # statistically present. NOT a target cap (target is the unbounded r_H).
    k = max(10, len(p) // 10)
    med = np.median(p)
    top = np.argsort(-np.abs(p - med))[:k]
    top_absmove = float(np.abs(rH[top]).mean())
    # magnitude-aware CAPTURE (user 2026-05-24, winner-selection criterion): the
    # SIGNED return harvested in the top-confidence decile (the points you'd
    # actually trade), gross and net of the 0.13% round-trip cost floor. This is
    # what the loss screen ranks on -- it rewards catching big *correctly-called*
    # moves, so a 0.30% catcher beats a 2x-less-profitable 0.13% catcher. rank-IC
    # is rank-only (magnitude-blind) and would mis-rank them. preds are raw
    # logits -> directional call = sign(logit - median); capture = dir * r_H.
    cap_gross = float((np.sign(p[top] - med) * rH[top]).mean())
    return {"n": int(m.sum()), "rank_ic": _rank_ic(p, rH),
            "auc": core.auc(upm, p), "placebo": core.placebo_auc(upm, p),
            "boot_se": core.block_bootstrap_auc_se(upm, p, blk),
            "top_decile_absmove": top_absmove,
            "econ_pass": int(top_absmove >= 0.0013),
            "cap_edge_gross": cap_gross, "cap_edge_net": cap_gross - 0.0013}
