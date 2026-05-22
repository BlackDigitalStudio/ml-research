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


def _day_paths(cache_dir, symbol):
    d = Path(cache_dir) / "hd2" / symbol
    return sorted(str(p) for p in d.glob("*.npz"))


def load_meta(day_paths, Hs):
    """Per-day metadata (NO stream) + global concatenated decision-point arrays
    in chronological order. day[i] = {path, n_ticks, n, g_lo, g_hi}."""
    days, g_t0, g_glob, g_day = [], [], [], []
    g_y = {H: [] for H in Hs}; g_rH = {H: [] for H in Hs}; g_rc = {H: [] for H in Hs}
    off = 0
    for di, p in enumerate(day_paths):
        d = np.load(p, allow_pickle=True)
        t0 = d["t0"].astype(np.int64)
        n = len(t0)
        meta = json.loads(str(d["meta"]))
        days.append({"path": p, "n_ticks": int(meta["n_ticks"]), "n": n,
                     "g_lo": off, "g_hi": off + n})
        g_t0.append(t0); g_glob.append(d["globals"].astype(np.float32))
        g_day.append(np.full(n, di, np.int64))
        for H in Hs:
            g_y[H].append(d[f"y_{H}"].astype(np.int64))
            g_rH[H].append(d[f"rH_{H}"].astype(np.float32))
            g_rc[H].append(d[f"rc_{H}"].astype(bool))
        d.close()
        off += n
    G = {"t0": np.concatenate(g_t0), "globals": np.concatenate(g_glob),
         "day": np.concatenate(g_day), "n": off,
         "y": {H: np.concatenate(g_y[H]) for H in Hs},
         "rH": {H: np.concatenate(g_rH[H]) for H in Hs},
         "rc": {H: np.concatenate(g_rc[H]) for H in Hs}}
    return days, G


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
    day_paths = _day_paths(cache_dir, cfg.symbol)
    assert day_paths, f"no cache for {cfg.symbol} in {cache_dir}"
    days, G = load_meta(day_paths, Hs)
    n = G["n"]

    tr, te, n_tr = core.honest_split(n)
    fit, val = core.train_val_split(tr)
    fit_day_ids = sorted(set(G["day"][fit].tolist()))
    test_day_ids = sorted(set(G["day"][te].tolist()))

    s_mu, s_sd = lob_stats(day_paths, fit_day_ids)
    g_mu, g_sd = (G["globals"][fit].mean(0), G["globals"][fit].std(0))
    g_sd = np.where(g_sd > 1e-8, g_sd, 1.0)
    glob_std = (G["globals"] - g_mu) / g_sd

    # R1 weights + up labels per H (global)
    w1 = {H: core.r1_weights(G["rH"][H], fit & G["rc"][H]) for H in Hs}
    up = {H: (G["y"][H] == 1).astype(np.float32) for H in Hs}

    # warmup-floor / deep-context per global decision point
    warm = min(cfg.L, WARMUP)
    ctx = G["t0"] - (G["t0"] // cfg.L) * cfg.L
    scored = ctx >= warm
    deep = ctx >= int(DEEP_CTX_FRAC * cfg.L)

    model = HD2Mamba2(cell_kind="mamba2", d_model=cfg.d_model,
                      n_layers=cfg.n_layers, d_state=cfg.d_state,
                      dropout=cfg.dropout, n_out=len(Hs)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    use_ckpt = cfg.L >= cfg.grad_ckpt_min_L
    H_idx = {H: k for k, H in enumerate(Hs)}

    start_ep, best_val, best_state, since = 0, -1e9, None, 0
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        sd = torch.load(cfg.ckpt_path, map_location=dev)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        start_ep = sd["epoch"] + 1; best_val = sd["best_val"]; since = sd["since"]
        torch.set_rng_state(sd["rng"].cpu())
        log(f"[resume] epoch {start_ep} best_val={best_val:.5f}")

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
    for ep in range(start_ep, cfg.epochs):
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
                    terms = []
                    for H in Hs:
                        rc = G["rc"][H][gdp]
                        if rc.sum() == 0:
                            continue
                        sel = torch.from_numpy(rc).to(dev)
                        lg = logits[:, H_idx[H]].float()[sel]
                        tg = torch.from_numpy(up[H][gdp][rc]).to(dev)
                        wv = torch.from_numpy(w1[H][gdp][rc].astype(np.float32)).to(dev)
                        terms.append(nn.functional.binary_cross_entropy_with_logits(
                            lg, tg, weight=wv))
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

        # ---- validation rank_IC (sum over H) for early stop ----
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
        if since >= cfg.patience:
            log("[early-stop]"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    # ---- final OOS metrics per H (all + deep) + persist preds ----
    preds = _eval_preds(model, G, te & scored, cfg, dev, _day_stream,
                        _batches, _forward)
    out = {"symbol": cfg.symbol, "L": cfg.L, "seed": cfg.seed, "n_dp": int(n),
           "n_tr": int(n_tr), "best_val_ricsum": float(best_val),
           "elapsed_s": time.time() - t_start, "by_H": {}}
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
    return {"n": int(m.sum()), "rank_ic": _rank_ic(p, rH),
            "auc": core.auc(upm, p), "placebo": core.placebo_auc(upm, p),
            "boot_se": core.block_bootstrap_auc_se(upm, p, blk)}
