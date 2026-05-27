#!/usr/bin/env python3
"""HD2 Mamba-2 streaming-stateful trainer (HD2 rev1 frozen spec).

Realized mechanism (faithful refinement of the pre-reg, settled by smoke):
each L-reset-period is ONE forward pass with zero-init state (state is RESET
every L ticks, so it never crosses a reset boundary -> no cross-period carry
needed); per-layer gradient checkpointing bounds activation memory for long L
(segment = CKPT_SEG ticks). Compute = O(total_ticks) INDEPENDENT of L. Within
a period the readout head fires at the STRIDE=1 decision points that lie past
the warmup floor (min(L, WARMUP) ticks); those carry R1-weighted BCE loss.

The LOB encoder block is PLUGGABLE behind one (x, ) -> y interface so the exact
same streaming orchestration is validated on CPU (GRU stub, no CUDA) and run on
H100 (real mamba_ssm.Mamba2 CUDA kernels):
    cell_kind="stub"   -> nn.GRU         (CPU/GPU, deterministic, for logic test)
    cell_kind="mamba2" -> Mamba2 stack   (CUDA only)

Label/scope/split/R1/AUC/placebo/bootstrap come from the FROZEN hd1_seq_core.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hd1_seq_core as core  # noqa: E402

core.STRIDE = 1
WARMUP = 2048          # warmup-floor cap (ticks); context-floor per reset-period
CKPT_SEG = 2048        # gradient-checkpoint segment (ticks)
DEEP_CTX_FRAC = 0.75   # "deep-context" readout subset: context >= 0.75*L


# =========================================================================
# Model
# =========================================================================
def _make_blocks(cell_kind, d_model, n_layers, d_state,
                 dt_min=0.001, dt_max=0.1, A_init_range=(1, 16), d_conv=4):
    import torch.nn as nn
    if cell_kind == "stub":
        return nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
    if cell_kind == "mamba2":
        from mamba_ssm import Mamba2
        return nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mix": Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2,
                              dt_min=dt_min, dt_max=dt_max, A_init_range=A_init_range),
            }) for _ in range(n_layers)
        ])
    raise ValueError(cell_kind)


class HD2Mamba2(__import__("torch").nn.Module):
    """2-stream: LOB stream (80ch -> d_model -> blocks) read at decision points,
    + 6 last-tick globals -> MLP, fused -> 1 BCE logit."""

    def __init__(self, cell_kind="mamba2", d_model=128, n_layers=4,
                 d_state=128, n_glob=6, dropout=0.1, n_out=1,
                 dt_min=0.001, dt_max=0.1, A_init_range=(1, 16), d_conv=4):
        import torch.nn as nn
        super().__init__()
        self.cell_kind = cell_kind
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_out = n_out          # multitask: one logit per horizon H
        self.in_proj = nn.Linear(80, d_model)
        self.blocks = _make_blocks(cell_kind, d_model, n_layers, d_state,
                                   dt_min=dt_min, dt_max=dt_max, A_init_range=A_init_range,
                                   d_conv=d_conv)
        self.lob_norm = nn.LayerNorm(d_model)
        self.glob = nn.Sequential(
            nn.Linear(n_glob, 32), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(d_model + 32, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_out))

    def encode_period(self, x, use_ckpt=False):
        """x: (1, T, 80) -> hidden (1, T, d_model). Zero-init state (period
        boundary = reset). Per-layer gradient checkpointing when use_ckpt."""
        import torch
        from torch.utils.checkpoint import checkpoint
        h = self.in_proj(x)
        if self.cell_kind == "stub":
            h, _ = self.blocks(h)
        else:
            for blk in self.blocks:
                def _f(z, _b=blk):
                    return z + _b["mix"](_b["norm"](z))
                h = checkpoint(_f, h, use_reentrant=False) if use_ckpt else _f(h)
        return self.lob_norm(h)

    def readout(self, hidden_at_t0, globals6):
        import torch
        g = self.glob(globals6)
        out = self.head(torch.cat([hidden_at_t0, g], dim=-1))   # (k, n_out)
        return out.squeeze(-1) if self.n_out == 1 else out


# =========================================================================
# Streaming orchestration
# =========================================================================
@dataclass
class RunCfg:
    cell_kind: str = "mamba2"
    L: int = 6000
    H: int = 600
    seed: int = 0
    d_model: int = 256   # causal_conv1d channel-last kernel needs this width;
                          # 128/192/384 hit a stride-multiple-of-8 assert (smoke-verified)
    n_layers: int = 4
    d_state: int = 128
    batch_periods: int = 8     # pack equal-length periods into one forward (GPU util)
    epochs: int = 6
    lr: float = 1e-3
    wd: float = 1e-3
    dropout: float = 0.1
    bf16: bool = True
    grad_ckpt_min_L: int = 20000   # checkpoint only when L is large
    device: str = "cuda"
    ckpt_path: str = ""
    max_minutes: float = 0.0       # 0 = no wall cap


def _periods(n_ticks, L):
    """Reset-period boundaries [start, end) partitioning [0, n_ticks)."""
    return [(s, min(s + L, n_ticks)) for s in range(0, n_ticks, L)]


def _standardize_fit(arr, fit_hi_row):
    """Per-channel mean/std over rows [:fit_hi_row]; return (mean, std)."""
    sl = arr[:fit_hi_row]
    mu = sl.mean(0)
    sd = sl.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def _rank_ic(pred, target):
    """Spearman rank-IC = Pearson of ranks (program convention)."""
    from scipy.stats import rankdata
    if len(pred) < 3 or np.all(pred == pred[0]):
        return 0.0
    rp, rt = rankdata(pred), rankdata(target)
    c = np.corrcoef(rp, rt)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def load_day(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return d, meta


def run(npz_path, cfg: RunCfg, log=print):
    """Train + eval one symbol-day stream. Returns metrics dict. Smoke-scale
    by default; the full tier concatenates day streams per (symbol, window)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    d, meta = load_day(npz_path)
    stream = d["stream"].astype(np.float32)        # (n_ticks, 80)
    t0 = d["t0"].astype(np.int64)                  # (n_dp,) book indices
    globals6 = d["globals"].astype(np.float32)     # (n_dp, 6)
    y = d[f"y_{cfg.H}"].astype(np.int64)
    rH = d[f"rH_{cfg.H}"].astype(np.float32)
    reached = d[f"rc_{cfg.H}"].astype(bool)
    n_ticks, n_dp = stream.shape[0], len(t0)

    # frozen honest split over decision points
    tr, te, n_tr = core.honest_split(n_dp)
    fit, val = core.train_val_split(tr)
    fit_hi_tick = int(t0[fit].max()) + 1 if fit.any() else n_ticks

    # standardize (fit-region only)
    s_mu, s_sd = _standardize_fit(stream, fit_hi_tick)
    g_mu, g_sd = _standardize_fit(globals6, max(1, int(fit.sum())))
    stream = (stream - s_mu) / s_sd
    globals6 = (globals6 - g_mu) / g_sd

    # R1 weights (fit-split stats), per decision point
    w1 = core.r1_weights(rH, fit & reached)
    up = (y == 1).astype(np.float32)

    # warmup-floor + deep-context masks per decision point (context since reset)
    warm = min(cfg.L, WARMUP)
    ctx = t0 - (t0 // cfg.L) * cfg.L            # ticks since last reset
    scored = ctx >= warm                         # past warmup -> eligible readout
    deep = ctx >= int(DEEP_CTX_FRAC * cfg.L)

    model = HD2Mamba2(cfg.cell_kind, cfg.d_model, cfg.n_layers, cfg.d_state,
                      dropout=cfg.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    use_ckpt = cfg.L >= cfg.grad_ckpt_min_L and cfg.cell_kind == "mamba2"

    periods = _periods(n_ticks, cfg.L)
    # map each decision point to its period index for readout grouping
    t0_period = (t0 // cfg.L).astype(np.int64)

    start_ep, t_start = 0, time.time()
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        st = torch.load(cfg.ckpt_path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_ep = st["epoch"] + 1
        torch.set_rng_state(st["rng"].cpu())
        log(f"[resume] from epoch {start_ep}")

    def _save_ckpt(ep):
        if not cfg.ckpt_path:
            return
        tmp = cfg.ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, cfg.ckpt_path)             # atomic

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if (cfg.bf16 and dev == "cuda")
                else torch.autocast("cpu", enabled=False))

    # period -> decision-point index lists (precomputed once)
    def _period_dps(mask):
        m = {}
        sel = np.where(mask)[0]
        for di in sel:
            m.setdefault(int(t0_period[di]), []).append(int(di))
        return {k: np.asarray(v, np.int64) for k, v in m.items()}
    fit_pd = _period_dps(fit & scored & reached)
    te_pd = _period_dps(te & scored)

    def _batched(pd_map):
        """Group equal-length (==L) periods into batches of cfg.batch_periods;
        shorter tail periods go solo. Batching the forward is the GPU-util win
        (B=1 small periods leave the H100 idle; smoke: 240k vs 2.16M tok/s)."""
        full, rest = [], []
        for pi, dps in pd_map.items():
            a, b = periods[pi]
            (full if (b - a) == cfg.L else rest).append((pi, a, b, dps))
        out = [full[i:i + cfg.batch_periods]
               for i in range(0, len(full), cfg.batch_periods)]
        out += [[p] for p in rest]
        return out

    def _forward_batch(grp, use_ckpt):
        T = grp[0][2] - grp[0][1]
        xs = np.stack([stream[a:b] for (_, a, b, _) in grp])   # (k, T, 80)
        x = torch.from_numpy(xs).to(dev)
        hidden = model.encode_period(x, use_ckpt=use_ckpt)     # (k, T, d)
        rows, poss, dps_all = [], [], []
        for j, (_pi, a, _b, dps) in enumerate(grp):
            rows.append(np.full(len(dps), j, np.int64))
            poss.append((t0[dps] - a).astype(np.int64))
            dps_all.append(dps)
        rows = np.concatenate(rows); poss = np.concatenate(poss)
        dps_all = np.concatenate(dps_all)
        hsel = hidden[torch.from_numpy(rows).to(dev), torch.from_numpy(poss).to(dev)]
        g = torch.from_numpy(globals6[dps_all]).to(dev)
        return model.readout(hsel, g), dps_all

    n_tok_total = 0
    for ep in range(start_ep, cfg.epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        rng = np.random.default_rng(cfg.seed * 1000 + ep)
        batches = _batched(fit_pd)
        for bi in rng.permutation(len(batches)):
            grp = batches[bi]
            with autocast:
                logit, dps_all = _forward_batch(grp, use_ckpt)
                tgt = torch.from_numpy(up[dps_all]).to(dev)
                w = torch.from_numpy(w1[dps_all].astype(np.float32)).to(dev)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logit.float(), tgt, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss) * len(dps_all); ep_n += len(dps_all)
            n_tok_total += sum(b - a for (_, a, b, _) in grp)
        _save_ckpt(ep)
        log(f"[ep {ep}] fit_loss={ep_loss/max(1,ep_n):.4f} "
            f"n_fit_pts={ep_n} toks={n_tok_total:,} "
            f"elapsed={time.time()-t_start:.1f}s")
        if cfg.max_minutes and (time.time() - t_start) / 60 > cfg.max_minutes:
            log("[wall-cap hit]"); break

    # ---- eval on OOS test points (all-scored + deep-context) ----
    model.eval()
    preds = np.full(n_dp, np.nan, np.float32)
    with torch.no_grad():
        for grp in _batched(te_pd):
            with autocast:
                logit, dps_all = _forward_batch(grp, use_ckpt=False)
            preds[dps_all] = logit.float().cpu().numpy()

    def _metrics(sub):
        m = te & scored & reached & sub & np.isfinite(preds)
        if m.sum() < 50:
            return {"n": int(m.sum())}
        p, u, r = preds[m], up[m], rH[m]
        blk = core.block_size(cfg.H)
        return {"n": int(m.sum()), "rank_ic": _rank_ic(p, r),
                "auc": core.auc(u, p),
                "placebo": core.placebo_auc(u, p),
                "boot_se": core.block_bootstrap_auc_se(u, p, blk)}

    res = {"cfg": cfg.__dict__, "symbol": meta["symbol"], "day": meta["day"],
           "n_dp": n_dp, "n_tr": int(n_tr),
           "n_fit_scored": int((fit & scored & reached).sum()),
           "tok_per_epoch": int(n_ticks), "all": _metrics(np.ones(n_dp, bool)),
           "deep_ctx": _metrics(deep), "elapsed_s": time.time() - t_start}
    log("[result] " + json.dumps({k: res[k] for k in ("all", "deep_ctx")}))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--cell", default="stub")
    ap.add_argument("--L", type=int, default=6000)
    ap.add_argument("--H", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    cfg = RunCfg(cell_kind=a.cell, L=a.L, H=a.H, seed=a.seed, epochs=a.epochs,
                 d_model=a.d_model, n_layers=a.n_layers, device=a.device,
                 ckpt_path=a.ckpt, bf16=(a.device == "cuda"))
    out = run(a.npz, cfg)
    print(json.dumps(out, indent=2, default=float))
