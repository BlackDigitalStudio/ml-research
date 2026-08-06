#!/usr/bin/env python3
"""HD2 #3 — sub-60s 2×2 cascade (Model A FLAT/NON-FLAT, Model B UP/DN), each with
2 streams (raw LOB + curated feats). Spec: mamba2_arch.md.

Key sub-60s choices (vs old HD2): SHORT context (bounded L), decision-stride to
control n_eff (rev24), capacity SIZED to n_eff (A ~250k per-symbol, B ~50k pooled),
A=weighted-BCE, B=IC/capture objective (rev10) on non-flat only. Pluggable cell:
  cell="stub"   -> GRU       (CPU smoke, deterministic)
  cell="mamba2" -> Mamba2    (CUDA)

Cache (per symbol-day, from subs60_cache_build): lob(n_ticks,80)f16 stream-1;
t0(n_dp) book idx; feat(n_dp,F) stream-2; rH60,y60(nonflat),updn(up),v60.
"""
from __future__ import annotations
import argparse, io, json, glob, os
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------- model
def make_cell(kind, d, n_layers, d_state=64):
    """cell='stub' -> GRU (any d). cell='mamba2' -> REAL Mamba2 on the PROVEN fused
    path. mamba_ssm 2.2.2's small-d/non-mem-eff path is broken (conv stride-lock,
    dconv typo, dt-shape assert), so the fused kernel requires d_model=256. We
    therefore keep d=256 and SHRINK via n_layers (=1) + small d_state instead of
    via d_model -> ~0.45M params/stream (vs ~1.9M at n1=4), real Mamba 2 unchanged."""
    import torch.nn as nn
    if kind == "stub":
        return nn.GRU(d, d, num_layers=n_layers, batch_first=True)
    if kind == "mamba2":
        assert d == 256, f"mamba2 fused kernel requires d_model=256 (got {d})"
        from mamba_ssm import Mamba2
        return nn.ModuleList([nn.ModuleDict({
            "norm": nn.LayerNorm(d),
            "mix": Mamba2(d_model=d, d_state=d_state, d_conv=4, expand=2),
        }) for _ in range(n_layers)])
    raise ValueError(kind)


class Cascade2Stream(__import__("torch").nn.Module):
    """2-stream readout: LOB-stream (80→d1, bounded-L cell) gathered at decisions
    + feat-stream (F→d2, cell over the 1s decision sequence) → fuse → 1 logit.
    Optional symbol-embedding (for pooled Model B)."""
    def __init__(self, F, cell="stub", d1=128, n1=2, d2=64, n2=1,
                 n_sym=0, sym_dim=8, dropout=0.1):
        import torch.nn as nn
        super().__init__()
        self.cell = cell
        self.in1 = nn.Linear(80, d1); self.enc1 = make_cell(cell, d1, n1)
        self.norm1 = nn.LayerNorm(d1)
        self.in2 = nn.Linear(F, d2); self.enc2 = make_cell(cell, d2, n2)
        self.norm2 = nn.LayerNorm(d2)
        self.sym = nn.Embedding(n_sym, sym_dim) if n_sym > 0 else None
        fuse_in = d1 + d2 + (sym_dim if n_sym > 0 else 0)
        self.head = nn.Sequential(nn.Linear(fuse_in, d1), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d1, 1))

    def _run(self, enc, x):
        import torch
        if self.cell == "stub":
            h, _ = enc(x); return h
        for blk in enc:
            x = x + blk["mix"](blk["norm"](x))
        return x

    def encode_lob(self, x):          # x:(1,T,80) one bounded-L period
        return self.norm1(self._run(self.enc1, self.in1(x)))

    def encode_feat(self, fseq):      # fseq:(1,M,F) the period's decision feats
        return self.norm2(self._run(self.enc2, self.in2(fseq)))

    def head_logit(self, h1, h2, sym_id=None):
        import torch
        parts = [h1, h2]
        if self.sym is not None and sym_id is not None:
            parts.append(self.sym(sym_id))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


# ---------------------------------------------------------------- data
@dataclass
class Stream:
    lob: np.ndarray; t0: np.ndarray; feat: np.ndarray
    rH: np.ndarray; y: np.ndarray; up: np.ndarray; v: np.ndarray
    sym_id: int; day: int
    pl: np.ndarray = None; ps: np.ndarray = None   # per-window executed payoff (long/short), bp (Stage-2 B2)


def load_cache(paths, sym_ids):
    out = []
    for p, (sid, di) in zip(paths, sym_ids):
        d = np.load(p) if os.path.exists(p) else np.load(io.BytesIO(p))
        # keep lob f16 in RAM (promote per-period in forward) -> ~half memory,
        # so pooled multi-symbol training fits.
        pl = d["pl"].astype(np.float32) if "pl" in d.files else None     # bracket payoff sidecar
        ps = d["ps"].astype(np.float32) if "ps" in d.files else None
        out.append(Stream(d["lob"].astype(np.float16), d["t0"].astype(np.int64),
                          d["feat"].astype(np.float32), d["rH60"].astype(np.float32),
                          d["y60"].astype(np.int64), d["updn"].astype(np.int64),
                          d["v60"].astype(bool), sid, di, pl, ps))
    return out


def standardize(streams, attr):
    """Per-channel mean/std over all train streams for `attr`, STREAMING (one
    stream at a time, no big concatenate) so pooled multi-day fits in RAM."""
    s = ss = None; n = 0
    for st in streams:
        a = getattr(st, attr).astype(np.float64)        # one day, transient
        sa = a.sum(0); sq = (a * a).sum(0)
        s = sa if s is None else s + sa
        ss = sq if ss is None else ss + sq
        n += len(a)
    mu = s / max(n, 1); var = ss / max(n, 1) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0)); sd = np.where(sd > 1e-8, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


# ---------------------------------------------------------------- run
@dataclass
class Cfg:
    head: str = "A"            # "A" (nonflat BCE) | "B" (updn IC) | "B2" (executed-payoff fine-tune)
    cell: str = "stub"
    L: int = 3000             # bounded reset-period (ticks) — SHORT context
    warmup: int = 300         # ticks before a decision is scored
    dec_stride_s: int = 25    # decision stride (seconds) — n_eff control
    d1: int = 128; n1: int = 2; d2: int = 64; n2: int = 1
    epochs: int = 4; lr: float = 1e-3; wd: float = 1e-3; dropout: float = 0.1
    patience: int = 0          # early-stop: stop if no best-val gain for `patience` epochs (0=off)
    device: str = "cpu"; seed: int = 0
    pooled: bool = False       # B: pool symbols + symbol-embed
    ckpt_path: str = ""        # resume/checkpoint (preemption-robust)
    # Stage-2 (B2) — executed-payoff fine-tune: L = -mean[ sigmoid(z)*PL + (1-sigmoid(z))*PS ]
    payoff: str = "hold"       # "hold" -> PL=+rH,PS=-rH (ETH); "bracket" -> use stream.pl/ps sidecar
    init_from: str = ""        # base .best.pt to warm-start the fine-tune (Stage-2)
    comm_mm: float = 4.0       # maker-maker RT bp (reporting only; side-independent -> not in grad)


def _periods(n, L):
    return [(s, min(s + L, n)) for s in range(0, n, L)]


def run(streams, cfg: Cfg, log=print, on_ckpt=None):
    import torch, torch.nn as nn
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    F = streams[0].feat.shape[1]
    base_sd = None
    if cfg.init_from:                                  # Stage-2: warm-start from base B
        base_sd = torch.load(cfg.init_from, map_location=dev)["model"]
    base_nsym = base_sd["sym.weight"].shape[0] if (base_sd is not None and "sym.weight" in base_sd) else 0
    n_sym = base_nsym if cfg.init_from else ((max(s.sym_id for s in streams) + 1) if cfg.pooled else 0)
    if cfg.head == "B2":                               # executed-payoff target per window
        for s in streams:                              # hold -> +/-rH; bracket -> sidecar pl/ps (else hold)
            if cfg.payoff == "hold" or s.pl is None:
                s.pl = s.rH.astype(np.float32); s.ps = (-s.rH).astype(np.float32)

    # purged day split (embargo handled by per-day streams; train earlier 65%)
    days = sorted(set(s.day for s in streams))
    cut = days[int(len(days) * 0.65)]; emb_hi = days[min(int(len(days) * 0.68), len(days)-1)]
    tr = [s for s in streams if s.day < cut]; te = [s for s in streams if s.day >= emb_hi]
    if not tr or not te:
        log("[split] insufficient"); return {}

    # standardize on train; keep GPU copies (standardize batches on-device -> less CPU)
    lob_mu, lob_sd = standardize(tr, "lob"); ft_mu, ft_sd = standardize(tr, "feat")
    lob_mu_t = torch.from_numpy(lob_mu).to(dev); lob_sd_t = torch.from_numpy(lob_sd).to(dev)
    ft_mu_t = torch.from_numpy(ft_mu).to(dev); ft_sd_t = torch.from_numpy(ft_sd).to(dev)
    KBATCH = 32                                   # full-L periods packed per LOB forward

    model = Cascade2Stream(F, cfg.cell, cfg.d1, cfg.n1, cfg.d2, cfg.n2,
                           n_sym=n_sym, dropout=cfg.dropout).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    if base_sd is not None:                            # warm-start BEFORE resume (resume overrides)
        model.load_state_dict(base_sd); log(f"[init_from] warm-started from base (n_sym={n_sym})")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # tracking + resume. The checkpoint CARRIES curve+best so a preemption-resumed
    # run yields a COMPLETE curve and correct best-val (not truncated). B1/B2 fix.
    pkey = {"A": "auc", "B": "cap@20.0%bp", "B2": "exec_soft_bp"}[cfg.head]   # primary selection metric
    curve = []; best = -1e18; best_ep = -1; best_state = None; best_metrics = {}
    start_ep = 0
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        st = torch.load(cfg.ckpt_path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_ep = st["epoch"] + 1
        curve = st.get("curve", []); best = st.get("best", -1e18)
        best_ep = st.get("best_ep", -1); best_metrics = st.get("best_metrics", {})
        best_state = st.get("best_state", None)
        log(f"[resume] ep{start_ep} (best so far ep{best_ep} {pkey}={best:.4f})")

    def _save_ckpt(ep):
        if not cfg.ckpt_path:
            return
        tmp = cfg.ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "curve": curve, "best": best, "best_ep": best_ep,
                    "best_metrics": best_metrics, "best_state": best_state}, tmp)
        os.replace(tmp, cfg.ckpt_path)          # atomic
        if on_ckpt:
            on_ckpt()                            # VOL.commit() -> persists across preempt

    def scored_decisions(s):
        """decision indices kept: stride by dec_stride_s, past warmup, valid."""
        # ~1s grid -> stride in steps
        step = max(1, cfg.dec_stride_s)
        keep = np.zeros(len(s.t0), bool); keep[::step] = True
        ctx = s.t0 - (s.t0 // cfg.L) * cfg.L
        keep &= (ctx >= min(cfg.warmup, cfg.L - 1)) & s.v
        if cfg.head in ("B", "B2"):
            keep &= (s.y == 1)               # non-flat only
        return np.where(keep)[0]

    def forward_stream(s, train=True):
        """(logits, dp) for kept decisions. BATCHED: all full-L periods packed into
        few (K,L,80) LOB forwards (GPU-util); standardization on-device."""
        dp = scored_decisions(s)
        if len(dp) == 0:
            return None
        L = cfg.L; per = _periods(len(s.lob), L); pidx = (s.t0[dp] // L)
        logits = torch.empty(len(dp), device=dev)
        sym_id = (torch.full((len(dp),), s.sym_id, device=dev, dtype=torch.long)
                  if n_sym > 0 else None)
        groups = {}
        for di, k in enumerate(pidx.tolist()):
            groups.setdefault(k, []).append(di)
        full = [(k, per[k][0], per[k][1], np.asarray(v)) for k, v in groups.items()
                if per[k][1] - per[k][0] == L]
        tail = [(k, per[k][0], per[k][1], np.asarray(v)) for k, v in groups.items()
                if per[k][1] - per[k][0] != L]

        def _emit(batch):                          # batch: list of (k,a,b,sel)
            xs = np.stack([s.lob[a:b] for _, a, b, _ in batch]).astype(np.float32)
            x = (torch.from_numpy(xs).to(dev) - lob_mu_t) / lob_sd_t       # (K,Lk,80)
            h1all = model.encode_lob(x)                                    # (K,Lk,d1)
            for row, (k, a, b, sel) in enumerate(batch):
                pos = torch.from_numpy((s.t0[dp[sel]] - a)).to(dev)
                h1 = h1all[row, pos]                                        # (m,d1)
                fb = (torch.from_numpy(s.feat[dp[sel]]).to(dev) - ft_mu_t) / ft_sd_t
                h2 = model.encode_feat(fb[None])[0]                         # (m,d2)
                logits[sel] = model.head_logit(
                    h1, h2, sym_id[sel] if sym_id is not None else None)

        for i in range(0, len(full), KBATCH):       # full-L periods packed
            _emit(full[i:i + KBATCH])
        for g in tail:                               # variable-length tails: solo
            _emit([g])
        return logits, dp

    def loss_fn(logits, s, dp):
        if cfg.head == "A":
            tgt = torch.from_numpy(s.y[dp].astype(np.float32)).to(dev)
            pw = torch.tensor([(len(tgt) - tgt.sum()) / max(tgt.sum(), 1)], device=dev)
            return nn.functional.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw)
        if cfg.head == "B2":
            # executed-payoff: maximize E[ p*PL + (1-p)*PS ], p=sigmoid(logit). Commission is
            # side-independent -> drops from the gradient (train on gross PL/PS).
            pl = torch.from_numpy(s.pl[dp].astype(np.float32)).to(dev)
            ps = torch.from_numpy(s.ps[dp].astype(np.float32)).to(dev)
            p = torch.sigmoid(logits)
            return -(p * pl + (1 - p) * ps).mean()
        # B: IC/capture — maximize corr(tanh(logit), rH) on non-flat
        r = torch.from_numpy(s.rH[dp].astype(np.float32)).to(dev)
        p = torch.tanh(logits)
        p = p - p.mean(); r = r - r.mean()
        denom = (p.norm() * r.norm() + 1e-8)
        return -(p * r).sum() / denom

    def _auc(sc, lb):
        o = np.argsort(sc); r = np.empty(len(sc)); r[o] = np.arange(len(sc))
        n1 = lb.sum(); n0 = len(lb) - n1
        return float((r[lb == 1].sum() - n1*(n1-1)/2) / (n1*n0)) if n1 > 20 and n0 > 20 else float("nan")

    def eval_oos():
        model.eval(); P, R, Y, PL, PS = [], [], [], [], []
        with torch.no_grad():
            for s in te:
                r = forward_stream(s, False)
                if r is None:
                    continue
                logits, dp = r
                P.append(logits.cpu().numpy()); R.append(s.rH[dp]); Y.append(s.y[dp])
                if cfg.head == "B2":
                    PL.append(s.pl[dp]); PS.append(s.ps[dp])
        if not P:
            return None
        P = np.concatenate(P); R = np.concatenate(R); Y = np.concatenate(Y)
        m = {"n_test": int(len(P))}
        if cfg.head == "A":
            order = np.argsort(-P); m["auc"] = _auc(P, Y)
            for q in (1.0, 0.5, 0.2):
                k = max(20, int(len(P)*q/100)); m[f"prec@{q}%"] = float(Y[order[:k]].mean())
        elif cfg.head == "B2":
            PL = np.concatenate(PL); PS = np.concatenate(PS)
            p = 1.0 / (1.0 + np.exp(-P))
            m["exec_soft_bp"] = float((p * PL + (1 - p) * PS).mean())   # the objective on val (best-val key)
            hard = np.where(P > 0, PL, PS)                              # deployed hard side
            m["exec_gross_bp"] = float(hard.mean()); m["wr"] = float((hard > 0).mean())
            m["exec_net_mm_bp"] = float(hard.mean() - cfg.comm_mm)
            for q in (20.0, 10.0, 5.0, 2.0):                           # selectivity by conviction |logit|
                k = max(20, int(len(P)*q/100)); top = np.argsort(-np.abs(P))[:k]
                m[f"net_mm@{q}%"] = float(hard[top].mean() - cfg.comm_mm)
        else:
            sgn = np.sign(P)
            for q in (50.0, 20.0, 10.0):
                k = max(20, int(len(P)*q/100)); top = np.argsort(-np.abs(P))[:k]
                m[f"dacc@{q}%"] = float(np.mean(sgn[top] == np.sign(R[top])))
                m[f"cap@{q}%bp"] = float(np.mean(sgn[top] * R[top]))
        return m

    # ALWAYS eval OOS every epoch + keep BEST-VAL model (never report last epoch).
    for ep in range(start_ep, cfg.epochs):
        model.train(); rng = np.random.default_rng(cfg.seed * 100 + ep)
        order = rng.permutation(len(tr)); tot = 0.0; nb = 0
        for i in order:
            r = forward_stream(tr[i], True)
            if r is None:
                continue
            logits, dp = r
            loss = loss_fn(logits, tr[i], dp)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        tl = tot / max(nb, 1); ev = eval_oos() or {}
        curve.append({"ep": ep, "train_loss": tl, **ev})
        log(f"[{cfg.head} ep{ep}] train_loss={tl:.4f} {pkey}={ev.get(pkey, float('nan')):.4f} "
            f"params={nparam:,}")
        if ev and np.isfinite(ev.get(pkey, np.nan)) and ev[pkey] > best:
            best = ev[pkey]; best_ep = ep; best_metrics = ev
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # persist the DEPLOYABLE best model: weights + standardization + cfg
            # (the asset). Atomic + commit -> survives preemption, usable for the
            # cascade/grid_sim and inference.
            if cfg.ckpt_path:
                bp = cfg.ckpt_path.replace(".ckpt", ".best.pt")
                torch.save({"model": best_state, "lob_mu": lob_mu, "lob_sd": lob_sd,
                            "ft_mu": ft_mu, "ft_sd": ft_sd, "cfg": dict(cfg.__dict__),
                            "F": int(F), "best_ep": ep, "metrics": ev}, bp + ".tmp")
                os.replace(bp + ".tmp", bp)
                if on_ckpt:
                    on_ckpt()
        _save_ckpt(ep)            # AFTER best-update -> ckpt carries the latest best/curve
        if cfg.patience and best_ep >= 0 and (ep - best_ep) >= cfg.patience:
            log(f"[early-stop] no {pkey} gain for {cfg.patience} ep (best ep{best_ep}={best:.4f})"); break

    if best_state is not None:
        model.load_state_dict(best_state)         # restore best-val weights
    res = {"head": cfg.head, "params": int(nparam), "best_ep": best_ep,
           "best_by": pkey, **best_metrics, "last_metrics": (curve[-1] if curve else {}),
           "curve": curve}
    log("[result-best] " + json.dumps({k: res[k] for k in res if k != "curve"}))
    return res


# ---------------------------------------------------------------- cli (local smoke)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="dir of {sym}/{day}.npz")
    ap.add_argument("--head", default="A", choices=["A", "B"])
    ap.add_argument("--cell", default="stub")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-days", type=int, default=8)
    a = ap.parse_args()
    syms = sorted(os.listdir(a.cache_dir))
    paths, ids = [], []
    for si, sym in enumerate(syms):
        ds = sorted(glob.glob(os.path.join(a.cache_dir, sym, "*.npz")))[: a.max_days]
        for di, p in enumerate(ds):
            paths.append(p); ids.append((si, di))
    streams = load_cache(paths, ids)
    print(f"[loaded] {len(streams)} streams, {len(syms)} syms")
    cfg = Cfg(head=a.head, cell=a.cell, device=a.device, epochs=a.epochs,
              pooled=(a.head == "B"))
    run(streams, cfg)
