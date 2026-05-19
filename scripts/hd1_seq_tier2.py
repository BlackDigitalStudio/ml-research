#!/usr/bin/env python3
"""HD1 Tier-2 sweep (FROZEN spec HD1 rev45; runner rev4 substrate).

Tier 2 probes the TCN LONG-CONTEXT axis: the conditional alpha surface
as a function of USABLE context length L (the literature/historical-best
lookback~1000 regime that rev32 flagged HD1-seq was structurally
under-tested on at the old MAX_L=512 cap).

PRIMARY DELIVERABLE (exploratory conditional-optimization, repo
CLAUDE.md rules 1-3): oos rank_IC mean +- seed-sd as a function of L per
(symbol,horizon) cell, the L that MAXIMIZES alpha per cell, its
magnitude and seed-stability -- the headline, ALWAYS. The §5 / delta-
vs-baseline machinery is a SECONDARY economic-deploy annotation only
(rules 2/4): kept verbatim, demoted, never the framing/verdict.

FROZEN rev45 axis & regime:
  * L in {512, 1024, 1536}, obtained by the right-causal slice
    X[:, -L:, :] of a SINGLE MAX_L=1536 pack (bit-identical to a native
    MAX_L=L build; gather is right-aligned causal with left zero-pad).
  * D forced per L to the min depth with receptive field
    RF = 1+4*(2^D-1) >= L  ->  FROZEN MAP {512:8, 1024:9, 1536:9}
    (D7 RF=509<512). This lifts the rev29 D=4 lock FOR TIER-2 ONLY
    (context-integration IS the Tier-2 axis); the L=512 point uses D=8
    and is the WITHIN-TIER-2 RF-matched anchor (NOT Tier-0/1's
    RF-starved D=4 512). The surface is alpha vs USABLE context.
  * Locked non-swept regime (rev44 outcome: no objective/head combo
    beat BCE|last; rev30/rev39/rev41 capacity lock): objective = bce
    (R1-weighted), head = last-step, W=16, dropout=0.5, wd=1e-3,
    kernel=3, dilation=2^b, Adam lr=1e-3; select+early-stop on VAL
    rank_ic, patience=2, restore-best, cosine T_max=12, abs cap 40.
  * Cells = rev32 4-cell cross-tier set; seeds {0,1,2}.
    Units = 4 cells x 3 L x 3 seeds = 36 (D determined by L).

FROZEN PARITY GUARD (rev45, MANDATORY pre-sweep, BINDING): the new
1536 pack sliced [:, -512:, :] MUST equal the existing persisted frozen
MAX_L=512 packed/{sym}.npz X bit-for-bit per symbol, and t0/n/labels
(which are L-independent) MUST be identical. Divergence = a BUILD BUG,
NOT a DOF -> ABORT, no sweep, no result. The frozen 512 pack is NOT
overwritten (the 1536 pack lands at a SEPARATE path).

Data contract (produced by the GCS->Modal transfer stage, NOT here):
  /cache/packed_l1536/{sym}_X.npy     (n,1536,46) f32, mmap-able
  /cache/packed_l1536/{sym}_meta.npz  t0,n,n_tr,y0_{H},rH_{H}
  /cache/packed/{sym}.npz             FROZEN 512 pack (parity ref)

Frozen hd1_seq_modal / hd1_seq_core are UNCHANGED; this runner only
adds the L axis + D(L) map + the rev45 parity guard. experiments.jsonl
untouched; rev28 refuted/shelved unchanged; HM1 continuous.

Run:  modal run scripts/hd1_seq_tier2.py            (guard->smoke->full)
      modal run scripts/hd1_seq_tier2.py --collect <run_id>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# rev32 4-cell cross-tier continuity set (L is now the swept axis).
CELLS = [("BTC-USDT-PERP", 180), ("BTC-USDT-PERP", 300),
         ("ETH-USDT-PERP", 180), ("LTC-USDT-PERP", 300)]
WEAK_CONTROL = ("LTC-USDT-PERP", 300)
SYMBOLS = sorted({s for s, _ in CELLS})            # BTC, ETH, LTC

# FROZEN rev45 primary axis + the receptive-field-forced D map.
L_GRID = (512, 1024, 1536)
D_FOR_L = {512: 8, 1024: 9, 1536: 9}               # min D with RF>=L
L_ANCHOR = 512                                     # RF-matched anchor

# Locked non-swept regime (rev44 outcome + rev30/rev39/rev41 lock).
OBJECTIVE, HEAD = "bce", "last"
W_FIXED, DROPOUT, WD = 16, 0.5, 1e-3
PATIENCE, MIN_DELTA, T_MAX, EP_CAP = 2, 1e-4, 12, 40
SEEDS = (0, 1, 2)

L4_USD_PER_S, BUDGET = 1.0e-3, 40.0
N_GPU = 4
SHM = "/tmp/hd1tier2"
PACK_L = 1536

_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
app = modal.App("hd1-tier2")
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"
PACK_DIR = f"{MNT}/packed_l1536"
FROZEN_512_DIR = f"{MNT}/packed"


def _build_tcn_t2(F, W, D, dropout):
    """Tier-2 TCN. Block topology + last-step head BYTE-IDENTICAL to the
    frozen hd1_seq_modal._build_tcn (head='last'); the ONLY variation vs
    Tier-0/1 is D (the rev45 RF>=L map), which the frozen Block stack
    already parameterizes (D blocks, dilation 2^b)."""
    import torch.nn as nn

    class Chomp(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, x):
            return x[:, :, :-self.c].contiguous() if self.c else x

    class Block(nn.Module):
        def __init__(self, ci, co, dil):
            super().__init__()
            pad = (3 - 1) * dil
            self.net = nn.Sequential(
                nn.Conv1d(ci, co, 3, padding=pad, dilation=dil),
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(co, co, 3, padding=pad, dilation=dil),
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
            self.down = nn.Conv1d(ci, co, 1) if ci != co else None
            self.relu = nn.ReLU()

        def forward(self, x):
            r = x if self.down is None else self.down(x)
            return self.relu(self.net(x) + r)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ci = [], F
            for b in range(D):
                layers.append(Block(ci, W, 2 ** b))
                ci = W
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(W, 1)

        def forward(self, x):                        # x: (B, L, F)
            h = self.tcn(x.transpose(1, 2))          # (B, W, L)
            return self.head(h[:, :, -1]).squeeze(-1)

    return TCN()


def _mark(run_id, name, txt):
    import os
    d = f"{MNT}/tier2/{run_id}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/{name}", "w").write(str(txt))
    os.makedirs(f"{MNT}/tier2", exist_ok=True)
    open(f"{MNT}/tier2/LATEST", "w").write(run_id)
    VOL.commit()


def _parity_guard_core():
    """rev45 BINDING guard. For each Tier-2 symbol assert the new 1536
    pack sliced [:,-512:,:] == the frozen 512 pack X bit-for-bit, and
    t0/n/labels (L-independent) identical. Returns (ok, report)."""
    import os
    import numpy as np
    rep = []
    ok = True
    for sym in SYMBOLS:
        xp = f"{PACK_DIR}/{sym}_X.npy"
        mp = f"{PACK_DIR}/{sym}_meta.npz"
        fp = f"{FROZEN_512_DIR}/{sym}.npz"
        if not (os.path.exists(xp) and os.path.exists(mp)):
            return False, f"{sym}: 1536 pack missing ({xp})"
        if not os.path.exists(fp):
            return False, (f"{sym}: FROZEN 512 pack missing ({fp}) -> "
                           f"cannot run the rev45 bit-exact guard; "
                           f"ABORT (do not skip the guard)")
        Xn = np.load(xp, mmap_mode="r")
        meta = np.load(mp)
        F = np.load(fp)
        if Xn.shape[1] != PACK_L:
            return False, f"{sym}: pack L={Xn.shape[1]} != {PACK_L}"
        A = np.ascontiguousarray(Xn[:, -512:, :])
        B = F["X"]
        if A.shape != B.shape:
            return False, (f"{sym}: slice shape {A.shape} != frozen "
                           f"512 {B.shape}")
        if not np.array_equal(A, B):
            d = int(np.argmax(np.any(A != B, axis=(1, 2))))
            return False, (f"{sym}: 1536[:,-512:] != frozen 512 X "
                           f"(first differing dp index {d}) -> BUILD "
                           f"BUG, not DOF (rev45)")
        for k in ("n",):
            if int(meta[k]) != int(F[k]):
                return False, f"{sym}: meta {k} != frozen 512 {k}"
        if not np.array_equal(meta["t0"], F["t0"]):
            return False, f"{sym}: t0 != frozen 512 t0"
        for H in (180, 300, 600):
            for lab in (f"y0_{H}", f"rH_{H}"):
                if not np.array_equal(meta[lab], F[lab]):
                    return False, f"{sym}: {lab} != frozen 512"
        rep.append(f"{sym}:OK(n={int(meta['n'])})")
    return ok, "PARITY_GUARD_OK " + " ".join(rep)


@app.function(image=GPU_IMG, timeout=1800, volumes={MNT: VOL})
def parity_guard(run_id: str):
    """Stand-alone fail-fast guard the entrypoint waits on BEFORE smoke
    so no GPU is ever spent on a mis-built pack. Also emits a memory
    projection (rev45 engineering-risk discipline: per-symbol n is
    unknown until the build DONE marker -> reconfirm tier2_all
    `memory=` against the worst-case L=1536 footprint before the paid
    run)."""
    import os
    import numpy as np
    VOL.reload()
    ok, msg = _parity_guard_core()
    _mark(run_id, "PARITY_OK" if ok else "PARITY_FAIL", msg)
    proj = []
    for sym in SYMBOLS:
        mp = f"{PACK_DIR}/{sym}_meta.npz"
        if os.path.exists(mp):
            n = int(np.load(mp)["n"])
            ub_gib = round(n * PACK_L * 46 * 4 / 2**30, 1)
            proj.append(f"{sym}: n={n} L1536-fullpack~{ub_gib}GiB "
                        f"(standardized fit/val/te subsets are a "
                        f"fraction; tier2_all memory=131072MB)")
    _mark(run_id, "MEM_PROJECTION", " | ".join(proj))
    return {"ok": ok, "msg": msg, "mem_projection": proj}


@app.function(image=GPU_IMG, gpu=f"L4:{N_GPU}", timeout=1800,
              volumes={MNT: VOL}, retries=2)
def smoke(run_id: str):
    """~$0 path probe on the EXACT instance (L4:N): ALIVE first
    (pre-import), then import the full stack + a 5-step tiny TCN at the
    DEEPEST D in the rev45 map (D=9) per visible GPU. Proves the
    multi-GPU allocation schedules and the deep long-context net builds
    and steps before the full run commits."""
    t = time.time()
    _mark(run_id, "SMOKE_ALIVE", t)
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    ngpu = torch.cuda.device_count()
    out = []
    Dmax = max(D_FOR_L.values())
    for gi in range(max(ngpu, 1)):
        dev = f"cuda:{gi}" if ngpu else "cpu"
        x = torch.randn(64, 1536, C.N_TICK_FEAT, device=dev)
        y = torch.randint(0, 2, (64,), device=dev).float()
        net = _build_tcn_t2(C.N_TICK_FEAT, W_FIXED, Dmax,
                            DROPOUT).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            loss = Fnn.binary_cross_entropy_with_logits(net(x), y)
            loss.backward()
            opt.step()
        out.append(f"g{gi}|D{Dmax}|L1536={round(float(loss),4)}")
    msg = (f"SMOKE_OK n_gpu={ngpu} {out[-N_GPU:]} "
           f"t={round(time.time()-t,2)}s")
    _mark(run_id, "SMOKE_OK", msg)
    return msg


def _pool_init(gpu_q):
    import os
    gid = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    os.environ["_HD1_GPU"] = str(gid)


def _unit_worker(task):
    """ONE (cell,L,seed) on this worker's pinned GPU. Training body is
    the Tier-0/1 BCE branch byte-for-byte (R1-weighted BCE, last-step
    head, rev31 schedule); the ONLY Tier-2 change is D = D_FOR_L[L]."""
    import os
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    (celld, key, L, D, seed) = task
    u0 = time.time()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    sd = celld["shm"]

    def _ld(n):
        return np.load(f"{sd}/{n}.npy", mmap_mode="r")

    Xfit = torch.from_numpy(np.ascontiguousarray(_ld("Xfit"))).to(dev)
    Xval = torch.from_numpy(np.ascontiguousarray(_ld("Xval"))).to(dev)
    Xte = torch.from_numpy(np.ascontiguousarray(_ld("Xte"))).to(dev)
    yfit = torch.from_numpy(np.ascontiguousarray(_ld("yfit"))).to(dev)
    wfit = torch.from_numpy(np.ascontiguousarray(_ld("wfit"))).to(dev)
    yval_i = np.ascontiguousarray(_ld("yval_i"))
    yte_i = np.ascontiguousarray(_ld("yte_i"))
    block = celld["block"]
    n_fit = Xfit.shape[0]
    idx = np.arange(n_fit)

    def _logits(net, Xg):
        o = []
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev != "cpu"):
            for s in range(0, Xg.shape[0], 16384):
                o.append(net(Xg[s:s + 16384]).float().cpu())
        return torch.cat(o)

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _build_tcn_t2(C.N_TICK_FEAT, W_FIXED, D, DROPOUT).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX)
    scaler = torch.amp.GradScaler("cuda", enabled=dev != "cpu")
    best_ric, pat, best_state = -1e9, 0, None
    for ep in range(EP_CAP):
        net.train()
        np.random.shuffle(idx)
        for s in range(0, n_fit, 1024):
            jt = torch.as_tensor(idx[s:s + 1024], device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev != "cpu"):
                lo = net(Xfit[jt])
                loss = (Fnn.binary_cross_entropy_with_logits(
                    lo, yfit[jt], reduction="none") * wfit[jt]
                ).sum() / (wfit[jt].sum() + 1e-9)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if ep < T_MAX:
            sch.step()
        net.eval()
        vp = _logits(net, Xval)
        v_ric = C.auc(yval_i, torch.sigmoid(vp).numpy()) - 0.5
        if v_ric > best_ric + MIN_DELTA:
            best_ric, pat = v_ric, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    p = torch.sigmoid(_logits(net, Xte)).numpy()
    se = C.block_bootstrap_auc_se(yte_i, p, block)
    return {"key": key, "sym": celld["sym"], "H": celld["H"],
            "L": L, "D": D, "obj": OBJECTIVE, "head": HEAD,
            "W": W_FIXED, "dropout": DROPOUT, "wd": WD,
            "seed": seed, "ric": round(C.auc(yte_i, p) - 0.5, 6),
            "placebo_ric": round(C.placebo_auc(yte_i, p) - 0.5, 6),
            "boot_se": (None if not np.isfinite(se)
                        else round(float(se), 6)),
            "n_tr": celld["ntr"], "n_oos": celld["nte"],
            "block": block, "gpu_s": round(time.time() - u0, 2),
            "gpu": os.environ.get("_HD1_GPU", "?")}


@app.function(image=GPU_IMG, gpu=f"L4:{N_GPU}", timeout=21600,
              memory=131072, volumes={MNT: VOL}, retries=3)
def tier2_all(run_id: str):
    """ONE container, 4 GPUs. rev45 parity guard FIRST (abort on fail).
    Per symbol: mmap {sym}_X.npy + load meta. Per (H,L): right-causal
    slice X[:,-L:,:], standardize on fit rows (train-split stats only),
    /tmp memmap, ProcessPool(4) dispatches the L x seeds grid at
    D=D_FOR_L[L]. Per-unit Volume checkpoint + resume-skip."""
    import os
    import shutil
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t_run = time.time()
    _mark(run_id, "ALIVE", t_run)
    import numpy as np
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    VOL.reload()
    ok, msg = _parity_guard_core()
    if not ok:
        _mark(run_id, "ABORT", f"PARITY_GUARD_FAIL: {msg}")
        return {"run_id": run_id, "aborted": True, "msg": msg}
    _mark(run_id, "PARITY_OK", msg)

    pdir = f"{MNT}/tier2/{run_id}/parts"
    os.makedirs(pdir, exist_ok=True)
    per_cell = len(L_GRID) * len(SEEDS)
    tot = len(CELLS) * per_cell

    def _hb(extra=""):
        d = sum(sum(1 for _ in open(f"{pdir}/{fn}"))
                for fn in (os.listdir(pdir) if os.path.isdir(pdir)
                           else []))
        _mark(run_id, "HB",
              f"units={d}/{tot} t={round(time.time()-t_run)}s {extra}")

    _hb(f"start ngpu={N_GPU}")
    ctx = mp.get_context("spawn")
    groups = {}
    for (s, H) in CELLS:
        groups.setdefault(s, []).append(H)

    for sym, Hs in groups.items():
        ppath = f"{pdir}/{sym}.jsonl"
        done = set()
        if os.path.exists(ppath):
            for ln in open(ppath):
                try:
                    done.add(json.loads(ln)["key"])
                except Exception:
                    pass
        Xfull = np.load(f"{PACK_DIR}/{sym}_X.npy", mmap_mode="r")
        meta = np.load(f"{PACK_DIR}/{sym}_meta.npz")
        n = int(meta["n"])
        tr, te, _ = C.honest_split(n)
        for H in Hs:
            y0 = meta[f"y0_{H}"]
            rH = meta[f"rH_{H}"].astype(np.float64)
            reached = (y0 != 0) & np.isfinite(rH)
            up = (y0 == 1).astype(np.float32)
            s_tr_all = tr & reached
            s_te = te & reached
            ntr, nte = int(s_tr_all.sum()), int(s_te.sum())
            fit_m, val_m = C.train_val_split(s_tr_all)
            w1 = C.r1_weights(rH, s_tr_all).astype(np.float32)
            for L in L_GRID:
                keys = [f"L{L}|s{seed}" for seed in SEEDS]
                if all(k in done for k in keys):
                    continue
                XL = Xfull[:, -L:, :]
                fr = np.ascontiguousarray(
                    XL[fit_m]).reshape(-1, C.N_TICK_FEAT)
                mu = fr.mean(0).astype(np.float32)
                sg = fr.std(0).astype(np.float32) + 1e-6
                del fr
                sdp = f"{SHM}/{run_id}/{sym}_{H}_L{L}"
                os.makedirs(sdp, exist_ok=True)
                np.save(f"{sdp}/Xfit.npy",
                        (np.ascontiguousarray(XL[fit_m]).astype(
                            np.float32) - mu) / sg)
                np.save(f"{sdp}/Xval.npy",
                        (np.ascontiguousarray(XL[val_m]).astype(
                            np.float32) - mu) / sg)
                np.save(f"{sdp}/Xte.npy",
                        (np.ascontiguousarray(XL[s_te]).astype(
                            np.float32) - mu) / sg)
                np.save(f"{sdp}/yfit.npy", up[fit_m])
                np.save(f"{sdp}/wfit.npy", w1[fit_m])
                np.save(f"{sdp}/yval_i.npy", up[val_m].astype(int))
                np.save(f"{sdp}/yte_i.npy", up[s_te].astype(int))
                celld = {"sym": sym, "H": H, "L": L,
                         "block": C.block_size(H),
                         "ntr": ntr, "nte": nte, "shm": sdp}
                tasks = [(celld, f"L{L}|s{seed}", L, D_FOR_L[L], seed)
                         for seed in SEEDS
                         if f"L{L}|s{seed}" not in done]
                gq = ctx.Queue()
                for g in range(N_GPU):
                    gq.put(g)
                done_n = 0
                with ProcessPoolExecutor(
                        max_workers=N_GPU, mp_context=ctx,
                        initializer=_pool_init,
                        initargs=(gq,)) as ex:
                    futs = [ex.submit(_unit_worker, t) for t in tasks]
                    for fu in as_completed(futs):
                        rec = fu.result()
                        with open(ppath, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        done.add(rec["key"])
                        done_n += 1
                        VOL.commit()
                        _hb(f"{sym}-H{H}-L{L}")
                shutil.rmtree(sdp, ignore_errors=True)
                _hb(f"{sym}-H{H}-L{L} cell-done")
        del Xfull, meta
    _finalize(run_id, t_run)
    return {"run_id": run_id, "done": True}


def _finalize(run_id, t_run):
    import os
    import numpy as np
    sys.path.insert(0, "/root/proj")
    from scripts.hd1_seq_modal import BASELINE_REF_RIC
    from scripts import hd1_seq_core as C

    pdir = f"{MNT}/tier2/{run_id}/parts"
    units = []
    if os.path.isdir(pdir):
        for fn in os.listdir(pdir):
            for ln in open(f"{pdir}/{fn}"):
                try:
                    units.append(json.loads(ln))
                except Exception:
                    pass
    spent = round(sum(u.get("gpu_s", 0) for u in units) *
                  L4_USD_PER_S, 2)
    per_cell = len(L_GRID) * len(SEEDS)

    def _stat(cu, L):
        us = [u for u in cu if u["L"] == L]
        if len(us) < len(SEEDS):
            return None
        a = np.array([x["ric"] for x in us], float)
        return {"ric_mean": round(float(a.mean()), 6),
                "ric_sd": round(float(a.std()), 6),
                "ric_seeds": sorted(round(x["ric"], 6) for x in us),
                "placebo_ric": us[0]["placebo_ric"],
                "boot_se": us[0]["boot_se"]}

    # ---- PRIMARY: the conditional alpha surface over L (headline) ----
    alpha_surface, incomplete = [], []
    for (s, H) in CELLS:
        cu = [u for u in units if u["sym"] == s and u["H"] == H]
        if len(cu) < per_cell:
            incomplete.append(f"{s}-H{H}({len(cu)}/{per_cell})")
        per_L = {}
        for L in L_GRID:
            st = _stat(cu, L)
            if st is not None:
                per_L[L] = st
        if not per_L:
            alpha_surface.append({"sym": s, "H": H,
                                  "error": "incomplete"})
            continue
        ranked = sorted(per_L.items(),
                        key=lambda kv: kv[1]["ric_mean"],
                        reverse=True)
        Lmax, emax = ranked[0]
        alpha_surface.append({
            "sym": s, "H": H,
            "is_weak_control": (s, H) == WEAK_CONTROL,
            "alpha_max_L": Lmax,
            "alpha_max_D": D_FOR_L[Lmax],
            "alpha_max_ric_mean": emax["ric_mean"],
            "alpha_max_seed_sd": emax["ric_sd"],
            "alpha_max_seeds": emax["ric_seeds"],
            "ranked_surface": [
                {"L": L, "D": D_FOR_L[L],
                 "ric_mean": e["ric_mean"], "seed_sd": e["ric_sd"],
                 "ric_seeds": e["ric_seeds"]}
                for L, e in sorted(per_L.items())]})

    valid = [a for a in alpha_surface if "error" not in a]
    from collections import Counter
    best_L = {
        "per_cell_alpha_max": {
            f"{a['sym'].split('-')[0]}-H{a['H']}"
            + ("[weak]" if a["is_weak_control"] else ""):
            f"L{a['alpha_max_L']}(D{a['alpha_max_D']}) "
            f"{a['alpha_max_ric_mean']:+.4f}"
            f"±{a['alpha_max_seed_sd']:.4f}" for a in valid},
        "alpha_max_L_vote": dict(
            Counter(a["alpha_max_L"] for a in valid)),
        "reading": ("the context length L that MAXIMIZES rank_IC per "
                    "(symbol,horizon), its magnitude and seed-"
                    "stability -- this IS the Tier-2 result (CLAUDE.md "
                    "rule 1). Long-context helps iff alpha_max_L>512 "
                    "with a seed-stable margin over the L512 anchor.")}

    # ---- SECONDARY economic-deploy annotation (DEMOTED, rules 2/4) ---
    # (i) rev45 pre-registered continuous lift vs the L=512 RF-matched
    #     anchor; (ii) frozen §5 (gate_cell/status_for_cell vs
    #     BASELINE_REF_RIC) on the alpha-max-L config, as a DIAGNOSTIC.
    deploy = []
    for a in valid:
        s, H = a["sym"], a["H"]
        byL = {r["L"]: r for r in a["ranked_surface"]}
        anc = byL.get(L_ANCHOR)
        amax = byL[a["alpha_max_L"]]
        lift = (None if anc is None else
                round(amax["ric_mean"] - anc["ric_mean"], 6))
        base = BASELINE_REF_RIC.get((s, H))
        d_ic = (None if base is None else
                C.cell_delta_ic(amax["ric_mean"], base))
        cu = [u for u in units if u["sym"] == s and u["H"] == H
              and u["L"] == a["alpha_max_L"]]
        plac = cu[0]["placebo_ric"] if cu else None
        bse = cu[0]["boot_se"] if cu else None
        passes_ab, why = (C.gate_cell(d_ic, plac, bse)
                          if d_ic is not None else (False, "no base"))
        deploy.append({
            "cell": f"{s}-H{H}", "is_weak_control": a["is_weak_control"],
            "alpha_max_L": a["alpha_max_L"],
            "lift_vs_L512_anchor": lift,
            "anchor_L512_ric_mean": None if anc is None
            else anc["ric_mean"],
            "delta_ic_vs_baseline_ref": d_ic,
            "passes_5ab": passes_ab, "gate_reason": why})

    doc = {"run_id": run_id,
           "spec": "HD1 rev45 Tier-2 (FROZEN); runner rev4 substrate; "
                   "single MAX_L=1536 pack, L right-causal-sliced, "
                   "D forced per L by RF>=L {512:8,1024:9,1536:9}",
           "PRIMARY_alpha_surface": alpha_surface,
           "PRIMARY_best_L_per_cell": best_L,
           "locked": {"objective": OBJECTIVE, "head": HEAD,
                      "W": W_FIXED, "dropout": DROPOUT, "wd": WD,
                      "D_for_L": D_FOR_L, "L_anchor": L_ANCHOR,
                      "note": "rev44 outcome (bce|last) + rev30/rev39/"
                              "rev41 capacity lock; D lifted per rev45 "
                              "RF>=L for Tier-2 only"},
           "secondary_economic_deploy_annotation": {
               "purpose": "SEPARATE confirmatory deploy question "
                          "(HM1 §5 family) -- NOT the tier conclusion "
                          "(CLAUDE.md rule 2). Pre-registered rev45 "
                          "continuous lift vs the L512 RF-matched "
                          "anchor + the frozen §5 gate on the "
                          "alpha-max-L config, as a DIAGNOSTIC, kept "
                          "verbatim and demoted (rule 4). rev28 "
                          "refuted/shelved + experiments.jsonl "
                          "UNCHANGED; HM1 continuous, not a §5 verdict.",
               "per_cell": deploy},
           "incomplete_cells": incomplete,
           "approx_gpu_usd": spent, "n_units": len(units),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime())}
    _mark(run_id, "tier2.json", json.dumps(doc, indent=2, default=str))
    if spent > BUDGET:
        _mark(run_id, "ABORT", f"spent~${spent}>{BUDGET} (partial ok)")
    else:
        _mark(run_id, "DONE", run_id)
    return doc


def _vol_text(p):
    import subprocess
    g = subprocess.run([sys.executable, "-m", "modal", "volume", "get",
                        "hd1-seq-cache", p, "-"],
                       capture_output=True, text=True)
    return g.stdout if g.returncode == 0 else None


def _ls(p):
    import subprocess
    return (subprocess.run([sys.executable, "-m", "modal", "volume",
                            "ls", "hd1-seq-cache", p],
                           capture_output=True, text=True).stdout or "")


def _collect(run_id):
    import subprocess
    import tempfile
    import re
    m = re.findall(r"tier2-\d{8}-\d{6}", run_id)
    run_id = m[0] if m else run_id
    tmp = tempfile.mkdtemp(prefix="tier2_")
    subprocess.run([sys.executable, "-m", "modal", "volume", "get",
                    "hd1-seq-cache", f"/tier2/{run_id}", tmp],
                   check=True)
    doc = json.loads(next(Path(tmp).rglob("tier2.json")).read_text())
    art = REPO / "research_runs" / run_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "tier2.json").write_text(json.dumps(doc, indent=2,
                                               default=str))
    # Headline = the conditional alpha surface over L (CLAUDE.md 1/3).
    asf = doc["PRIMARY_alpha_surface"]
    surf = " ;; ".join(
        f"{a['sym'].split('-')[0]}-H{a['H']}"
        + ("[weak]" if a.get("is_weak_control") else "")
        + f": alpha-max L{a['alpha_max_L']}(D{a['alpha_max_D']})="
        f"{a['alpha_max_ric_mean']:+.4f}±{a['alpha_max_seed_sd']:.4f}"
        f"; surface=[" + ", ".join(
            f"L{x['L']} {x['ric_mean']:+.4f}±{x['seed_sd']:.4f}"
            for x in a["ranked_surface"]) + "]"
        for a in asf if "error" not in a)
    dep = doc["secondary_economic_deploy_annotation"]
    inc = doc.get("incomplete_cells")
    rec = {"hypothesis_id": "HD1", "rev": 46,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "statement": (
               f"TIER-2 CONDITIONAL ALPHA SURFACE over context length "
               f"(run {run_id}, FROZEN rev45 spec, runner rev4; single "
               f"MAX_L=1536 pack right-causal-sliced; D forced per L "
               f"by RF>=L {{512:8,1024:9,1536:9}}; locked bce|last "
               f"W=16 do0.5 wd1e-3; rev45 bit-exact parity guard "
               f"PASSED). PRIMARY -- rank_IC mean±seed-sd vs USABLE "
               f"context L, and the alpha-max L per (symbol,horizon) "
               f"condition: {surf}. incomplete={inc}. approx GPU "
               f"${doc['approx_gpu_usd']} ({doc['n_units']} units). "
               f"SECONDARY economic-deploy annotation (separate HM1 "
               f"§5 question, NOT the tier conclusion -- CLAUDE.md "
               f"rule 2/4; continuous lift vs the L512 RF-matched "
               f"anchor + frozen §5 diagnostic): {dep['per_cell']}"),
           "status": "testing", "priority_rank": 1,
           "result_experiment_id": run_id,
           "note": (f"Tier-2 PRIMARY = conditional alpha surface over "
                    f"L (rank_IC vs usable context per cell + "
                    f"alpha-max L). Deploy-gate demoted to a secondary "
                    f"annotation per CLAUDE.md (kept, not deleted). "
                    f"rev45 1536[:,-512:]==frozen512 parity guard "
                    f"binding. experiments.jsonl untouched; "
                    f"rev28/rev32/rev44/rev45 unchanged.")}
    with open(REPO / "research" / "hypotheses.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[tier2] alpha-surface: {surf[:140]} -> "
          f"{art/'tier2.json'} ; rev46 appended")


@app.local_entrypoint()
def main(collect: str = "", skip_smoke: int = 0):
    if collect:
        _collect(collect)
        print("local entrypoint completed")
        return
    sys.path.insert(0, str(REPO))
    from scripts.hd1_seq_modal import _parity_gate
    _parity_gate()                                   # frozen Py+Rust
    import time as _t
    run_id = f"tier2-{_t.strftime('%Y%m%d-%H%M%S', _t.gmtime())}"

    # rev45 BINDING guard: 1536[:,-512:] == frozen 512, fail-fast
    # BEFORE any GPU spend.
    parity_guard.spawn(run_id)
    print(f"[guard] spawned run_id={run_id}")
    t0, gok = _t.time(), False
    while _t.time() - t0 < 1500:
        o = _ls(f"/tier2/{run_id}")
        if "PARITY_FAIL" in o:
            print("[guard] FAIL: " + (
                _vol_text(f"/tier2/{run_id}/PARITY_FAIL") or "").strip())
            raise SystemExit(4)
        if "PARITY_OK" in o:
            print("[guard] OK: " + (
                _vol_text(f"/tier2/{run_id}/PARITY_OK") or "").strip())
            gok = True
            break
        _t.sleep(15)
    if not gok:
        print("[guard] no PARITY_OK in 25min -> NOT launching.")
        raise SystemExit(4)

    if not skip_smoke:
        smoke.spawn(run_id)
        print(f"[smoke] spawned run_id={run_id} (L4:{N_GPU})")
        t0, ok = _t.time(), False
        while _t.time() - t0 < 1500:
            o = _ls(f"/tier2/{run_id}")
            if "SMOKE_ALIVE" in o:
                print("[smoke] ALIVE (multi-GPU container reached code)")
            if "SMOKE_OK" in o:
                print("[smoke] OK: " + (
                    _vol_text(f"/tier2/{run_id}/SMOKE_OK") or "").strip())
                ok = True
                break
            _t.sleep(20)
        if not ok:
            print("[smoke] FAILED (no SMOKE_OK in 25min) -> NOT "
                  "launching full run.")
            raise SystemExit(3)
    tier2_all.spawn(run_id)
    print(f"[tier2] spawned run_id={run_id} (1 container, {N_GPU} GPU)")
    t0 = _t.time()
    while _t.time() - t0 < 7 * 3600:
        o = _ls(f"/tier2/{run_id}")
        if "ABORT" in o or "DONE" in o:
            _collect(run_id)
            print("local entrypoint completed"
                  + (" (budget-capped)" if "ABORT" in o else ""))
            return
        _t.sleep(30)
    print(f"[poll] still running; collect later: --collect {run_id}.")
