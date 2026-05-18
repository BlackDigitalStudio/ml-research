#!/usr/bin/env python3
"""HD1 Tier-1 sweep (FROZEN spec HD1 rev43; runner rev4 substrate).

Tier 1 probes the TCN OBJECTIVE/HEAD axis -- the last "have we
really exhausted TCN" lever before escalating "TCN ~= snapshot
fundamentally" (rev28/rev30/rev39). It REUSES the rev4 multi-GPU
substrate verbatim (ONE container gpu='L4:4', spawn ProcessPool 1
proc/GPU, /tmp memmap shm, per-unit Volume checkpoint/resume, ALIVE
marker, $0 smoke gate, local parity gate). The frozen numeric core
(label/scope/split/r1-weights/AUC/bootstrap, hd1_seq_core) is
UNCHANGED -- only TRAINING (objective, head) varies; EVAL is
byte-identical to Tier-0 (rank_IC = auc(up/down)-0.5).

FROZEN 2x2 (rev43):
  objective {bce [=reference], pairwise} x head {last [=reference], mean}
at LOCKED W=16, D=4 and the rev30 principled-default regularization
that ACCOMPANIES the W=16 lock (Tier-0 lock recommendation verbatim:
"smallest W (16), cheapest + MOST REGULARIZED per rev30") =>
dropout=0.5, wd=1e-3. Frozen rev32 data regime: same 4 cells, same
honest split / r1-weights / standardization / EP_CAP/PATIENCE/T_MAX.
Seeds (0,1,2). Units = 4 combos x 4 cells x 3 seeds = 48.

FROZEN operator defs (rev43):
  * pairwise = RankNet over ALL in-batch pairs with sign(rH_i-rH_j)!=0,
    target=1{rH_i>rH_j}, ranking key = CONTINUOUS rH (NOT the binary
    up/down label), loss = mean BCE(sigmoid(s_i-s_j), target), s =
    model scalar score; UNWEIGHTED pair sampling (no r1 weights on the
    pairwise branch). The (B,B) score difference is computed in fp32
    for numerical stability (deliberate, at the new operator).
  * bce branch = UNCHANGED from Tier-0 (r1-weighted BCE on up/down).
  * head: last = h[:,:,-1] (== frozen _build_tcn); mean = h.mean(2)
    over the temporal axis before the final linear. Block topology is
    otherwise byte-identical to the frozen _build_tcn.

PRE-REGISTERED decision rule (rev43): in-run (bce,last) is the
REFERENCE corner (computed FRESH this run -> controls the CUDA
run-to-run noise quantified in rev41). For each non-ref combo C and
cell: lift = ric_mean(C,cell) - ric_mean(bce|last,cell) (BOTH from
THIS run); C "wins" a cell iff lift >= +0.004 AND seed-sd(C,cell) <=
0.004; C is a REAL EFFECT iff it wins on >=3/4 cells INCLUDING the
weak control LTC-H300. seed-sd ALWAYS reported; reliance on
3-seed-mean + 0.004 noise-floor + weak-control (rev41: robust to
CUDA nondeterminism -> NO torch deterministic-algo flag).

PRE-REGISTERED rev42 interaction clause: if any combo is a REAL
EFFECT, a confirmatory re-probe of W in {16,32,64} at that winning
(objective,head) is MANDATORY before any final architecture lock
(capacity x objective/head interaction). Emitted as a binding
recommendation in tier1.json.

OUTCOME framing (HM1 continuous, NOT a §5 hard gate, rev35): NO combo
a real effect => objective/head axis also inert at locked capacity in
this regime => materially strengthens "TCN ~= snapshot fundamentally"
(escalate). Some combo a real effect => that is the lever => execute
the rev42(b) W re-probe, then lock. NO experiments.jsonl writes;
rev32/rev35/rev28/rev39/rev41/rev42 ALL UNCHANGED.

Run:  modal run scripts/hd1_seq_tier1.py            (smoke -> full)
      modal run scripts/hd1_seq_tier1.py --collect <run_id>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

CELLS = [("BTC-USDT-PERP", 180, 512), ("BTC-USDT-PERP", 300, 512),
         ("ETH-USDT-PERP", 180, 256), ("LTC-USDT-PERP", 300, 128)]
WEAK_CONTROL = ("LTC-USDT-PERP", 300)
# FROZEN rev43 axes (2x2). Reference corner = ("bce","last").
OBJECTIVES = ("bce", "pairwise")
HEADS = ("last", "mean")
REF_COMBO = ("bce", "last")
SEEDS = (0, 1, 2)
# LOCKED capacity (rev39/rev41) + rev30 principled-default reg that
# accompanies the W=16 lock (Tier-0 lock recommendation, verbatim).
W_FIXED, D_FIXED, DROPOUT, WD = 16, 4, 0.5, 1e-3
PATIENCE, MIN_DELTA, T_MAX, EP_CAP = 2, 1e-4, 12, 40
LIFT_THRESH = 0.004
L4_USD_PER_S, BUDGET = 1.0e-3, 12.0
N_GPU = 4
SHM = "/tmp/hd1tier1"

_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
app = modal.App("hd1-tier1")
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"


def _build_tcn_t1(F, W, D, dropout, head):
    """Tier-1 TCN. Block topology BYTE-IDENTICAL to the frozen
    hd1_seq_modal._build_tcn; the ONLY variation is the temporal
    reduction `head`: 'last' = h[:,:,-1] (== frozen), 'mean' =
    h.mean(2) (FROZEN rev43 mean-pool, arithmetic mean over the
    temporal axis before the final linear)."""
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
            self._mode = head

        def forward(self, x):                    # x: (B, L, F)
            h = self.tcn(x.transpose(1, 2))      # (B, W, L)
            z = h[:, :, -1] if self._mode == "last" else h.mean(2)
            return self.head(z).squeeze(-1)

    return TCN()


def _mark(run_id, name, txt):
    import os
    d = f"{MNT}/tier1/{run_id}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/{name}", "w").write(str(txt))
    os.makedirs(f"{MNT}/tier1", exist_ok=True)
    open(f"{MNT}/tier1/LATEST", "w").write(run_id)
    VOL.commit()


@app.function(image=GPU_IMG, gpu=f"L4:{N_GPU}", timeout=1800,
              volumes={MNT: VOL}, retries=2)
def smoke(run_id: str):
    """~$0 path probe on the EXACT instance (L4:N): ALIVE first
    (pre-import), then import the full stack + a 5-step tiny TCN for
    EACH (objective,head) combo per visible GPU + Volume write.
    Proves the multi-GPU allocation schedules and BOTH new operators
    (pairwise loss, mean-pool head) run before the full run commits."""
    t = time.time()
    _mark(run_id, "SMOKE_ALIVE", t)
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    ngpu = torch.cuda.device_count()
    out = []
    for gi in range(max(ngpu, 1)):
        dev = f"cuda:{gi}" if ngpu else "cpu"
        x = torch.randn(256, 64, C.N_TICK_FEAT, device=dev)
        y = torch.randint(0, 2, (256,), device=dev).float()
        r = torch.randn(256, device=dev)
        for obj in OBJECTIVES:
            for hd in HEADS:
                net = _build_tcn_t1(C.N_TICK_FEAT, 32, D_FIXED,
                                    DROPOUT, hd).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    s = net(x)
                    if obj == "bce":
                        loss = Fnn.binary_cross_entropy_with_logits(s, y)
                    else:
                        sf = s.float()
                        ds = sf[:, None] - sf[None, :]
                        dr = r[:, None] - r[None, :]
                        m = dr != 0
                        loss = Fnn.binary_cross_entropy_with_logits(
                            ds[m], (dr[m] > 0).float())
                    loss.backward()
                    opt.step()
                out.append(f"{obj}|{hd}={round(float(loss),4)}")
    msg = (f"SMOKE_OK n_gpu={ngpu} last_gpu_combos={out[-4:]} "
           f"t={round(time.time()-t,2)}s")
    _mark(run_id, "SMOKE_OK", msg)
    return msg


def _pool_init(gpu_q):
    """ProcessPool initializer: pin THIS worker to one GPU by setting
    CUDA_VISIBLE_DEVICES BEFORE torch is imported in the worker."""
    import os
    gid = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    os.environ["_HD1_GPU"] = str(gid)


def _unit_worker(task):
    """Runs ONE (cell,combo,seed) on this worker's pinned GPU. Inner
    training mirrors Tier-0's rev3 body byte-for-byte for the bce
    branch; the pairwise branch + mean-pool head are the ONLY frozen
    rev43 additions. Process isolation => global RNG re-seeded per
    unit reproduces the Tier-0 numeric regime."""
    import os
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C

    (celld, key, obj, hd, seed) = task
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
    rfit = torch.from_numpy(np.ascontiguousarray(_ld("rfit"))).to(dev)
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
    net = _build_tcn_t1(C.N_TICK_FEAT, W_FIXED, D_FIXED, DROPOUT,
                        hd).to(dev)
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
                if obj == "bce":
                    loss = (Fnn.binary_cross_entropy_with_logits(
                        lo, yfit[jt], reduction="none") * wfit[jt]
                    ).sum() / (wfit[jt].sum() + 1e-9)
                else:
                    # FROZEN rev43 RankNet: all in-batch pairs with
                    # sign(rH_i-rH_j)!=0, target=1{rH_i>rH_j}, key =
                    # continuous rH; unweighted; (B,B) diff in fp32.
                    sf = lo.float()
                    rj = rfit[jt]
                    ds = sf[:, None] - sf[None, :]
                    dr = rj[:, None] - rj[None, :]
                    m = dr != 0
                    if m.any():
                        loss = Fnn.binary_cross_entropy_with_logits(
                            ds[m], (dr[m] > 0).float())
                    else:
                        loss = (sf * 0.0).sum()
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
            "L": celld["L"], "obj": obj, "head": hd,
            "W": W_FIXED, "dropout": DROPOUT, "wd": WD,
            "seed": seed, "ric": round(C.auc(yte_i, p) - 0.5, 6),
            "placebo_ric": round(C.placebo_auc(yte_i, p) - 0.5, 6),
            "boot_se": (None if not np.isfinite(se)
                        else round(float(se), 6)),
            "n_tr": celld["ntr"], "n_oos": celld["nte"],
            "block": block, "gpu_s": round(time.time() - u0, 2),
            "gpu": os.environ.get("_HD1_GPU", "?")}


@app.function(image=GPU_IMG, gpu=f"L4:{N_GPU}", timeout=21600,
              memory=65536, volumes={MNT: VOL}, retries=3)
def tier1_all(run_id: str):
    """ONE container, 4 GPUs. ALIVE before import. Per cell: CPU
    standardize once -> /tmp memmap -> ProcessPool(4, GPU-pinned)
    dispatches the 2x2xseeds grid. Per-unit Volume checkpoint +
    resume-skip. Aggregates from durable parts; writes tier1.json +
    DONE."""
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
    pdir = f"{MNT}/tier1/{run_id}/parts"
    os.makedirs(pdir, exist_ok=True)
    groups = {}
    for (s, H, L) in CELLS:
        groups.setdefault(s, []).append((H, L))
    per_cell = len(OBJECTIVES) * len(HEADS) * len(SEEDS)
    tot = len(CELLS) * per_cell

    def _hb(extra=""):
        d = sum(sum(1 for _ in open(f"{pdir}/{fn}"))
                for fn in (os.listdir(pdir) if os.path.isdir(pdir)
                           else []))
        _mark(run_id, "HB",
              f"units={d}/{tot} t={round(time.time()-t_run)}s {extra}")

    _hb(f"start ngpu={N_GPU}")
    ctx = mp.get_context("spawn")

    for sym, hl in groups.items():
        ppath = f"{pdir}/{sym}.jsonl"
        done = set()
        if os.path.exists(ppath):
            for ln in open(ppath):
                try:
                    done.add(json.loads(ln)["key"])
                except Exception:
                    pass
        P = np.load(f"{MNT}/packed/{sym}.npz")
        Xfull = P["X"]
        n = int(P["n"])
        for (H, L) in hl:
            XL = Xfull if L == C.MAX_L else np.ascontiguousarray(
                Xfull[:, -L:, :])
            tr, te, _ = C.honest_split(n)
            y0 = P[f"y0_{H}"]
            rH = P[f"rH_{H}"].astype(np.float64)
            reached = (y0 != 0) & np.isfinite(rH)
            up = (y0 == 1).astype(np.float32)
            s_tr_all = tr & reached
            s_te = te & reached
            ntr, nte = int(s_tr_all.sum()), int(s_te.sum())
            fit_m, val_m = C.train_val_split(s_tr_all)
            w1 = C.r1_weights(rH, s_tr_all).astype(np.float32)
            fr = XL[fit_m].reshape(-1, C.N_TICK_FEAT)
            mu = fr.mean(0).astype(np.float32)
            sg = fr.std(0).astype(np.float32) + 1e-6
            sd = f"{SHM}/{run_id}/{sym}_{H}"
            os.makedirs(sd, exist_ok=True)
            np.save(f"{sd}/Xfit.npy",
                    (XL[fit_m].astype(np.float32) - mu) / sg)
            np.save(f"{sd}/Xval.npy",
                    (XL[val_m].astype(np.float32) - mu) / sg)
            np.save(f"{sd}/Xte.npy",
                    (XL[s_te].astype(np.float32) - mu) / sg)
            np.save(f"{sd}/yfit.npy", up[fit_m])
            np.save(f"{sd}/wfit.npy", w1[fit_m])
            np.save(f"{sd}/rfit.npy",
                    rH[fit_m].astype(np.float32))
            np.save(f"{sd}/yval_i.npy", up[val_m].astype(int))
            np.save(f"{sd}/yte_i.npy", up[s_te].astype(int))
            celld = {"sym": sym, "H": H, "L": L,
                     "block": C.block_size(H),
                     "ntr": ntr, "nte": nte, "shm": sd}
            tasks = [(celld, f"H{H}|{obj}|{hd}|s{seed}", obj, hd, seed)
                     for obj in OBJECTIVES for hd in HEADS
                     for seed in SEEDS
                     if f"H{H}|{obj}|{hd}|s{seed}" not in done]
            if tasks:
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
                        if done_n % N_GPU == 0:
                            VOL.commit()
                            _hb(f"{sym}-H{H}")
                VOL.commit()
            shutil.rmtree(sd, ignore_errors=True)
            _hb(f"{sym}-H{H} cell-done")
        del Xfull, P
    _finalize(run_id, t_run)
    return {"run_id": run_id, "done": True}


def _finalize(run_id, t_run):
    import os
    import numpy as np
    pdir = f"{MNT}/tier1/{run_id}/parts"
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
    per_cell = len(OBJECTIVES) * len(HEADS) * len(SEEDS)
    combos = [(o, h) for o in OBJECTIVES for h in HEADS]

    def _stat(cu, o, h):
        us = [u for u in cu if u["obj"] == o and u["head"] == h]
        if len(us) < len(SEEDS):
            return None
        a = np.array([x["ric"] for x in us], float)
        return {"ric_mean": round(float(a.mean()), 6),
                "ric_sd": round(float(a.std()), 6),
                "ric_seeds": sorted(round(x["ric"], 6) for x in us),
                "placebo_ric": us[0]["placebo_ric"],
                "boot_se": us[0]["boot_se"]}

    rows, incomplete = [], []
    # win[combo] -> list of cells won (incl weak control flag)
    wins = {c: [] for c in combos if c != REF_COMBO}
    ctrl_wins = {c: False for c in combos if c != REF_COMBO}
    for (s, H, L) in CELLS:
        cu = [u for u in units if u["sym"] == s and u["H"] == H]
        if len(cu) < per_cell:
            incomplete.append(f"{s}-H{H}({len(cu)}/{per_cell})")
        ref = _stat(cu, *REF_COMBO)
        cellrow = {"sym": s, "H": H, "L": L,
                   "is_weak_control": (s, H) == WEAK_CONTROL,
                   "ref_combo": "|".join(REF_COMBO),
                   "ref_ric_mean": None if ref is None
                   else ref["ric_mean"],
                   "ref_ric_sd": None if ref is None
                   else ref["ric_sd"],
                   "combos": {}}
        for c in combos:
            st = _stat(cu, *c)
            ck = "|".join(c)
            if st is None:
                cellrow["combos"][ck] = {"error": "incomplete"}
                continue
            entry = {"ric_mean": st["ric_mean"],
                     "ric_sd": st["ric_sd"],
                     "ric_seeds": st["ric_seeds"],
                     "placebo_ric": st["placebo_ric"],
                     "boot_se": st["boot_se"]}
            if c != REF_COMBO and ref is not None:
                lift = round(st["ric_mean"] - ref["ric_mean"], 6)
                win = (lift >= LIFT_THRESH
                       and st["ric_sd"] <= LIFT_THRESH)
                entry["lift_vs_ref"] = lift
                entry["wins_cell"] = win
                if win:
                    wins[c].append(f"{s}-H{H}")
                    if (s, H) == WEAK_CONTROL:
                        ctrl_wins[c] = True
            cellrow["combos"][ck] = entry
        rows.append(cellrow)

    real_effects = []
    for c in combos:
        if c == REF_COMBO:
            continue
        nwin = len(wins[c])
        is_real = nwin >= 3 and ctrl_wins[c]
        if is_real:
            real_effects.append({"combo": "|".join(c),
                                  "cells_won": wins[c],
                                  "n_won": nwin,
                                  "weak_control_won": ctrl_wins[c]})

    if real_effects:
        winner = real_effects[0]["combo"]
        decision = (
            f"REAL EFFECT(S) on the objective/head axis: "
            f"{real_effects}. PRE-REGISTERED rev42 INTERACTION CLAUSE "
            f"TRIGGERED -> a confirmatory re-probe of W in {{16,32,64}} "
            f"at the winning (objective,head)={winner} is MANDATORY "
            f"BEFORE any final architecture lock (capacity x "
            f"objective/head interaction). Do NOT lock yet.")
    elif incomplete:
        decision = ("INCOMPLETE cells -> inconclusive; resume/re-run "
                    "needed before reading the rule.")
    else:
        decision = (
            "NO combo is a real effect (no non-ref combo lifts "
            f">=+{LIFT_THRESH} with seed-sd<={LIFT_THRESH} on >=3/4 "
            "cells incl the LTC-H300 weak control). The TCN "
            "objective/head axis is ALSO inert at locked W=16/D=4 in "
            "the frozen rev32 regime -> materially STRENGTHENS 'TCN "
            "~= snapshot fundamentally' (rev28/rev30/rev39 reinforced; "
            "escalate). HM1 continuous, NOT a §5 gate (rev35).")

    # PRIMARY DELIVERABLE (CLAUDE.md rule 1): the conditional alpha
    # surface. rank_IC mean +- seed-sd per (objective,head) regime,
    # per cell; the regime that MAXIMIZES alpha per cell, its
    # magnitude and seed-stability. This is the headline; the
    # confirmatory deploy gate is demoted to a secondary annotation
    # (rule 2/4 -- kept verbatim, NOT deleted, just not the framing).
    from collections import Counter
    alpha_surface = []
    for r in rows:
        ranked = sorted(
            ((ck, e) for ck, e in r["combos"].items()
             if "error" not in e),
            key=lambda kv: kv[1]["ric_mean"], reverse=True)
        if not ranked:
            alpha_surface.append({"sym": r["sym"], "H": r["H"],
                                  "error": "incomplete"})
            continue
        top_ck, top_e = ranked[0]
        alpha_surface.append({
            "sym": r["sym"], "H": r["H"], "L": r["L"],
            "is_weak_control": r["is_weak_control"],
            "alpha_max_regime": top_ck,
            "alpha_max_ric_mean": top_e["ric_mean"],
            "alpha_max_seed_sd": top_e["ric_sd"],
            "alpha_max_seeds": top_e["ric_seeds"],
            "ranked_surface": [{"regime": ck,
                                "ric_mean": e["ric_mean"],
                                "seed_sd": e["ric_sd"]}
                               for ck, e in ranked]})
    valid_as = [a for a in alpha_surface if "error" not in a]
    best_regime = {
        "per_cell_alpha_max": {
            f"{a['sym'].split('-')[0]}-H{a['H']}"
            + ("[weak]" if a["is_weak_control"] else ""):
            f"{a['alpha_max_regime']} {a['alpha_max_ric_mean']:+.4f}"
            f"±{a['alpha_max_seed_sd']:.4f}" for a in valid_as},
        "alpha_max_regime_vote": dict(
            Counter(a["alpha_max_regime"] for a in valid_as)),
        "reading": ("which (objective,head) regime yields the most "
                    "alpha, under which (symbol,horizon) conditions, "
                    "and how seed-stable -- this IS the tier result")}

    doc = {"run_id": run_id,
           "spec": "HD1 rev43 Tier-1 (FROZEN); runner rev4 substrate "
                   "(per-unit numerics == Tier-0 rev3/rev4 regime)",
           "PRIMARY_alpha_surface": alpha_surface,
           "PRIMARY_best_regime_per_cell": best_regime,
           "locked": {"W": W_FIXED, "D": D_FIXED, "dropout": DROPOUT,
                      "wd": WD,
                      "note": "rev30 principled default (smallest W, "
                              "most regularized) accompanying the "
                              "rev39/rev41 W=16 lock"},
           "rows": rows,
           "secondary_economic_deploy_annotation": {
               "purpose": "SEPARATE confirmatory deploy question "
                          "(HM1 §5 family) -- NOT the tier conclusion "
                          "(CLAUDE.md rule 2). Pre-registered rev43 "
                          "rule kept verbatim, demoted not deleted "
                          "(rule 4). rev43 itself framed HM1 as "
                          "continuous / NOT a §5 hard gate (rev35).",
               "reference_corner": "|".join(REF_COMBO),
               "rule": (f"per non-ref combo & cell: lift = ric_mean(C) "
                        f"- ric_mean(bce|last) in-run; win iff lift>="
                        f"+{LIFT_THRESH} AND seed-sd<={LIFT_THRESH}; "
                        f"real effect iff wins>=3/4 incl LTC-H300"),
               "real_effects": real_effects,
               "rev42_interaction_clause": decision},
           "incomplete_cells": incomplete,
           "approx_gpu_usd": spent, "n_units": len(units),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime())}
    _mark(run_id, "tier1.json", json.dumps(doc, indent=2, default=str))
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
    m = re.findall(r"tier1-\d{8}-\d{6}", run_id)
    run_id = m[0] if m else run_id
    tmp = tempfile.mkdtemp(prefix="tier1_")
    subprocess.run([sys.executable, "-m", "modal", "volume", "get",
                    "hd1-seq-cache", f"/tier1/{run_id}", tmp], check=True)
    doc = json.loads(next(Path(tmp).rglob("tier1.json")).read_text())
    art = REPO / "research_runs" / run_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "tier1.json").write_text(json.dumps(doc, indent=2,
                                               default=str))
    # Headline = the conditional alpha surface (CLAUDE.md rule 1/3).
    asf = doc["PRIMARY_alpha_surface"]
    surf = " ;; ".join(
        f"{a['sym'].split('-')[0]}-H{a['H']}"
        + ("[weak]" if a.get("is_weak_control") else "")
        + f": alpha-max {a['alpha_max_regime']}="
        f"{a['alpha_max_ric_mean']:+.4f}±{a['alpha_max_seed_sd']:.4f}"
        f"; surface=[" + ", ".join(
            f"{x['regime']} {x['ric_mean']:+.4f}±{x['seed_sd']:.4f}"
            for x in a["ranked_surface"]) + "]"
        for a in asf if "error" not in a)
    dep = doc["secondary_economic_deploy_annotation"]
    inc = doc.get("incomplete_cells")
    rec = {"hypothesis_id": "HD1", "rev": 44,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "statement": (
               f"TIER-1 CONDITIONAL ALPHA SURFACE (run {run_id}, "
               f"FROZEN rev43 spec, runner rev4; locked W=16/D=4 + "
               f"rev30 reg do0.5/wd1e-3). PRIMARY -- rank_IC mean±"
               f"seed-sd by objective×head regime, and the alpha-max "
               f"regime per (symbol,horizon) condition: {surf}. "
               f"incomplete={inc}. approx GPU "
               f"${doc['approx_gpu_usd']} ({doc['n_units']} units). "
               f"SECONDARY economic-deploy annotation (separate HM1 "
               f"§5 question, NOT the tier conclusion -- CLAUDE.md "
               f"rule 2/4; rev43 framed HM1 continuous, not a §5 "
               f"gate): {dep['rev42_interaction_clause']}"),
           "status": "testing", "priority_rank": 1,
           "result_experiment_id": run_id,
           "note": (f"Tier-1 PRIMARY = conditional alpha surface "
                    f"(rank_IC by objective×head per cell + alpha-max "
                    f"regime). Deploy-gate demoted to a secondary "
                    f"annotation per CLAUDE.md (kept, not deleted). "
                    f"experiments.jsonl untouched; "
                    f"rev28/rev32/rev39/rev41/rev42 unchanged.")}
    with open(REPO / "research" / "hypotheses.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[tier1] alpha-surface: {surf[:140]} -> "
          f"{art/'tier1.json'} ; rev appended")


@app.local_entrypoint()
def main(collect: str = "", skip_smoke: int = 0):
    if collect:
        _collect(collect)
        print("local entrypoint completed")
        return
    sys.path.insert(0, str(REPO))
    from scripts.hd1_seq_modal import _parity_gate
    _parity_gate()
    import time as _t
    run_id = f"tier1-{_t.strftime('%Y%m%d-%H%M%S', _t.gmtime())}"
    if not skip_smoke:
        smoke.spawn(run_id)
        print(f"[smoke] spawned run_id={run_id} (L4:{N_GPU})")
        t0, ok = _t.time(), False
        while _t.time() - t0 < 1500:
            o = _ls(f"/tier1/{run_id}")
            if "SMOKE_ALIVE" in o:
                print("[smoke] ALIVE (multi-GPU container reached code)")
            if "SMOKE_OK" in o:
                print("[smoke] OK: " + (
                    _vol_text(f"/tier1/{run_id}/SMOKE_OK") or "").strip())
                ok = True
                break
            _t.sleep(20)
        if not ok:
            print("[smoke] FAILED (no SMOKE_OK in 25min) -> NOT "
                  "launching full run; agent diagnoses autonomously.")
            raise SystemExit(3)
    tier1_all.spawn(run_id)
    print(f"[tier1] spawned run_id={run_id} (1 container, {N_GPU} GPU)")
    t0 = _t.time()
    while _t.time() - t0 < 7 * 3600:
        o = _ls(f"/tier1/{run_id}")
        if "ABORT" in o or "DONE" in o:
            _collect(run_id)
            print("local entrypoint completed"
                  + (" (budget-capped)" if "ABORT" in o else ""))
            return
        _t.sleep(30)
    print(f"[poll] still running; collect later: --collect {run_id}.")
