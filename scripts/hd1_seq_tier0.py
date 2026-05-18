#!/usr/bin/env python3
"""HD1 Tier-0 sweep (FROZEN spec HD1 rev32; runner rev3 per HD1 rev38).

rev3 infra fix (root cause rev37 = preemptible/multi-container GPU
cold-start churn; CLI logs unusable here so the code self-reports via
Volume markers). Changes vs rev2 are TOPOLOGY/OBSERVABILITY ONLY --
numerics BIT-IDENTICAL (same data prep mirrored from frozen
train_cell, same RNG/seed/shuffle/bs=1024/schedule/loss):
  * SINGLE container does ALL symbols sequentially (no coordinator +
    3-way starmap fan-out -> removes the GPU scheduling contention);
  * SINGLE GPU type "L4" (no ["L4","T4"] fallback-list churn);
  * an ALIVE marker is the VERY FIRST thing written (before any heavy
    import / 28GiB load) -> the container reaching user code is
    immediately observable (rev2's _hb was after imports -> blind);
  * a ~$0 smoke() proves the path (start/image/import/GPU/Volume) on
    this exact instance BEFORE the full run is committed;
  * per-(cell,config,seed) Volume checkpoint + resume-skip, retries
    -> an interrupt resumes, never restarts.
rev35 design-lock framing; HM1 continuous Delta-rank_ic; binary §5
gate NOT applied; rev28 refuted/shelved UNCHANGED.

Run:  modal run scripts/hd1_seq_tier0.py            (smoke -> full)
      modal run scripts/hd1_seq_tier0.py --collect <run_id>
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
W_SWEEP = (16, 32, 64)
DO_SWEEP = (0.1, 0.3, 0.5)
WD_SWEEP = (1e-4, 1e-3)
SEEDS = (0, 1, 2)
D_FIXED = 4
PATIENCE, MIN_DELTA, T_MAX, EP_CAP = 2, 1e-4, 12, 40
LIFT_THRESH = 0.004
L4_USD_PER_S, BUDGET = 1.0e-3, 8.0

_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
app = modal.App("hd1-tier0")
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"


def _mark(run_id, name, txt):
    import os
    d = f"{MNT}/tier0/{run_id}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/{name}", "w").write(str(txt))
    os.makedirs(f"{MNT}/tier0", exist_ok=True)
    open(f"{MNT}/tier0/LATEST", "w").write(run_id)
    VOL.commit()


@app.function(image=GPU_IMG, gpu="L4", timeout=900,
              volumes={MNT: VOL}, retries=2)
def smoke(run_id: str):
    """~$0 path probe: ALIVE first (pre-import), then import the full
    stack + a 5-step tiny TCN on a synthetic tensor (NO packed load) +
    Volume write. Proves start/image/import/GPU/commit on this exact
    instance before the full run is committed."""
    t = time.time()
    _mark(run_id, "SMOKE_ALIVE", t)               # before heavy import
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    from scripts.hd1_seq_modal import _build_tcn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(256, 64, C.N_TICK_FEAT, device=dev)
    y = torch.randint(0, 2, (256,), device=dev).float()
    net = _build_tcn(C.N_TICK_FEAT, 32, D_FIXED, dropout=0.1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = Fnn.binary_cross_entropy_with_logits(net(x), y)
        loss.backward()
        opt.step()
    msg = (f"SMOKE_OK dev={dev} stack+train5={round(time.time()-t,2)}s "
           f"loss={float(loss):.4f}")
    _mark(run_id, "SMOKE_OK", msg)
    return msg


@app.function(image=GPU_IMG, gpu="L4", timeout=21600,
              volumes={MNT: VOL}, retries=3)
def tier0_all(run_id: str):
    """SINGLE container, ALL symbols sequentially, resumable. ALIVE
    first (pre-import). Aggregates from durable parts and writes
    tier0.json + DONE itself (no separate coordinator)."""
    import os
    t_run = time.time()
    _mark(run_id, "ALIVE", t_run)                  # before heavy import
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    from scripts.hd1_seq_modal import _build_tcn, BASELINE_REF_RIC

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    VOL.reload()
    pdir = f"{MNT}/tier0/{run_id}/parts"
    os.makedirs(pdir, exist_ok=True)
    groups = {}
    for (s, H, L) in CELLS:
        groups.setdefault(s, []).append((H, L))
    per_cell = len(W_SWEEP) * len(DO_SWEEP) * len(WD_SWEEP) * len(SEEDS)

    def _hb(extra=""):
        tot_done = 0
        for fn in os.listdir(pdir) if os.path.isdir(pdir) else []:
            tot_done += sum(1 for _ in open(f"{pdir}/{fn}"))
        _mark(run_id, "HB", f"units={tot_done}/{len(CELLS)*per_cell} "
              f"t={round(time.time()-t_run)}s {extra}")

    _hb("start")
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
            sd = fr.std(0).astype(np.float32) + 1e-6

            def _G(mask):
                x = (XL[mask].astype(np.float32) - mu) / sd
                return torch.from_numpy(x).to(dev)

            Xfit, Xval, Xte = _G(fit_m), _G(val_m), _G(s_te)
            yfit = torch.from_numpy(up[fit_m]).to(dev)
            wfit = torch.from_numpy(w1[fit_m]).to(dev)
            yval_i = up[val_m].astype(int)
            yte_i = up[s_te].astype(int)
            block = C.block_size(H)
            n_fit = Xfit.shape[0]
            idx = np.arange(n_fit)

            def _logits(net, Xg):
                o = []
                with torch.no_grad(), torch.amp.autocast(
                        "cuda", enabled=dev == "cuda"):
                    for s in range(0, Xg.shape[0], 4096):
                        o.append(net(Xg[s:s + 4096]).float().cpu())
                return torch.cat(o)

            for W in W_SWEEP:
                for DO in DO_SWEEP:
                    for WD in WD_SWEEP:
                        for seed in SEEDS:
                            key = f"H{H}|W{W}|do{DO}|wd{WD:g}|s{seed}"
                            if key in done:
                                continue
                            u0 = time.time()
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            net = _build_tcn(C.N_TICK_FEAT, W, D_FIXED,
                                             dropout=DO).to(dev)
                            opt = torch.optim.Adam(
                                net.parameters(), lr=1e-3,
                                weight_decay=WD)
                            sch = torch.optim.lr_scheduler.\
                                CosineAnnealingLR(opt, T_MAX)
                            scaler = torch.amp.GradScaler(
                                "cuda", enabled=dev == "cuda")
                            best_ric, pat, best_state = -1e9, 0, None
                            for ep in range(EP_CAP):
                                net.train()
                                np.random.shuffle(idx)
                                for s in range(0, n_fit, 1024):
                                    jt = torch.as_tensor(
                                        idx[s:s + 1024], device=dev)
                                    opt.zero_grad(set_to_none=True)
                                    with torch.amp.autocast(
                                            "cuda",
                                            enabled=dev == "cuda"):
                                        lo = net(Xfit[jt])
                                        loss = (
                                            Fnn.
                                            binary_cross_entropy_with_logits(
                                                lo, yfit[jt],
                                                reduction="none")
                                            * wfit[jt]).sum() / (
                                            wfit[jt].sum() + 1e-9)
                                    scaler.scale(loss).backward()
                                    scaler.step(opt)
                                    scaler.update()
                                if ep < T_MAX:
                                    sch.step()
                                net.eval()
                                vp = _logits(net, Xval)
                                v_ric = C.auc(
                                    yval_i,
                                    torch.sigmoid(vp).numpy()) - 0.5
                                if v_ric > best_ric + MIN_DELTA:
                                    best_ric, pat = v_ric, 0
                                    best_state = {
                                        k: v.detach().cpu().clone()
                                        for k, v in
                                        net.state_dict().items()}
                                else:
                                    pat += 1
                                    if pat >= PATIENCE:
                                        break
                            if best_state:
                                net.load_state_dict(best_state)
                            net.eval()
                            p = torch.sigmoid(
                                _logits(net, Xte)).numpy()
                            se = C.block_bootstrap_auc_se(
                                yte_i, p, block)
                            rec = {"key": key, "sym": sym, "H": H,
                                   "L": L, "W": W, "dropout": DO,
                                   "wd": WD, "seed": seed,
                                   "ric": round(
                                       C.auc(yte_i, p) - 0.5, 6),
                                   "placebo_ric": round(
                                       C.placebo_auc(yte_i, p) - 0.5,
                                       6),
                                   "boot_se": (
                                       None if not np.isfinite(se)
                                       else round(float(se), 6)),
                                   "n_tr": ntr, "n_oos": nte,
                                   "block": block,
                                   "gpu_s": round(time.time() - u0,
                                                  2)}
                            with open(ppath, "a") as fh:
                                fh.write(json.dumps(rec) + "\n")
                            VOL.commit()
                            done.add(key)
                            if len(done) % 4 == 0:
                                _hb(f"{sym} {len(done)}/{per_cell*len(hl)}")
            del Xfit, Xval, Xte, yfit, wfit
            if dev == "cuda":
                torch.cuda.empty_cache()
        del Xfull, P
    _finalize(run_id, t_run, BASELINE_REF_RIC)
    return {"run_id": run_id, "done": True}


def _finalize(run_id, t_run, BASELINE_REF_RIC):
    """Aggregate durable parts -> rev35 design-lock + tier0.json +
    DONE/ABORT. Runs in-container at the end of tier0_all (also safe
    to import-call from --collect recovery)."""
    import os
    import numpy as np
    pdir = f"{MNT}/tier0/{run_id}/parts"
    units = []
    if os.path.isdir(pdir):
        for fn in os.listdir(pdir):
            for ln in open(f"{pdir}/{fn}"):
                try:
                    units.append(json.loads(ln))
                except Exception:
                    pass
    rev25 = {}
    for ln in open("/root/proj/research/experiments.jsonl"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("model_family") == "tcn" and r.get("symbols"):
            rev25[(r["symbols"][0], r.get("horizon_sec"))] = \
                r.get("rank_ic_oos")
    spent = round(sum(u.get("gpu_s", 0) for u in units) *
                  L4_USD_PER_S, 2)
    per_cell = len(W_SWEEP) * len(DO_SWEEP) * len(WD_SWEEP) * len(SEEDS)
    rows, n_pass, ctrl_pass, incomplete = [], 0, False, []
    for (s, H, L) in CELLS:
        cu = [u for u in units if u["sym"] == s and u["H"] == H]
        if len(cu) < per_cell:
            incomplete.append(f"{s}-H{H}({len(cu)}/{per_cell})")
        cfg = {}
        for u in cu:
            cfg.setdefault((u["W"], u["dropout"], u["wd"]), []).append(u)
        best = None
        for (W, DO, WD), us in cfg.items():
            if len(us) < len(SEEDS):
                continue
            arr = np.array([x["ric"] for x in us], float)
            cand = {"W": W, "dropout": DO, "wd": WD,
                    "ric_mean": round(float(arr.mean()), 6),
                    "ric_sd": round(float(arr.std()), 6),
                    "ric_seeds": sorted(round(x["ric"], 6) for x in us),
                    "placebo_ric": us[0]["placebo_ric"],
                    "boot_se": us[0]["boot_se"]}
            if best is None or cand["ric_mean"] > best["ric_mean"]:
                best = cand
        if best is None:
            rows.append({"sym": s, "H": H, "error": "incomplete"})
            continue
        base = BASELINE_REF_RIC[(s, H)]
        r25 = rev25.get((s, H))
        d_now = round(best["ric_mean"] - base, 6)
        lift = None if r25 is None else round(best["ric_mean"] - r25, 6)
        robust = (best["ric_sd"] <= LIFT_THRESH and lift is not None
                  and lift >= LIFT_THRESH)
        if robust:
            n_pass += 1
            if (s, H) == WEAK_CONTROL:
                ctrl_pass = True
        rows.append({"sym": s, "H": H, "L": L, "baseline_ref": base,
                     "rev25_ric": r25,
                     "locked_config": {"W": best["W"],
                                       "dropout": best["dropout"],
                                       "wd": best["wd"]},
                     "ric_mean": best["ric_mean"],
                     "ric_sd": best["ric_sd"],
                     "ric_seeds": best["ric_seeds"],
                     "delta_ic_now": d_now, "lift_vs_rev25": lift,
                     "placebo_ric": best["placebo_ric"],
                     "boot_se": best["boot_se"],
                     "robust_vs_noise": robust,
                     "is_weak_control": (s, H) == WEAK_CONTROL})
    rule_ok = (n_pass >= 3 and ctrl_pass)
    valid = [r for r in rows if "error" not in r]
    if rule_ok:
        lock = ("ROBUST WINNER -> lock per-cell (W,dropout,wd); stop "
                "sweeping the axis")
    elif valid:
        from collections import Counter
        ws = Counter(r["locked_config"]["W"] for r in valid)
        lock = (f"NO config robustly clears the ~{LIFT_THRESH} noise "
                f"floor on >=3/4 incl weak control -> capacity/reg "
                f"axis INERT in this range; principled default = "
                f"smallest W ({min(W_SWEEP)}, cheapest + most "
                f"regularized per rev30); stop sweeping. "
                f"(best-W vote {dict(ws)})")
    else:
        lock = "NO complete cells -> inconclusive; re-run needed"
    doc = {"run_id": run_id,
           "spec": "HD1 rev32 Tier-0 (FROZEN); runner rev3 (rev38 "
                   "single-container infra fix, numerics bit-identical)",
           "framing": "rev35 design-lock; HM1 continuous; §5 gate NOT "
                      "applied; rev28 unchanged",
           "DESIGN_LOCK_RECOMMENDATION": lock,
           "secondary_rule_check": (
               f"rev32 lift>=+{LIFT_THRESH} & sd<={LIFT_THRESH} on "
               f">=3/4 incl LTC-H300: pass={n_pass}/4 ctrl={ctrl_pass} "
               f"-> edge_real={rule_ok}"),
           "incomplete_cells": incomplete, "rows": rows,
           "approx_gpu_usd": spent, "n_units": len(units),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime())}
    _mark(run_id, "tier0.json", json.dumps(doc, indent=2, default=str))
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
    m = re.findall(r"tier0-\d{8}-\d{6}", run_id)
    run_id = m[0] if m else run_id
    tmp = tempfile.mkdtemp(prefix="tier0_")
    subprocess.run([sys.executable, "-m", "modal", "volume", "get",
                    "hd1-seq-cache", f"/tier0/{run_id}", tmp], check=True)
    doc = json.loads(next(Path(tmp).rglob("tier0.json")).read_text())
    art = REPO / "research_runs" / run_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "tier0.json").write_text(json.dumps(doc, indent=2,
                                               default=str))
    tbl = "; ".join(
        f"{r['sym'].split('-')[0]}-H{r['H']} lock={r['locked_config']} "
        f"ric={r['ric_mean']:+.4f}±{r['ric_sd']:.4f} "
        f"lift={r['lift_vs_rev25']} robust={r['robust_vs_noise']}"
        for r in doc["rows"] if "error" not in r)
    dl = doc["DESIGN_LOCK_RECOMMENDATION"]
    sc = doc["secondary_rule_check"]
    inc = doc.get("incomplete_cells")
    rec = {"hypothesis_id": "HD1", "rev": 34,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "statement": (
               f"TIER-0 RESULT (run {run_id}, FROZEN rev32 spec, "
               f"runner rev3 single-container infra fix, numerics "
               f"bit-identical; rev35 design-lock; HM1 continuous, §5 "
               f"gate NOT applied; rev28 unchanged). DESIGN-LOCK: "
               f"{dl}. PER-CELL: {tbl}. Secondary rev32-rule check: "
               f"{sc}. incomplete={inc}. approx GPU "
               f"${doc['approx_gpu_usd']} ({doc['n_units']} units)."),
           "status": "testing", "priority_rank": 1,
           "result_experiment_id": run_id,
           "note": (f"Tier-0 design-lock outcome: {dl[:120]}. "
                    f"Continuous HM1 (no §5 gate; experiments.jsonl "
                    f"untouched); rev28 unchanged. Tier 1 decision "
                    f"next.")}
    with open(REPO / "research" / "hypotheses.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[tier0] {dl[:90]} -> {art/'tier0.json'} ; rev34 appended")


@app.local_entrypoint()
def main(collect: str = "", skip_smoke: int = 0):
    if collect:
        _collect(collect)
        print("local entrypoint completed")
        return
    sys.path.insert(0, str(REPO))
    from scripts.hd1_seq_modal import _parity_gate
    _parity_gate()
    import re
    import time as _t
    run_id = f"tier0-{_t.strftime('%Y%m%d-%H%M%S', _t.gmtime())}"

    if not skip_smoke:
        sh = smoke.spawn(run_id)
        print(f"[smoke] spawned fc={getattr(sh,'object_id','?')} "
              f"run_id={run_id}")
        t0, ok = _t.time(), False
        while _t.time() - t0 < 720:                # 12 min smoke budget
            o = _ls(f"/tier0/{run_id}")
            if "SMOKE_ALIVE" in o:
                print("[smoke] ALIVE (container reached user code)")
            if "SMOKE_OK" in o:
                print(f"[smoke] OK: "
                      f"{(_vol_text(f'/tier0/{run_id}/SMOKE_OK') or '').strip()}")
                ok = True
                break
            _t.sleep(20)
        if not ok:
            print("[smoke] FAILED (no SMOKE_OK in 12min) -> infra "
                  "still bad on this profile; NOT launching full run. "
                  "Agent will diagnose/switch autonomously.")
            raise SystemExit(3)

    h = tier0_all.spawn(run_id)
    print(f"[tier0] spawned fc={getattr(h,'object_id','?')} "
          f"run_id={run_id} (single container, resumable)")
    t0 = _t.time()
    while _t.time() - t0 < 7 * 3600:
        o = _ls(f"/tier0/{run_id}")
        if "ABORT" in o:
            _collect(run_id)
            print("local entrypoint completed (budget-capped)")
            return
        if "DONE" in o:
            _collect(run_id)
            print("local entrypoint completed")
            return
        _t.sleep(30)
    print(f"[poll] still running server-side; collect later: "
          f"--collect {run_id}.")
