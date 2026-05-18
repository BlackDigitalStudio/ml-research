#!/usr/bin/env python3
"""HD1 Tier-0 sweep (FROZEN spec HD1 rev32; runner rev2 per HD1 rev36).

Capacity/regularization design-lock probe (rev35): W x dropout x wd on
the 4 pre-registered cells (BTC-H180/H300, ETH-H180 + LTC-H300
falsification control), 3 seeds, select+early-stop on VAL rank_ic
(rev31 schedule), judged by CONTINUOUS Delta-rank_ic vs HM6
baseline_ref (HM1 lens; binary §5 gate NOT applied; rev28 unchanged).

rev36 runner-only fixes (numerics BIT-IDENTICAL to runner rev1 -- same
data prep, RNG, shuffle order, bs=1024, schedule, loss; speed comes
ONLY from the data pipeline, NOT any hyperparameter change):
  (a) whole standardized Xfit/Xval/Xte to GPU ONCE per cell; minibatch
      by GPU-side index of the SAME np.random-shuffled idx -> kills the
      per-step CPU->GPU transfer starvation (the dominant slowdown);
  (b) per-(cell,config,seed) checkpoint to Volume parts/<sym>.jsonl +
      resume-skip -> preemption never re-does completed units;
  (c) working guard: per-worker wall cap + coordinator sums durable
      gpu-seconds from parts and ABORTs if > budget (not post-symbol);
  (d) hb_<sym> HEARTBEAT markers -> telemetry not hidden behind BTC.
Coordinator aggregates from the durable parts so a dead worker still
yields a partial verdict + flags incomplete cells.

Run:  modal run scripts/hd1_seq_tier0.py
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
L4_USD_PER_S, BUDGET = 1.0e-3, 5.0
PER_WORKER_SECS = 7200                            # wall cap / attempt

_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
app = modal.App("hd1-tier0")
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"


@app.function(image=GPU_IMG, gpu=["L4", "T4"], timeout=9000,
              volumes={MNT: VOL}, retries=4)
def run_symbol(sym: str, hl_list: list, run_id: str):
    """One container per symbol (amortize the packed load). Per-unit
    Volume checkpoint + resume-skip => preemption-resilient. Tensors
    live on GPU; minibatch by GPU index of the same shuffled idx."""
    import os
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    from scripts.hd1_seq_modal import _build_tcn

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    VOL.reload()
    pdir = f"{MNT}/tier0/{run_id}/parts"
    os.makedirs(pdir, exist_ok=True)
    ppath = f"{pdir}/{sym}.jsonl"
    done = set()
    if os.path.exists(ppath):
        for ln in open(ppath):
            try:
                done.add(json.loads(ln)["key"])
            except Exception:
                pass
    total = len(hl_list) * len(W_SWEEP) * len(DO_SWEEP) * \
        len(WD_SWEEP) * len(SEEDS)

    def _hb(extra=""):
        open(f"{MNT}/tier0/{run_id}/hb_{sym}", "w").write(
            f"{sym} done={len(done)}/{total} {extra}".strip())
        VOL.commit()

    _hb("resumed" if done else "start")
    P = np.load(f"{MNT}/packed/{sym}.npz")
    Xfull = P["X"]
    n = int(P["n"])

    for (H, L) in hl_list:
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

        def _G(mask):                                # standardize -> GPU
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
                        if time.time() - t0 > PER_WORKER_SECS:
                            _hb("WALLCAP -> partial; resumable")
                            return {"sym": sym, "stopped": "wallcap",
                                    "done": len(done), "total": total}
                        u0 = time.time()
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        net = _build_tcn(C.N_TICK_FEAT, W, D_FIXED,
                                         dropout=DO).to(dev)
                        opt = torch.optim.Adam(net.parameters(), lr=1e-3,
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
                                xb = Xfit[jt]
                                yb = yfit[jt]
                                wb = wfit[jt]
                                opt.zero_grad(set_to_none=True)
                                with torch.amp.autocast(
                                        "cuda", enabled=dev == "cuda"):
                                    lo = net(xb)
                                    loss = (
                                        Fnn.binary_cross_entropy_with_logits(
                                            lo, yb, reduction="none") * wb
                                    ).sum() / (wb.sum() + 1e-9)
                                scaler.scale(loss).backward()
                                scaler.step(opt)
                                scaler.update()
                            if ep < T_MAX:          # cosine clamped;
                                sch.step()          # cap is ceiling only
                            net.eval()
                            vp = _logits(net, Xval)
                            v_ric = C.auc(
                                yval_i, torch.sigmoid(vp).numpy()) - 0.5
                            if v_ric > best_ric + MIN_DELTA:
                                best_ric, pat = v_ric, 0
                                best_state = {k: v.detach().cpu().clone()
                                              for k, v in
                                              net.state_dict().items()}
                            else:
                                pat += 1
                                if pat >= PATIENCE:
                                    break
                        if best_state:
                            net.load_state_dict(best_state)
                        net.eval()
                        p = torch.sigmoid(_logits(net, Xte)).numpy()
                        se = C.block_bootstrap_auc_se(yte_i, p, block)
                        rec = {"key": key, "sym": sym, "H": H, "L": L,
                               "W": W, "dropout": DO, "wd": WD,
                               "seed": seed,
                               "ric": round(C.auc(yte_i, p) - 0.5, 6),
                               "placebo_ric": round(
                                   C.placebo_auc(yte_i, p) - 0.5, 6),
                               "boot_se": (None if not np.isfinite(se)
                                           else round(float(se), 6)),
                               "n_tr": ntr, "n_oos": nte, "block": block,
                               "gpu_s": round(time.time() - u0, 2)}
                        with open(ppath, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        VOL.commit()
                        done.add(key)
                        if len(done) % 5 == 0:
                            _hb()
        # free GPU before the next (heavier) cell
        del Xfit, Xval, Xte, yfit, wfit
        if dev == "cuda":
            torch.cuda.empty_cache()
    _hb("DONE")
    return {"sym": sym, "done": len(done), "total": total,
            "gpu_seconds": round(time.time() - t0, 1)}


@app.function(image=GPU_IMG, volumes={MNT: VOL}, timeout=18000)
def coordinator():
    """Server-side: starmap symbols (return_exceptions so one dead
    worker can't kill aggregation), then AGGREGATE FROM THE DURABLE
    parts/*.jsonl (survives a dead worker -> partial verdict + flags
    incomplete cells). rev35 framing: lead with the locked
    (W,dropout,wd); rev32 numeric rule = secondary 'is the edge real'
    check. NO §5 gate, NO experiments.jsonl write."""
    import os
    import numpy as np
    sys.path.insert(0, "/root/proj")
    from scripts.hd1_seq_modal import BASELINE_REF_RIC

    run_id = f"tier0-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    outd = f"{MNT}/tier0/{run_id}"

    def _w(name, txt):
        os.makedirs(outd, exist_ok=True)
        open(f"{outd}/{name}", "w").write(txt)
        os.makedirs(f"{MNT}/tier0", exist_ok=True)
        open(f"{MNT}/tier0/LATEST", "w").write(run_id)
        VOL.commit()

    _w("STARTED", run_id)
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

    groups = {}
    for (s, H, L) in CELLS:
        groups.setdefault(s, []).append((H, L))
    args = [(s, hl, run_id) for s, hl in groups.items()]
    for o in run_symbol.starmap(args, return_exceptions=True):
        print(f"[coordinator] worker -> {o}")

    # aggregate from durable parts (truth, survives dead workers)
    units = []
    pdir = f"{outd}/parts"
    if os.path.isdir(pdir):
        for fn in os.listdir(pdir):
            for ln in open(f"{pdir}/{fn}"):
                try:
                    units.append(json.loads(ln))
                except Exception:
                    pass
    spent = round(sum(u.get("gpu_s", 0) for u in units) *
                  L4_USD_PER_S, 2)
    exp_per_cell = len(W_SWEEP) * len(DO_SWEEP) * len(WD_SWEEP) * \
        len(SEEDS)

    rows, n_pass, ctrl_pass, incomplete = [], 0, False, []
    for (s, H, L) in CELLS:
        cu = [u for u in units if u["sym"] == s and u["H"] == H]
        if len(cu) < exp_per_cell:
            incomplete.append(f"{s}-H{H}({len(cu)}/{exp_per_cell})")
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
            rows.append({"sym": s, "H": H, "error": "no complete config"})
            continue
        base = BASELINE_REF_RIC[(s, H)]
        r25 = rev25.get((s, H))
        d_now = round(best["ric_mean"] - base, 6)
        lift = None if r25 is None else round(best["ric_mean"] - r25, 6)
        # robustness flag (rev35): edge must clear the ~0.004 noise floor
        robust = (best["ric_sd"] <= LIFT_THRESH and
                  lift is not None and lift >= LIFT_THRESH)
        if robust:
            n_pass += 1
            if (s, H) == WEAK_CONTROL:
                ctrl_pass = True
        rows.append({
            "sym": s, "H": H, "L": L, "baseline_ref": base,
            "rev25_ric": r25, "locked_config": {
                "W": best["W"], "dropout": best["dropout"],
                "wd": best["wd"]},
            "ric_mean": best["ric_mean"], "ric_sd": best["ric_sd"],
            "ric_seeds": best["ric_seeds"], "delta_ic_now": d_now,
            "lift_vs_rev25": lift, "placebo_ric": best["placebo_ric"],
            "boot_se": best["boot_se"],
            "robust_vs_noise": robust,
            "is_weak_control": (s, H) == WEAK_CONTROL})

    rule_ok = (n_pass >= 3 and ctrl_pass)
    # rev35: lead with the design-lock recommendation
    valid = [r for r in rows if "error" not in r]
    if rule_ok:
        lock = ("ROBUST WINNER -> lock per-cell (W,dropout,wd); "
                "stop sweeping the axis")
    elif valid:
        from collections import Counter
        ws = Counter(r["locked_config"]["W"] for r in valid)
        lock = (f"NO config robustly clears the ~{LIFT_THRESH} noise "
                f"floor on >=3/4 incl weak control -> capacity/reg "
                f"axis INERT in this range; principled default = "
                f"smallest W ({min(W_SWEEP)}, cheapest + most "
                f"regularized per rev30), dropout/wd at regularizing "
                f"end; stop sweeping. (best-W vote {dict(ws)})")
    else:
        lock = "NO complete cells -> inconclusive; re-run needed"

    doc = {"run_id": run_id, "spec": "HD1 rev32 Tier-0 (FROZEN); "
           "runner rev2 (rev36 perf+robustness, numerics bit-identical)",
           "framing": "rev35 design-lock; HM1 continuous Delta-rank_ic; "
                      "binary §5 gate NOT applied; rev28 unchanged",
           "DESIGN_LOCK_RECOMMENDATION": lock,
           "secondary_rule_check": (
               f"rev32 lift>=+{LIFT_THRESH} & sd<= {LIFT_THRESH} on "
               f">=3/4 incl LTC-H300: pass={n_pass}/4 "
               f"ctrl={ctrl_pass} -> edge_real={rule_ok}"),
           "incomplete_cells": incomplete, "rows": rows,
           "approx_gpu_usd": spent, "n_units": len(units),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime())}
    if spent > BUDGET:
        doc["BUDGET_NOTE"] = (f"gpu spend ~${spent} exceeded "
                              f"${BUDGET}; result is partial-but-valid")
        _w("tier0.json", json.dumps(doc, indent=2, default=str))
        _w("ABORT", f"spent~${spent}>{BUDGET}; partial result saved")
        return {"run_id": run_id, "budget_exceeded": True}
    _w("tier0.json", json.dumps(doc, indent=2, default=str))
    _w("DONE", run_id)
    print(f"[tier0] DONE lock=({lock[:60]}...) pass={n_pass}/4 "
          f"ctrl={ctrl_pass} incomplete={incomplete} ~${spent}")
    return {"run_id": run_id, "rule_ok": rule_ok}


def _vol_text(p):
    import subprocess
    g = subprocess.run([sys.executable, "-m", "modal", "volume", "get",
                        "hd1-seq-cache", p, "-"],
                       capture_output=True, text=True)
    return g.stdout if g.returncode == 0 else None


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
    rec = {"hypothesis_id": "HD1", "rev": 34,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "statement": (
               f"TIER-0 RESULT (run {run_id}, FROZEN rev32 spec, "
               f"runner rev2 rev36 perf+robustness, numerics "
               f"bit-identical; rev35 design-lock framing; HM1 "
               f"continuous, §5 gate NOT applied; rev28 unchanged). "
               f"DESIGN-LOCK RECOMMENDATION: {doc['DESIGN_LOCK_"
               f"RECOMMENDATION']}. PER-CELL: {tbl}. Secondary "
               f"rev32-rule check (is the edge real vs rev25/noise): "
               f"{doc['secondary_rule_check']}. incomplete="
               f"{doc.get('incomplete_cells')}. approx GPU "
               f"${doc['approx_gpu_usd']} ({doc['n_units']} units). "
               f"Lock follows the rule, not a post-hoc read."),
           "status": "testing",
           "priority_rank": 1, "result_experiment_id": run_id,
           "note": (f"Tier-0 design-lock outcome: "
                    f"{doc['DESIGN_LOCK_RECOMMENDATION'][:120]}. "
                    f"Continuous HM1 evidence (no §5 gate; "
                    f"experiments.jsonl untouched); rev28 unchanged. "
                    f"Tier 1 decision is the next user-owned step.")}
    with open(REPO / "research" / "hypotheses.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[tier0] {doc['DESIGN_LOCK_RECOMMENDATION'][:80]} "
          f"-> {art/'tier0.json'} ; HD1 rev34 appended")


@app.local_entrypoint()
def main(collect: str = ""):
    if collect:
        _collect(collect)
        print("local entrypoint completed")
        return
    sys.path.insert(0, str(REPO))
    from scripts.hd1_seq_modal import _parity_gate
    _parity_gate()
    h = coordinator.spawn()
    print(f"[spawn] tier0 fc={getattr(h, 'object_id', '?')} — "
          f"server-side; resume: modal run scripts/hd1_seq_tier0.py "
          f"--collect <run_id> (run_id at Volume /tier0/LATEST).")
    import re
    import subprocess
    import time as _t
    t0, rid = _t.time(), None
    while _t.time() - t0 < 5 * 3600:
        if rid is None:
            s = _vol_text("/tier0/LATEST")
            mm = re.findall(r"tier0-\d{8}-\d{6}", s or "")
            if mm:
                rid = mm[-1]
                print(f"[poll] tier0 run_id={rid}")
        if rid:
            o = subprocess.run([sys.executable, "-m", "modal", "volume",
                                "ls", "hd1-seq-cache", f"/tier0/{rid}"],
                               capture_output=True, text=True).stdout or ""
            if "ABORT" in o:
                _collect(rid)
                print("local entrypoint completed (budget-capped)")
                return
            if "DONE" in o:
                _collect(rid)
                print("local entrypoint completed")
                return
        _t.sleep(30)
    print(f"[poll] still running server-side; collect later: "
          f"--collect {rid}.")
