#!/usr/bin/env python3
"""HD1 Tier-0 sweep (FROZEN spec HD1 rev32; user-launched 2026-05-18).

Capacity/regularization probe: W x dropout x wd, on the 4 pre-registered
cells (BTC-H180/H300, ETH-H180 + LTC-H300 falsification control),
3 seeds, select+early-stop on VAL rank_ic (rev31 schedule), judged by
CONTINUOUS Delta-rank_ic vs HM6 baseline_ref (HM1 lens -- the binary §5
GATE is NOT applied; rev28 refuted/shelved UNCHANGED).

Reuses the EXACT frozen scripts.hd1_seq_core + scripts.hd1_seq_modal.
{_build_tcn, _parity_gate, BASELINE_REF_RIC} and the PERSISTED f32
MAX_L=512 packed cache on Volume hd1-seq-cache (NO re-egress/build).
Does NOT touch the frozen hd1_seq_modal pipeline/verdict. Server-side
.spawn + Volume marker = preemption/disconnect-immune (rev27 pattern).

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

# FROZEN rev32: 4 cells, fixed per-(sym,H) L = rev28 winner; D=4 (rev29)
CELLS = [("BTC-USDT-PERP", 180, 512), ("BTC-USDT-PERP", 300, 512),
         ("ETH-USDT-PERP", 180, 256), ("LTC-USDT-PERP", 300, 128)]
WEAK_CONTROL = ("LTC-USDT-PERP", 300)            # must also lift (rev32)
W_SWEEP = (16, 32, 64)
DO_SWEEP = (0.1, 0.3, 0.5)
WD_SWEEP = (1e-4, 1e-3)
SEEDS = (0, 1, 2)
D_FIXED = 4
# rev31 early-stop/select schedule (select+judge on val rank_ic)
PATIENCE, MIN_DELTA, T_MAX, EP_CAP = 2, 1e-4, 12, 40
LIFT_THRESH = 0.004                              # rev32 decision rule
L4_USD_PER_S, BUDGET = 1.0e-3, 5.0               # GPU-only guard ceiling

_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
app = modal.App("hd1-tier0")
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"


@app.function(image=GPU_IMG, gpu=["L4", "T4"], timeout=10800,
              volumes={MNT: VOL}, retries=1)
def run_symbol(sym: str, hl_list: list):
    """One container per symbol (amortize the packed-cache load).
    Data prep is byte-mirrored from frozen hd1_seq_modal.train_cell;
    only the schedule (rev31: select/stop on val rank_ic) + the
    W/dropout/wd sweep differ."""
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    from scripts.hd1_seq_modal import _build_tcn

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    VOL.reload()
    P = np.load(f"{MNT}/packed/{sym}.npz")
    Xfull = P["X"]
    n = int(P["n"])
    out = {}
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
        if ntr < C.N_TR_FLOOR or nte < C.N_OOS_FLOOR:
            out[H] = {"error": f"underpowered n_tr={ntr} n_oos={nte}"}
            continue
        fit_m, val_m = C.train_val_split(s_tr_all)
        w1 = C.r1_weights(rH, s_tr_all).astype(np.float32)
        fr = XL[fit_m].reshape(-1, C.N_TICK_FEAT)
        mu = fr.mean(0).astype(np.float32)
        sd = fr.std(0).astype(np.float32) + 1e-6

        def _T(mask):
            return torch.from_numpy((XL[mask].astype(np.float32) - mu) / sd)

        Xfit, Xval, Xte = _T(fit_m), _T(val_m), _T(s_te)
        yfit = torch.from_numpy(up[fit_m])
        wfit = torch.from_numpy(w1[fit_m])
        yval_i = up[val_m].astype(int)
        yte_i = up[s_te].astype(int)
        block = C.block_size(H)
        idx = np.arange(Xfit.shape[0])

        def _logits(net, Xm):
            o = []
            with torch.no_grad(), torch.amp.autocast(
                    "cuda", enabled=dev == "cuda"):
                for s in range(0, Xm.shape[0], 4096):
                    o.append(net(Xm[s:s + 4096].to(dev)).float().cpu())
            return torch.cat(o)

        configs = {}
        for W in W_SWEEP:
            for DO in DO_SWEEP:
                for WD in WD_SWEEP:
                    rics = []
                    last_p = None
                    for seed in SEEDS:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        net = _build_tcn(C.N_TICK_FEAT, W, D_FIXED,
                                         dropout=DO).to(dev)
                        opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                                               weight_decay=WD)
                        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_MAX)
                        scaler = torch.amp.GradScaler(
                            "cuda", enabled=dev == "cuda")
                        best_ric, pat, best_state = -1e9, 0, None
                        for ep in range(EP_CAP):
                            net.train()
                            np.random.shuffle(idx)
                            for s in range(0, len(idx), 1024):
                                j = idx[s:s + 1024]
                                xb = Xfit[j].to(dev, non_blocking=True)
                                yb = yfit[j].to(dev)
                                wb = wfit[j].to(dev)
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
                            # cosine clamped to T_MAX; abs-cap is ceiling
                            # only (rev31) -- LR stays at floor after.
                            if ep < T_MAX:
                                sch.step()
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
                        rics.append(C.auc(yte_i, p) - 0.5)
                        last_p = p
                    arr = np.array(rics, float)
                    key = f"W{W}_do{DO}_wd{WD:g}"
                    rec = {"W": W, "dropout": DO, "wd": WD,
                           "ric_seeds": [round(x, 6) for x in rics],
                           "ric_mean": round(float(arr.mean()), 6),
                           "ric_sd": round(float(arr.std()), 6)}
                    # diagnostics (NOT a gate) on the last seed's preds
                    rec["placebo_ric"] = round(
                        C.placebo_auc(yte_i, last_p) - 0.5, 6)
                    se = C.block_bootstrap_auc_se(yte_i, last_p, block)
                    rec["boot_se"] = (None if not np.isfinite(se)
                                      else round(float(se), 6))
                    configs[key] = rec
        best = max(configs.values(), key=lambda r: r["ric_mean"])
        out[H] = {"L": L, "n_tr": ntr, "n_oos": nte, "block": block,
                  "best": best, "configs": configs}
    out["_gpu_seconds"] = round(time.time() - t0, 1)
    return {"sym": sym, "res": out}


@app.function(image=GPU_IMG, volumes={MNT: VOL}, timeout=14400)
def coordinator():
    """Server-side: starmap symbols, aggregate, apply the FROZEN rev32
    decision rule, write continuous result to Volume /tier0/<run_id>.
    NO §5 gate, NO experiments.jsonl write (exploratory HM1 probe)."""
    import os
    import numpy as np  # noqa: F401
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

    # rev25 recorded per-cell rank_ic (frozen baseline to beat) from
    # the uncontaminated experiments.jsonl
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
    args = [(s, hl) for s, hl in groups.items()]

    results, spent = {}, 0.0
    for o in run_symbol.starmap(args):
        results[o["sym"]] = o["res"]
        spent += o["res"].get("_gpu_seconds", 0) * L4_USD_PER_S
        _w("PROGRESS", f"{o['sym']} done; spent~${spent:.2f}")
        if spent > BUDGET:
            _w("ABORT", f"spent~${spent:.2f} > ${BUDGET}")
            return {"run_id": run_id, "aborted": True}

    rows, n_pass, ctrl_pass = [], 0, False
    for (s, H, L) in CELLS:
        cell = results.get(s, {}).get(H, {})
        if "error" in cell:
            rows.append({"sym": s, "H": H, "error": cell["error"]})
            continue
        b = cell["best"]
        base = BASELINE_REF_RIC[(s, H)]
        r25 = rev25.get((s, H))
        d_now = round(b["ric_mean"] - base, 6)
        d_r25 = None if r25 is None else round(r25 - base, 6)
        lift = None if r25 is None else round(b["ric_mean"] - r25, 6)
        passed = lift is not None and lift >= LIFT_THRESH
        if passed:
            n_pass += 1
            if (s, H) == WEAK_CONTROL:
                ctrl_pass = True
        rows.append({
            "sym": s, "H": H, "L": L, "baseline_ref": base,
            "rev25_ric": r25, "rev25_delta_ic": d_r25,
            "best_config": {k: b[k] for k in ("W", "dropout", "wd")},
            "ric_mean": b["ric_mean"], "ric_sd": b["ric_sd"],
            "ric_seeds": b["ric_seeds"], "delta_ic_now": d_now,
            "lift_vs_rev25": lift, "placebo_ric": b["placebo_ric"],
            "boot_se": b["boot_se"], "passes_rule": passed,
            "is_weak_control": (s, H) == WEAK_CONTROL})

    verdict = (n_pass >= 3 and ctrl_pass)
    doc = {"run_id": run_id, "spec": "HD1 rev32 Tier-0 (FROZEN)",
           "framing": "HM1 continuous Delta-rank_ic; binary §5 gate NOT "
                      "applied; rev28 refuted/shelved unchanged",
           "decision_rule": (f"capacity/reg IS the lever IFF lift "
                             f">= +{LIFT_THRESH} on >=3/4 cells "
                             f"INCLUDING the LTC-H300 weak control"),
           "cells_passing": n_pass, "weak_control_passes": ctrl_pass,
           "VERDICT": ("LEVER_SUPPORTED -> unlock Tier 1"
                       if verdict else
                       "LEVER_NOT_SUPPORTED -> capacity is not the "
                       "lever; next = Tier 1 (obj/head) or escalate "
                       "TCN==snapshot"),
           "rows": rows, "raw": results,
           "approx_gpu_usd": round(spent, 2),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime())}
    _w("tier0.json", json.dumps(doc, indent=2, default=str))
    _w("DONE", run_id)
    print(f"[tier0] DONE verdict={doc['VERDICT']} "
          f"pass={n_pass}/4 ctrl={ctrl_pass} ~${spent:.2f}")
    return {"run_id": run_id, "verdict": doc["VERDICT"]}


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
    cj = next(Path(tmp).rglob("tier0.json"))
    doc = json.loads(cj.read_text())
    art = REPO / "research_runs" / run_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "tier0.json").write_text(json.dumps(doc, indent=2,
                                               default=str))
    # append the OUTCOME vs the pre-registered rev32 rule (HD1 rev34);
    # NO experiments.jsonl write (exploratory HM1 probe, no §5 gate)
    tbl = "; ".join(
        f"{r['sym'].split('-')[0]}-H{r['H']} "
        f"best={r['best_config']} ric={r['ric_mean']:+.4f}±{r['ric_sd']:.4f} "
        f"d_now={r['delta_ic_now']:+.4f} lift_vs_rev25="
        f"{r['lift_vs_rev25']:+.4f} pass={r['passes_rule']}"
        for r in doc["rows"] if "error" not in r)
    rec = {"hypothesis_id": "HD1", "rev": 34,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "statement": (
               f"TIER-0 RESULT (run {run_id}, FROZEN rev32 spec; "
               f"continuous HM1 Delta-rank_ic, binary §5 gate NOT "
               f"applied; rev28 refuted/shelved UNCHANGED). "
               f"Capacity/reg sweep W{W_SWEEP}xdropout{DO_SWEEP}x"
               f"wd{WD_SWEEP}, 4 cells x3 seeds, select+stop on val "
               f"rank_ic (rev31). PER-CELL best: {tbl}. "
               f"Pre-registered rule (lift>=+{LIFT_THRESH} 3-seed-mean "
               f"vs rev25 on >=3/4 cells INCL LTC-H300 control): "
               f"cells_pass={doc['cells_passing']}/4 "
               f"weak_control_pass={doc['weak_control_passes']} -> "
               f"VERDICT {doc['VERDICT']}. approx GPU "
               f"${doc['approx_gpu_usd']}. Decision follows the "
               f"pre-registered rule, not a post-hoc read."),
           "status": ("testing" if "unlock Tier 1" in doc["VERDICT"]
                      else "refuted"),
           "priority_rank": 1, "result_experiment_id": run_id,
           "note": (f"Tier-0 capacity/reg probe outcome vs frozen rev32 "
                    f"rule -> {doc['VERDICT']}. Continuous HM1 evidence "
                    f"(no §5 gate, experiments.jsonl untouched); rev28 "
                    f"unchanged. Tier 1 lock/decision is the next "
                    f"user-owned step.")}
    with open(REPO / "research" / "hypotheses.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[tier0] {doc['VERDICT']}  (pass={doc['cells_passing']}/4 "
          f"ctrl={doc['weak_control_passes']})  -> "
          f"{art/'tier0.json'} ; HD1 rev34 appended")


@app.local_entrypoint()
def main(collect: str = ""):
    if collect:
        _collect(collect)
        print("local entrypoint completed")
        return
    sys.path.insert(0, str(REPO))
    from scripts.hd1_seq_modal import _parity_gate
    _parity_gate()                       # frozen contract, $0, pre-spend
    h = coordinator.spawn()
    print(f"[spawn] tier0 fc={getattr(h, 'object_id', '?')} — "
          f"server-side; resume: modal run scripts/hd1_seq_tier0.py "
          f"--collect <run_id> (run_id at Volume /tier0/LATEST).")
    import re
    import subprocess
    import time as _t
    t0, rid = _t.time(), None
    while _t.time() - t0 < 4 * 3600:
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
                print("[tier0] ABORT marker (budget guard).")
                return
            if "DONE" in o:
                _collect(rid)
                print("local entrypoint completed")
                return
        _t.sleep(30)
    print(f"[poll] still running server-side; collect later: "
          f"--collect {rid}.")
