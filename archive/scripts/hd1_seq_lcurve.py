#!/usr/bin/env python3
"""HD1-seq learning-curve DIAGNOSTIC (pre-registered HD1 rev29).

NOT a re-judgement of HD1-seq (stays refuted, rev28). Trains ONE cell
with NO early-stop for a generous fixed epoch budget, logging per-epoch
{train_loss, R1 val_logloss, val_auc, val_ric, test_auc, test_ric} and
the epoch the frozen rev25 rule (patience=5, <=30 ep) WOULD have
stopped. Purpose: see whether under-training and/or a logloss-vs-AUC
divergence confounds the refuted negative, to inform future design.

Reuses the EXACT frozen scripts.hd1_seq_core + scripts.hd1_seq_modal.
_build_tcn (identical architecture) and the persisted f32 packed cache
on Volume hd1-seq-cache (no re-egress/build). Server-side .spawn +
Volume marker = preemption/disconnect-immune (HD1 rev27 pattern).

Run:  modal run scripts/hd1_seq_lcurve.py
      modal run scripts/hd1_seq_lcurve.py --collect <run_id>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# Same (sym,H,L,D) as the rev28 LTC-H180 winner; LTC = lightest packed
# (3.84 GiB). seed 0 (curve SHAPE vs epochs is the question, not seed
# variance). EPOCHS = 4x the rev25 cap to see the full trajectory.
SYM, H, L, D, SEED = "LTC-USDT-PERP", 180, 64, 4, 0
EPOCHS = 120
RUN_TAG = "lcurve"

_IGN = ["**/.git/**", "**/__pycache__/**", "data/**", "models/**",
        "**/*.db", "research_runs/**", "**/target/**"]
app = modal.App("hd1-lcurve")
GPU_IMG = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy==2.2.4", "scikit-learn", "torch")
           .add_local_dir(str(REPO), "/root/proj", ignore=_IGN))
VOL = modal.Volume.from_name("hd1-seq-cache", create_if_missing=True)
MNT = "/cache"


@app.function(image=GPU_IMG, gpu=["L4", "T4"], timeout=10800,
              volumes={MNT: VOL}, retries=1)
def lcurve():
    import os
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    sys.path.insert(0, "/root/proj")
    from scripts import hd1_seq_core as C
    from scripts.hd1_seq_modal import _build_tcn

    run_id = f"{RUN_TAG}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    outd = f"{MNT}/lcurve/{run_id}"

    def _w(name, txt):
        os.makedirs(outd, exist_ok=True)
        with open(f"{outd}/{name}", "w") as fh:
            fh.write(txt)
        os.makedirs(f"{MNT}/lcurve", exist_ok=True)
        with open(f"{MNT}/lcurve/LATEST", "w") as fh:
            fh.write(run_id)
        VOL.commit()

    _w("STARTED", run_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    VOL.reload()
    P = np.load(f"{MNT}/packed/{SYM}.npz")
    Xfull = P["X"]
    n = int(P["n"])
    XL = np.ascontiguousarray(Xfull[:, -L:, :])      # causal slice
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

    def _T(mask):
        return torch.from_numpy((XL[mask].astype(np.float32) - mu) / sd)

    Xfit, Xval, Xte = _T(fit_m), _T(val_m), _T(s_te)
    yfit = torch.from_numpy(up[fit_m])
    yval = torch.from_numpy(up[val_m])
    wfit = torch.from_numpy(w1[fit_m])
    wval = torch.from_numpy(w1[val_m])
    yval_i = up[val_m].astype(int)
    yte_i = up[s_te].astype(int)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = _build_tcn(C.N_TICK_FEAT, C.W_FIXED, D).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                           weight_decay=C.WD_FIXED)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    bs = 1024
    idx = np.arange(Xfit.shape[0])

    def _logits(Xm):
        out = []
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev == "cuda"):
            for s in range(0, Xm.shape[0], 4096):
                out.append(net(Xm[s:s + 4096].to(dev)).float().cpu())
        return torch.cat(out)

    curve = []
    for ep in range(EPOCHS):
        net.train()
        np.random.shuffle(idx)
        tl = 0.0
        nb = 0
        for s in range(0, len(idx), bs):
            j = idx[s:s + bs]
            xb = Xfit[j].to(dev, non_blocking=True)
            yb = yfit[j].to(dev)
            wb = wfit[j].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                lo = net(xb)
                loss = (Fnn.binary_cross_entropy_with_logits(
                    lo, yb, reduction="none") * wb).sum() / (wb.sum() + 1e-9)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += float(loss)
            nb += 1
        sch.step()
        net.eval()
        vp = _logits(Xval)
        vloss = float((Fnn.binary_cross_entropy_with_logits(
            vp, yval, reduction="none") * wval).sum() / (wval.sum() + 1e-9))
        v_auc = C.auc(yval_i, torch.sigmoid(vp).numpy())
        t_auc = C.auc(yte_i, torch.sigmoid(_logits(Xte)).numpy())
        curve.append({"epoch": ep + 1, "train_loss": round(tl / nb, 6),
                      "val_logloss": round(vloss, 6),
                      "val_auc": round(v_auc, 6),
                      "val_ric": round(v_auc - 0.5, 6),
                      "test_auc": round(t_auc, 6),
                      "test_ric": round(t_auc - 0.5, 6)})
        if (ep + 1) % 10 == 0:
            _w("PROGRESS", f"epoch {ep+1}/{EPOCHS} "
               f"val_ll={vloss:.5f} val_ric={v_auc-0.5:+.4f} "
               f"test_ric={t_auc-0.5:+.4f}")

    # simulate the FROZEN rev25 stop rule on the logged val_logloss
    best, pat, stop_ep = 1e9, 0, EPOCHS
    for c in curve[:30]:
        if c["val_logloss"] < best - 1e-5:
            best, pat = c["val_logloss"], 0
        else:
            pat += 1
            if pat >= 5:
                stop_ep = c["epoch"]
                break
    rev25 = next(c for c in curve if c["epoch"] == stop_ep)
    best_val_ric = max(curve, key=lambda c: c["val_ric"])
    best_test_ric = max(curve, key=lambda c: c["test_ric"])

    doc = {"run_id": run_id, "cell": {"sym": SYM, "H": H, "L": L,
           "D": D, "seed": SEED}, "epochs": EPOCHS,
           "n_tr": ntr, "n_oos": nte,
           "rev25_would_stop_epoch": stop_ep,
           "at_rev25_stop": rev25,
           "best_by_val_ric": best_val_ric,
           "best_by_test_ric": best_test_ric,
           "final": curve[-1], "curve": curve,
           "freeze": "HD1 rev29 diagnostic (rev25 verdict unchanged)",
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime())}
    _w("curve.json", json.dumps(doc, indent=2))
    _w("DONE", run_id)
    print(f"[lcurve] DONE rev25_stop_ep={stop_ep} "
          f"at_stop test_ric={rev25['test_ric']:+.4f} | "
          f"best_test_ric={best_test_ric['test_ric']:+.4f}"
          f"@ep{best_test_ric['epoch']} | final test_ric="
          f"{curve[-1]['test_ric']:+.4f}")
    return doc


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
    m = re.findall(r"lcurve-\d{8}-\d{6}", run_id)
    run_id = m[0] if m else run_id
    tmp = tempfile.mkdtemp(prefix="lcurve_")
    subprocess.run([sys.executable, "-m", "modal", "volume", "get",
                    "hd1-seq-cache", f"/lcurve/{run_id}", tmp], check=True)
    cj = next(Path(tmp).rglob("curve.json"))
    art = REPO / "research_runs" / run_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "curve.json").write_text(cj.read_text())
    print(f"[lcurve] curve -> {art/'curve.json'}")


@app.local_entrypoint()
def main(collect: str = ""):
    if collect:
        _collect(collect)
        print("local entrypoint completed")
        return
    h = lcurve.spawn()
    print(f"[spawn] lcurve fc={getattr(h, 'object_id', '?')} — server-side; "
          f"collect: modal run scripts/hd1_seq_lcurve.py --collect <run_id> "
          f"(run_id at Volume /lcurve/LATEST).")
    import re
    import time as _t
    t0, rid = _t.time(), None
    while _t.time() - t0 < 3 * 3600:
        if rid is None:
            s = _vol_text("/lcurve/LATEST")
            mm = re.findall(r"lcurve-\d{8}-\d{6}", s or "")
            if mm:
                rid = mm[-1]
                print(f"[poll] lcurve run_id={rid}")
        if rid:
            import subprocess
            o = subprocess.run([sys.executable, "-m", "modal", "volume",
                                "ls", "hd1-seq-cache", f"/lcurve/{rid}"],
                               capture_output=True, text=True).stdout or ""
            if "DONE" in o:
                _collect(rid)
                print("local entrypoint completed")
                return
        _t.sleep(30)
    print(f"[poll] still running server-side; collect later with "
          f"--collect {rid}.")
