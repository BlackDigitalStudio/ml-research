#!/usr/bin/env python3
"""H100 utilization probe for the HD2 Mamba-2 model. Samples nvidia-smi
(GPU util % + VRAM) during real forward/backward of our exact model shape
(d_model=256, n_layers=4, d_state=128, L=216000) at several batch_periods,
to see how loaded the H100 is and whether bigger batch pushes it toward 90%.

Uses a REAL normalized day stream (untrained model + random input over L=216000
explodes the SSM scan -> CUBLAS failure; real z-scored data is stable, as in
the actual trainer).

  modal run scripts/hd2_gpuprobe.py
"""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent
_CCV = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/"
        "causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
_MAMBA = ("https://github.com/state-spaces/mamba/releases/download/v2.2.2/"
          "mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
IMG = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy==2.2.4", "scipy", "scikit-learn",
                 "einops", "packaging", "transformers==4.43.3")
    .pip_install(_CCV, _MAMBA)
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True)
)
VOL = modal.Volume.from_name("hd2-cache")
MNT = "/cache"
app = modal.App("hd2-gpuprobe")


@app.function(image=IMG, gpu="H100", timeout=1800, volumes={MNT: VOL})
def probe(bps=(2, 8, 16), L=216000, steps=10):
    import sys, os, glob, subprocess, threading, time, statistics
    sys.path.insert(0, "/root/scripts")
    import numpy as np
    import torch
    from hd2_mamba_stream import HD2Mamba2
    dev = "cuda"

    # one real day, z-scored, tiled to length L (stable input for the SSM scan)
    day = sorted(glob.glob(f"{MNT}/hd2/SOL-USDT-PERP/*.npz"))[0]
    s = np.load(day)["stream"].astype(np.float32)
    s = (s - s.mean(0)) / (s.std(0) + 1e-6)
    reps = (L // len(s)) + 1
    w = np.tile(s, (reps, 1))[:L]                       # (L, 80)
    print(f"probe data: {os.path.basename(day)} n={len(s)} -> window L={L}")
    out = []

    def sample_into(buf, stop):
        while not stop[0]:
            try:
                q = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"]).decode().strip().splitlines()[0]
                u, mu, mt = [int(x.strip()) for x in q.split(",")]
                buf.append((u, mu, mt))
            except Exception:
                pass
            time.sleep(0.5)

    for bp in bps:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = HD2Mamba2(cell_kind="mamba2", d_model=256, n_layers=4,
                          d_state=128, n_out=3).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.from_numpy(np.broadcast_to(w, (bp, L, 80)).copy()).to(dev)
        buf = []; stop = [False]
        th = threading.Thread(target=sample_into, args=(buf, stop), daemon=True)
        th.start()
        t0 = time.time(); err = None
        try:
            for _ in range(steps):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model.encode_period(x, use_ckpt=True)
                    loss = h.float().pow(2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize()
        except RuntimeError as e:
            err = str(e)[:120]
        dt = time.time() - t0
        time.sleep(1.0); stop[0] = True; time.sleep(0.7)
        us = [a[0] for a in buf]
        peak = torch.cuda.max_memory_allocated() / 1e9
        mt = buf[0][2] if buf else 0
        rec = {"bp": bp, "err": err, "s_per_step": (dt / steps) if not err else None,
               "util_mean": round(statistics.mean(us), 1) if us else None,
               "util_p50": (statistics.median(us) if us else None),
               "util_max": (max(us) if us else None),
               "vram_peak_gb": round(peak, 1), "vram_total_gb": round(mt / 1024, 1),
               "n_samples": len(us)}
        print("PROBE", rec)
        out.append(rec)
        del model, opt, x; torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main():
    import json
    res = probe.remote()
    print("GPUPROBE_DONE " + json.dumps(res, indent=2, default=float))
