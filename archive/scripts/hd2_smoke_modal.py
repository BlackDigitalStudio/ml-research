#!/usr/bin/env python3
"""HD2 SMOKE on Modal H100 (HD2 rev1 pre-reg, execution order pre-reg->smoke).

Validates, on 1 day of SOL, that the streaming-stateful Mamba-2 pipeline:
  (1) stub orchestration runs (reset/warmup/readout/R1-BCE/metrics) - logic;
  (2) real mamba_ssm.Mamba2 trains on H100 at L=6000  (kernel + readout + loss);
  (3) the long-context cell L=216000 (~whole day, 1 period) fits H100 memory
      with per-layer gradient checkpointing  (the O(N) feasibility claim);
  (4) atomic checkpoint -> resume advances the epoch cursor (preemption-safe);
  (5) measures real H100 fwd+bwd tokens/s -> exact full-tier cost projection.

Run:  python -m modal run scripts/hd2_smoke_modal.py
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
    .add_local_file(str(REPO / ".smoke" / "SOL_2025-05-13.npz"),
                    "/root/SOL_2025-05-13.npz", copy=True)
)
VOL = modal.Volume.from_name("hd2-smoke", create_if_missing=True)
app = modal.App("hd2-smoke")
NPZ = "/root/SOL_2025-05-13.npz"


@app.function(image=IMG, gpu="H100", timeout=2400, volumes={"/ck": VOL})
def smoke():
    import sys, json, time, torch
    sys.path.insert(0, "/root/scripts")
    import hd2_mamba_stream as M

    out = {}
    dev_name = torch.cuda.get_device_name(0)
    print("device:", dev_name, "torch", torch.__version__)

    # (1) stub orchestration on GPU (logic), tiny
    torch.cuda.reset_peak_memory_stats()
    c = M.RunCfg(cell_kind="stub", L=6000, H=600, seed=0, d_model=64,
                 n_layers=2, epochs=2, device="cuda", bf16=False)
    out["stub"] = M.run(NPZ, c)

    # (2) real Mamba2, L=6000, H=600
    torch.cuda.reset_peak_memory_stats()
    c = M.RunCfg(cell_kind="mamba2", L=6000, H=600, seed=0, d_model=256,
                 n_layers=4, d_state=128, epochs=2, device="cuda", bf16=True,
                 ckpt_path="/ck/m2_L6000_H600_s0.pt")
    t = time.time(); r = M.run(NPZ, c); r["wall_s"] = time.time() - t
    r["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    r["tok_per_s_fwdbwd"] = r["tok_per_epoch"] * c.epochs / max(1e-9, r["elapsed_s"])
    out["mamba2_L6000"] = r

    # (3) long-context cell L=216000 (whole-day period) + grad checkpointing
    torch.cuda.reset_peak_memory_stats()
    c = M.RunCfg(cell_kind="mamba2", L=216000, H=1800, seed=0, d_model=256,
                 n_layers=4, d_state=128, epochs=1, device="cuda", bf16=True,
                 grad_ckpt_min_L=20000)
    try:
        t = time.time(); r = M.run(NPZ, c); r["wall_s"] = time.time() - t
        r["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["mamba2_L216000"] = r
    except RuntimeError as e:
        out["mamba2_L216000"] = {"error": str(e)[:300]}

    # (4) resume: re-run (2) with more epochs -> must resume from ckpt
    c = M.RunCfg(cell_kind="mamba2", L=6000, H=600, seed=0, d_model=256,
                 n_layers=4, d_state=128, epochs=4, device="cuda", bf16=True,
                 ckpt_path="/ck/m2_L6000_H600_s0.pt")
    VOL.reload()
    out["resume"] = M.run(NPZ, c)

    print("SMOKE_RESULT_JSON " + json.dumps(out, default=float))
    return out


@app.function(image=IMG, gpu="H100", timeout=2400, volumes={"/ck": VOL})
def bench():
    """Steady-state throughput (excludes one-time Triton compile) for the
    period-batching win + per-L cost calibration. Prints [ep] lines with
    cumulative elapsed; steady-state tok/s = toks_delta / elapsed_delta."""
    import sys, json, torch
    sys.path.insert(0, "/root/scripts")
    import hd2_mamba_stream as M
    res = {}
    configs = [("L6000_bp1", 6000, 1), ("L6000_bp16", 6000, 16),
               ("L36000_bp4", 36000, 4), ("L216000_bp1", 216000, 1)]
    for tag, L, bp in configs:
        torch.cuda.reset_peak_memory_stats()
        logs = []
        c = M.RunCfg(cell_kind="mamba2", L=L, H=600, seed=0, d_model=256,
                     n_layers=4, d_state=128, batch_periods=bp, epochs=4,
                     device="cuda", bf16=True, grad_ckpt_min_L=20000)
        r = M.run(NPZ, c, log=lambda s: (logs.append(s), print(tag, s)))
        res[tag] = {"ep_logs": [x for x in logs if x.startswith("[ep")],
                    "tok_per_epoch": r["tok_per_epoch"],
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    print("BENCH_JSON " + json.dumps(res, default=float))
    return res


@app.local_entrypoint()
def main():
    import json
    r = smoke.remote()
    print(json.dumps(r, indent=2, default=float))


@app.local_entrypoint()
def bench_main():
    import json
    print(json.dumps(bench.remote(), indent=2, default=float))
