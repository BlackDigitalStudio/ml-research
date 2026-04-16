"""Modal app for the one-shot bakeoff_v3 22-arch sweep.

Architecture:
  * One Modal Volume `bakeoff-v3-cache` holding the v3 sample cache + repo
    code. Uploaded once via `modal volume put` from the host.
  * One Modal Volume `bakeoff-v3-runs` holding `runs/bakeoff_v3/` output
    (per-arch `.pt` + metrics JSON).
  * One image (CUDA 12.1 devel + torch 2.4 + full training stack). First
    build ~15 min, layer-cached afterwards.
  * `train_arch` function — single arch, subprocesses `bakeoff_v3.py`.
    GPU tier is passed at call time via `.with_options(gpu=...)`, so one
    function decorator covers all 22 archs at their correct tier.
  * `infer_primaries` function — re-inference on the full cache (A10).
  * Local `orchestrate()` driver — kicks off all 22 archs in parallel with
    `.spawn()`, waits for each `FunctionCall`, aggregates metrics.

The runtime contract matches `scripts/bakeoff_v3.py` exactly — the Modal
function is a thin wrapper. Everything parity-testable on Contabo stays
parity-testable; the Modal layer adds only orchestration + GPU rental.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


APP_NAME = "scalper-bakeoff-v3"
REPO_ROOT_LOCAL = Path(__file__).resolve().parents[1]   # /home/scalper/scalper-bot

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# CUDA devel base is needed for mamba-ssm --no-build-isolation. Everything
# else is pip-installable. bitsandbytes pinned for time_llm_7b_4bit stability.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .uv_pip_install(
        # core — torch 2.5.1 required for torch.float8_e8m0fnu (Qwen/Time-LLM dep)
        "torch==2.5.1",
        "numpy==2.2.4",
        "pandas==2.2.3",
        "pyarrow==19.0.1",
        "xgboost==2.1.4",
        "lightgbm==4.3.0",
        "scikit-learn==1.6.1",
        "joblib>=1.4",
        # neural archs
        "transformers>=4.43",
        "accelerate>=0.33",
        "peft>=0.12",
        "bitsandbytes>=0.43",
        "chronos-forecasting",
        # foundation model packages (missing in v1 image; caused 9/21 failures)
        "mambapy",      # mamba, hybrid_mamba_attn — Python fallback alongside mamba-ssm CUDA kernels
        "timesfm",      # timesfm_2p5_{200m, multi, unfrozen}
        "momentfm",     # moment_large{, _multi, _unfrozen}
        # observability
        "tqdm",
    )
    # mamba-ssm needs --no-build-isolation and CUDA dev headers; if it fails
    # on the Modal GPU class's CUDA version, the arch is simply skipped by
    # `--skip-on-error` downstream. Don't let it kill the whole image build.
    .run_commands(
        "pip install packaging wheel",
        "pip install mamba-ssm==2.2.2 causal-conv1d==1.4.0 --no-build-isolation "
        "|| echo '[image] mamba-ssm failed — mamba/hybrid_mamba_attn will skip'",
    )
    .env({
        "PYTHONPATH": "/root/scalper-bot",
        "PYTHONUNBUFFERED": "1",
        # HF cache goes to the Volume so foundation-model downloads amortise
        # across all 22 arch invocations (chronos/timesfm/moment/time_llm
        # share tokenisers + weights). Saves 5-15 min per arch on warm runs.
        "HF_HOME": "/vol/cache/hf_home",
        "HUGGINGFACE_HUB_CACHE": "/vol/cache/hf_home/hub",
    })
    # Local-file adds MUST come last — Modal enforces this so that file
    # changes do not invalidate the expensive build layers above.
    .add_local_dir(
        str(REPO_ROOT_LOCAL / "src"),
        remote_path="/root/scalper-bot/src",
    )
    .add_local_file(
        str(REPO_ROOT_LOCAL / "scripts" / "bakeoff_v1.py"),
        remote_path="/root/scalper-bot/scripts/bakeoff_v1.py",
    )
    .add_local_file(
        str(REPO_ROOT_LOCAL / "scripts" / "bakeoff_v3.py"),
        remote_path="/root/scalper-bot/scripts/bakeoff_v3.py",
    )
    .add_local_file(
        str(REPO_ROOT_LOCAL / "scripts" / "infer_primaries_v3.py"),
        remote_path="/root/scalper-bot/scripts/infer_primaries_v3.py",
    )
    .add_local_file(
        str(REPO_ROOT_LOCAL / "main.py"),
        remote_path="/root/scalper-bot/main.py",
    )
)


app = modal.App(APP_NAME, image=image)

cache_volume = modal.Volume.from_name("bakeoff-v3-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("bakeoff-v3-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

CACHE_MOUNT = "/vol/cache"
RUNS_MOUNT = "/vol/runs"
CACHE_PREFIX_DEFAULT = "/vol/cache/samples_v3_999h_1776165949"


# ---------------------------------------------------------------------------
# Arch → GPU tier mapping.
#
# Informed by scripts/bakeoff_v3.py::ARCH_RECIPES (batch_size_target,
# lora_cfg) + per-tier cost table in STRATEGY.md.
# ---------------------------------------------------------------------------
GPU_TIERS = {
    # From-scratch classifiers (≤ 3M params, bs 512 * bs_scale). L4 24 GB fits.
    "transformer":           "L4",
    "tcn":                   "L4",
    "patchtst":              "L4",
    "mamba":                 "L4",
    "hybrid_mamba_attn":     "L4",
    # patchtst_pretrained disabled — it takes a `patchtst_pretrained:<path>`
    # spec where <path> is an SSL-pretrained backbone. We do not run SSL
    # pretraining on Modal in the one-shot; re-enable only if/when we train
    # the SSL backbone first (scripts/pretrain_ssl.py).
    # Frozen foundation — weights frozen, head fine-tunes. VRAM bounded by the
    # full foundation weights loaded; L40S 48 GB covers the largest (MOMENT
    # ~385M, TimesFM ~200M, Chronos-Bolt-base ~100M).
    "chronos_bolt_tiny":     "L4",
    "chronos_bolt_mini":     "L4",
    "chronos_bolt_small":    "L4",
    "chronos_bolt_base":     "L40S",
    "chronos_base_multi":    "L40S",
    "timesfm_2p5_200m":      "L40S",
    "timesfm_2p5_multi":     "L40S",
    "moment_large":          "L40S",
    "moment_large_multi":    "L40S",
    # Unfrozen foundation — LoRA + backbone gradients. A100-40 matches the
    # bakeoff_v1 reference (PRO 6000 48 GB equivalent).
    "chronos_base_unfrozen": "A100-40GB",
    "timesfm_2p5_unfrozen":  "A100-40GB",
    "moment_large_unfrozen": "A100-40GB",
    # Time-LLM — 0.5B/1.5B fit A100-40. 7B-4bit needs A100-80 for NF4 + LoRA.
    "time_llm_0p5b":         "A100-40GB",
    "time_llm_1p5b":         "A100-40GB",
    "time_llm_7b_4bit":      "A100-80GB",
}


# ---------------------------------------------------------------------------
# Training functions — one per GPU tier.
#
# Modal 1.4 `@app.function` is declaration-time only: there is no runtime
# `with_options(gpu=...)` override. So we stamp out one function per GPU
# class, all sharing the same body via `_train_impl`, and the orchestrator
# dispatches by `GPU_TIERS[arch]`.
# ---------------------------------------------------------------------------
def _train_impl(
    arch: str,
    cache_prefix: str,
    epochs_override: int | None,
    batch_size_override: int | None,
    seed: int,
) -> dict:
    import os as _os
    import subprocess as _sp
    import json as _json
    import time as _time
    from pathlib import Path as _P

    out_dir = _P(f"{RUNS_MOUNT}/bakeoff_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    _P("/vol/cache/hf_home/hub").mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "/root/scalper-bot/scripts/bakeoff_v3.py",
        "--cache-prefix", cache_prefix,
        "--archs", arch,
        "--out", str(out_dir),
        "--seed", str(seed),
        "--skip-on-error",
    ]
    if epochs_override is not None:
        cmd += ["--epochs-override", str(epochs_override)]
    if batch_size_override is not None:
        cmd += ["--batch-size-override", str(batch_size_override)]

    env = _os.environ.copy()
    env["PYTHONPATH"] = "/root/scalper-bot"
    env["SCALPER_ENABLE_HEAVY_ARCHS"] = "1"
    env["SCALPER_USE_RUST"] = "0"  # cache already built on Contabo; no
                                     # Rust binaries in the Modal image.

    t0 = _time.monotonic()
    print(f"[{arch}] START gpu={_os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')} "
          f"cmd={' '.join(cmd)}")
    result = _sp.run(cmd, env=env, cwd="/root/scalper-bot")
    wall_min = (_time.monotonic() - t0) / 60.0

    runs_volume.commit()

    # Prefer per-arch `{tag}_metrics.json` — no race between parallel
    # functions. Fall back to summary.json (races under parallelism but
    # at least the per-call write will land if our subprocess was the
    # only active one when it flushed).
    metrics_path = out_dir / f"{arch}_metrics.json"
    summary_path = out_dir / "summary.json"
    arch_result: dict
    if metrics_path.exists():
        try:
            arch_result = _json.loads(metrics_path.read_text())
        except Exception as e:
            arch_result = {"error": f"metrics-read: {type(e).__name__}: {e}"}
    elif summary_path.exists():
        try:
            summary = _json.loads(summary_path.read_text())
            arch_result = summary.get("results", {}).get(arch, {})
        except Exception as e:
            arch_result = {"error": f"summary-read: {type(e).__name__}: {e}"}
    else:
        arch_result = {"error": "no metrics.json and no summary.json"}

    print(f"[{arch}] DONE rc={result.returncode} wall_min={wall_min:.1f} "
          f"result={_json.dumps(arch_result)[:300]}")
    return {
        "arch": arch,
        "returncode": result.returncode,
        "wall_min": wall_min,
        "result": arch_result,
    }


_FN_KWARGS = dict(
    volumes={CACHE_MOUNT: cache_volume, RUNS_MOUNT: runs_volume},
    secrets=[hf_secret],
    timeout=86400,
    cpu=4.0,
    memory=32 * 1024,
    startup_timeout=300,
)


@app.function(gpu="L4", **_FN_KWARGS)
def train_l4(arch, cache_prefix=CACHE_PREFIX_DEFAULT,
              epochs_override=None, batch_size_override=None, seed=42):
    return _train_impl(arch, cache_prefix, epochs_override, batch_size_override, seed)


@app.function(gpu="L40S", **_FN_KWARGS)
def train_l40s(arch, cache_prefix=CACHE_PREFIX_DEFAULT,
                epochs_override=None, batch_size_override=None, seed=42):
    return _train_impl(arch, cache_prefix, epochs_override, batch_size_override, seed)


@app.function(gpu="A100-40GB", **_FN_KWARGS)
def train_a100_40(arch, cache_prefix=CACHE_PREFIX_DEFAULT,
                   epochs_override=None, batch_size_override=None, seed=42):
    return _train_impl(arch, cache_prefix, epochs_override, batch_size_override, seed)


@app.function(gpu="A100-80GB", **_FN_KWARGS)
def train_a100_80(arch, cache_prefix=CACHE_PREFIX_DEFAULT,
                   epochs_override=None, batch_size_override=None, seed=42):
    return _train_impl(arch, cache_prefix, epochs_override, batch_size_override, seed)


TIER_FN = {
    "L4":          train_l4,
    "L40S":        train_l40s,
    "A100-40GB":   train_a100_40,
    "A100-80GB":   train_a100_80,
}


# ---------------------------------------------------------------------------
# Inference function — re-infer primary softmaxes on the full cache.
# ---------------------------------------------------------------------------
@app.function(
    volumes={CACHE_MOUNT: cache_volume, RUNS_MOUNT: runs_volume},
    secrets=[hf_secret],
    gpu="A10",
    timeout=3 * 3600,
    cpu=4.0,
    memory=32 * 1024,
)
def infer_primaries(
    cache_prefix: str = CACHE_PREFIX_DEFAULT,
    archs: str = "all",
) -> dict:
    """Run scripts/infer_primaries_v3.py on the full cache."""
    import os as _os
    import subprocess as _sp
    import time as _time
    from pathlib import Path as _P

    out = _P(f"{RUNS_MOUNT}/primary_softs_v4.npz")
    weights_dir = f"{RUNS_MOUNT}/bakeoff_v3"

    # Expand `all` into whichever archs actually produced a `.pt` file on
    # the Volume — skip misses without crashing.
    if archs == "all":
        pts = sorted(_P(weights_dir).glob("*_best.pt"))
        archs = ",".join(p.stem.replace("_best", "") for p in pts) or "transformer"

    cmd = [
        "python", "/root/scalper-bot/scripts/infer_primaries_v3.py",
        "--cache-dir", f"{CACHE_MOUNT}",
        "--weights-dir", weights_dir,
        "--archs", archs,
        "--out", str(out),
        "--batch-size", "256",
        "--device", "cuda",
    ]
    env = _os.environ.copy()
    env["PYTHONPATH"] = "/root/scalper-bot"
    env["SCALPER_ENABLE_HEAVY_ARCHS"] = "1"

    t0 = _time.monotonic()
    rc = _sp.run(cmd, env=env, cwd="/root/scalper-bot").returncode
    wall_min = (_time.monotonic() - t0) / 60.0
    runs_volume.commit()

    return {
        "returncode": rc,
        "wall_min": wall_min,
        "out": str(out),
        "archs": archs,
    }


# ---------------------------------------------------------------------------
# Smoke test — cheapest possible sanity check.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def smoke():
    """One epoch of `transformer` on L4 (~$0.01, ~1 min)."""
    result = train_l4.remote(
        arch="transformer",
        cache_prefix=CACHE_PREFIX_DEFAULT,
        epochs_override=1,
    )
    print("\n=== SMOKE RESULT ===")
    print(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Full 22-arch orchestrator.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def full_sweep(archs: str = "all", wait: bool = True):
    """Kick off every arch in parallel, tiered by GPU class.

    Progress logged per-completion. Final `aggregated_summary.json` lands on
    the `bakeoff-v3-runs` Volume.
    """
    if archs == "all":
        arch_list = list(GPU_TIERS.keys())
    else:
        arch_list = [a.strip() for a in archs.split(",") if a.strip()]

    print(f"[orchestrate] starting {len(arch_list)} archs in parallel")
    print(f"[orchestrate] tiers: "
          f"{json.dumps({a: GPU_TIERS[a] for a in arch_list}, indent=2)}")

    call_by_arch: dict[str, modal.FunctionCall] = {}
    for arch in arch_list:
        gpu = GPU_TIERS[arch]
        fn = TIER_FN[gpu]
        call = fn.spawn(arch=arch, cache_prefix=CACHE_PREFIX_DEFAULT)
        call_by_arch[arch] = call
        print(f"[orchestrate] spawned {arch}  gpu={gpu}  call_id={call.object_id}")

    if not wait:
        print("[orchestrate] wait=False, returning FunctionCall IDs")
        return {a: c.object_id for a, c in call_by_arch.items()}

    done: dict[str, dict] = {}
    t_start = time.monotonic()
    for arch, call in call_by_arch.items():
        print(f"[orchestrate] waiting on {arch}...")
        try:
            done[arch] = call.get()
        except Exception as e:
            done[arch] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[orchestrate] {arch} FAILED: {e}")
        elapsed = (time.monotonic() - t_start) / 60.0
        print(f"[orchestrate] {arch} finished — {len(done)}/{len(arch_list)} "
              f"  elapsed_total_min={elapsed:.1f}")

    # Cost ballpark — public Modal GPU rates (per hour).
    _GPU_COST_PER_HOUR = {
        "T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95,
        "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95,
        "RTX-PRO-6000": 3.03,
    }
    cost_total = 0.0
    per_arch_cost: dict[str, float] = {}
    for a, res in done.items():
        wall_h = float(res.get("wall_min", 0.0)) / 60.0
        rate = _GPU_COST_PER_HOUR.get(GPU_TIERS.get(a, ""), 1.0)
        c = wall_h * rate
        per_arch_cost[a] = round(c, 3)
        cost_total += c

    summary = {
        "archs": arch_list,
        "gpu_tiers": {a: GPU_TIERS[a] for a in arch_list},
        "results": done,
        "per_arch_cost_usd": per_arch_cost,
        "cost_total_usd_estimate": round(cost_total, 2),
        "total_wall_min": (time.monotonic() - t_start) / 60.0,
    }

    print("\n=== AGGREGATED SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))

    local_out = REPO_ROOT_LOCAL / "runs" / "bakeoff_v3_modal" / "aggregated_summary.json"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[orchestrate] wrote {local_out}  (~${cost_total:.2f} est.)")
    return summary


@app.local_entrypoint()
def infer(archs: str = "all"):
    """Re-infer primary softmaxes on the full cache (A10, ~30 min)."""
    result = infer_primaries.remote(
        cache_prefix=CACHE_PREFIX_DEFAULT,
        archs=archs,
    )
    print("\n=== INFER RESULT ===")
    print(json.dumps(result, indent=2, default=str))
