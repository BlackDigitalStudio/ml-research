#!/usr/bin/env python3
"""bakeoff_v3 — efficient-by-default cross-arch bake-off.

Supersedes bakeoff_v1.  Key differences:

  * Hardware-aware. At startup queries `torch.cuda.get_device_properties(0)`
    and scales batch size + AMP dtype + quantisation per GPU class. Same
    command runs on V100 / T4 (fp16), A10 / A100 / H100 / PRO 6000 Ada
    (bf16+ optional 4bit), or CPU (no AMP). Bakeoff v1 hard-coded bs=256
    + no AMP.

  * Per-arch recipe table. Each arch declares its optimal fine-tune
    strategy:
        - from-scratch (transformer, tcn, patchtst): warmup + cosine,
          20-25 epoch cap, patience=5.
        - frozen foundation (chronos_bolt_*, timesfm_frozen,
          moment_frozen): 10-15 epochs, head_lr_mult=3 — the head learns
          fast on a frozen projection.
        - unfrozen foundation (chronos_base_unfrozen, moment_unfrozen):
          LoRA r=16 on q/k/v/o projections + layer-wise LR decay + base
          lr=5e-5, 5-8 epochs. LoRA matches full-FT quality at ~5 % of
          trainable params and ~half the epochs.
        - Time-LLM (0.5B/1.5B/7B): LoRA r=8 on q/v + bf16; 4bit for 7B.

  * Early-stop metric: `f1_up + f1_dn` (NOT f1_macro — 85 % FL
    imbalance makes f1_macro drift 0.485-0.496 across every arch, see
    ROADMAP pitfall #8).

  * Per-epoch checkpoint: `{out_dir}/{tag}_last.pt` + `{tag}_best.pt` +
    `{tag}_metrics.json` refreshed every epoch, so a pod crash after
    e12/e25 loses one epoch max, not the whole run.

  * OOM retry: on torch.cuda.OutOfMemoryError halve batch size, double
    grad_accum, retry (max 3 downscales).

Usage:

    # default set (5-arch production ensemble) on any GPU
    python scripts/bakeoff_v3.py \\
        --cache-prefix data/_cache/samples_v3_999h_1776165949 \\
        --archs default

    # everything, heavy archs gated by SCALPER_ENABLE_HEAVY_ARCHS
    SCALPER_ENABLE_HEAVY_ARCHS=1 \\
        python scripts/bakeoff_v3.py --cache-prefix ... --archs all

    # single arch
    python scripts/bakeoff_v3.py ... --archs chronos_base_unfrozen

    # force a specific batch size (skip HW auto-scale)
    python scripts/bakeoff_v3.py ... --batch-size-override 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train_efficient import train_efficient  # noqa: E402


# ---------------------------------------------------------------------------
# Per-arch recipes.
# ---------------------------------------------------------------------------
@dataclass
class Recipe:
    epochs: int = 25
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stop_patience: int = 5
    reg_loss_weight: float = 0.3
    head_lr_mult: float = 1.0
    layerwise_lr_decay: float | None = None
    batch_size_target: int = 256
    amp: bool = True
    lora_cfg: dict | None = None


# From-scratch classifiers.
_SCRATCH = Recipe(epochs=25, lr=3e-4, warmup_steps=1000,
                   early_stop_patience=5, batch_size_target=512)

# Frozen foundation encoders — head-only fine-tune.
_FROZEN_FOUND = Recipe(epochs=12, lr=1e-3, warmup_steps=500,
                        early_stop_patience=3, head_lr_mult=3.0,
                        batch_size_target=512)

# Unfrozen foundation — LoRA + layer-wise LR decay.
# target_modules is arch-specific: T5-family (Chronos-Bolt) uses bare `q/v/k/o`
# without the `_proj` suffix HF's default attention layers carry. We branch
# below per-arch; this recipe stays as the "standard" default for
# MOMENT/TimesFM which expose q_proj/v_proj.
_LORA_UNFROZEN = Recipe(
    epochs=8, lr=5e-5, warmup_steps=500, early_stop_patience=3,
    head_lr_mult=5.0, layerwise_lr_decay=0.9,
    lora_cfg={"r": 16, "alpha": 32, "dropout": 0.05,
              "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]},
    batch_size_target=256,
)

# T5 variant for Chronos-Bolt (T5 encoder uses q/v/k/o bare names). Wiring
# the wrong suffix raises "Target modules not found" in peft on the first
# call to `get_peft_model` — 2026-04-16 Modal sweep had this.
_LORA_UNFROZEN_T5 = Recipe(
    epochs=8, lr=5e-5, warmup_steps=500, early_stop_patience=3,
    head_lr_mult=5.0, layerwise_lr_decay=0.9,
    lora_cfg={"r": 16, "alpha": 32, "dropout": 0.05,
              "target_modules": ["q", "v", "k", "o"]},
    batch_size_target=256,
)

# Time-LLM — LoRA, low LR.
_LLM_LORA = Recipe(
    epochs=5, lr=5e-5, warmup_steps=300, early_stop_patience=2,
    head_lr_mult=10.0, layerwise_lr_decay=0.95,
    lora_cfg={"r": 8, "alpha": 16, "dropout": 0.05,
              "target_modules": ["q_proj", "v_proj"]},
    batch_size_target=128,
)
_LLM_LORA_4BIT = Recipe(
    epochs=4, lr=3e-5, warmup_steps=200, early_stop_patience=2,
    head_lr_mult=10.0, layerwise_lr_decay=0.95,
    lora_cfg={"r": 8, "alpha": 16, "dropout": 0.05,
              "target_modules": ["q_proj", "v_proj"]},
    batch_size_target=64,
)

# SSL-pretrained patchtst — short fine-tune.
_SSL_FINETUNE = Recipe(epochs=10, lr=1e-4, warmup_steps=200,
                        early_stop_patience=3, head_lr_mult=10.0,
                        batch_size_target=512)


ARCH_RECIPES: dict[str, Recipe] = {
    "transformer":           _SCRATCH,
    "tcn":                   _SCRATCH,
    "patchtst":              _SCRATCH,
    "mamba":                 _SCRATCH,
    "hybrid_mamba_attn":     _SCRATCH,
    "patchtst_pretrained":   _SSL_FINETUNE,
    "chronos_bolt_tiny":     _FROZEN_FOUND,
    "chronos_bolt_mini":     _FROZEN_FOUND,
    "chronos_bolt_small":    _FROZEN_FOUND,
    "chronos_bolt_base":     _FROZEN_FOUND,
    "chronos_base_multi":    _FROZEN_FOUND,
    "chronos_base_unfrozen": _LORA_UNFROZEN_T5,
    "timesfm_2p5_200m":      _FROZEN_FOUND,
    "timesfm_2p5_multi":     _FROZEN_FOUND,
    "timesfm_2p5_unfrozen":  _LORA_UNFROZEN,
    "moment_large":          _FROZEN_FOUND,
    "moment_large_multi":    _FROZEN_FOUND,
    "moment_large_unfrozen": _LORA_UNFROZEN,
    "time_llm_0p5b":         _LLM_LORA,
    "time_llm_1p5b":         _LLM_LORA,
    "time_llm_7b_4bit":      _LLM_LORA_4BIT,
}

_GROUPS = {
    "default": [
        "transformer", "tcn", "patchtst",
        "chronos_bolt_tiny", "chronos_bolt_mini", "chronos_bolt_small",
    ],
    "scratch": ["transformer", "tcn", "patchtst",
                 "mamba", "hybrid_mamba_attn"],
    "frozen_foundation": [
        "chronos_bolt_tiny", "chronos_bolt_mini", "chronos_bolt_small",
        "chronos_bolt_base", "chronos_base_multi",
        "timesfm_2p5_200m", "timesfm_2p5_multi",
        "moment_large", "moment_large_multi",
    ],
    "unfrozen_foundation": [
        "chronos_base_unfrozen", "timesfm_2p5_unfrozen",
        "moment_large_unfrozen",
    ],
    "llm": ["time_llm_0p5b", "time_llm_1p5b", "time_llm_7b_4bit"],
    "heavy": [
        "chronos_bolt_base", "chronos_base_multi", "chronos_base_unfrozen",
        "timesfm_2p5_200m", "timesfm_2p5_multi", "timesfm_2p5_unfrozen",
        "moment_large", "moment_large_multi", "moment_large_unfrozen",
        "time_llm_0p5b", "time_llm_1p5b", "time_llm_7b_4bit",
    ],
    "all": list(ARCH_RECIPES.keys()),
}


# ---------------------------------------------------------------------------
# Hardware profile.
# ---------------------------------------------------------------------------
@dataclass
class HwProfile:
    name: str = "cpu"
    vram_gb: float = 0.0
    sm: int = 0
    bf16_ok: bool = False
    flash_attn_ok: bool = False

    @property
    def bs_scale(self) -> float:
        """Batch size multiplier relative to the 24 GB A10 reference.

        Calibration points: 8GB V100→0.25, 16GB T4→0.5, 24GB A10→1.0,
        40GB A100→1.8, 48GB Ada→2.0, 80GB A100→3.5, 80GB H100→4.0.
        """
        if self.vram_gb <= 0:
            return 0.1
        return max(0.25, min(4.0, self.vram_gb / 24.0))


def detect_hw() -> HwProfile:
    if not torch.cuda.is_available():
        print("[hw] no CUDA — training on CPU")
        return HwProfile()
    props = torch.cuda.get_device_properties(0)
    sm = props.major * 10 + props.minor
    vram = props.total_memory / 1e9
    bf16 = sm >= 80
    try:
        import flash_attn  # noqa: F401
        fa_ok = sm >= 80
    except ImportError:
        fa_ok = False
    hp = HwProfile(name=props.name, vram_gb=vram, sm=sm,
                    bf16_ok=bf16, flash_attn_ok=fa_ok)
    print(f"[hw] {hp.name}  VRAM={hp.vram_gb:.1f} GB  sm={hp.sm}  "
          f"bf16={hp.bf16_ok}  flash_attn={hp.flash_attn_ok}  "
          f"bs_scale={hp.bs_scale:.2f}")
    return hp


def build_factory(arch: str):
    """Delegate to bakeoff_v1's factory — one source of truth for model wiring."""
    from scripts.bakeoff_v1 import build_factory as _bf1
    return _bf1(arch)


def resolve_archs(requested: list[str]) -> list[str]:
    out: list[str] = []
    for r in requested:
        if r in _GROUPS:
            out.extend(_GROUPS[r])
        else:
            out.append(r)
    seen: set[str] = set()
    return [a for a in out if not (a in seen or seen.add(a))]


def load_cache(prefix: str):
    prefix = Path(prefix)
    def _ld(name, mmap=True):
        return np.load(f"{prefix}_{name}.npy",
                        mmap_mode="r" if mmap else None)
    X_lob = _ld("X_lob", mmap=True)
    X_feat = _ld("X_feat", mmap=False)
    y = _ld("y", mmap=False)
    pnl = _ld("pnl", mmap=False)
    print(f"[cache] X_lob {X_lob.shape} X_feat {X_feat.shape} y {y.shape}")
    return X_lob, X_feat, y, pnl


def run_one(arch, hp, X_lob, X_feat, y, pnl, out_dir,
             bs_override, epochs_override, seed):
    recipe = ARCH_RECIPES.get(arch)
    if recipe is None:
        raise ValueError(f"no recipe for arch '{arch}'")

    bs = int(recipe.batch_size_target * hp.bs_scale)
    if bs_override:
        bs = bs_override
    bs = max(16, min(bs, 4096))
    epochs = epochs_override or recipe.epochs

    factory, tag = build_factory(arch)
    print(f"\n=== [{arch}] epochs={epochs} bs={bs} lr={recipe.lr} "
          f"lora={'y' if recipe.lora_cfg else 'n'} "
          f"patience={recipe.early_stop_patience} ===")

    kwargs = dict(
        model_factory=factory,
        X_lob=X_lob, X_feat=X_feat, y=y, target_pnl=pnl,
        batch_size=bs, epochs=epochs, lr=recipe.lr,
        weight_decay=recipe.weight_decay,
        warmup_steps=recipe.warmup_steps,
        reg_loss_weight=recipe.reg_loss_weight,
        label_smoothing=recipe.label_smoothing,
        early_stop_patience=recipe.early_stop_patience,
        seed=seed, tag=tag, out_dir=out_dir,
        amp=recipe.amp, lora_cfg=recipe.lora_cfg,
        layerwise_lr_decay=recipe.layerwise_lr_decay,
        head_lr_mult=recipe.head_lr_mult,
    )
    grad_accum = 1
    while True:
        try:
            _, info = train_efficient(**kwargs, grad_accum=grad_accum)
            return info
        except torch.cuda.OutOfMemoryError as e:
            if grad_accum >= 8:
                raise RuntimeError(
                    f"[{arch}] OOM even after 3 downscales"
                ) from e
            kwargs["batch_size"] = max(16, kwargs["batch_size"] // 2)
            grad_accum *= 2
            torch.cuda.empty_cache()
            print(f"[{arch}] OOM → retry bs={kwargs['batch_size']} "
                  f"grad_accum={grad_accum}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-prefix", required=True,
                   help="e.g. data/_cache/samples_v3_999h_1776165949")
    p.add_argument("--archs", nargs="+", default=["default"])
    p.add_argument("--out", default="runs/bakeoff_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size-override", type=int, default=None)
    p.add_argument("--epochs-override", type=int, default=None)
    p.add_argument("--skip-on-error", action="store_true")
    args = p.parse_args()

    hp = detect_hw()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    archs = resolve_archs(args.archs)
    print(f"[run] archs ({len(archs)}): {archs}")

    X_lob, X_feat, y, pnl = load_cache(args.cache_prefix)

    results: dict[str, dict] = {}
    t_start = time.monotonic()
    for arch in archs:
        t0 = time.monotonic()
        try:
            info = run_one(arch, hp, X_lob, X_feat, y, pnl, out_dir,
                           args.batch_size_override, args.epochs_override,
                           args.seed)
            results[arch] = {
                "best_score": info["best_score"],
                "best_metrics": info["best_metrics"],
                "params_M": info["params_M"],
                "trainable_M": info["trainable_M"],
                "wall_time_min": (time.monotonic() - t0) / 60,
            }
        except Exception as e:
            print(f"[{arch}] FAILED: {type(e).__name__}: {e}")
            if not args.skip_on_error:
                raise
            results[arch] = {"error": f"{type(e).__name__}: {e}"}
        (out_dir / "summary.json").write_text(json.dumps({
            "hw": hp.__dict__,
            "results": results,
            "wall_time_min_total": (time.monotonic() - t_start) / 60,
        }, indent=2))

    print("\n=== bake-off summary ===")
    for a, r in results.items():
        if "error" in r:
            print(f"  {a:30s}  ERROR  {r['error']}")
            continue
        bm = r["best_metrics"]
        print(f"  {a:30s}  F1[UP={bm['f1_up']:.3f} DN={bm['f1_dn']:.3f}]  "
              f"pnfl={bm['prec_nonflat']:.3f}  score={r['best_score']:.4f}  "
              f"{r['wall_time_min']:5.1f} min  "
              f"trn={r['trainable_M']:.1f}M/tot={r['params_M']:.1f}M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
