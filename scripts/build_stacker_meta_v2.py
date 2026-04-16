#!/usr/bin/env python3
"""Stacker + Meta v2 — retargeted on REALIZED direction-aware PnL.

The old meta target was `primary_pred == y AND target_pnl > 0`.
`target_pnl` is the TB-winner's PnL (oracular direction), not the realized
PnL when primary picks direction X and we execute a trade. As a result the
old meta predicts "classification correctness" (AUC 0.88) but not
"trade profitability" — grid_live showed WR 38% vs BE 45.8% at R:R 2:1
despite that "strong" meta.

New target: for a representative (TP, SL, timeout) config, compute the
direction-aware realized PnL via Rust simulate_labels and label
`y_meta = pnl_actual > BE_margin`. Meta now learns "will this specific
trade turn a profit after commissions", which is what we actually trade on.

Output: `models/stacker_meta_v2.npz` with stacker_soft, meta_prob, meta_metrics
(over a grid of representative configs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge                                 # noqa: E402
from scripts.grid_live import _walk_forward_stacker_meta    # noqa: E402


CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
SOFTS_PATH = Path("/home/scalper/scalper-bot/models/primary_softs_v4.npz")
OUT = Path("/home/scalper/scalper-bot/models/stacker_meta_v2.npz")
TRAIN_FRAC = 0.75
UP, DOWN, FLAT = 0, 1, 2

# The "anchor" config — meta is trained to predict profitability under this
# strategy config. We pick one that wide-grid identified as lowest BE gap
# (R:R 3:1, timeout 120 s). The trained meta generalises to nearby configs.
ANCHOR_TP = 0.45
ANCHOR_SL = 0.15
ANCHOR_TIMEOUT = 1200      # 120 s at 100 ms ticks
BE_MARGIN_PCT = 0.03       # require net PnL > 3 bps (covers commissions + buffer)
COMMISSION_WIN = 0.04
COMMISSION_LOSS = 0.07


def _load_cache():
    cand = sorted(CACHE_DIR.glob("samples_v3_*_y.npy"))
    if not cand:
        raise FileNotFoundError(f"No v3 cache in {CACHE_DIR}")
    # Pick the LARGEST cache by file size — avoids lex-sort preferring
    # `samples_v3_leakfree70k_*` over the full `samples_v3_999h_*` (which
    # holds the complete 93k including the eval tail we want softs for).
    cand.sort(key=lambda p: p.stat().st_size, reverse=True)
    prefix = str(cand[0])[: -len("_y.npy")]
    print(f"[smo2] using cache prefix: {prefix}")
    return {
        "prefix": prefix,
        "y": np.load(f"{prefix}_y.npy"),
        "pnl": np.load(f"{prefix}_pnl.npy"),
        "mid_paths": np.load(f"{prefix}_mid_paths.npy"),
        "entry_long": np.load(f"{prefix}_entry_long.npy"),
        "entry_short": np.load(f"{prefix}_entry_short.npy"),
    }


def _train_stacker_only(primary_softs, y, n_tr, seed=42):
    """Just the stacker step — we need its softs + argmax to build the new
    direction-aware meta target. Reuses logic from grid_live's walk-forward
    but we drop its meta training (we'll train a new meta here)."""
    # Call existing function and just discard the meta it trains. The
    # stacker inside is trained on the first n_tr samples only.
    stacker_soft, _, _ = _walk_forward_stacker_meta(
        primary_softs, y, y.astype(np.float32), n_tr, seed=seed,
    )
    return stacker_soft


def _realized_pnl_at_anchor(c: dict) -> np.ndarray:
    """Run Rust simulate_labels with the anchor (TP, SL, timeout) config
    and return (N, 2) = pnl_long + pnl_short per sample."""
    N = c["y"].shape[0]
    tp_arr = np.full(N, ANCHOR_TP, dtype=np.float64)
    sl_arr = np.full(N, ANCHOR_SL, dtype=np.float64)
    to_arr = np.full(N, ANCHOR_TIMEOUT, dtype=np.int64)
    out = rust_bridge.simulate_labels(
        c["entry_long"], c["entry_short"], c["mid_paths"],
        tp_arr, sl_arr, to_arr,
        commission_win_pct=COMMISSION_WIN,
        commission_loss_pct=COMMISSION_LOSS,
        partial_enabled=True, trailing_enabled=True,
        fill_latency_ms=150.0,
    )
    return np.stack([out["pnl_long"], out["pnl_short"]], axis=1).astype(np.float32)


def _build_retargeted_meta_dataset(stacker_soft, pnl_lg_sh, X_feat=None):
    """Return (X_meta, y_meta_bin, weight) — meta target is realized PnL > 0."""
    primary_pred = stacker_soft.argmax(axis=-1)
    non_flat = primary_pred != FLAT
    if not non_flat.any():
        raise ValueError("stacker predicts 100% FLAT — no meta samples")

    # Realised PnL for the direction stacker picked
    pnl_actual = np.where(
        primary_pred == UP, pnl_lg_sh[:, 0],
        np.where(primary_pred == DOWN, pnl_lg_sh[:, 1], 0.0),
    )
    pnl_actual = pnl_actual.astype(np.float32)

    # Build meta features on non-FLAT rows only
    soft = stacker_soft[non_flat]
    pred = primary_pred[non_flat]
    pnl_nf = pnl_actual[non_flat]

    p_max = soft.max(axis=-1, keepdims=True)
    p_marg = (np.sort(soft, axis=-1)[:, -1] - np.sort(soft, axis=-1)[:, -2])[:, None]
    p_ent = -(soft * np.log(soft.clip(1e-9))).sum(axis=-1, keepdims=True)

    parts = [soft, p_max, p_marg, p_ent,
             (pred == UP).astype(np.float32)[:, None]]
    if X_feat is not None:
        parts.append(X_feat[non_flat].astype(np.float32))
    X_meta = np.concatenate(parts, axis=-1).astype(np.float32)

    # NEW TARGET: realised net PnL above break-even margin
    y_meta = (pnl_nf > BE_MARGIN_PCT).astype(np.int64)

    # Weight by |pnl| so model focuses on large wins/losses
    w = np.maximum(np.abs(pnl_nf), 1e-3)
    return X_meta, y_meta, w, non_flat, pnl_actual


def main():
    c = _load_cache()
    d = np.load(SOFTS_PATH, allow_pickle=False)
    soft_keys = sorted(k for k in d.files if k.startswith("soft_"))
    primary_softs = [d[k] for k in soft_keys]
    arch_keys = [k[len("soft_"):] for k in soft_keys]

    n = c["y"].shape[0]
    n_tr = int(TRAIN_FRAC * n)
    print(f"[smo2] N={n:,}  n_tr={n_tr:,}  archs={len(arch_keys)}")

    # Load X_feat from cache prefix
    X_feat = np.load(f"{c['prefix']}_X_feat.npy")

    # 1. Stacker (walk-forward on first n_tr)
    print("[smo2] training stacker on 75% split...")
    stacker_soft = _train_stacker_only(primary_softs, c["y"], n_tr)

    # 2. Simulate anchor config PnL for all samples
    print(f"[smo2] simulating anchor config TP={ANCHOR_TP} SL={ANCHOR_SL} "
          f"timeout={ANCHOR_TIMEOUT} ticks...")
    pnl_lg_sh = _realized_pnl_at_anchor(c)
    print(f"[smo2] pnl_long mean={pnl_lg_sh[:, 0].mean():.4f} "
          f"pnl_short mean={pnl_lg_sh[:, 1].mean():.4f}")

    # 3. Build NEW meta dataset with realized-PnL target
    X_meta, y_meta, w, non_flat, pnl_actual = _build_retargeted_meta_dataset(
        stacker_soft, pnl_lg_sh, X_feat
    )
    print(f"[smo2] meta samples={len(y_meta):,} "
          f"pos_rate={y_meta.mean():.3f}  mean|w|={w.mean():.3f}")

    # 4. Train meta using the SAME XGBoost trainer but on the new target
    from src.models.meta_label import train_meta, MetaConfig
    meta_model, meta_metrics = train_meta(X_meta, y_meta, w,
                                            cfg=MetaConfig(), val_frac=0.2)

    # 5. Inference over all 93k — meta_prob on non-FLAT rows
    meta_prob = np.zeros(n, dtype=np.float32)
    primary_pred = stacker_soft.argmax(axis=-1)
    nf_all = primary_pred != FLAT
    if nf_all.any():
        soft_nf = stacker_soft[nf_all]
        p_max = soft_nf.max(axis=-1, keepdims=True)
        p_marg = (np.sort(soft_nf, axis=-1)[:, -1] - np.sort(soft_nf, axis=-1)[:, -2])[:, None]
        p_ent = -(soft_nf * np.log(soft_nf.clip(1e-9))).sum(axis=-1, keepdims=True)
        parts = [soft_nf, p_max, p_marg, p_ent,
                 (primary_pred[nf_all] == UP).astype(np.float32)[:, None]]
        if X_feat is not None:
            parts.append(X_feat[nf_all].astype(np.float32))
        X_infer = np.concatenate(parts, axis=-1).astype(np.float32)
        meta_prob[nf_all] = meta_model.predict_proba(X_infer)[:, 1]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        stacker_soft=stacker_soft.astype(np.float32),
        meta_prob=meta_prob,
        pnl_actual_anchor=pnl_actual,
        pnl_long_anchor=pnl_lg_sh[:, 0],
        pnl_short_anchor=pnl_lg_sh[:, 1],
        arch_keys=np.array(arch_keys),
        y=c["y"],
        meta_metrics=json.dumps(meta_metrics, default=float),
        anchor_config=json.dumps({
            "tp": ANCHOR_TP, "sl": ANCHOR_SL, "timeout_ticks": ANCHOR_TIMEOUT,
            "be_margin_pct": BE_MARGIN_PCT,
        }),
        n_train=n_tr,
    )
    print(f"[smo2] wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"[smo2] meta metrics: {meta_metrics}")

    # Quick eval — take trades where meta_prob > 0.5 on TAIL split
    tail_start = n_tr
    tail_pred = primary_pred[tail_start:]
    tail_meta = meta_prob[tail_start:]
    tail_pnl = pnl_actual[tail_start:]
    gate = (tail_pred != FLAT) & (tail_meta > 0.5)
    n_take = int(gate.sum())
    if n_take > 0:
        taken_pnl = tail_pnl[gate]
        print(f"\n[smo2] TAIL EVAL @ anchor config + meta_thr=0.5:")
        print(f"  n_trades={n_take}  "
              f"WR={100*(taken_pnl > 0).mean():.1f}%  "
              f"mean_pnl={taken_pnl.mean():.4f}%  "
              f"sum_pnl={taken_pnl.sum():+.2f}%")


if __name__ == "__main__":
    main()
