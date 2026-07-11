#!/usr/bin/env python3
"""Stacker v3 — CatBoost L2 + retargeted meta (net_pnl > BE).

Same retargeted meta target as v2, but L2 stacker is CatBoost instead of
XGBoost. CatBoost's ordered boosting reduces target leakage in stacking
(where train and inference both see primary softmaxes), and its native
NaN handling + better calibration on correlated features often yields
+0.5-2% AUC over XGBoost on tabular meta-stacks.

Output: `models/stacker_meta_v3.npz` with same keys as v2 plus a
`backend` field marking "catboost".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge                                         # noqa: E402
from src.models.meta_label import train_meta, MetaConfig            # noqa: E402
from src.models.stacking import StackerConfig, train_stacker        # noqa: E402


CACHE_DIR = Path("/home/scalper/scalper-bot/data/_cache")
SOFTS_PATH = Path("/home/scalper/scalper-bot/models/primary_softs_v4.npz")
OUT = Path("/home/scalper/scalper-bot/models/stacker_meta_v3.npz")
TRAIN_FRAC = 0.75
UP, DOWN, FLAT = 0, 1, 2

ANCHOR_TP = 0.45
ANCHOR_SL = 0.15
ANCHOR_TIMEOUT = 1200
BE_MARGIN_PCT = 0.03
COMMISSION_WIN = 0.04
COMMISSION_LOSS = 0.07


def _load_cache():
    cand = sorted(CACHE_DIR.glob("samples_v3_*_y.npy"))
    if not cand:
        raise FileNotFoundError(f"No v3 cache in {CACHE_DIR}")
    # Pick largest (avoids leakfree70k being picked over full 999h).
    cand.sort(key=lambda p: p.stat().st_size, reverse=True)
    prefix = str(cand[0])[: -len("_y.npy")]
    print(f"[smo3] using cache prefix: {prefix}")
    return {
        "prefix": prefix,
        "y": np.load(f"{prefix}_y.npy"),
        "pnl": np.load(f"{prefix}_pnl.npy"),
        "mid_paths": np.load(f"{prefix}_mid_paths.npy"),
        "entry_long": np.load(f"{prefix}_entry_long.npy"),
        "entry_short": np.load(f"{prefix}_entry_short.npy"),
    }


def _realized_pnl_at_anchor(c):
    N = c["y"].shape[0]
    tp = np.full(N, ANCHOR_TP, dtype=np.float64)
    sl = np.full(N, ANCHOR_SL, dtype=np.float64)
    to = np.full(N, ANCHOR_TIMEOUT, dtype=np.int64)
    out = rust_bridge.simulate_labels(
        c["entry_long"], c["entry_short"], c["mid_paths"],
        tp, sl, to,
        commission_win_pct=COMMISSION_WIN, commission_loss_pct=COMMISSION_LOSS,
        partial_enabled=True, trailing_enabled=True, fill_latency_ms=150.0,
    )
    return np.stack([out["pnl_long"], out["pnl_short"]], axis=1).astype(np.float32)


def main():
    c = _load_cache()
    d = np.load(SOFTS_PATH, allow_pickle=False)
    soft_keys = sorted(k for k in d.files if k.startswith("soft_"))
    primary_softs = [d[k] for k in soft_keys]
    arch_keys = [k[len("soft_"):] for k in soft_keys]

    n = c["y"].shape[0]
    n_tr = int(TRAIN_FRAC * n)
    print(f"[smo3] N={n:,}  n_tr={n_tr:,}  archs={len(arch_keys)}  backend=catboost")

    X_feat = np.load(f"{c['prefix']}_X_feat.npy")

    # Train CatBoost L2 stacker on first n_tr (walk-forward)
    print("[smo3] training CatBoost stacker on 75% split...")
    cb_cfg = StackerConfig(backend="catboost", n_estimators=500, max_depth=6,
                            learning_rate=0.05)
    # train_stacker expects primary_softmaxes + y_true with internal val split;
    # we pass only train slice to get stacker FIT ON TRAIN ONLY, then predict
    # on the full range.
    softs_train = [s[:n_tr] for s in primary_softs]
    stk_model, stk_metrics = train_stacker(softs_train, c["y"][:n_tr],
                                             X_feat=X_feat[:n_tr],
                                             cfg=cb_cfg, val_frac=0.2)
    # Inference on all 93k via the stacker trained on train slice
    from src.models.stacking import stack_inputs
    X_stack_full = stack_inputs(primary_softs, X_feat=X_feat)
    stacker_soft = stk_model.predict_proba(X_stack_full).astype(np.float32)
    # CatBoost returns (N, 1, 3) for multiclass sometimes — squeeze safely
    if stacker_soft.ndim == 3:
        stacker_soft = stacker_soft.squeeze(axis=1)
    print(f"[smo3] stacker val metrics: {stk_metrics}")

    # Anchor-config PnL for retargeted meta
    print(f"[smo3] simulating anchor TP={ANCHOR_TP} SL={ANCHOR_SL} to={ANCHOR_TIMEOUT}ticks")
    pnl_lg_sh = _realized_pnl_at_anchor(c)

    # Retargeted meta dataset
    primary_pred = stacker_soft.argmax(axis=-1)
    pnl_actual = np.where(
        primary_pred == UP, pnl_lg_sh[:, 0],
        np.where(primary_pred == DOWN, pnl_lg_sh[:, 1], 0.0),
    ).astype(np.float32)

    nf = primary_pred != FLAT
    soft_nf = stacker_soft[nf]
    pred_nf = primary_pred[nf]
    pnl_nf = pnl_actual[nf]

    p_max = soft_nf.max(axis=-1, keepdims=True)
    p_marg = (np.sort(soft_nf, axis=-1)[:, -1] - np.sort(soft_nf, axis=-1)[:, -2])[:, None]
    p_ent = -(soft_nf * np.log(soft_nf.clip(1e-9))).sum(axis=-1, keepdims=True)
    parts = [soft_nf, p_max, p_marg, p_ent,
             (pred_nf == UP).astype(np.float32)[:, None], X_feat[nf].astype(np.float32)]
    X_meta = np.concatenate(parts, axis=-1).astype(np.float32)

    y_meta = (pnl_nf > BE_MARGIN_PCT).astype(np.int64)
    w = np.maximum(np.abs(pnl_nf), 1e-3)
    print(f"[smo3] meta samples={len(y_meta):,}  pos_rate={y_meta.mean():.3f}")

    meta_model, meta_metrics = train_meta(X_meta, y_meta, w, cfg=MetaConfig(), val_frac=0.2)
    print(f"[smo3] meta metrics: {meta_metrics}")

    # Meta infer on all 93k
    meta_prob = np.zeros(n, dtype=np.float32)
    if nf.any():
        soft_all_nf = stacker_soft[nf]
        p_max_all = soft_all_nf.max(axis=-1, keepdims=True)
        p_marg_all = (np.sort(soft_all_nf, axis=-1)[:, -1]
                       - np.sort(soft_all_nf, axis=-1)[:, -2])[:, None]
        p_ent_all = -(soft_all_nf * np.log(soft_all_nf.clip(1e-9))).sum(axis=-1, keepdims=True)
        X_inf = np.concatenate([
            soft_all_nf, p_max_all, p_marg_all, p_ent_all,
            (primary_pred[nf] == UP).astype(np.float32)[:, None],
            X_feat[nf].astype(np.float32),
        ], axis=-1).astype(np.float32)
        meta_prob[nf] = meta_model.predict_proba(X_inf)[:, 1]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        stacker_soft=stacker_soft,
        meta_prob=meta_prob,
        pnl_actual_anchor=pnl_actual,
        arch_keys=np.array(arch_keys),
        y=c["y"],
        meta_metrics=json.dumps(meta_metrics, default=float),
        stacker_metrics=json.dumps(stk_metrics, default=float),
        anchor_config=json.dumps({
            "tp": ANCHOR_TP, "sl": ANCHOR_SL, "timeout_ticks": ANCHOR_TIMEOUT,
            "be_margin_pct": BE_MARGIN_PCT, "backend": "catboost",
        }),
        n_train=n_tr,
    )
    print(f"[smo3] wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")

    # Quick tail eval at meta_thr=0.80
    tail_pred = primary_pred[n_tr:]
    tail_meta = meta_prob[n_tr:]
    tail_pnl = pnl_actual[n_tr:]
    gate = (tail_pred != FLAT) & (tail_meta > 0.80)
    n_take = int(gate.sum())
    if n_take > 0:
        t = tail_pnl[gate]
        wr = (t > 0).mean() * 100
        print(f"\n[smo3] TAIL EVAL @ anchor + meta_thr=0.80:")
        print(f"  n_trades={n_take}  WR={wr:.1f}%  "
              f"avg={t.mean()*100:.2f}bp  sum={t.sum()*100:.2f}%")


if __name__ == "__main__":
    main()
