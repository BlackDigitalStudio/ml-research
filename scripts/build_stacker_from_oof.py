#!/usr/bin/env python3
"""Phase 2.1.b — L2 + L3 fit from GCS OOF softmax (TPU-built).

Reads gs://{bucket}/oof/{arch}/fold_{N}/softmax.npy and
gs://{bucket}/oof/{arch}/holdout_softmax.npy uploaded by tpu_inference_oof.sh,
fits XGBoost L2 stacker (multiclass) and L3 LdP meta gate, persists models
and a summary JSON.

Direction-aware reward via rust_bridge.simulate_labels (NOT target_pnl).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rust_bridge  # noqa: E402
from src.cv import CPCV  # noqa: E402

UP, DOWN, FLAT = 0, 1, 2
BE_MARGIN = 0.03


def gcs_ls(prefix: str) -> list[str]:
    r = subprocess.run(["gcloud", "storage", "ls", prefix],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return []
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def gcs_download(src: str, dst: str) -> bool:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["gcloud", "storage", "cp", src, dst],
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def discover_oof_archs(bucket: str, oof_prefix: str = "oof") -> list[str]:
    paths = gcs_ls(f"gs://{bucket}/{oof_prefix}/")
    out = []
    for p in paths:
        arch = p.rstrip("/").split("/")[-1]
        if arch:
            out.append(arch)
    return out


def load_arch_oof(bucket: str, arch: str, n_working: int,
                  fold_splits, cache_dir: Path,
                  oof_prefix: str = "oof") -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (oof_softmax, holdout_softmax, coverage_pct).

    oof_softmax: (n_working, 3) - per-sample test softmax via fold's test_idx
    holdout_softmax: (n_holdout, 3) or empty array if missing
    coverage_pct: 0..1 fraction of n_working covered
    """
    oof = np.zeros((n_working, 3), dtype=np.float32)
    mask = np.zeros(n_working, dtype=bool)
    for fold in range(len(fold_splits)):
        local = cache_dir / arch / f"fold_{fold}" / "softmax.npy"
        gcs_path = f"gs://{bucket}/{oof_prefix}/{arch}/fold_{fold}/softmax.npy"
        if not local.exists():
            if not gcs_download(gcs_path, str(local)):
                continue
        soft = np.load(local)
        _, test_idx = fold_splits[fold]
        if soft.shape[0] != len(test_idx):
            print(f"  WARN {arch} fold={fold}: shape mismatch "
                  f"{soft.shape[0]} vs {len(test_idx)}", flush=True)
            continue
        oof[test_idx] = soft
        mask[test_idx] = True

    cov = float(mask.mean())

    holdout_local = cache_dir / arch / "holdout_softmax.npy"
    holdout_gcs = f"gs://{bucket}/{oof_prefix}/{arch}/holdout_softmax.npy"
    holdout = np.zeros((0, 3), dtype=np.float32)
    if not holdout_local.exists():
        gcs_download(holdout_gcs, str(holdout_local))
    if holdout_local.exists():
        holdout = np.load(holdout_local)

    return oof, holdout, cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="scalper-bot-research-data")
    ap.add_argument("--oof-prefix", default="oof",
                    help="GCS subfolder, e.g. oof or oof_volaware")
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--cache-dir", default="data/_oof_cache")
    ap.add_argument("--cv-n-groups", type=int, default=6)
    ap.add_argument("--cv-k-test", type=int, default=1)
    ap.add_argument("--cv-holdout-frac", type=float, default=0.20)
    ap.add_argument("--cv-purge-indices", type=int, default=2000)
    ap.add_argument("--tp", type=float, default=0.25)
    ap.add_argument("--sl", type=float, default=0.12)
    ap.add_argument("--timeout-ticks", type=int, default=1300)
    ap.add_argument("--only-archs", nargs="+", default=None)
    ap.add_argument("--discover-gcs", action="store_true",
                    help="Force GCS discover even when --only-archs provided")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="Drop arch if OOF coverage < this fraction")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache (only for y, mid_paths, entry_long, entry_short — softmax из GCS)
    p = args.cache_prefix
    print(f"[stk_oof] loading cache {p}", flush=True)
    y = np.load(f"{p}_y.npy")
    mid_paths = np.load(f"{p}_mid_paths.npy")
    entry_long = np.load(f"{p}_entry_long.npy")
    entry_short = np.load(f"{p}_entry_short.npy")
    n_total = len(y)
    n_holdout = int(n_total * args.cv_holdout_frac)
    n_working = n_total - n_holdout
    print(f"[stk_oof] n_total={n_total} working={n_working} holdout={n_holdout}",
          flush=True)

    # CPCV splits
    sample_pseudo_ts = np.arange(n_working).astype(np.int64)
    cpcv = CPCV(n_groups=args.cv_n_groups, k_test=args.cv_k_test,
                sample_ts=sample_pseudo_ts,
                label_horizon_ms=args.cv_purge_indices,
                embargo_pct=0.005)
    fold_splits = list(cpcv.split(n_working))
    print(f"[stk_oof] CPCV {len(fold_splits)} folds", flush=True)

    # Discover archs in GCS oof/
    if args.only_archs and not args.discover_gcs:
        archs = list(args.only_archs)
        print(f"[stk_oof] using --only-archs (skip GCS discover): {archs}", flush=True)
    else:
        archs = discover_oof_archs(args.bucket, args.oof_prefix)
        if args.only_archs:
            archs = [a for a in archs if a in args.only_archs]
        print(f"[stk_oof] OOF archs in GCS ({args.oof_prefix}): {archs}", flush=True)

    arch_oof: dict[str, np.ndarray] = {}
    arch_holdout: dict[str, np.ndarray] = {}
    for arch in sorted(archs):
        t0 = time.time()
        oof, holdout, cov = load_arch_oof(
            args.bucket, arch, n_working, fold_splits, cache_dir,
            oof_prefix=args.oof_prefix)
        dt = time.time() - t0
        if cov < args.min_coverage:
            print(f"[stk_oof] skip {arch}: coverage {cov*100:.1f}% < {args.min_coverage*100:.0f}%",
                  flush=True)
            continue
        print(f"[stk_oof] {arch}: cov={cov*100:.1f}% holdout_n={len(holdout)} "
              f"({dt:.1f}s)", flush=True)
        arch_oof[arch] = oof
        arch_holdout[arch] = holdout

    if not arch_oof:
        print("[stk_oof] no usable archs — abort", flush=True)
        return 1

    # Direction-aware reward via rust_bridge.simulate_labels
    print(f"[stk_oof] simulate_labels TP={args.tp}% SL={args.sl}% "
          f"timeout={args.timeout_ticks}", flush=True)
    sim = rust_bridge.simulate_labels(
        entry_long=entry_long[:n_working],
        entry_short=entry_short[:n_working],
        mid_paths=mid_paths[:n_working],
        tp_pct=np.full(n_working, args.tp, dtype=np.float64),
        sl_pct=np.full(n_working, args.sl, dtype=np.float64),
        timeout_ticks=np.full(n_working, args.timeout_ticks, dtype=np.int64),
        partial_enabled=True, trailing_enabled=True,
    )
    pnl_long_w = sim["pnl_long"]
    pnl_short_w = sim["pnl_short"]
    print(f"[stk_oof] pnl_long mean={pnl_long_w.mean():.4f}% "
          f"pos={(pnl_long_w>0).mean():.3f}", flush=True)
    print(f"[stk_oof] pnl_short mean={pnl_short_w.mean():.4f}% "
          f"pos={(pnl_short_w>0).mean():.3f}", flush=True)

    # Stack per-fold OOF softmaxes
    arch_names = sorted(arch_oof.keys())
    print(f"[stk_oof] stacking {len(arch_names)} archs: {arch_names}", flush=True)
    X_stack = np.concatenate([arch_oof[a] for a in arch_names], axis=1)
    y_w = y[:n_working]
    print(f"[stk_oof] X_stack {X_stack.shape} y_w {y_w.shape}", flush=True)

    # Walk-forward 75/25 — последние 25% working для stacker validation
    n_stk_train = int(n_working * 0.75)
    Xs_tr, ys_tr = X_stack[:n_stk_train], y_w[:n_stk_train]
    Xs_va, ys_va = X_stack[n_stk_train:], y_w[n_stk_train:]
    print(f"[stk_oof] L2 fit: train={len(Xs_tr)} val={len(Xs_va)}", flush=True)

    # sqrt-inv-freq class weights — балансирует FLAT доминирование
    cls_freq = np.array([(ys_tr == c).sum() for c in (UP, DOWN, FLAT)],
                        dtype=np.float64)
    cls_w = 1.0 / np.sqrt(np.maximum(cls_freq, 1.0))
    cls_w = cls_w / cls_w.mean()
    sample_w = cls_w[ys_tr]
    print(f"[stk_oof] class freq UP={int(cls_freq[0])} DN={int(cls_freq[1])} "
          f"FL={int(cls_freq[2])} → weights {cls_w.round(3)}", flush=True)

    stacker = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=20,
    )
    stacker.fit(Xs_tr, ys_tr, sample_weight=sample_w,
                eval_set=[(Xs_va, ys_va)], verbose=False)
    val_proba = stacker.predict_proba(Xs_va)
    val_pred = val_proba.argmax(axis=1)
    val_acc = float((val_pred == ys_va).mean())
    nonflat = val_pred != FLAT
    val_pnfl = float(((val_pred == ys_va) & nonflat).sum() / max(nonflat.sum(), 1))
    print(f"[stk_oof] L2 val acc={val_acc:.3f} prec_nonflat={val_pnfl:.3f}",
          flush=True)
    stacker.save_model(str(out_dir / "stacker_v3.json"))
    print(f"[stk_oof] saved {out_dir/'stacker_v3.json'}", flush=True)

    # L3 meta — fit на val portion (25%), используя direction-aware reward
    val_proba_full = stacker.predict_proba(X_stack)
    val_pred_full = val_proba_full.argmax(axis=1)
    realized = np.where(val_pred_full == UP, pnl_long_w,
                np.where(val_pred_full == DOWN, pnl_short_w, 0.0))
    meta_y = (realized > BE_MARGIN).astype(np.int8)

    meta_mask = val_pred_full != FLAT
    n_meta = int(meta_mask.sum())
    if n_meta < 100:
        print(f"[stk_oof] WARN meta has only {n_meta} non-FLAT — skip L3",
              flush=True)
        return 0

    proba_meta = val_proba_full[meta_mask]
    max_prob = proba_meta.max(axis=1, keepdims=True)
    entropy = (-proba_meta * np.log(proba_meta + 1e-12)).sum(axis=1, keepdims=True)
    X_meta = np.hstack([proba_meta, max_prob, entropy])
    y_meta = meta_y[meta_mask]
    print(f"[stk_oof] L3 meta n={n_meta} pos_rate={y_meta.mean()*100:.1f}%",
          flush=True)

    n_meta_train = int(n_meta * 0.75)
    Xm_tr, ym_tr = X_meta[:n_meta_train], y_meta[:n_meta_train]
    Xm_va, ym_va = X_meta[n_meta_train:], y_meta[n_meta_train:]
    meta = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=15,
    )
    meta.fit(Xm_tr, ym_tr, eval_set=[(Xm_va, ym_va)], verbose=False)
    meta_proba = meta.predict_proba(Xm_va)[:, 1]
    meta_auc_pos = float((meta_proba > 0.5).mean())
    print(f"[stk_oof] L3 meta val: pos_pred_rate={meta_auc_pos:.3f} "
          f"true_pos_rate={ym_va.mean():.3f}", flush=True)
    joblib.dump(meta, out_dir / "meta_v3.pkl")
    print(f"[stk_oof] saved {out_dir/'meta_v3.pkl'}", flush=True)

    # Сохраняем holdout softmax matrix для grid_live (37k, K*3)
    holdout_lens = [len(arch_holdout[a]) for a in arch_names]
    if all(n == holdout_lens[0] and n > 0 for n in holdout_lens):
        H = np.concatenate([arch_holdout[a] for a in arch_names], axis=1)
        np.save(out_dir / "holdout_X_stack.npy", H)
        print(f"[stk_oof] saved {out_dir/'holdout_X_stack.npy'} {H.shape}",
              flush=True)
    else:
        print(f"[stk_oof] WARN holdout shapes mismatch: {holdout_lens}", flush=True)

    summary = {
        "archs_used": arch_names,
        "n_archs": len(arch_names),
        "n_working": n_working,
        "n_holdout": n_holdout,
        "stacker_val_acc": val_acc,
        "stacker_val_pnfl": val_pnfl,
        "meta_val_pos_rate": meta_auc_pos,
        "meta_val_true_rate": float(ym_va.mean()),
        "meta_n": n_meta,
        "tp": args.tp, "sl": args.sl, "timeout_ticks": args.timeout_ticks,
        "BE_MARGIN": BE_MARGIN,
    }
    (out_dir / "stacker_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[stk_oof] saved {out_dir/'stacker_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
