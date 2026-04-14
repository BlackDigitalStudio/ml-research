#!/usr/bin/env python3
"""Per-class F1 audit — check if bake-off F1 convergence is bug or saturation.

Hypothesis A (saturation): F1_macro is dominated by F1[FL] (stable ~0.88 on
  85% FL dataset). UP/DN F1 vary but are averaged away.
Hypothesis B (bug): metric or training code has a leak that makes all
  archs converge to identical F1.

This reads val_predictions.npz from each GPU's bake-off output and computes:
 1. per-class precision/recall/F1 for every arch
 2. pairwise softmax correlation — if ~1.0 means archs produce the same
    predictions (would be a bug); if <0.9 archs are genuinely different

Usage:
    python scripts/audit_f1_bug.py
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


def f1_per_class(soft, y_true):
    pred = soft.argmax(axis=-1)
    cm = np.zeros((3, 3), dtype=np.int64)
    for t in range(3):
        for p in range(3):
            cm[t, p] = int(((y_true == t) & (pred == p)).sum())
    totals = cm.sum(axis=1)
    preds = cm.sum(axis=0)
    diag = np.diag(cm)
    recalls = np.where(totals > 0, diag / np.maximum(totals, 1), 0.0)
    precisions = np.where(preds > 0, diag / np.maximum(preds, 1), 0.0)
    f1 = np.where((precisions + recalls) > 0,
                  2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12),
                  0.0)
    return precisions, recalls, f1


def main():
    bakeoff_roots = [
        "/workspace/scalper-bot/runs/bakeoff_v2_gpu0",
        "/workspace/scalper-bot/runs/bakeoff_v2_gpu1",
        "/workspace/scalper-bot/runs/bakeoff_v2_gpu2",
    ]

    hdr = ("arch".rjust(22) + "  " +
           "P_UP  P_DN  P_FL   R_UP  R_DN  R_FL   F1_UP F1_DN F1_FL  F1_macro")
    print(hdr)

    all_softs = {}
    y_val_ref = None

    for root in bakeoff_roots:
        vp = Path(root) / "val_predictions.npz"
        if not vp.exists():
            print(f"  ({vp}: not yet present)")
            continue
        data = dict(np.load(vp, allow_pickle=False))
        y_val = data["y_val"]
        if y_val_ref is None:
            y_val_ref = y_val
        for k in data:
            if not k.startswith("soft_"):
                continue
            arch = k[5:]
            soft = data[k]
            p, r, f1 = f1_per_class(soft, y_val)
            macro = float(f1.mean())
            print(f"{arch:>22s}  "
                  f"{p[0]:4.2f}  {p[1]:4.2f}  {p[2]:4.2f}   "
                  f"{r[0]:4.2f}  {r[1]:4.2f}  {r[2]:4.2f}   "
                  f"{f1[0]:4.2f}  {f1[1]:4.2f}  {f1[2]:4.2f}   "
                  f"{macro:.4f}")
            all_softs[arch] = soft

    print()
    print("=== cross-arch softmax pearson correlation ===")
    archs = list(all_softs.keys())
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            a = all_softs[archs[i]].flatten()
            b = all_softs[archs[j]].flatten()
            corr = float(np.corrcoef(a, b)[0, 1])
            flag = "BUG?" if corr > 0.99 else ""
            print(f"  {archs[i]:>22s} vs {archs[j]:<22s}  corr={corr:.4f}  {flag}")

    # Also check argmax agreement
    print()
    print("=== argmax agreement across archs (same prediction fraction) ===")
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            pa = all_softs[archs[i]].argmax(axis=-1)
            pb = all_softs[archs[j]].argmax(axis=-1)
            agree = float((pa == pb).mean())
            flag = "BUG?" if agree > 0.995 else ""
            print(f"  {archs[i]:>22s} vs {archs[j]:<22s}  agree={agree:.4f}  {flag}")


if __name__ == "__main__":
    main()
