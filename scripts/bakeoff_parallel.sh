#!/bin/bash
# Launch two bakeoff_v2 runs in parallel on separate GPUs.
# Splits the arch list into two halves by weight (heavy on one GPU, light on
# other). Outputs to runs/bakeoff_v2_gpu{0,1}/. Merging step combines them
# into runs/bakeoff_v2/ for downstream stacker + backtest.
#
# Usage:
#   bash scripts/bakeoff_parallel.sh [CACHE_NPZ] [EPOCHS]
#
# Invoked on pod with 2× GPU available.

set -euo pipefail

CACHE="${1:-runs/pipeline_v1/cache.npz}"
EPOCHS="${2:-40}"
OUT_MERGED="runs/bakeoff_v2"
OUT_GPU0="runs/bakeoff_v2_gpu0"
OUT_GPU1="runs/bakeoff_v2_gpu1"

# Balance split — heavy models on GPU 1 (which is idle longer), light on GPU 0.
# Each split has ~equal wall-time estimate (light has 4 small archs, heavy has
# 4 larger + foundation archs).
ARCHS_GPU0=(transformer patchtst mamba tcn)                     # ~60 min total
ARCHS_GPU1=(hybrid_mamba_attn chronos_bolt_base timesfm_2p5_200m moment_large)  # ~3 hr total

log() { echo "[$(date -Iseconds)] $*"; }

mkdir -p "${OUT_GPU0}" "${OUT_GPU1}" "${OUT_MERGED}"

log "GPU 0 archs: ${ARCHS_GPU0[*]}"
log "GPU 1 archs: ${ARCHS_GPU1[*]}"

# Launch in parallel, each pinned to its GPU.
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/bakeoff_v2.py \
    --cache "${CACHE}" --archs "${ARCHS_GPU0[@]}" \
    --epochs "${EPOCHS}" --out "${OUT_GPU0}" \
    > "${OUT_GPU0}/gpu0.log" 2>&1 &
PID0=$!
log "launched GPU 0 pid=${PID0}"

CUDA_VISIBLE_DEVICES=1 python3 -u scripts/bakeoff_v2.py \
    --cache "${CACHE}" --archs "${ARCHS_GPU1[@]}" \
    --epochs "${EPOCHS}" --out "${OUT_GPU1}" \
    > "${OUT_GPU1}/gpu1.log" 2>&1 &
PID1=$!
log "launched GPU 1 pid=${PID1}"

# Tail both logs while running
tail -F "${OUT_GPU0}/gpu0.log" "${OUT_GPU1}/gpu1.log" &
TAIL_PID=$!

# Wait for both
log "waiting for both to finish..."
wait ${PID0}
RC0=$?
wait ${PID1}
RC1=$?
kill ${TAIL_PID} 2>/dev/null || true

log "GPU 0 exit: ${RC0}; GPU 1 exit: ${RC1}"
if [[ ${RC0} -ne 0 || ${RC1} -ne 0 ]]; then
    log "one of the halves failed, abort merge"
    exit 1
fi

# Merge val_predictions.npz + state_dicts + leaderboards into unified
# runs/bakeoff_v2/ for downstream stacker + backtest.
log "merging halves into ${OUT_MERGED}/"
python3 - <<PYEOF
import numpy as np, json, os, shutil
from pathlib import Path

gpu0 = Path("${OUT_GPU0}")
gpu1 = Path("${OUT_GPU1}")
out  = Path("${OUT_MERGED}")

# Leaderboards
lbs = []
for d in (gpu0, gpu1):
    lb_path = d / "leaderboard.json"
    if lb_path.exists():
        with lb_path.open() as f:
            lbs.append(json.load(f))
merged_rows = []
for lb in lbs:
    merged_rows.extend(lb.get("rows", []))
with (out / "leaderboard.json").open("w") as f:
    json.dump({"rows": merged_rows, "source": "bakeoff_parallel"}, f, indent=2)
print(f"merged leaderboard: {len(merged_rows)} rows")

# State dicts — just copy per-model .pt files
for d in (gpu0, gpu1):
    for pt in d.glob("*.pt"):
        shutil.copy2(pt, out / pt.name)
        print(f"copied {pt.name}")

# Merge val_predictions — take first y_val (must match), concat soft_* per model
def _load(d):
    p = d / "val_predictions.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=False))

pred0 = _load(gpu0)
pred1 = _load(gpu1)
if pred0 is None or pred1 is None:
    print("one of halves missing val_predictions.npz, skipping merge")
else:
    # y_val + pnl_val should be identical (same cache + same train_generic split).
    assert np.array_equal(pred0["y_val"], pred1["y_val"]), "y_val mismatch"
    merged = {"y_val": pred0["y_val"]}
    # stacker_soft: average of the two halves' stacker outputs — but stacker
    # was trained on different L1 sets per half. For merged, rebuild below.
    # Carry the raw L1 softmaxes; stacker will be refit on all of them.
    for k in list(pred0.keys()) + list(pred1.keys()):
        if k.startswith("soft_"):
            merged[k] = pred0.get(k, pred1.get(k))
    # Need pnl_val too
    merged["pnl_val"] = pred0.get("pnl_val", pred1.get("pnl_val"))
    # stacker_soft placeholder (to be refit); use gpu0's if present.
    if "stacker_soft" in pred0:
        merged["stacker_soft"] = pred0["stacker_soft"]
    elif "stacker_soft" in pred1:
        merged["stacker_soft"] = pred1["stacker_soft"]
    np.savez(out / "val_predictions.npz", **merged)
    print(f"merged val_predictions: keys={list(merged.keys())}")
PYEOF

log "merge done — outputs in ${OUT_MERGED}/"

# Re-fit stacker on COMBINED L1 softmaxes for proper ensemble.
log "re-fitting stacker on combined L1 outputs..."
python3 - <<PYEOF
import numpy as np, json
from pathlib import Path
from src.models.stacking import train_stacker
from src.models.meta_label import build_meta_dataset, train_meta, MetaConfig

out = Path("${OUT_MERGED}")
preds = np.load(out / "val_predictions.npz", allow_pickle=False)
y_val = preds["y_val"]
pnl_val = preds["pnl_val"]
soft_keys = [k for k in preds.files if k.startswith("soft_")]
print(f"using {len(soft_keys)} L1 softmaxes: {soft_keys}")
softs = [preds[k] for k in soft_keys]

X_feat_val = None  # we don't have X_feat_val here; caller can add later
stacker, stk_metrics = train_stacker(softs, y_val, X_feat=X_feat_val)
stacker.save_model(str(out / "stacker.json"))
print(f"stacker val_acc={stk_metrics['val_acc']:.4f} bal_acc={stk_metrics['val_bal_acc']:.4f}")

from src.models.stacking import predict_stacked
stacker_soft = predict_stacked(stacker, softs, X_feat=X_feat_val, use_feats=False)

X_m, y_m, w_m = build_meta_dataset(stacker_soft, y_val, pnl_val)
meta, meta_metrics = train_meta(X_m, y_m, w_m)
meta.save_model(str(out / "meta.json"))
print(f"meta val_auc={meta_metrics['val_auc']:.4f}")

# Update val_predictions with final stacker_soft
np.savez(out / "val_predictions.npz", **{**dict(preds), "stacker_soft": stacker_soft})

with (out / "ensemble_metrics.json").open("w") as f:
    json.dump({"stacker": stk_metrics, "meta": meta_metrics}, f, indent=2, default=float)
PYEOF

log "PARALLEL BAKE-OFF DONE — results in ${OUT_MERGED}/"
