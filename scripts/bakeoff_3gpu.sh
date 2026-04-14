#!/bin/bash
# Launch bakeoff_v2 across 3 GPUs in parallel.
# Each GPU gets an independent arch subset + its own output dir. After all
# three finish, merges val_predictions.npz + refits stacker on the combined
# L1 softmaxes.
#
# Designed for the 3× PRO 6000 WK pod (306 GB VRAM, 755 GB RAM). Archs
# split by wall-time estimate; heaviest stays on GPU 0 where DataLoader
# warm-up is fastest.
#
# Usage:
#   bash scripts/bakeoff_3gpu.sh [CACHE_NPZ] [EPOCHS]

set -euo pipefail

CACHE="${1:-/workspace/scalper-bot/data/_cache/samples.npz}"
EPOCHS="${2:-40}"
OUT_MERGED="runs/bakeoff_v2"
OUT0="runs/bakeoff_v2_gpu0"
OUT1="runs/bakeoff_v2_gpu1"
OUT2="runs/bakeoff_v2_gpu2"

# Arch split — all CPU-deployable (see HEAVY_ARCHS in bakeoff_v1.py).
# GPU 0: heaviest pair (hybrid + patchtst)
# GPU 1: transformer + chronos_bolt_small (medium)
# GPU 2: mamba + tcn + chronos_bolt_tiny + chronos_bolt_mini (light x4)
ARCHS_GPU0=(hybrid_mamba_attn patchtst)
ARCHS_GPU1=(transformer chronos_bolt_small)
ARCHS_GPU2=(mamba tcn chronos_bolt_tiny chronos_bolt_mini)

log() { echo "[$(date -Iseconds)] $*"; }

mkdir -p "${OUT0}" "${OUT1}" "${OUT2}" "${OUT_MERGED}"

log "GPU 0 archs: ${ARCHS_GPU0[*]}"
log "GPU 1 archs: ${ARCHS_GPU1[*]}"
log "GPU 2 archs: ${ARCHS_GPU2[*]}"

CUDA_VISIBLE_DEVICES=0 python3 -u scripts/bakeoff_v2.py \
    --cache "${CACHE}" --archs "${ARCHS_GPU0[@]}" \
    --epochs "${EPOCHS}" --out "${OUT0}" \
    > "${OUT0}/gpu0.log" 2>&1 &
PID0=$!
log "launched GPU 0 pid=${PID0}"

CUDA_VISIBLE_DEVICES=1 python3 -u scripts/bakeoff_v2.py \
    --cache "${CACHE}" --archs "${ARCHS_GPU1[@]}" \
    --epochs "${EPOCHS}" --out "${OUT1}" \
    > "${OUT1}/gpu1.log" 2>&1 &
PID1=$!
log "launched GPU 1 pid=${PID1}"

CUDA_VISIBLE_DEVICES=2 python3 -u scripts/bakeoff_v2.py \
    --cache "${CACHE}" --archs "${ARCHS_GPU2[@]}" \
    --epochs "${EPOCHS}" --out "${OUT2}" \
    > "${OUT2}/gpu2.log" 2>&1 &
PID2=$!
log "launched GPU 2 pid=${PID2}"

tail -F "${OUT0}/gpu0.log" "${OUT1}/gpu1.log" "${OUT2}/gpu2.log" &
TAIL_PID=$!

log "waiting for all three..."
wait ${PID0}
RC0=$?
wait ${PID1}
RC1=$?
wait ${PID2}
RC2=$?
kill ${TAIL_PID} 2>/dev/null || true

log "GPU 0 exit: ${RC0}; GPU 1 exit: ${RC1}; GPU 2 exit: ${RC2}"
if [[ ${RC0} -ne 0 || ${RC1} -ne 0 || ${RC2} -ne 0 ]]; then
    log "one of three halves failed, abort merge"
    exit 1
fi

log "merging three halves into ${OUT_MERGED}/"
python3 - <<PYEOF
import numpy as np, json, shutil
from pathlib import Path

shares = [Path("${OUT0}"), Path("${OUT1}"), Path("${OUT2}")]
out = Path("${OUT_MERGED}")

# Leaderboards
merged_rows = []
for d in shares:
    lb = d / "leaderboard.json"
    if lb.exists():
        with lb.open() as f:
            merged_rows.extend(json.load(f).get("rows", []))
with (out / "leaderboard.json").open("w") as f:
    json.dump({"rows": merged_rows, "source": "bakeoff_3gpu"}, f, indent=2)
print(f"merged leaderboard: {len(merged_rows)} rows")

# Checkpoints
for d in shares:
    for pt in d.glob("*.pt"):
        shutil.copy2(pt, out / pt.name)
        print(f"copied {pt.name}")

# Val predictions — each half has its own soft_* keys; y_val/pnl_val must match.
def _load(d):
    p = d / "val_predictions.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None

preds = [_load(d) for d in shares]
preds = [p for p in preds if p is not None]
if not preds:
    print("no val_predictions.npz found in any half, skipping merge")
else:
    ref = preds[0]
    merged = {"y_val": ref["y_val"], "pnl_val": ref["pnl_val"]}
    for p in preds:
        assert np.array_equal(p["y_val"], ref["y_val"]), "y_val mismatch across halves"
        for k in p:
            if k.startswith("soft_"):
                merged[k] = p[k]
    # carry stacker_soft from first half as placeholder; refit below
    if "stacker_soft" in ref:
        merged["stacker_soft"] = ref["stacker_soft"]
    np.savez(out / "val_predictions.npz", **merged)
    print(f"merged val_predictions: keys={list(merged.keys())}")
PYEOF

log "refitting stacker + meta on combined L1 softmaxes..."
python3 - <<PYEOF
import numpy as np, json
from pathlib import Path
from src.models.stacking import train_stacker, predict_stacked
from src.models.meta_label import build_meta_dataset, train_meta, MetaConfig

out = Path("${OUT_MERGED}")
preds = np.load(out / "val_predictions.npz", allow_pickle=False)
y_val = preds["y_val"]
pnl_val = preds["pnl_val"]
soft_keys = [k for k in preds.files if k.startswith("soft_")]
print(f"stacker input: {len(soft_keys)} L1 softmaxes — {soft_keys}")
softs = [preds[k] for k in soft_keys]

stacker, stk_metrics = train_stacker(softs, y_val, X_feat=None)
stacker.save_model(str(out / "stacker.json"))
print(f"stacker val_acc={stk_metrics['val_acc']:.4f} bal_acc={stk_metrics['val_bal_acc']:.4f}")

stacker_soft = predict_stacked(stacker, softs, X_feat=None, use_feats=False)

X_m, y_m, w_m = build_meta_dataset(stacker_soft, y_val, pnl_val)
meta, meta_metrics = train_meta(X_m, y_m, w_m)
meta.save_model(str(out / "meta.json"))
print(f"meta val_auc={meta_metrics['val_auc']:.4f}")

np.savez(out / "val_predictions.npz", **{**dict(preds), "stacker_soft": stacker_soft})
with (out / "ensemble_metrics.json").open("w") as f:
    json.dump({"stacker": stk_metrics, "meta": meta_metrics}, f, indent=2, default=float)
PYEOF

log "DONE — results in ${OUT_MERGED}/"
