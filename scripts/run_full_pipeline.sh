#!/bin/bash
# Full research pipeline: cache → SSL pretrain → bake-off → backtest.
# Run on the pod where SCALPER_USE_RUST=1 + GPU are available.
#
# Usage:
#     bash scripts/run_full_pipeline.sh [TP_PCT] [SL_PCT]
#
# Outputs go to /workspace/scalper-bot/runs/pipeline_v1/

set -euo pipefail

ROOT="${ROOT:-/workspace/scalper-bot}"
DATA="${DATA:-${ROOT}/data}"
RUNS="${RUNS:-${ROOT}/runs/pipeline_v1}"
TP_PCT="${1:-0.20}"
SL_PCT="${2:-0.10}"
EPOCHS="${EPOCHS:-40}"
HOURS="${HOURS:-200}"

mkdir -p "${RUNS}"
export SCALPER_USE_RUST=1

log() { echo "[$(date -Iseconds)] $*" >&2; }

# ─── Stage 1: training cache ───────────────────────────────────────────────
CACHE_DIR="${DATA}/_cache"
if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A ${CACHE_DIR} 2>/dev/null)" ]]; then
    log "STAGE 1: building training cache (hours=${HOURS})"
    python3 "${ROOT}/scripts/build_cache.py" \
        --hours "${HOURS}" --data-dir "${DATA}" --force \
        > "${RUNS}/stage1_build_cache.log" 2>&1
    log "STAGE 1: done"
else
    log "STAGE 1: cache exists, skip"
fi

# Identify the cache file we just built.
CACHE_NPZ="${RUNS}/cache.npz"
if [[ ! -f "${CACHE_NPZ}" ]]; then
    log "STAGE 1b: repackaging cache into single .npz for bakeoff"
    python3 -c "
import numpy as np, glob, os
cache_dir = '${CACHE_DIR}'
X_lob_file = glob.glob(os.path.join(cache_dir, 'samples_*_X_lob.npy'))[0]
key = os.path.basename(X_lob_file).replace('_X_lob.npy', '').replace('samples_', '')
X_lob = np.load(X_lob_file, mmap_mode='r')
X_feat = np.load(os.path.join(cache_dir, f'samples_{key}_X_feat.npy'))
y = np.load(os.path.join(cache_dir, f'samples_{key}_y.npy'))
pnl = np.load(os.path.join(cache_dir, f'samples_{key}_pnl.npy'))
np.savez('${CACHE_NPZ}', X_lob=X_lob, X_feat=X_feat, y=y, target_pnl=pnl)
print(f'wrote cache.npz: X_lob {X_lob.shape} X_feat {X_feat.shape}')
" > "${RUNS}/stage1b_repack.log" 2>&1
fi

# ─── Stage 2: SSL pretraining (optional, one-time) ─────────────────────────
SSL_DIR="${RUNS}/ssl_pretrain"
SSL_WEIGHTS="${SSL_DIR}/final_backbone.pt"
if [[ ! -f "${SSL_WEIGHTS}" ]]; then
    log "STAGE 2: SSL pretraining on flat Tardis depth"
    python3 "${ROOT}/scripts/pretrain_ssl.py" \
        --data-dir "${DATA}" --output "${SSL_DIR}" \
        --epochs 10 --batch 256 --window 256 --samples-per-epoch 20000 \
        > "${RUNS}/stage2_ssl_pretrain.log" 2>&1 || {
            log "STAGE 2: SSL failed — proceeding without pretrain"
            SSL_WEIGHTS=""
        }
else
    log "STAGE 2: SSL weights exist, skip"
fi

# ─── Stage 3: bake-off ─────────────────────────────────────────────────────
BAKEOFF_DIR="${RUNS}/bakeoff_v2"
if [[ ! -f "${BAKEOFF_DIR}/leaderboard.json" ]]; then
    log "STAGE 3: bake-off (${EPOCHS} epochs per arch)"
    ARCHS=(transformer patchtst mamba hybrid_mamba_attn tcn
           chronos_bolt_tiny chronos_bolt_small chronos_bolt_base
           timesfm_2p5_200m moment_large)
    if [[ -n "${SSL_WEIGHTS}" && -f "${SSL_WEIGHTS}" ]]; then
        ARCHS+=("patchtst_pretrained:${SSL_WEIGHTS}")
    fi
    python3 "${ROOT}/scripts/bakeoff_v2.py" \
        --cache "${CACHE_NPZ}" --archs "${ARCHS[@]}" \
        --epochs "${EPOCHS}" --out "${BAKEOFF_DIR}" \
        > "${RUNS}/stage3_bakeoff.log" 2>&1
    log "STAGE 3: done → ${BAKEOFF_DIR}/leaderboard.json"
else
    log "STAGE 3: leaderboard exists, skip"
fi

# ─── Stage 4: end-to-end backtest ──────────────────────────────────────────
log "STAGE 4: backtest on val predictions"
python3 "${ROOT}/scripts/backtest_ensemble.py" \
    --bakeoff-dir "${BAKEOFF_DIR}" \
    --tp-pct "${TP_PCT}" --sl-pct "${SL_PCT}" \
    --out "${RUNS}/backtest_tp${TP_PCT}_sl${SL_PCT}.json" \
    > "${RUNS}/stage4_backtest.log" 2>&1
log "STAGE 4: done"

log "PIPELINE COMPLETE → ${RUNS}/"
ls -la "${RUNS}/"
