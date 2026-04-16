#!/bin/bash
# Post-training: pull weights + softmaxes from Modal Volume, inline-train
# L2 stacker + L3 meta on Contabo CPU via grid_live.py, land a
# JSON leaderboard of strategy configs.
#
# Run AFTER:
#   1. `modal run modal_bakeoff/app.py::full_sweep` completes
#   2. `modal run modal_bakeoff/app.py::infer` completes (writes
#      primary_softs_v4.npz to the bakeoff-v3-runs volume)
#
# Output:
#   /home/scalper/scalper-bot/backups/pod/recover_v3/*.pt         (neural primaries)
#   /home/scalper/scalper-bot/models/primary_softs_v4.npz         (softmax cache)
#   /home/scalper/scalper-bot/models/grid_live_v4.json            (strategy grid)

set -euo pipefail
cd /home/scalper/scalper-bot
source venv/bin/activate

RECOVER_DIR="backups/pod/recover_v3"
MODELS_DIR="models"
mkdir -p "${RECOVER_DIR}" "${MODELS_DIR}"

echo "=== 1/3: pulling per-arch weights from Modal Volume ==="
modal volume get bakeoff-v3-runs bakeoff_v3/ "${RECOVER_DIR}/"
ls -la "${RECOVER_DIR}/bakeoff_v3/" | tail -30

echo "=== 2/3: pulling primary_softs_v4.npz ==="
modal volume get bakeoff-v3-runs primary_softs_v4.npz "${MODELS_DIR}/primary_softs_v4.npz"
ls -la "${MODELS_DIR}/primary_softs_v4.npz"

echo "=== 3/3: running grid_live.py in the timing zone ==="
# Cache dir has the 999h cache already. grid_live reads the v3 sidecars +
# primary_softs + runs walk-forward stacker/meta + Rust simulate_labels.
python scripts/grid_live.py \
    --cache-dir data/_cache \
    --primaries "${MODELS_DIR}/primary_softs_v4.npz" \
    --out "${MODELS_DIR}/grid_live_v4.json" \
    --tp 0.15 0.20 0.25 0.30 0.35 \
    --sl 0.10 0.12 0.15 0.18 0.20 \
    --timeout 60 90 120 150 180 \
    --kelly 0.10 0.15 0.20 0.25 \
    --meta-thr 0.50 0.60 0.70 \
    --min-prob 0.50 0.55 0.60 \
    --spread-bps 0 2 \
    --fill-prob 1.0 0.8 \
    2>&1 | tee logs/grid_live_v4.log | tail -60

echo "=== DONE ==="
echo "Top-K configs: jq '.top_by_net[0:5]' ${MODELS_DIR}/grid_live_v4.json"
