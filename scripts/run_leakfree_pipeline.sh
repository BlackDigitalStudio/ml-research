#!/bin/bash
# End-to-end pipeline after leak-free bakeoff finishes on Modal:
#   1. Pull retrained weights from bakeoff-v3-runs Volume to Contabo.
#   2. Re-infer primaries on the FULL 93k cache (Modal A10, retrained
#      weights haven't seen samples 70k..93k).
#   3. Pull primary_softs_v5_leakfree.npz back.
#   4. Train retargeted meta v2 on fresh softs.
#   5. Grid sweep at anchor + wide.
#   6. Validate honest tail.
#
# Expects leak-free bakeoff weights already on bakeoff-v3-runs Volume
# (functions would have written `{arch}_best.pt` to
#  /vol/runs/bakeoff_v3/ — the same filename as before, now from the
#  retrained network).

set -euo pipefail
cd /home/scalper/scalper-bot
source venv/bin/activate

OUT_DIR=/home/scalper/scalper-bot/models/leakfree
mkdir -p "$OUT_DIR"

echo "=== 1/6 pull retrained weights ==="
rm -rf /tmp/metrics_lf
mkdir -p /tmp/metrics_lf
modal volume get bakeoff-v3-runs bakeoff_v3/ /tmp/metrics_lf/ --force 2>&1 | tail -3
ls /tmp/metrics_lf/bakeoff_v3/*_best.pt | wc -l

echo ""
echo "=== 2/6 re-infer primaries on FULL 93k cache ==="
modal run modal_bakeoff/app.py::infer --archs all 2>&1 | tail -5

echo ""
echo "=== 3/6 pull softmaxes ==="
modal volume get bakeoff-v3-runs primary_softs_v4.npz "$OUT_DIR/primary_softs_v5_leakfree.npz" --force 2>&1 | tail -2
ls -la "$OUT_DIR/primary_softs_v5_leakfree.npz"

echo ""
echo "=== 4/6 train retargeted meta v2 on fresh softs ==="
python -c "
import sys, shutil
# Temp-point the stacker_meta_v2 to leakfree softs by symlink
shutil.copy('$OUT_DIR/primary_softs_v5_leakfree.npz', 'models/primary_softs_v4.npz')
print('[lf] symlinked primary_softs_v4.npz → leakfree version')
"
python scripts/build_stacker_meta_v2.py 2>&1 | tail -10
# Save the retargeted stacker/meta with a leakfree tag
cp models/stacker_meta_v2.npz "$OUT_DIR/stacker_meta_v2_leakfree.npz"
echo "[lf] saved $OUT_DIR/stacker_meta_v2_leakfree.npz"

echo ""
echo "=== 5/6 grid sweep with leakfree retargeted meta ==="
python scripts/grid_live_retargeted.py 2>&1 | tail -20
cp models/grid_live_v5_retarget.json "$OUT_DIR/grid_live_v5_leakfree.json"

echo ""
echo "=== 6/6 validate honest tail ==="
python scripts/validate_honest_tail.py 2>&1 | tail -25 | tee "$OUT_DIR/validate_honest_tail.txt"

echo ""
echo "=== DONE ==="
echo "Artifacts in $OUT_DIR:"
ls -la "$OUT_DIR"
