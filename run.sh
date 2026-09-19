#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
ROOT="$PWD"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
INPUT="${1:-$ROOT/GOAI_eval_public}"
OUTPUT="${2:-$ROOT/../GOAI_pred_xxxxxm429}"
test -x "$PYTHON"
test -f "$INPUT/protocol.json"
if [ ! -f runtime/predict_trajectory_adapter.py ]; then
  "$PYTHON" tools/build_runtime.py --output runtime
fi
"$PYTHON" runtime/predict_trajectory_adapter.py \
  --base models/E3.pt --adapter models/epoch_02.pt \
  --public-root "$INPUT" \
  --output outputs/reproduction \
  --geometry-calibration configs/ligand_geometry_calibration_v1.json \
  --t4-geometry-calibration configs/phys_calibration.json \
  --seed 2026091101 --verify-replay
"$PYTHON" tools/export_submission_predictions.py \
  --public-root "$INPUT" --predictions outputs/reproduction --output "$OUTPUT" \
  --review reproduction_verification.json
