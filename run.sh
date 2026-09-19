#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python runtime/predict_trajectory_adapter.py \
  --base models/E3.pt --adapter models/epoch_02.pt \
  --public-root "${1:?Usage: bash run.sh INPUT_DIRECTORY OUTPUT_DIRECTORY}" \
  --output "${2:?Usage: bash run.sh INPUT_DIRECTORY OUTPUT_DIRECTORY}" \
  --geometry-calibration configs/ligand_geometry_calibration_v1.json \
  --t4-geometry-calibration configs/phys_calibration.json \
  --seed 2026091101 --verify-replay
