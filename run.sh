#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
DATA_DIR="$ROOT_DIR/GOAI_eval_public"
OUTPUT_DIR="$ROOT_DIR/GOAI_pred_xxxxxm429"
SUMMARY="$ROOT_DIR/reproduction_verification.json"

test -x "$PYTHON"
test -f "$DATA_DIR/README.md"
test -f "$DATA_DIR/protocol.json"
test ! -e "$OUTPUT_DIR"
test ! -e "$SUMMARY"

for artifact in \
  quantum_encoder.pt quantum_statistics.hdf5 trajectory_model.pt \
  trajectory_model_t1.pt trajectory_model_t2.pt history_trajectory_t1.pt \
  history_trajectory_t2.pt velocity_correlation_t1.csv \
  t1_phase_targets.csv velocity_correlation.csv expert_gate.pt \
  geometry_coefficients.csv; do
  test -f "$ROOT_DIR/artifacts/$artifact"
done
test -f "$ROOT_DIR/config/model.json"

"$PYTHON" -c "import h5py, MDAnalysis, numpy, rdkit, scipy, torch; assert torch.cuda.is_available()"

cd "$ROOT_DIR"
"$PYTHON" -m tools.predict_public \
  --data "$DATA_DIR" \
  --output "$OUTPUT_DIR" \
  --tiers T1 T2 T3 T4 \
  --device cuda \
  --seed 20260825

"$PYTHON" -m tools.verify_public_output \
  --data "$DATA_DIR" \
  --predictions "$OUTPUT_DIR" \
  --summary "$SUMMARY"
