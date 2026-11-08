#!/usr/bin/env bash
# Export ground-truth and per-model Excel columns after all test predictions succeed.
# Read the entire block first so edits during execution cannot change pending commands.
{
set -euo pipefail
if [[ $# -eq 0 || ${1:-} == --help ]]; then
  echo "Usage: bash scripts/predict_and_organize.sh TRAIN_DIR [NEW_OUTPUT_DIR] [--id-tolerance N] [--check-overlap]"
  echo "Example: bash scripts/predict_and_organize.sh results/train/20261006_095933 --id-tolerance 2"
  if [[ ${1:-} == --help ]]; then exit 0; else exit 2; fi
fi
HRM_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HRM_TRAIN_DIR="$1"
shift
HRM_PREDICT_OUTPUT="$HRM_PROJECT_DIR/results/predict/$(date +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && $1 != --* ]]; then
  HRM_PREDICT_OUTPUT="$1"
  shift
fi
HRM_ORGANIZE_ARGS=()
HRM_PREDICT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-overlap)
      HRM_PREDICT_ARGS=(--check-overlap)
      shift ;;
    --id-tolerance)
      if [[ $# -lt 2 || ! ${2:-} =~ ^([0-9]|1[0-7])$ ]]; then
        echo "--id-tolerance requires an integer from 0 to 17." >&2
        exit 2
      fi
      HRM_ORGANIZE_ARGS=(--id-tolerance "$2")
      shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2 ;;
  esac
done
bash "$HRM_PROJECT_DIR/scripts/hrm_python.sh" "$HRM_PROJECT_DIR/predict.py" \
  --train-dir "$HRM_TRAIN_DIR" --output "$HRM_PREDICT_OUTPUT" "${HRM_PREDICT_ARGS[@]}"
exec bash "$HRM_PROJECT_DIR/scripts/hrm_python.sh" "$HRM_PROJECT_DIR/organize_results.py" \
  "$HRM_PREDICT_OUTPUT/predictions.xlsx" "${HRM_ORGANIZE_ARGS[@]}"
}
