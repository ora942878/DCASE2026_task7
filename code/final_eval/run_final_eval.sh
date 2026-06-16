#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${HERE}/.." && pwd)"
RELEASE_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
SUBMISSION_LABEL="Gao_SHNU_task7_1"
SUBMISSION_DIR="${HERE}/submission/${SUBMISSION_LABEL}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${HERE}/predict_official_eval.py" \
  --audio-dir "${RELEASE_ROOT}/data/eval" \
  --checkpoint "${RELEASE_ROOT}/checkpoints/ours/${SUBMISSION_LABEL}_D3_dictionary.pth" \
  --output-csv "${SUBMISSION_DIR}/${SUBMISSION_LABEL}.output.csv" \
  --manifest "${SUBMISSION_DIR}/${SUBMISSION_LABEL}_manifest.json" \
  --preview-csv "${SUBMISSION_DIR}/${SUBMISSION_LABEL}_preview.csv" \
  --task-id 2 \
  --window-sec 3.0 \
  --centers "0.1,0.3,0.5,0.7,0.9" \
  --batch-size 256
