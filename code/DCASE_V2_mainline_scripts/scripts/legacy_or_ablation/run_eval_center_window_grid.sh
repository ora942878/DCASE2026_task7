#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON:-python}"
OUT="serial120_checkpoint3_center_window_grid"

cd "${ROOT}"
mkdir -p "runs/${OUT}"
"${PY}" -m py_compile scripts/eval_center_window_grid.py
nohup "${PY}" scripts/eval_center_window_grid.py \
  --checkpoint runs/serial120_checkpoint3_beta080_mean5/checkpoint_D3_method_beta080_mean5.pth \
  --out-name "${OUT}" \
  --task-id 2 \
  --batch-size 256 \
  > "runs/${OUT}/pipeline.log" 2>&1 < /dev/null &
echo "$!" > "runs/${OUT}/pipeline.pid"
echo "started pid=$(cat "runs/${OUT}/pipeline.pid")"
