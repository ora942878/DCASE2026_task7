#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON:-python}"
OUT="serial120_checkpoint3_final_inference_ablation"

cd "${ROOT}"
mkdir -p "runs/${OUT}"
"${PY}" -m py_compile scripts/eval_final_inference_ablation.py
nohup "${PY}" scripts/eval_final_inference_ablation.py \
  --checkpoint runs/serial120_checkpoint3_beta080_mean5/checkpoint_D3_method_beta080_mean5.pth \
  --out-name "${OUT}" \
  --task-id 2 \
  --window-sec 3.0 4.0 5.0 \
  --center-sets quint5:0.1,0.3,0.5,0.7,0.9 dense7:0.05,0.2,0.35,0.5,0.65,0.8,0.95 \
  --batch-size 256 \
  > "runs/${OUT}/pipeline.log" 2>&1 < /dev/null &
echo "$!" > "runs/${OUT}/pipeline.pid"
echo "started pid=$(cat "runs/${OUT}/pipeline.pid")"
