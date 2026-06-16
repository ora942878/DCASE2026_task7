#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON:-python}"
OUT="serial120_checkpoint2_w3_quint5_maxconf"

cd "${ROOT}"
mkdir -p "runs/${OUT}"
"${PY}" -m py_compile scripts/eval_final_inference_ablation.py
"${PY}" scripts/eval_final_inference_ablation.py \
  --checkpoint runs/serial120_checkpoint2_beta080_mean5/checkpoint_D2_method_beta080_mean5.pth \
  --out-name "${OUT}" \
  --task-id 1 \
  --no-full-clip \
  --window-sec 3.0 \
  --center-sets quint5:0.1,0.3,0.5,0.7,0.9 \
  --batch-size 256 \
  > "runs/${OUT}/pipeline.log" 2>&1
cat "runs/${OUT}/summary.csv"
