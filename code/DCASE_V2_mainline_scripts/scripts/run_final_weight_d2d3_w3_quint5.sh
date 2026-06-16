#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PY="${PYTHON:-python}"

OUT="final_weight_d2d3_w3_quint5"
CHECKPOINT="${REPO_ROOT}/checkpoints/ours/Gao_SHNU_task7_1_D3_dictionary.pth"

cd "${ROOT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/data/D2/d2-dev-test" ]]; then
  echo "Missing D2 test wav directory: ${REPO_ROOT}/data/D2/d2-dev-test" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/data/D3/d3-dev-test" ]]; then
  echo "Missing D3 test wav directory: ${REPO_ROOT}/data/D3/d3-dev-test" >&2
  exit 1
fi

mkdir -p "runs/${OUT}"

"${PY}" -m py_compile scripts/eval_final_inference_ablation.py

"${PY}" scripts/eval_final_inference_ablation.py \
  --checkpoint "${CHECKPOINT}" \
  --out-name "${OUT}" \
  --task-id 2 \
  --no-full-clip \
  --window-sec 3.0 \
  --center-sets quint5:0.1,0.3,0.5,0.7,0.9 \
  --batch-size 256 \
  | tee "runs/${OUT}/pipeline.log"

echo
echo "Summary:"
cat "runs/${OUT}/summary.csv"
