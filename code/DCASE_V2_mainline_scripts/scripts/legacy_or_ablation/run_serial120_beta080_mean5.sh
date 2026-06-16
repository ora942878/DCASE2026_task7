#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON:-python}"
SEEDS=(101 202 303 404 505)
EPOCHS=120
BATCH_SIZE=128
NUM_WORKERS=16
LR="1e-5"
ETA_MIN="1e-6"
BETA="0.8"

D2_PREFIX="serial120_d2_ccsg"
D3_PREFIX="serial120_d3_ccsg"
D2_RUN="serial120_d1bn1_to_d2_5paths_balanced_bs128"
D3_RUN="serial120_d2method_to_d3_5paths_balanced_bs128"
INIT_RUN="serial120_stage_inits_beta080_mean5"
D2_COMBINE_RUN="serial120_checkpoint2_beta080_mean5"
D3_COMBINE_RUN="serial120_checkpoint3_beta080_mean5"
PIPE_RUN="serial120_beta080_mean5_pipeline"
STATUS_FILE="${ROOT}/runs/${PIPE_RUN}/status.log"

cd "${ROOT}"
mkdir -p "${ROOT}/runs/${PIPE_RUN}"

status() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "${STATUS_FILE}"
}

train_one() {
  local domain="$1"
  local task_id="$2"
  local run_name="$3"
  local method="$4"
  local init_checkpoint="$5"
  local seed="$6"
  status "TRAIN START domain=${domain} task_id=${task_id} method=${method} seed=${seed} init=${init_checkpoint}"
  "${PY}" scripts/train_full_ft_domain_bn.py \
    --method "${method}" \
    --domain "${domain}" \
    --run-name "${run_name}" \
    --init-checkpoint "${init_checkpoint}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --eta-min "${ETA_MIN}" \
    --task-id "${task_id}" \
    --class-weight balanced \
    --final-checkpoint last \
    --seed "${seed}" \
    --resume \
    --no-progress
  status "TRAIN DONE domain=${domain} method=${method} seed=${seed}"
}

status "PIPELINE START serial 120epoch beta=${BETA} seeds=${SEEDS[*]} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
nvidia-smi || true

status "BUILD D2 RANDOM VIEWS"
"${PY}" scripts/build_random_domain_views.py \
  --domain D2 \
  --seeds "${SEEDS[@]}" \
  --prefix "${D2_PREFIX}" \
  --p-concat 0.5 \
  --p-shift 0.5 \
  --p-gain 0.5 \
  --aug-ratio 1.0 \
  --minority-power 0.5

status "CREATE D2 INIT: official checkpoint_D1 BN1 -> BN2"
"${PY}" scripts/copy_bn_branch.py \
  --checkpoint checkpoint_D1.pth \
  --out-name "${INIT_RUN}" \
  --checkpoint-name checkpoint_D1_bn1_to_bn2_init.pth \
  --source-task-id 0 \
  --target-task-id 1
D2_INIT="runs/${INIT_RUN}/checkpoint_D1_bn1_to_bn2_init.pth"

D2_CKPTS=()
for seed in "${SEEDS[@]}"; do
  method="${D2_PREFIX}_s${seed}"
  train_one "D2" "1" "${D2_RUN}" "${method}" "${D2_INIT}" "${seed}"
  D2_CKPTS+=("runs/${D2_RUN}/${method}/checkpoint_D2_fullft_bn2_last.pth")
done

status "COMBINE CHECKPOINT2: 0.2 checkpoint1+BN1 + 0.8 mean(D2 paths+BN2)"
"${PY}" scripts/combine_serial_stage_paths.py \
  --out-name "${D2_COMBINE_RUN}" \
  --anchor-checkpoint checkpoint_D1.pth \
  --path-checkpoints "${D2_CKPTS[@]}" \
  --source-task-id 0 \
  --target-task-id 1 \
  --beta "${BETA}" \
  --checkpoint-name checkpoint_D2_method_beta080_mean5.pth \
  --eval-name checkpoint2_method_beta080_mean5
D2_METHOD="runs/${D2_COMBINE_RUN}/checkpoint_D2_method_beta080_mean5.pth"

status "BUILD D3 RANDOM VIEWS"
"${PY}" scripts/build_random_domain_views.py \
  --domain D3 \
  --seeds "${SEEDS[@]}" \
  --prefix "${D3_PREFIX}" \
  --p-concat 0.5 \
  --p-shift 0.5 \
  --p-gain 0.5 \
  --aug-ratio 1.0 \
  --minority-power 0.5

status "CREATE D3 INIT: method checkpoint2 BN2 -> BN3"
"${PY}" scripts/copy_bn_branch.py \
  --checkpoint "${D2_METHOD}" \
  --out-name "${INIT_RUN}" \
  --checkpoint-name checkpoint_D2_method_bn2_to_bn3_init.pth \
  --source-task-id 1 \
  --target-task-id 2
D3_INIT="runs/${INIT_RUN}/checkpoint_D2_method_bn2_to_bn3_init.pth"

D3_CKPTS=()
for seed in "${SEEDS[@]}"; do
  method="${D3_PREFIX}_s${seed}"
  train_one "D3" "2" "${D3_RUN}" "${method}" "${D3_INIT}" "${seed}"
  D3_CKPTS+=("runs/${D3_RUN}/${method}/checkpoint_D3_fullft_bn3_last.pth")
done

status "COMBINE CHECKPOINT3: 0.2 checkpoint2_method+BN2 + 0.8 mean(D3 paths+BN3)"
"${PY}" scripts/combine_serial_stage_paths.py \
  --out-name "${D3_COMBINE_RUN}" \
  --anchor-checkpoint "${D2_METHOD}" \
  --path-checkpoints "${D3_CKPTS[@]}" \
  --source-task-id 1 \
  --target-task-id 2 \
  --beta "${BETA}" \
  --checkpoint-name checkpoint_D3_method_beta080_mean5.pth \
  --eval-name checkpoint3_method_beta080_mean5

status "PIPELINE DONE"
