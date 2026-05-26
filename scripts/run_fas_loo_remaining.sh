#!/usr/bin/env bash
set -euo pipefail

# Run remaining OCIM leave-one-out experiments in parallel.
# Completed already:
#   OCI -> M
# Remaining here:
#   CIM -> O
#   OIM -> C
#   OCM -> I

export RUN_CONFIG_TEXT="$(cat "$0")"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

ROOT="/share/rongss/test_time/IAPL"
DATASET_PATH="${ROOT}/Datasets/FasImage"
CLIP_PATH="${ROOT}/models/clip/ViT-L-14.pt"
RESULT_ROOT="${ROOT}/results"
LOG_ROOT="${ROOT}/results/loo_ocim_logs"

mkdir -p "${LOG_ROOT}"

run_one() {
    local gpu="$1"
    local test_set="$2"
    local test_tag="$3"
    shift 3
    local train_sets=("$@")

    local train_tag=""
    local s
    for s in "${train_sets[@]}"; do
        case "${s}" in
            OULU_NPU) train_tag+="O" ;;
            CASIA_FASD) train_tag+="C" ;;
            ReplayAttack) train_tag+="I" ;;
            MSU_MFSD) train_tag+="M" ;;
            *)
                echo "[ERROR] Unknown dataset: ${s}" >&2
                return 1
                ;;
        esac
    done

    local train_model_name="fas_${train_tag,,}"
    local test_model_name="tta_${train_tag,,}2${test_tag,,}"
    local train_log="${LOG_ROOT}/${train_tag,,}2${test_tag,,}_train.log"
    local test_log="${LOG_ROOT}/${train_tag,,}2${test_tag,,}_test.log"

    echo "[INFO] GPU ${gpu}: ${train_tag} -> ${test_tag} start"

    CUDA_VISIBLE_DEVICES="${gpu}" WORLD_SIZE= RANK= LOCAL_RANK= python "${ROOT}/main.py" \
        --batchsize 32 \
        --evalbatchsize 32 \
        --clip_path "${CLIP_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --train_selected_subsets "${train_sets[@]}" \
        --test_selected_subsets "${train_sets[@]}" \
        --lr 0.00005 \
        --model_name "${train_model_name}" \
        --dataset FasImage \
        --epoch 5 \
        --lr_drop 10 \
        --gate True \
        --condition True \
        --cond_type dgpdl \
        --smooth True \
        > "${train_log}" 2>&1

    local run_root="${RESULT_ROOT}/${train_model_name}"
    local ckpt
    ckpt="$(find "${run_root}" -type f -path "${run_root}/*/checkpoint_best_hter.pth" | sort | tail -n1)"
    if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
        echo "[ERROR] Missing checkpoint under: ${run_root}" >&2
        return 1
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" WORLD_SIZE= RANK= LOCAL_RANK= python "${ROOT}/main.py" \
        --batchsize 32 \
        --evalbatchsize 32 \
        --clip_path "${CLIP_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --train_selected_subsets "${train_sets[@]}" \
        --test_selected_subsets "${test_set}" \
        --lr 0.005 \
        --model_name "${test_model_name}" \
        --dataset FasImage \
        --epoch 1 \
        --lr_drop 10 \
        --gate True \
        --condition True \
        --cond_type dgpdl \
        --pretrained_model "${ckpt}" \
        --eval \
        --smooth True \
        --tta True \
        --tta_steps 2 \
        --ois True \
        --num_workers 8 \
        > "${test_log}" 2>&1

    echo "[INFO] GPU ${gpu}: ${train_tag} -> ${test_tag} done"
}

# Remaining experiments and GPU binding.
# GPU IDs are editable if needed.
run_one 0 OULU_NPU O CASIA_FASD ReplayAttack MSU_MFSD &
pid1=$!

run_one 1 CASIA_FASD C OULU_NPU ReplayAttack MSU_MFSD &
pid2=$!

run_one 2 ReplayAttack I OULU_NPU CASIA_FASD MSU_MFSD &
pid3=$!

run_one 3 MSU_MFSD M OULU_NPU CASIA_FASD ReplayAttack &
pid4=$!

fails=0
for p in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
    if ! wait "${p}"; then
        fails=$((fails + 1))
    fi
done

if [[ "${fails}" -gt 0 ]]; then
    echo "[ERROR] ${fails} experiment(s) failed. Check logs in ${LOG_ROOT}." >&2
    exit 1
fi

echo "[INFO] All remaining OCIM leave-one-out experiments finished successfully."
