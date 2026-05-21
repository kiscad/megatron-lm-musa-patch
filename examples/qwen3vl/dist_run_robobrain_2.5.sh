#!/bin/bash
set -euo pipefail

# Launch run_robobrain_2.5.sh on every node listed in hostfile.
# The workspace is expected to be mounted at the same path on all nodes.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_HOME=${WORK_HOME:-${SCRIPT_DIR}}
HOSTFILE=${HOSTFILE:-${SCRIPT_DIR}/hostfile}
ROBOBRAIN_SCRIPT=${ROBOBRAIN_SCRIPT:-${SCRIPT_DIR}/run_robobrain_2.5.sh}

TIMESTAMP=${TIMESTAMP:-$(date "+%Y%m%d_%H%M%S")}
JOB_NAME=${JOB_NAME:-qwen3_vl_robobrain}
JOB_ID=${JOB_ID:-${JOB_NAME}_${TIMESTAMP}}
MASTER_PORT=${MASTER_PORT:-7788}
SSH_CMD=${SSH_CMD:-ssh}
SSH_OPTS=${SSH_OPTS:-}
DIST_LAUNCH_DRY_RUN=${DIST_LAUNCH_DRY_RUN:-0}
REDIRECT_REMOTE_LOGS=${REDIRECT_REMOTE_LOGS:-1}
WORK_HOME=$(cd "${WORK_HOME}" && pwd)
HOSTFILE=$(cd "$(dirname "${HOSTFILE}")" && pwd)/$(basename "${HOSTFILE}")
ROBOBRAIN_SCRIPT=$(cd "$(dirname "${ROBOBRAIN_SCRIPT}")" && pwd)/$(basename "${ROBOBRAIN_SCRIPT}")

if [[ ! -f "${HOSTFILE}" ]]; then
    echo "Hostfile not found: ${HOSTFILE}" >&2
    exit 1
fi

if [[ ! -f "${ROBOBRAIN_SCRIPT}" ]]; then
    echo "Training script not found: ${ROBOBRAIN_SCRIPT}" >&2
    exit 1
fi

mapfile -t HOSTS < <(awk '$0 !~ /^[[:space:]]*(#|$)/ {print $1}' "${HOSTFILE}")
if [[ "${#HOSTS[@]}" -eq 0 ]]; then
    echo "No hosts found in ${HOSTFILE}" >&2
    exit 1
fi

NUM_NODES=${#HOSTS[@]}
MASTER_ADDR=${DIST_MASTER_ADDR:-${HOSTS[0]}}

FIRST_HOST_SLOTS=$(
    awk '
        $0 !~ /^[[:space:]]*(#|$)/ {
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^slots=/) {
                    sub(/^slots=/, "", $i)
                    print $i
                    exit
                }
            }
        }
    ' "${HOSTFILE}"
)
GPUS_PER_NODE=${GPUS_PER_NODE:-${TQ_GPU_NUM:-${FIRST_HOST_SLOTS:-8}}}

SLOT_MISMATCHES=$(
    awk -v expected="${GPUS_PER_NODE}" '
        $0 !~ /^[[:space:]]*(#|$)/ {
            host = $1
            slot = ""
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^slots=/) {
                    slot = $i
                    sub(/^slots=/, "", slot)
                }
            }
            if (slot != "" && slot != expected) {
                print host " slots=" slot
            }
        }
    ' "${HOSTFILE}"
)
if [[ -n "${SLOT_MISMATCHES}" ]]; then
    echo "All nodes must use the same GPUS_PER_NODE for torchrun." >&2
    echo "Expected slots=${GPUS_PER_NODE}, but found:" >&2
    echo "${SLOT_MISMATCHES}" >&2
    exit 1
fi

DIST_LOG_DIR=${DIST_LOG_DIR:-${SCRIPT_DIR}/output/${JOB_ID}/dist_launcher}
mkdir -p "${DIST_LOG_DIR}"

# Forward commonly tuned run_robobrain_2.5.sh variables when they are set in the
# launcher environment. Defaults still live in run_robobrain_2.5.sh.
DEFAULT_FORWARD_ENV_VARS=(
    PROJECT_ROOT PATCH_HOME MEGATRON_PATH
    PYTHON_BIN FLAGSCALE_HOME PIP_INDEX_URL AUTO_INSTALL_ENERGON_DEPS
    OUTPUT_BASE_DIR OUTPUT_DIR JOB_OUTPUT_DIR TENSORBOARD_DIR LOG_BASE_DIR WANDB_DIR CONFIG_DIR
    CHECKPOINT_DIR CHECKPOINT_SAVE_DIR CHECKPOINT_LOAD_DIR CHECKPOINT_LOAD_DIR_ENV CHECKPOINT_SAVE_DIR_ENV
    TRAINING_PATH LD_LIBRARY_PATH PYTHONPATH LOGLEVEL
    MUSA_VISIBLE_DEVICES CUDA_DEVICE_MAX_CONNECTIONS MUSA_EXECUTION_TIMEOUT ACCELERATOR_BACKEND OMP_NUM_THREADS
    MCCL_PROTOS MCCL_CHECK_POINTERS MCCL_ALGOS MCCL_BUFFSIZE MCCL_CROSS_NIC MCCL_IB_TC MCCL_IB_TIMEOUT
    MCCL_IB_RETRY_CNT MCCL_IB_GID_INDEX MCCL_IB_HCA MCCL_NET_SHARED_BUFFERS
    MUSA_BLOCK_SCHEDULE_MODE MUSA_BLOCK_DISTRIBUTION_GRANULARITY
    TRAIN_SCRIPT PRETRAINED_CHECKPOINT DATA_PATH VISION_ROOT TOKENIZER_MODEL DATALOADER_SAVE_DIR
    TP_SIZE PP_SIZE CP_SIZE VISION_RATION NUM_WORKERS KV_CHANNELS NUM_LAYERS DECODER_FIRST_PIPELINE_NUM_LAYERS
    HIDDEN_SIZE FFN_HIDDEN_SIZE NUM_ATTENTION_HEADS NUM_QUERY_GROUPS SEQ_LENGTH SEQ_LEN MAX_PADDING_LENGTH
    MAX_POSITION_EMBEDDINGS NORM_EPSILON INIT_METHOD_STD ATTENTION_DROPOUT HIDDEN_DROPOUT ROTARY_PERCENT
    ROTARY_BASE ROTARY_SEQ_LEN_INTERPOLATION_FACTOR PATCH_SIZE MAKE_VOCAB_SIZE_DIVISIBLE_BY EXTRA_VOCAB_SIZE
    TRAIN_ITERS EXIT_INTERVAL EVAL_ITERS SAVE_INTERVAL MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE GBS SEED
    MANUAL_GC_INTERVAL CLIP_GRAD WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 LR MIN_LR LR_WARMUP_FRACTION
    LR_DECAY_STYLE RERUN_MODE DRY_RUN
)

if [[ -n "${FORWARD_ENV_VARS:-}" ]]; then
    read -r -a EXTRA_FORWARD_ENV_VARS <<< "${FORWARD_ENV_VARS}"
else
    EXTRA_FORWARD_ENV_VARS=()
fi

build_remote_payload() {
    local host=$1
    local node_rank=$2
    local remote_log=$3
    local env_args=(
        "TIMESTAMP=${TIMESTAMP}"
        "JOB_NAME=${JOB_NAME}"
        "JOB_ID=${JOB_ID}"
        "HOSTFILE=${HOSTFILE}"
        "MASTER_ADDR=${MASTER_ADDR}"
        "MASTER_PORT=${MASTER_PORT}"
        "WORLD_SIZE=${NUM_NODES}"
        "num_nodes=${NUM_NODES}"
        "NODE_RANK=${node_rank}"
        "RANK=${node_rank}"
        "POD_RANK=${node_rank}"
        "GPUS_PER_NODE=${GPUS_PER_NODE}"
        "TQ_GPU_NUM=${GPUS_PER_NODE}"
        "NODE_ADDR=${host}"
    )
    local var
    for var in "${DEFAULT_FORWARD_ENV_VARS[@]}" "${EXTRA_FORWARD_ENV_VARS[@]}"; do
        if [[ -n "${!var+x}" ]]; then
            env_args+=("${var}=${!var}")
        fi
    done

    local env_command payload
    printf -v env_command '%q ' env "${env_args[@]}" bash "${ROBOBRAIN_SCRIPT}"
    printf -v payload 'cd %q && %s' "${WORK_HOME}" "${env_command}"

    if [[ "${REDIRECT_REMOTE_LOGS}" == "1" ]]; then
        printf -v payload '%s > %q 2>&1' "${payload}" "${remote_log}"
    fi

    printf '%s' "${payload}"
}

echo "=========================================="
echo "Qwen3-VL RoboBrain distributed launcher"
echo "Hostfile: ${HOSTFILE}"
echo "Training script: ${ROBOBRAIN_SCRIPT}"
echo "Work home: ${WORK_HOME}"
echo "Job ID: ${JOB_ID}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NUM_NODES: ${NUM_NODES}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "Launcher log dir: ${DIST_LOG_DIR}"
echo "=========================================="

for idx in "${!HOSTS[@]}"; do
    host=${HOSTS[${idx}]}
    safe_host=${host//[^A-Za-z0-9_.-]/_}
    remote_log="${DIST_LOG_DIR}/node_${idx}_${safe_host}.log"
    remote_payload=$(build_remote_payload "${host}" "${idx}" "${remote_log}")
    remote_cmd="nohup bash -lc $(printf '%q' "${remote_payload}") < /dev/null &"
    echo "Launching rank ${idx} on ${host}"
    echo "Remote log: ${remote_log}"
    if [[ "${DIST_LAUNCH_DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: '
        # shellcheck disable=SC2086
        printf '%q ' "${SSH_CMD}" ${SSH_OPTS} "${host}" "${remote_cmd}"
        printf '\n'
    else
        # shellcheck disable=SC2086
        if ! "${SSH_CMD}" ${SSH_OPTS} -f -n "${host}" "${remote_cmd}"; then
            echo "ERROR: failed to submit training command on ${host}" >&2
            exit 1
        fi
    fi
done

echo "Launch requests submitted."
