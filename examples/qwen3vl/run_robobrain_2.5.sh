#!/bin/bash
set -eo pipefail

# Qwen3-VL RoboBrain 2.5 training entry.
# Override the defaults below with environment variables when launching on a
# different cluster, dataset, checkpoint, or parallelism layout.

# -----------------------------------------------------------------------------
# Script locations and job identity
# -----------------------------------------------------------------------------
TIMESTAMP=${TIMESTAMP:-$(date "+%Y%m%d_%H%M")}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/seed17/001688/cchen/kimi-k25}
PATCH_HOME=${PATCH_HOME:-${PROJECT_ROOT}/megatron-lm-musa-patch}
MEGATRON_PATH=${MEGATRON_PATH:-${PROJECT_ROOT}/Megatron-LM}

JOB_NAME=${JOB_NAME:-qwen3_vl}
JOB_ID=${JOB_ID:-${JOB_NAME}_${TIMESTAMP}}

# -----------------------------------------------------------------------------
# Distributed topology
# -----------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-7788}
GPUS_PER_NODE=${GPUS_PER_NODE:-${TQ_GPU_NUM:-8}}
WORLD_SIZE=${WORLD_SIZE:-${num_nodes:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-${POD_RANK:-0}}}
NODE_ADDR=${NODE_ADDR:-$(ip a 2>/dev/null | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1 || true)}
NODE_ADDR=${NODE_ADDR:-$(hostname)}

# -----------------------------------------------------------------------------
# Output, checkpoint, and log paths
# -----------------------------------------------------------------------------
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-${OUTPUT_DIR:-/mnt/seed17/001688/cchen/kimi-k25/tmp}}
JOB_OUTPUT_DIR=${JOB_OUTPUT_DIR:-${OUTPUT_BASE_DIR}/${JOB_ID}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${OUTPUT_BASE_DIR}/tf_logs/${JOB_ID}}
LOG_BASE_DIR=${LOG_BASE_DIR:-${JOB_OUTPUT_DIR}/logs}
WANDB_DIR=${WANDB_DIR:-${JOB_OUTPUT_DIR}/wandb}
CONFIG_DIR=${CONFIG_DIR:-${JOB_OUTPUT_DIR}/configs}

if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    CHECKPOINT_SAVE_DIR=${CHECKPOINT_SAVE_DIR:-${CHECKPOINT_DIR}}
    CHECKPOINT_LOAD_DIR=${CHECKPOINT_LOAD_DIR:-${CHECKPOINT_DIR}}
else
    CHECKPOINT_SAVE_DIR=${CHECKPOINT_SAVE_DIR:-${JOB_OUTPUT_DIR}/checkpoints}
    CHECKPOINT_LOAD_DIR=${CHECKPOINT_LOAD_DIR:-${JOB_OUTPUT_DIR}/checkpoints}
fi

if [[ -n "${CHECKPOINT_LOAD_DIR_ENV:-}" ]]; then
    CHECKPOINT_LOAD_DIR="${CHECKPOINT_LOAD_DIR_ENV}/${JOB_ID}"
fi
if [[ -n "${CHECKPOINT_SAVE_DIR_ENV:-}" ]]; then
    CHECKPOINT_SAVE_DIR="${CHECKPOINT_SAVE_DIR_ENV}/${JOB_ID}"
fi

LOG_TYPES=(training monitor system distributed pids)

# -----------------------------------------------------------------------------
# MUSA/MCCL runtime environment
# -----------------------------------------------------------------------------
DEFAULT_PATH=/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/musa/mudnn_bench/bin:/usr/local/musa/mccl_test:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
DEFAULT_LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/openmpi/lib:/usr/local/musa/lib

export PATH=${TRAINING_PATH:-${DEFAULT_PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-${DEFAULT_LD_LIBRARY_PATH}}
export PYTHONPATH="${PATCH_HOME}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

export LOGLEVEL=${LOGLEVEL:-INFO}
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export MUSA_EXECUTION_TIMEOUT=${MUSA_EXECUTION_TIMEOUT:-600000}
export ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND:-musa}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_CHECK_POINTERS=${MCCL_CHECK_POINTERS:-0}
export MCCL_ALGOS=${MCCL_ALGOS:-1}
export MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-20971520}
export MCCL_CROSS_NIC=${MCCL_CROSS_NIC:-0}
export MCCL_IB_TC=${MCCL_IB_TC:-136}
export MCCL_IB_TIMEOUT=${MCCL_IB_TIMEOUT:-20}
export MCCL_IB_RETRY_CNT=${MCCL_IB_RETRY_CNT:-7}
export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-3}
export MCCL_IB_HCA=${MCCL_IB_HCA:-}
export MCCL_NET_SHARED_BUFFERS=${MCCL_NET_SHARED_BUFFERS:-0}

export MUSA_BLOCK_SCHEDULE_MODE=${MUSA_BLOCK_SCHEDULE_MODE:-1}
export MUSA_BLOCK_DISTRIBUTION_GRANULARITY=${MUSA_BLOCK_DISTRIBUTION_GRANULARITY:-0}

# -----------------------------------------------------------------------------
# Training, model, data, and optimizer defaults
# -----------------------------------------------------------------------------
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_qwen3_vl.py}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/mnt/seed17/001688/cchen/kimi-k25/model/Qwen3-VL-32B-Instruct}
DATA_PATH=${DATA_PATH:-/mnt/seed17/001688/haoran.huang/OneThinker/wds-1}
VISION_ROOT=${VISION_ROOT:-/mnt/seed17/001688/haoran.huang/OneThinker}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/mnt/seed17/001688/cchen/kimi-k25/model/Qwen3-VL-32B-Instruct}
DATALOADER_SAVE_DIR=${DATALOADER_SAVE_DIR:-${CHECKPOINT_SAVE_DIR}/dataloader}

TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}

VISION_RATION=${VISION_RATION:-0.1}
NUM_WORKERS=${NUM_WORKERS:-1}
KV_CHANNELS=${KV_CHANNELS:-128}
NUM_LAYERS=${NUM_LAYERS:-16}
DECODER_FIRST_PIPELINE_NUM_LAYERS=${DECODER_FIRST_PIPELINE_NUM_LAYERS:-}
if [[ -z "${DECODER_FIRST_PIPELINE_NUM_LAYERS}" && "${PP_SIZE}" -gt 1 ]]; then
    DECODER_FIRST_PIPELINE_NUM_LAYERS=$((NUM_LAYERS / PP_SIZE))
fi
HIDDEN_SIZE=${HIDDEN_SIZE:-5120}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-25600}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-64}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-8}
SEQ_LENGTH=${SEQ_LENGTH:-${SEQ_LEN:-4096}}
MAX_PADDING_LENGTH=${MAX_PADDING_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-262144}
NORM_EPSILON=${NORM_EPSILON:-1e-06}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.02}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.0}
HIDDEN_DROPOUT=${HIDDEN_DROPOUT:-0.0}
ROTARY_PERCENT=${ROTARY_PERCENT:-1.0}
ROTARY_BASE=${ROTARY_BASE:-5000000}
ROTARY_SEQ_LEN_INTERPOLATION_FACTOR=${ROTARY_SEQ_LEN_INTERPOLATION_FACTOR:-1}
PATCH_SIZE=${PATCH_SIZE:-16}
EXTRA_VOCAB_SIZE=${EXTRA_VOCAB_SIZE:-293}

TRAIN_ITERS=${TRAIN_ITERS:-50}
EXIT_INTERVAL=${EXIT_INTERVAL:-6000}
EVAL_ITERS=${EVAL_ITERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${GBS:-128}}
SEED=${SEED:-42}
MANUAL_GC_INTERVAL=${MANUAL_GC_INTERVAL:-200}

CLIP_GRAD=${CLIP_GRAD:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
LR=${LR:-1e-05}
MIN_LR=${MIN_LR:-1e-06}
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.03}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
RERUN_MODE=${RERUN_MODE:-disabled}

# -----------------------------------------------------------------------------
# Directory setup and metadata
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Distributed Training Configuration:"
echo "Job Name: ${JOB_NAME}"
echo "Job ID: ${JOB_ID}"
echo "Timestamp: ${TIMESTAMP}"
echo "Output Base Directory: ${OUTPUT_BASE_DIR}"
echo "Job Output Directory: ${JOB_OUTPUT_DIR}"
echo "TensorBoard Directory: ${TENSORBOARD_DIR}"
echo "Checkpoint Load Directory: ${CHECKPOINT_LOAD_DIR}"
echo "Checkpoint Save Directory: ${CHECKPOINT_SAVE_DIR}"
echo "Log Base Directory: ${LOG_BASE_DIR}"
echo "NODE_RANK: ${NODE_RANK}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "NODE_ADDR: ${NODE_ADDR}"
echo "=========================================="

if [[ "${NODE_RANK}" == "0" ]]; then
    echo "Master node creating output directory structure..."
    mkdir -p "${OUTPUT_BASE_DIR}" "${JOB_OUTPUT_DIR}"
    for log_type in "${LOG_TYPES[@]}"; do
        mkdir -p "${LOG_BASE_DIR}/${log_type}"
        echo "Created log directory: ${LOG_BASE_DIR}/${log_type}"
    done
    mkdir -p "${CHECKPOINT_SAVE_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}" "${CONFIG_DIR}"

    env > "${CONFIG_DIR}/env_config.txt"
    cat > "${CONFIG_DIR}/job_metadata.json" <<EOF
{
    "job_id": "${JOB_ID}",
    "job_name": "${JOB_NAME}",
    "timestamp": "${TIMESTAMP}",
    "node_rank": ${NODE_RANK},
    "world_size": ${WORLD_SIZE},
    "master_addr": "${MASTER_ADDR}",
    "master_port": "${MASTER_PORT}",
    "gpus_per_node": ${GPUS_PER_NODE},
    "output_base_dir": "${OUTPUT_BASE_DIR}",
    "job_output_dir": "${JOB_OUTPUT_DIR}",
    "checkpoint_load_dir": "${CHECKPOINT_LOAD_DIR}",
    "checkpoint_save_dir": "${CHECKPOINT_SAVE_DIR}",
    "tensorboard_dir": "${TENSORBOARD_DIR}"
}
EOF

    echo "Output directory structure created successfully"
    echo "Job metadata saved to: ${CONFIG_DIR}/job_metadata.json"
else
    echo "Worker node (rank ${NODE_RANK}) creating necessary directories..."
    mkdir -p "${LOG_BASE_DIR}/distributed" "${LOG_BASE_DIR}/pids" "${CONFIG_DIR}"
    cat > "${CONFIG_DIR}/node_${NODE_RANK}_metadata.json" <<EOF
{
    "node_rank": ${NODE_RANK},
    "node_addr": "${NODE_ADDR}",
    "job_id": "${JOB_ID}",
    "timestamp": "${TIMESTAMP}"
}
EOF
fi

mkdir -p "${CHECKPOINT_SAVE_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}" "${DATALOADER_SAVE_DIR}"
if [[ "${CHECKPOINT_LOAD_DIR}" != "${CHECKPOINT_SAVE_DIR}" ]]; then
    mkdir -p "${CHECKPOINT_LOAD_DIR}"
fi

TRAINING_LOG_DIR="${LOG_BASE_DIR}/training"
DETAILED_LOG_DIR="${LOG_BASE_DIR}/distributed/rank_${NODE_RANK}_${NODE_ADDR}"
TRAINING_LOG_FILE="${TRAINING_LOG_DIR}/rank_${NODE_RANK}_${NODE_ADDR}_${TIMESTAMP}.log"
COMMAND_LOG_FILE="${DETAILED_LOG_DIR}/train_command_${TIMESTAMP}.cmd"
mkdir -p "${TRAINING_LOG_DIR}" "${DETAILED_LOG_DIR}"

# -----------------------------------------------------------------------------
# Megatron argument groups
# -----------------------------------------------------------------------------
DISTRIBUTED_ARGS=(
    --nnodes "${WORLD_SIZE}"
    --nproc_per_node "${GPUS_PER_NODE}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --log_dir "${DETAILED_LOG_DIR}"
    --redirects 0
    --tee 3
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
)

RUNTIME_ARGS=(
    --use-flash-attn
    --use-distributed-optimizer
    --use-mcore-models
    --transformer-impl transformer_engine
    --use-te
    --bf16
    --attention-softmax-in-fp32
    --calculate-per-token-loss
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-log-interval 1
    --log-throughput
    --log-params-norm
    --log-num-zeros-in-grad
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --wandb-save-dir "${WANDB_DIR}"
)

PRETRAINED_CHECKPOINT_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    if [[ -f "${PRETRAINED_CHECKPOINT}/latest_checkpointed_iteration.txt" ]]; then
        PRETRAINED_CHECKPOINT_ARGS=(--pretrained-checkpoint "${PRETRAINED_CHECKPOINT}")
    else
        echo "WARNING: PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT} is not a Megatron checkpoint; skipping --pretrained-checkpoint."
        echo "         Use a converted Megatron checkpoint directory here to initialize from pretrained weights."
    fi
fi

CHECKPOINT_ARGS=(
    --save-interval "${SAVE_INTERVAL}"
    "${PRETRAINED_CHECKPOINT_ARGS[@]}"
    --dataloader-save "${DATALOADER_SAVE_DIR}"
    --ckpt-format torch
    --save "${CHECKPOINT_SAVE_DIR}"
    --load "${CHECKPOINT_LOAD_DIR}"
)

MODEL_ARGS=(
    --vision-ration "${VISION_RATION}"
    --num-workers "${NUM_WORKERS}"
    --kv-channels "${KV_CHANNELS}"
    --qk-layernorm
    --attention-backend flash
    --disable-bias-linear
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --num-query-groups "${NUM_QUERY_GROUPS}"
    --seq-length "${SEQ_LENGTH}"
    --max-padding-length "${MAX_PADDING_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --swiglu
    --normalization RMSNorm
    --norm-epsilon "${NORM_EPSILON}"
    --init-method-std "${INIT_METHOD_STD}"
    --attention-dropout "${ATTENTION_DROPOUT}"
    --hidden-dropout "${HIDDEN_DROPOUT}"
    --group-query-attention
    --no-masked-softmax-fusion
    --untie-embeddings-and-output-weights
    --position-embedding-type mrope
    --rotary-percent "${ROTARY_PERCENT}"
    --rotary-base "${ROTARY_BASE}"
    --rotary-seq-len-interpolation-factor "${ROTARY_SEQ_LEN_INTERPOLATION_FACTOR}"
    --mrope-section 24 20 20
)
if [[ "${PP_SIZE}" -gt 1 && -n "${DECODER_FIRST_PIPELINE_NUM_LAYERS}" ]]; then
    MODEL_ARGS+=(--decoder-first-pipeline-num-layers "${DECODER_FIRST_PIPELINE_NUM_LAYERS}")
fi

VISION_ARGS=(
    --patch-size "${PATCH_SIZE}"
    --disable-vision-class-token
)

TRAINING_ARGS=(
    --seed "${SEED}"
    --train-iters "${TRAIN_ITERS}"
    --exit-interval "${EXIT_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --tp-only-amax-red
    --manual-gc
    --manual-gc-interval "${MANUAL_GC_INTERVAL}"
)

OPTIMIZER_ARGS=(
    --clip-grad "${CLIP_GRAD}"
    --weight-decay "${WEIGHT_DECAY}"
    --adam-beta1 "${ADAM_BETA1}"
    --adam-beta2 "${ADAM_BETA2}"
    --lr "${LR}"
    --min-lr "${MIN_LR}"
    --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
    --lr-decay-style "${LR_DECAY_STYLE}"
)

DATA_ARGS=(
    --no-use-system-prompt
    --data-path "${DATA_PATH}"
    --vision-root "${VISION_ROOT}"
    --dataloader-type external
    --split 100,0,0
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --extra-vocab-size "${EXTRA_VOCAB_SIZE}"
)

CMD=(
    torchrun
    "${DISTRIBUTED_ARGS[@]}"
    "${TRAIN_SCRIPT}"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${RUNTIME_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${CHECKPOINT_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${VISION_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    --rerun-mode "${RERUN_MODE}"
)

printf '%q ' "${CMD[@]}" | tee "${COMMAND_LOG_FILE}"
printf '\n' | tee -a "${COMMAND_LOG_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; command written to ${COMMAND_LOG_FILE}"
    exit 0
fi

echo "Training command constructed successfully, preparing to launch..."
set +e
"${CMD[@]}" 2>&1 | tee "${TRAINING_LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
