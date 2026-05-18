#!/bin/bash
set -eo pipefail

TIMESTAMP=$(date "+%Y%m%d_%H%M")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Set up environment variables
export PATH=/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/musa/mudnn_bench/bin:/usr/local/musa/mccl_test:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/lib/x86_64-linux-gnu:/usr/local/openmpi/lib

# Install required packages
# echo "=========================================="
# echo "Installing required packages..."
# echo "=========================================="

# # Install Megatron-Energon
# echo "Installing Megatron-Energon..."
# cd /mnt/seed17/001688/haoran.huang/FlagScale/ && pip install -r requirements/requirements-base.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# cd /mnt/seed17/001688/haoran.huang/FlagScale/third_party/Megatron-Energon/ && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# echo "=========================================="
# echo "Package installation completed successfully"
# echo "=========================================="

# Get configuration from environment variables
NODE_RANK=${RANK:-1}
# WORLD_SIZE=1
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# NODE_RANK=0
# MASTER_ADDR="127.0.0.1"
MASTER_PORT=${MASTER_PORT:-"7788"}
GPUS_PER_NODE=${TQ_GPU_NUM:-8}

# Get local IP address
NODE_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)

# Calculate number of nodes
if [ -z "$WORLD_SIZE" ] && [ -n "$num_nodes" ]; then
    WORLD_SIZE=$num_nodes
fi

if [ -z "$NODE_RANK" ] && [ -n "$POD_RANK" ]; then
    NODE_RANK=$POD_RANK
fi

# ============================================================
# Improved timestamp and directory structure
# ============================================================

# Generate timestamp with minute precision (format: YYYYMMDD_HHMM)
JOB_NAME=${JOB_NAME:-"qwen3_vl"}

# Create job identifier with timestamp
JOB_ID="${JOB_NAME}_${TIMESTAMP}"

# ============================================================
# 1. Support getting output directory from OUTPUT_DIR environment variable
# ============================================================
# Priority: use OUTPUT_DIR environment variable, otherwise use default path
if [ -n "${OUTPUT_DIR}" ]; then
    OUTPUT_BASE_DIR="${OUTPUT_DIR}"
    echo "Using OUTPUT_DIR from environment variable: ${OUTPUT_BASE_DIR}"
else
    OUTPUT_BASE_DIR="/mnt/seed17/001688/cchen/kimi-k25/tmp"
    echo "Using default OUTPUT_DIR: ${OUTPUT_BASE_DIR}"
fi

# Create output directory based on job ID (directly under output_base_dir)
JOB_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${JOB_ID}"

# ============================================================
# 2. Improved TensorBoard log directory structure
# ============================================================
# TensorBoard logs stored in ${OUTPUT_BASE_DIR}/tf_logs/${JOB_ID}
TENSORBOARD_DIR="${OUTPUT_BASE_DIR}/tf_logs/${JOB_ID}"

# ============================================================
# 3. Unified log directory structure
# ============================================================
# Main log directory
LOG_BASE_DIR="${JOB_OUTPUT_DIR}/logs"

# Various log subdirectory types
LOG_TYPES=(
    "training"      # Training process logs
    "monitor"       # Monitoring logs
    "system"        # System logs
    "distributed"   # Distributed training logs
    "pids"          # PID files
)

# Checkpoint directory configuration
# Use CHECKPOINT_DIR environment variable if set, otherwise use default location
if [ -n "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_SAVE_DIR="${CHECKPOINT_DIR}"
    CHECKPOINT_LOAD_DIR="${CHECKPOINT_DIR}"
    echo "Using custom checkpoint directory from environment variable: ${CHECKPOINT_DIR}"
else
    CHECKPOINT_SAVE_DIR="${JOB_OUTPUT_DIR}/checkpoints"
    CHECKPOINT_LOAD_DIR="${JOB_OUTPUT_DIR}/checkpoints"
    echo "Using default checkpoint directory: ${CHECKPOINT_SAVE_DIR}"
fi

# Optional: Separate load and save directories
if [ -n "${CHECKPOINT_LOAD_DIR_ENV}" ]; then
    CHECKPOINT_LOAD_DIR="${CHECKPOINT_LOAD_DIR_ENV}/${JOB_ID}"
    echo "Using separate checkpoint load directory: ${CHECKPOINT_LOAD_DIR}"
fi

if [ -n "${CHECKPOINT_SAVE_DIR_ENV}" ]; then
    CHECKPOINT_SAVE_DIR="${CHECKPOINT_SAVE_DIR_ENV}/${JOB_ID}"
    echo "Using separate checkpoint save directory: ${CHECKPOINT_SAVE_DIR}"
fi

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
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NODE_ADDR: $NODE_ADDR"
echo "=========================================="

# Create directory structure
# Only rank 0 creates the main structure, other nodes create necessary subdirectories
if [ ${NODE_RANK} -eq 0 ]; then
    echo "Master node creating output directory structure..."

    # Create base directories
    mkdir -p ${OUTPUT_BASE_DIR}
    mkdir -p ${JOB_OUTPUT_DIR}

    # Create all log subdirectories
    for log_type in "${LOG_TYPES[@]}"; do
        mkdir -p "${LOG_BASE_DIR}/${log_type}"
        echo "Created log directory: ${LOG_BASE_DIR}/${log_type}"
    done

    # Create other necessary directories
    mkdir -p ${CHECKPOINT_SAVE_DIR}
    mkdir -p ${TENSORBOARD_DIR}
    mkdir -p ${JOB_OUTPUT_DIR}/wandb
    mkdir -p ${JOB_OUTPUT_DIR}/configs

    # Save environment configuration and metadata
    env > ${JOB_OUTPUT_DIR}/configs/env_config.txt
    cat > ${JOB_OUTPUT_DIR}/configs/job_metadata.json << EOF
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
    echo "Job metadata saved to: ${JOB_OUTPUT_DIR}/configs/job_metadata.json"
else
    echo "Worker node (rank ${NODE_RANK}) creating necessary directories..."
    # Worker nodes only need to create directories they will use
    mkdir -p "${LOG_BASE_DIR}/distributed"
    mkdir -p "${LOG_BASE_DIR}/pids"
    mkdir -p "${JOB_OUTPUT_DIR}/configs"

    # Save node-specific metadata
    cat > ${JOB_OUTPUT_DIR}/configs/node_${NODE_RANK}_metadata.json << EOF
{
    "node_rank": ${NODE_RANK},
    "node_addr": "${NODE_ADDR}",
    "job_id": "${JOB_ID}",
    "timestamp": "${TIMESTAMP}"
}
EOF
fi

# Ensure checkpoint directories exist
if [ "${CHECKPOINT_SAVE_DIR}" != "${JOB_OUTPUT_DIR}/checkpoints" ]; then
    echo "Creating checkpoint save directory: ${CHECKPOINT_SAVE_DIR}"
    mkdir -p ${CHECKPOINT_SAVE_DIR}
fi

if [ "${CHECKPOINT_LOAD_DIR}" != "${JOB_OUTPUT_DIR}/checkpoints" ] && [ "${CHECKPOINT_LOAD_DIR}" != "${CHECKPOINT_SAVE_DIR}" ]; then
    echo "Creating checkpoint load directory: ${CHECKPOINT_LOAD_DIR}"
    mkdir -p ${CHECKPOINT_LOAD_DIR}
fi

# cd /mnt/seed17/001688/haoran.huang/megatron-lm-musa-patch/examples/qwen3vl

export PYTHONPATH=/mnt/seed17/001688/cchen/kimi-k25/megatron-lm-musa-patch:/mnt/seed17/001688/cchen/kimi-k25/Megatron-LM:${PYTHONPATH}

# Create node-specific training log path
TRAINING_LOG_DIR="${LOG_BASE_DIR}/training"
mkdir -p ${TRAINING_LOG_DIR}

# Training log file naming
TRAINING_LOG_FILE="${TRAINING_LOG_DIR}/rank_${NODE_RANK}_${NODE_ADDR}_${TIMESTAMP}.log"
DETAILED_LOG_DIR="${LOG_BASE_DIR}/distributed/rank_${NODE_RANK}_${NODE_ADDR}"
mkdir -p ${DETAILED_LOG_DIR}

# Construct torchrun command
#    --enable-variable-seq-lengths \    --fp8-format hybrid \

cmd="LOGLEVEL=INFO MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_DEVICE_MAX_CONNECTIONS=1 MUSA_EXECUTION_TIMEOUT=600000 ACCELERATOR_BACKEND=musa MCCL_PROTOS=2 MCCL_CHECK_POINTERS=0 OMP_NUM_THREADS=4 MCCL_ALGOS=1 MCCL_BUFFSIZE=20971520 MUSA_BLOCK_SCHEDULE_MODE=1 MUSA_BLOCK_DISTRIBUTION_GRANULARITY=0 MCCL_CROSS_NIC=0 MCCL_IB_TC=136 MCCL_IB_TIMEOUT=20 MCCL_IB_RETRY_CNT=7 MCCL_IB_GID_INDEX=3 MCCL_IB_HCA= MCCL_NET_SHARED_BUFFERS=0 LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/openmpi/lib:/usr/local/musa/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/openmpi/lib:/usr/local/musa/lib torchrun \
    --nnodes ${WORLD_SIZE} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --log_dir ${DETAILED_LOG_DIR} \
    --redirects 0 \
    --tee 3 \
    ${SCRIPT_DIR}/train_qwen3_vl.py \
    --vision-ration 0.1 \
    --num-workers 1 \
    --calculate-per-token-loss \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --context-parallel-size 1 \
    --use-flash-attn \
    --use-distributed-optimizer \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --use-te \
    --bf16 \
    --attention-softmax-in-fp32 \
    --log-interval 1 \
    --tensorboard-log-interval 1 \
    --log-throughput \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --wandb-save-dir ${JOB_OUTPUT_DIR}/wandb \
    --save-interval ${SAVE_INTERVAL:-2000} \
    --pretrained-checkpoint /mnt/seed17/001688/haoran.huang/Qwen3-VL-8B-Instruct-tp1-pp2 \
    --dataloader-save ${CHECKPOINT_SAVE_DIR}/dataloader \
    --ckpt-format torch \
    --save ${CHECKPOINT_SAVE_DIR} \
    --load ${CHECKPOINT_LOAD_DIR} \
    --kv-channels 128 \
    --qk-layernorm \
    --attention-backend flash \
    --disable-bias-linear \
    --num-layers 36 \
    --decoder-first-pipeline-num-layers 16 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --num-query-groups 8 \
    --seq-length ${SEQ_LEN:-4096} \
    --max-padding-length ${MAX_PADDING_LENGTH:-4096} \
    --max-position-embeddings 4096 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --train-iters ${TRAIN_ITERS:-6000} \
    --exit-interval ${EXIT_INTERVAL:-6000} \
    --eval-iters 0 \
    --micro-batch-size 1 \
    --global-batch-size ${GBS:-128} \
    --group-query-attention \
    --no-masked-softmax-fusion \
    --untie-embeddings-and-output-weights \
    --position-embedding-type mrope \
    --rotary-percent 1.0 \
    --rotary-base 5000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --mrope-section 24 20 20 \
    --patch-size 16 \
    --disable-vision-class-token \
    --seed 42 \
    --tp-only-amax-red  \
    --manual-gc \
    --manual-gc-interval 200 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --lr ${LR:-1e-05} \
    --min-lr 1e-06 \
    --lr-warmup-fraction 0.03 \
    --lr-decay-style cosine \
    --no-use-system-prompt \
    --data-path ${DATA_PATH:-/mnt/seed17/001688/haoran.huang/OneThinker/wds-1} \
    --vision-root /mnt/seed17/001688/haoran.huang/OneThinker \
    --dataloader-type external \
    --split 100,0,0 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /mnt/seed17/001688/haoran.huang/Qwen3-VL-8B-Instruct-tp2-pp2 \
    --extra-vocab-size 293 \
    --rerun-mode disabled" # don't save checkpoint when NaN

echo "Training command constructed successfully, preparing to launch..."

eval $cmd 2>&1 | tee ${TRAINING_LOG_FILE}
