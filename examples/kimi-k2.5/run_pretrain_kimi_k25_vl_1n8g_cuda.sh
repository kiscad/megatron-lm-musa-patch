#!/bin/bash
set -euo pipefail

# Kimi-K2.5-VL pretraining entry for CUDA/NCCL.
#
# Usage:
#   WORK_HOME=/path/to/work DATA_DIR=/path/to/jsonl_dir \
#     bash run_pretrain_kimi_k25_vl_1n8g_cuda.sh
#
# Positional fallback:
#   bash run_pretrain_kimi_k25_vl_1n8g_cuda.sh [WORK_HOME] [DATA_DIR] [EXPNAME] [extra Megatron args...]
# Env-style extra Megatron args:
#   WORK_HOME=... TRAIN_DATA=... bash run_pretrain_kimi_k25_vl_1n8g_cuda.sh -- --load /ckpt
#
# DATA_DIR defaults to the local Flickr30k debug set and expects
# train.simple.jsonl / valid.simple.jsonl / test.simple.jsonl. TRAIN_DATA,
# VALID_DATA, and TEST_DATA can override those paths explicitly.
#
# This script keeps tokenizer/vocab/RoPE/media-token defaults tied to
# kimi-k25-vl JSON files, but scales model depth and width down for one 8-GPU
# node. Override the environment variables below to scale it back up or out.

# -----------------------------------------------------------------------------
# Script locations and user inputs
# -----------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_HOME=${PATCH_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
MEGATRON_PATH=${MEGATRON_PATH:-${PATCH_HOME}/../Megatron-LM}

DEFAULT_MODEL_CONFIG_DIR="/data02/sam.chen/kimi-k25/Kimi-K2.5-tokenizer"
DEFAULT_WORK_HOME="/data02/sam.chen/kimi-k25/tmp"
DEFAULT_DATA_DIR="/data02/sam.chen/kimi-k25/data/flickr30k_kimi_full"
DEFAULT_EXPNAME="kimi_k25_vl_1n8g_cuda"

MODEL_CONFIG_DIR=${MODEL_CONFIG_DIR:-${DEFAULT_MODEL_CONFIG_DIR}}
WORK_HOME=${WORK_HOME:-}
DATA_DIR=${DATA_DIR:-}
EXPNAME=${EXPNAME:-}
EXTRA_ARGS=()

if (($# > 0)); then
    if [[ "$1" == "--" ]]; then
        EXTRA_ARGS=("${@:2}")
    else
        WORK_HOME=${WORK_HOME:-$1}
        DATA_DIR=${DATA_DIR:-${2:-}}
        EXPNAME=${EXPNAME:-${3:-}}
        if (($# > 3)); then
            EXTRA_ARGS=("${@:4}")
        fi
    fi
fi

EXPNAME=${EXPNAME:-${DEFAULT_EXPNAME}}
WORK_HOME=${WORK_HOME:-${DEFAULT_WORK_HOME}}
DATA_DIR=${DATA_DIR:-${DEFAULT_DATA_DIR}}

if [[ -z "${WORK_HOME}" || -z "${DATA_DIR}" ]]; then
    echo "Usage: WORK_HOME=/path DATA_DIR=/path bash $0"
    echo "   or: bash $0 [WORK_HOME] [DATA_DIR] [EXPNAME]"
    exit 1
fi

if [[ ! -d "${MODEL_CONFIG_DIR}" ]]; then
    echo "MODEL_CONFIG_DIR does not exist: ${MODEL_CONFIG_DIR}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Kimi config defaults extracted from tokenizer/config files
# -----------------------------------------------------------------------------
eval "$(
python - "${MODEL_CONFIG_DIR}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg_dir = Path(sys.argv[1])
with open(cfg_dir / "config.json", encoding="utf-8") as f:
    cfg = json.load(f)
with open(cfg_dir / "tokenizer_config.json", encoding="utf-8") as f:
    tok = json.load(f)
with open(cfg_dir / "preprocessor_config.json", encoding="utf-8") as f:
    pre = json.load(f)

text = cfg["text_config"]
vision = cfg["vision_config"]
media_cfg = pre.get("media_proc_cfg", {})
rope = text.get("rope_scaling") or {}
media_token_id = str(cfg.get("media_placeholder_token_id", text.get("media_placeholder_token_id", 163605)))
media_token = tok.get("added_tokens_decoder", {}).get(media_token_id, {}).get("content", "<|media_pad|>")

def emit(name, value):
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"{name}={shlex.quote(str(value))}")

patch_size = int(vision.get("patch_size", media_cfg.get("patch_size", 14)))
init_pos_emb_height = int(vision.get("init_pos_emb_height", 64))
init_pos_emb_width = int(vision.get("init_pos_emb_width", 64))
merge_kernel_size = vision.get("merge_kernel_size", [media_cfg.get("merge_kernel_size", 2)])
if isinstance(merge_kernel_size, list):
    merge_kernel_size = merge_kernel_size[0]

emit("CFG_VOCAB_SIZE", text["vocab_size"])
emit("CFG_NUM_LAYERS", text["num_hidden_layers"])
emit("CFG_HIDDEN_SIZE", text["hidden_size"])
emit("CFG_NUM_ATTENTION_HEADS", text["num_attention_heads"])
emit("CFG_FFN_HIDDEN_SIZE", text["intermediate_size"])
emit("CFG_NUM_EXPERTS", text["n_routed_experts"])
emit("CFG_NUM_SHARED_EXPERTS", text["n_shared_experts"])
emit("CFG_MOE_FFN_HIDDEN_SIZE", text["moe_intermediate_size"])
emit("CFG_FIRST_K_DENSE_REPLACE", text["first_k_dense_replace"])
emit("CFG_MOE_TOPK", text["num_experts_per_tok"])
emit("CFG_MOE_ROUTER_NUM_GROUPS", text["n_group"])
emit("CFG_MOE_ROUTER_GROUP_TOPK", text["topk_group"])
emit("CFG_MOE_ROUTER_TOPK_SCALING_FACTOR", text["routed_scaling_factor"])
emit("CFG_MOE_ROUTER_SCORE_FUNCTION", text["scoring_func"])
emit("CFG_MOE_ROUTER_NORM_TOPK_PROB", text["norm_topk_prob"])
emit("CFG_MOE_AUX_LOSS_COEFF", text["aux_loss_alpha"])
emit("CFG_TOPK_METHOD", text.get("topk_method", ""))
emit("CFG_Q_LORA_RANK", text["q_lora_rank"])
emit("CFG_KV_LORA_RANK", text["kv_lora_rank"])
emit("CFG_QK_HEAD_DIM", text["qk_nope_head_dim"])
emit("CFG_QK_POS_EMB_HEAD_DIM", text["qk_rope_head_dim"])
emit("CFG_V_HEAD_DIM", text["v_head_dim"])
emit("CFG_ROTARY_BASE", int(text["rope_theta"]))
emit("CFG_ROTARY_SCALING_FACTOR", rope.get("factor", 1.0))
emit("CFG_ORIGINAL_MAX_POSITION_EMBEDDINGS", rope.get("original_max_position_embeddings", text["max_position_embeddings"]))
emit("CFG_MSCALE", rope.get("mscale", 1.0))
emit("CFG_MSCALE_ALL_DIM", rope.get("mscale_all_dim", 1.0))
emit("CFG_MAX_POSITION_EMBEDDINGS", text["max_position_embeddings"])
emit("CFG_INIT_METHOD_STD", text["initializer_range"])
emit("CFG_NORM_EPSILON", text["rms_norm_eps"])
emit("CFG_ATTENTION_DROPOUT", text["attention_dropout"])
emit("CFG_IMG_H", init_pos_emb_height * patch_size)
emit("CFG_IMG_W", init_pos_emb_width * patch_size)
emit("CFG_PATCH_DIM", patch_size)
emit("CFG_KIMI_VISION_NUM_LAYERS", vision["vt_num_hidden_layers"])
emit("CFG_KIMI_VISION_HIDDEN_SIZE", vision["vt_hidden_size"])
emit("CFG_KIMI_VISION_FFN_HIDDEN_SIZE", vision["vt_intermediate_size"])
emit("CFG_KIMI_VISION_NUM_ATTENTION_HEADS", vision["vt_num_attention_heads"])
emit("CFG_KIMI_VISION_MERGE_KERNEL_SIZE", merge_kernel_size)
emit("CFG_KIMI_VISION_INIT_POS_EMB_HEIGHT", init_pos_emb_height)
emit("CFG_KIMI_VISION_INIT_POS_EMB_WIDTH", init_pos_emb_width)
emit("CFG_KIMI_VISION_INIT_POS_EMB_TIME", vision["init_pos_emb_time"])
emit("CFG_KIMI_MEDIA_PLACEHOLDER_TOKEN", media_token)
PY
)"

# -----------------------------------------------------------------------------
# CUDA/NCCL runtime environment
# -----------------------------------------------------------------------------
export ENABLE_PROFILER=${ENABLE_PROFILER:-0}
export PROFILER_FREQ=${PROFILER_FREQ:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND:-cuda}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export USE_RECOMPUTE_VARIANCE=${USE_RECOMPUTE_VARIANCE:-0}
export ENABLE_D2H_IN_PERMUTATION=${ENABLE_D2H_IN_PERMUTATION:-0}
export NO_LOSS_REDUCE=${NO_LOSS_REDUCE:-0}
export USE_MUSA_MOE=${USE_MUSA_MOE:-0}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-0}

export PYTHONPATH="${MEGATRON_PATH}:${PATCH_HOME}${PYTHONPATH:+:${PYTHONPATH}}"

unset MLFLOW_TRACKING_URI
unset MCCL_IB_HCA
unset TORCH_MCCL_AVOID_RECORD_STREAMS
unset PYTORCH_MUSA_ALLOC_CONF

# -----------------------------------------------------------------------------
# Megatron extension build
# -----------------------------------------------------------------------------
if [[ ! -d "${MEGATRON_PATH}/build" ]]; then
    pushd "${MEGATRON_PATH}" >/dev/null
    python setup.py build_ext --inplace
    popd >/dev/null
fi

# -----------------------------------------------------------------------------
# Distributed topology and parallelism
# -----------------------------------------------------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12356}
RDZV_ID=${RDZV_ID:-${EXPNAME}}

if (( NUM_NODES == 1 )); then
    TORCHRUN_STANDALONE=${TORCHRUN_STANDALONE:-1}
else
    TORCHRUN_STANDALONE=${TORCHRUN_STANDALONE:-0}
fi

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-4}
EP_SIZE=${EP_SIZE:-2}
ETP_SIZE=${ETP_SIZE:-${TP_SIZE}}
CP_SIZE=${CP_SIZE:-1}
USE_THD_ATTENTION=${USE_THD_ATTENTION:-0}

if (( TP_SIZE > 1 )); then
    ENABLE_SEQUENCE_PARALLEL=${ENABLE_SEQUENCE_PARALLEL:-1}
else
    ENABLE_SEQUENCE_PARALLEL=${ENABLE_SEQUENCE_PARALLEL:-0}
fi

if (( WORLD_SIZE % (TP_SIZE * PP_SIZE * CP_SIZE) != 0 )); then
    echo "WORLD_SIZE=${WORLD_SIZE} must be divisible by TP*PP*CP=$((TP_SIZE * PP_SIZE * CP_SIZE))."
    exit 1
fi
if (( WORLD_SIZE % (ETP_SIZE * EP_SIZE * PP_SIZE) != 0 )); then
    echo "WORLD_SIZE=${WORLD_SIZE} must be divisible by ETP*EP*PP=$((ETP_SIZE * EP_SIZE * PP_SIZE))."
    exit 1
fi

# -----------------------------------------------------------------------------
# Model architecture defaults
# -----------------------------------------------------------------------------
NUM_LAYERS=${NUM_LAYERS:-4}
FIRST_K_DENSE_REPLACE=${FIRST_K_DENSE_REPLACE:-1}
MOE_LAYER_FREQ=${MOE_LAYER_FREQ:-"([0]*${FIRST_K_DENSE_REPLACE}+[1]*$((NUM_LAYERS - FIRST_K_DENSE_REPLACE)))"}
DECODER_FIRST_PIPELINE_NUM_LAYERS=${DECODER_FIRST_PIPELINE_NUM_LAYERS:-${FIRST_K_DENSE_REPLACE}}

HIDDEN_SIZE=${HIDDEN_SIZE:-${CFG_HIDDEN_SIZE}}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-${CFG_NUM_ATTENTION_HEADS}}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-${CFG_FFN_HIDDEN_SIZE}}
VOCAB_SIZE=${VOCAB_SIZE:-${CFG_VOCAB_SIZE}}
NORM_EPSILON=${NORM_EPSILON:-${CFG_NORM_EPSILON}}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-${CFG_ATTENTION_DROPOUT}}
HIDDEN_DROPOUT=${HIDDEN_DROPOUT:-0.0}

if (( HIDDEN_SIZE % NUM_ATTENTION_HEADS != 0 )); then
    echo "HIDDEN_SIZE=${HIDDEN_SIZE} must be divisible by NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS}."
    exit 1
fi

NUM_EXPERTS=${NUM_EXPERTS:-16}
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-${CFG_MOE_FFN_HIDDEN_SIZE}}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-8}
MOE_ROUTER_NUM_GROUPS=${MOE_ROUTER_NUM_GROUPS:-${CFG_MOE_ROUTER_NUM_GROUPS}}
MOE_ROUTER_GROUP_TOPK=${MOE_ROUTER_GROUP_TOPK:-${CFG_MOE_ROUTER_GROUP_TOPK}}
MOE_ROUTER_TOPK_SCALING_FACTOR=${MOE_ROUTER_TOPK_SCALING_FACTOR:-${CFG_MOE_ROUTER_TOPK_SCALING_FACTOR}}
MOE_ROUTER_SCORE_FUNCTION=${MOE_ROUTER_SCORE_FUNCTION:-${CFG_MOE_ROUTER_SCORE_FUNCTION}}
MOE_ROUTER_NORM_TOPK_PROB=${MOE_ROUTER_NORM_TOPK_PROB:-${CFG_MOE_ROUTER_NORM_TOPK_PROB}}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-${CFG_MOE_AUX_LOSS_COEFF}}
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-${MOE_FFN_HIDDEN_SIZE}}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
MOE_ROUTER_DTYPE=${MOE_ROUTER_DTYPE:-fp32}
MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-0}
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-1}
MOE_ROUTER_ENABLE_EXPERT_BIAS=${MOE_ROUTER_ENABLE_EXPERT_BIAS:-$([[ "${CFG_TOPK_METHOD}" == noaux* ]] && echo 1 || echo 0)}
MOE_ROUTER_BIAS_UPDATE_RATE=${MOE_ROUTER_BIAS_UPDATE_RATE:-1e-3}

if (( NUM_EXPERTS % EP_SIZE != 0 )); then
    echo "NUM_EXPERTS=${NUM_EXPERTS} must be divisible by EP_SIZE=${EP_SIZE}."
    exit 1
fi

Q_LORA_RANK=${Q_LORA_RANK:-${CFG_Q_LORA_RANK}}
KV_LORA_RANK=${KV_LORA_RANK:-${CFG_KV_LORA_RANK}}
QK_HEAD_DIM=${QK_HEAD_DIM:-${CFG_QK_HEAD_DIM}}
QK_POS_EMB_HEAD_DIM=${QK_POS_EMB_HEAD_DIM:-${CFG_QK_POS_EMB_HEAD_DIM}}
V_HEAD_DIM=${V_HEAD_DIM:-${CFG_V_HEAD_DIM}}

ROTARY_BASE=${ROTARY_BASE:-${CFG_ROTARY_BASE}}
ROTARY_SCALING_FACTOR=${ROTARY_SCALING_FACTOR:-${CFG_ROTARY_SCALING_FACTOR}}
MSCALE=${MSCALE:-${CFG_MSCALE}}
MSCALE_ALL_DIM=${MSCALE_ALL_DIM:-${CFG_MSCALE_ALL_DIM}}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-1024}

IMG_H=${IMG_H:-224}
IMG_W=${IMG_W:-224}
PATCH_DIM=${PATCH_DIM:-${CFG_PATCH_DIM}}
KIMI_VISION_NUM_LAYERS=${KIMI_VISION_NUM_LAYERS:-2}
KIMI_VISION_HIDDEN_SIZE=${KIMI_VISION_HIDDEN_SIZE:-${CFG_KIMI_VISION_HIDDEN_SIZE}}
KIMI_VISION_FFN_HIDDEN_SIZE=${KIMI_VISION_FFN_HIDDEN_SIZE:-${CFG_KIMI_VISION_FFN_HIDDEN_SIZE}}
KIMI_VISION_NUM_ATTENTION_HEADS=${KIMI_VISION_NUM_ATTENTION_HEADS:-${CFG_KIMI_VISION_NUM_ATTENTION_HEADS}}
KIMI_VISION_MERGE_KERNEL_SIZE=${KIMI_VISION_MERGE_KERNEL_SIZE:-${CFG_KIMI_VISION_MERGE_KERNEL_SIZE}}
KIMI_VISION_INIT_POS_EMB_HEIGHT=${KIMI_VISION_INIT_POS_EMB_HEIGHT:-${CFG_KIMI_VISION_INIT_POS_EMB_HEIGHT}}
KIMI_VISION_INIT_POS_EMB_WIDTH=${KIMI_VISION_INIT_POS_EMB_WIDTH:-${CFG_KIMI_VISION_INIT_POS_EMB_WIDTH}}
KIMI_VISION_INIT_POS_EMB_TIME=${KIMI_VISION_INIT_POS_EMB_TIME:-${CFG_KIMI_VISION_INIT_POS_EMB_TIME}}
KIMI_MEDIA_PLACEHOLDER_TOKEN=${KIMI_MEDIA_PLACEHOLDER_TOKEN:-${CFG_KIMI_MEDIA_PLACEHOLDER_TOKEN}}

if (( KIMI_VISION_HIDDEN_SIZE % KIMI_VISION_NUM_ATTENTION_HEADS != 0 )); then
    echo "KIMI_VISION_HIDDEN_SIZE=${KIMI_VISION_HIDDEN_SIZE} must be divisible by KIMI_VISION_NUM_ATTENTION_HEADS=${KIMI_VISION_NUM_ATTENTION_HEADS}."
    exit 1
fi
if (( IMG_H % (PATCH_DIM * KIMI_VISION_MERGE_KERNEL_SIZE) != 0 || IMG_W % (PATCH_DIM * KIMI_VISION_MERGE_KERNEL_SIZE) != 0 )); then
    echo "IMG_H/IMG_W must be divisible by PATCH_DIM*KIMI_VISION_MERGE_KERNEL_SIZE."
    exit 1
fi

NUM_IMAGE_TOKENS=$((((IMG_H / PATCH_DIM) / KIMI_VISION_MERGE_KERNEL_SIZE) * (((IMG_W / PATCH_DIM) / KIMI_VISION_MERGE_KERNEL_SIZE))))
DECODER_SEQ_LENGTH=${DECODER_SEQ_LENGTH:-4096}
SEQ_LENGTH=${SEQ_LENGTH:-$((DECODER_SEQ_LENGTH - NUM_IMAGE_TOKENS + 1))}

if (( SEQ_LENGTH <= 0 )); then
    echo "DECODER_SEQ_LENGTH=${DECODER_SEQ_LENGTH} is too small for NUM_IMAGE_TOKENS=${NUM_IMAGE_TOKENS}."
    exit 1
fi
if (( MAX_POSITION_EMBEDDINGS < DECODER_SEQ_LENGTH )); then
    MAX_POSITION_EMBEDDINGS=${DECODER_SEQ_LENGTH}
fi

# -----------------------------------------------------------------------------
# Training schedule and optimizer defaults
# -----------------------------------------------------------------------------
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
TRAIN_ITERS=${TRAIN_ITERS:-2000}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-}
SEED=${SEED:-42}
INIT_METHOD_STD=${INIT_METHOD_STD:-${CFG_INIT_METHOD_STD}}
DETERMINISTIC_MODE=${DETERMINISTIC_MODE:-1}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-0}
DISABLE_ROPE_FUSION=${DISABLE_ROPE_FUSION:-1}

LR=${LR:-2.0e-4}
MIN_LR=${MIN_LR:-2.0e-5}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-10}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}

if [[ -n "${TRAIN_SAMPLES}" ]]; then
    echo "TRAIN_SAMPLES is not supported by this fixed iteration LR schedule. Use TRAIN_ITERS instead."
    exit 1
fi
if [[ "${USE_THD_ATTENTION}" == "1" && "${MICRO_BATCH_SIZE}" != "1" ]]; then
    echo "USE_THD_ATTENTION=1 currently requires MICRO_BATCH_SIZE=1."
    exit 1
fi
if (( LR_WARMUP_ITERS >= LR_DECAY_ITERS )); then
    echo "LR_WARMUP_ITERS=${LR_WARMUP_ITERS} must be < LR_DECAY_ITERS=${LR_DECAY_ITERS}."
    exit 1
fi
if [[ "${DETERMINISTIC_MODE}" == "1" && "${USE_FLASH_ATTN}" == "1" ]]; then
    echo "DETERMINISTIC_MODE=1 is incompatible with USE_FLASH_ATTN=1 because Megatron marks flash attention as non-deterministic."
    exit 1
fi

# Determinism controls for loss-curve alignment runs.
export PYTHONHASHSEED=${PYTHONHASHSEED:-${SEED}}
if [[ "${DETERMINISTIC_MODE}" == "1" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
    export NCCL_ALGO=${NCCL_ALGO:-Ring}
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}
fi

# -----------------------------------------------------------------------------
# Data, tokenizer, logging, and output paths
# -----------------------------------------------------------------------------
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
EVAL_ITERS=${EVAL_ITERS:-0}
LOG_INTERVAL=${LOG_INTERVAL:-1}
LOGGING_LEVEL=${LOGGING_LEVEL:-30}
NUM_WORKERS=${NUM_WORKERS:-2}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-120}
DISABLE_CHECKPOINT_SAVE=${DISABLE_CHECKPOINT_SAVE:-0}

TOKENIZER_MODEL=${TOKENIZER_MODEL:-${MODEL_CONFIG_DIR}}
TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR:+${DATA_DIR}/train.simple.jsonl}}
VALID_DATA=${VALID_DATA:-${DATA_DIR:+${DATA_DIR}/valid.simple.jsonl}}
TEST_DATA=${TEST_DATA:-${DATA_DIR:+${DATA_DIR}/test.simple.jsonl}}

if [[ -z "${TRAIN_DATA}" || ! -f "${TRAIN_DATA}" ]]; then
    echo "TRAIN_DATA is required and must point to a Kimi-K2.5-VL JSON/JSONL multimodal dataset."
    echo "Set TRAIN_DATA explicitly or set DATA_DIR containing train.simple.jsonl."
    exit 1
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-${WORK_HOME}/checkpoints/${EXPNAME}}
LOAD_PATH=${LOAD_PATH:-${CHECKPOINT_PATH}}
LOG_PATH=${LOG_PATH:-${WORK_HOME}/logs/${EXPNAME}}
TB_PATH=${TB_PATH:-${WORK_HOME}/tboard/${EXPNAME}}
WB_PATH=${WB_PATH:-${WORK_HOME}/wandb/${EXPNAME}}
TORCHRUN_LOG_PATH=${TORCHRUN_LOG_PATH:-${WORK_HOME}/output_log/${RDZV_ID}/${EXPNAME}}

mkdir -p "${CHECKPOINT_PATH}" "${LOG_PATH}" "${TB_PATH}" "${WB_PATH}" "${TORCHRUN_LOG_PATH}"
cp "$0" "${LOG_PATH}/"

# -----------------------------------------------------------------------------
# Megatron argument groups
# -----------------------------------------------------------------------------
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
)

if [[ "${TORCHRUN_STANDALONE}" == "1" ]]; then
    if (( NUM_NODES != 1 )); then
        echo "TORCHRUN_STANDALONE=1 requires NUM_NODES=1."
        exit 1
    fi
    DISTRIBUTED_ARGS+=(--standalone)
else
    DISTRIBUTED_ARGS+=(
        --nnodes "${NUM_NODES}"
        --node_rank "${NODE_RANK}"
        --master_addr "${MASTER_ADDR}"
        --master_port "${MASTER_PORT}"
    )
fi

DISTRIBUTED_ARGS+=(
    --log_dir "${TORCHRUN_LOG_PATH}"
    --redirects 3
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size "${ETP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
)

if (( PP_SIZE > 1 )); then
    MODEL_PARALLEL_ARGS+=(--decoder-first-pipeline-num-layers "${DECODER_FIRST_PIPELINE_NUM_LAYERS}")
fi

if [[ "${ENABLE_SEQUENCE_PARALLEL}" == "1" ]]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi

TEXT_MODEL_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --vocab-size "${VOCAB_SIZE}"
    --seq-length "${SEQ_LENGTH}"
    --decoder-seq-length "${DECODER_SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --norm-epsilon "${NORM_EPSILON}"
    --attention-dropout "${ATTENTION_DROPOUT}"
    --hidden-dropout "${HIDDEN_DROPOUT}"
    --normalization RMSNorm
    --position-embedding-type rope
    --no-position-embedding
    --rotary-base "${ROTARY_BASE}"
    --rotary-scaling-factor "${ROTARY_SCALING_FACTOR}"
    --mscale "${MSCALE}"
    --mscale-all-dim "${MSCALE_ALL_DIM}"
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    --num-experts "${NUM_EXPERTS}"
    --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
    --moe-shared-expert-intermediate-size "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE}"
    --moe-layer-freq "${MOE_LAYER_FREQ}"
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF}"
    --moe-router-topk "${MOE_ROUTER_TOPK}"
    --moe-router-score-function "${MOE_ROUTER_SCORE_FUNCTION}"
    --moe-router-topk-scaling-factor "${MOE_ROUTER_TOPK_SCALING_FACTOR}"
    --moe-router-num-groups "${MOE_ROUTER_NUM_GROUPS}"
    --moe-router-group-topk "${MOE_ROUTER_GROUP_TOPK}"
    --moe-router-dtype "${MOE_ROUTER_DTYPE}"
    --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}"
    # --moe-enable-deepep
)

if [[ "${MOE_ROUTER_NORM_TOPK_PROB}" == "1" ]]; then
    MOE_ARGS+=(--moe-router-norm-topk-prob)
fi
if [[ "${MOE_ROUTER_ENABLE_EXPERT_BIAS}" == "1" ]]; then
    MOE_ARGS+=(--moe-router-enable-expert-bias --moe-router-bias-update-rate "${MOE_ROUTER_BIAS_UPDATE_RATE}")
fi
if [[ "${MOE_GROUPED_GEMM}" == "1" ]]; then
    MOE_ARGS+=(--moe-grouped-gemm)
fi
if [[ "${MOE_PERMUTE_FUSION}" == "1" ]]; then
    MOE_ARGS+=(--moe-permute-fusion)
fi

MLA_ARGS=(
    --multi-latent-attention
    --qk-layernorm
    --q-lora-rank "${Q_LORA_RANK}"
    --kv-lora-rank "${KV_LORA_RANK}"
    --qk-head-dim "${QK_HEAD_DIM}"
    --qk-pos-emb-head-dim "${QK_POS_EMB_HEAD_DIM}"
    --v-head-dim "${V_HEAD_DIM}"
)

VISION_ARGS=(
    --vision-model-type moonvit
    --img-h "${IMG_H}"
    --img-w "${IMG_W}"
    --patch-dim "${PATCH_DIM}"
    --kimi-vision-num-layers "${KIMI_VISION_NUM_LAYERS}"
    --kimi-vision-hidden-size "${KIMI_VISION_HIDDEN_SIZE}"
    --kimi-vision-ffn-hidden-size "${KIMI_VISION_FFN_HIDDEN_SIZE}"
    --kimi-vision-num-attention-heads "${KIMI_VISION_NUM_ATTENTION_HEADS}"
    --kimi-vision-merge-kernel-size "${KIMI_VISION_MERGE_KERNEL_SIZE}"
    --kimi-vision-init-pos-emb-height "${KIMI_VISION_INIT_POS_EMB_HEIGHT}"
    --kimi-vision-init-pos-emb-width "${KIMI_VISION_INIT_POS_EMB_WIDTH}"
    --kimi-vision-init-pos-emb-time "${KIMI_VISION_INIT_POS_EMB_TIME}"
)

TOKENIZER_ARGS=(
    --tokenizer-type MultimodalTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --tokenizer-prompt-format chatml
    --language-model-type kimi_k25
    --special-tokens "${KIMI_MEDIA_PLACEHOLDER_TOKEN}"
    --kimi-media-placeholder-token "${KIMI_MEDIA_PLACEHOLDER_TOKEN}"
)

DATA_ARGS=(
    --simple-mm-train-data "${TRAIN_DATA}"
    --simple-mm-image-key "${SIMPLE_MM_IMAGE_KEY:-image}"
    --simple-mm-text-key "${SIMPLE_MM_TEXT_KEY:-text}"
    --simple-mm-prompt-key "${SIMPLE_MM_PROMPT_KEY:-prompt}"
    --simple-mm-default-prompt "${SIMPLE_MM_DEFAULT_PROMPT:-Describe the image.}"
    --num-workers "${NUM_WORKERS}"
)

if [[ -n "${VALID_DATA}" && -f "${VALID_DATA}" ]]; then
    DATA_ARGS+=(--simple-mm-valid-data "${VALID_DATA}")
fi
if [[ -n "${TEST_DATA}" && -f "${TEST_DATA}" ]]; then
    DATA_ARGS+=(--simple-mm-test-data "${TEST_DATA}")
fi

TRAINING_ARGS=(
    --seed "${SEED}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --init-method-std "${INIT_METHOD_STD}"
    --use-mcore-models
    --use-distributed-optimizer
    --enable-experimental
    --distributed-backend nccl
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}"
)

if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
    TRAINING_ARGS+=(--use-flash-attn)
fi
if [[ "${DETERMINISTIC_MODE}" == "1" ]]; then
    TRAINING_ARGS+=(--deterministic-mode)
fi
if [[ "${DISABLE_ROPE_FUSION}" == "1" ]]; then
    TRAINING_ARGS+=(--no-rope-fusion)
fi

if [[ "${USE_THD_ATTENTION}" == "1" ]]; then
    TRAINING_ARGS+=(--use-thd-attention)
fi

if [[ -n "${TRAIN_SAMPLES}" ]]; then
    TRAINING_ARGS+=(--train-samples "${TRAIN_SAMPLES}")
else
    TRAINING_ARGS+=(--train-iters "${TRAIN_ITERS}")
fi

REGULARIZATION_ARGS=(
    --weight-decay "${WEIGHT_DECAY}"
    --adam-beta1 "${ADAM_BETA1}"
    --adam-beta2 "${ADAM_BETA2}"
    --clip-grad "${CLIP_GRAD}"
)

LEARNING_RATE_ARGS=(
    --lr "${LR}"
    --lr-decay-style "${LR_DECAY_STYLE}"
    --min-lr "${MIN_LR}"
    --lr-decay-iters "${LR_DECAY_ITERS}"
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --initial-loss-scale "${INITIAL_LOSS_SCALE:-65536}"
    --min-loss-scale "${MIN_LOSS_SCALE:-1.0}"
)

MIXED_PRECISION_ARGS=(
    --bf16
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --accumulate-allreduce-grads-in-fp32
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)

if [[ "${ENABLE_FP8:-0}" == "1" ]]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --fp8-format e4m3
        --fp8-param-gather
        --fp8-recipe mxfp8
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval "${LOG_INTERVAL}"
    --logging-level "${LOGGING_LEVEL}"
    --log-throughput
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --load "${LOAD_PATH}"
    --tensorboard-dir "${TB_PATH}"
)

if [[ "${DISABLE_CHECKPOINT_SAVE}" != "1" ]]; then
    EVAL_AND_LOGGING_ARGS+=(--save "${CHECKPOINT_PATH}")
fi

CMD=(
    torchrun
    "${DISTRIBUTED_ARGS[@]}"
    "${SCRIPT_DIR}/pretrain_kimi_k25_vl.py"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${TEXT_MODEL_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${MLA_ARGS[@]}"
    "${VISION_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${EVAL_AND_LOGGING_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

printf '%q ' "${CMD[@]}" | tee "${LOG_PATH}/pretrain_kimi_k25_vl_1n8g_cuda.cmd"
printf '\n' | tee -a "${LOG_PATH}/pretrain_kimi_k25_vl_1n8g_cuda.cmd"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
set +e
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}/pretrain_kimi_k25_vl_1n8g_cuda.log"
status=${PIPESTATUS[0]}
set -e

if (( status != 0 )); then
    {
        echo
        echo "torchrun failed with exit code ${status}."
        echo "Recent rank stderr logs under ${TORCHRUN_LOG_PATH}:"
    } | tee -a "${LOG_PATH}/pretrain_kimi_k25_vl_1n8g_cuda.log"

    mapfile -t rank_stderr_logs < <(
        find "${TORCHRUN_LOG_PATH}" -path '*/attempt_*/*/stderr.log' -type f -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | head -8 \
            | cut -d' ' -f2-
    )

    for rank_stderr_log in "${rank_stderr_logs[@]}"; do
        [[ -s "${rank_stderr_log}" ]] || continue
        {
            echo
            echo "===== tail -120 ${rank_stderr_log}"
            tail -120 "${rank_stderr_log}"
        } | tee -a "${LOG_PATH}/pretrain_kimi_k25_vl_1n8g_cuda.log"
    done
fi

exit "${status}"
