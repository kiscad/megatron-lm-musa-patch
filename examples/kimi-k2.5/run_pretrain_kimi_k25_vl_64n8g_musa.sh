#!/bin/bash
set -euo pipefail

# Kimi-K2.5-VL 64-node pretraining entry.
#
# Usage:
#   WORK_HOME=/path/to/work HOSTFILE=/path/to/hostfile DATA_DIR=/path/to/jsonl_dir \
#     bash run_pretrain_kimi_k25_vl_64n8g_musa.sh
#
# Positional fallback:
#   bash run_pretrain_kimi_k25_vl_64n8g_musa.sh WORK_HOME HOSTFILE DATA_DIR [EXPNAME] [extra Megatron args...]
# Env-style extra Megatron args:
#   WORK_HOME=... HOSTFILE=... TRAIN_DATA=... bash run_pretrain_kimi_k25_vl_64n8g_musa.sh -- --load /ckpt
#
# DATA_DIR defaults to train.jsonl / valid.jsonl / test.jsonl. TRAIN_DATA,
# VALID_DATA, and TEST_DATA can be set explicitly.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_HOME=${PATCH_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
MEGATRON_PATH=${MEGATRON_PATH:-${PATCH_HOME}/../Megatron-LM}
MODEL_CONFIG_DIR=${MODEL_CONFIG_DIR:-${PATCH_HOME}/../kimi-k25-vl}

WORK_HOME=${WORK_HOME:-}
HOSTFILE=${HOSTFILE:-}
DATA_DIR=${DATA_DIR:-}
EXPNAME=${EXPNAME:-}
EXTRA_ARGS=()

if (($# > 0)); then
    if [[ "$1" == "--" ]]; then
        EXTRA_ARGS=("${@:2}")
    else
        WORK_HOME=${WORK_HOME:-$1}
        HOSTFILE=${HOSTFILE:-${2:-}}
        DATA_DIR=${DATA_DIR:-${3:-}}
        EXPNAME=${EXPNAME:-${4:-}}
        if (($# > 4)); then
            EXTRA_ARGS=("${@:5}")
        fi
    fi
fi

EXPNAME=${EXPNAME:-kimi_k25_vl_64n8g}

if [[ -z "${WORK_HOME}" || -z "${HOSTFILE}" ]]; then
    echo "Usage: WORK_HOME=/path HOSTFILE=/path DATA_DIR=/path bash $0"
    echo "   or: bash $0 WORK_HOME HOSTFILE DATA_DIR [EXPNAME]"
    exit 1
fi

if [[ ! -f "${HOSTFILE}" ]]; then
    echo "HOSTFILE does not exist: ${HOSTFILE}"
    exit 1
fi

if [[ ! -d "${MODEL_CONFIG_DIR}" ]]; then
    echo "MODEL_CONFIG_DIR does not exist: ${MODEL_CONFIG_DIR}"
    exit 1
fi

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

export ENABLE_PROFILER=${ENABLE_PROFILER:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MUSA_KERNEL_TIMEOUT=${MUSA_KERNEL_TIMEOUT:-3200000}
export ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND:-musa}
export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_CHECK_POINTERS=${MCCL_CHECK_POINTERS:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-3}
export MUSA_BLOCK_SCHEDULE_MODE=${MUSA_BLOCK_SCHEDULE_MODE:-1}
export MCCL_ALGOS=${MCCL_ALGOS:-1}
export MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-20480000}
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:${PYTHONPATH:-}

if [[ ! -d "${MEGATRON_PATH}/build" ]]; then
    pushd "${MEGATRON_PATH}" >/dev/null
    python setup.py build_ext --inplace
    popd >/dev/null
fi

EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-$(wc -l < "${HOSTFILE}")}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

if [[ "${NUM_NODES}" -ne "${EXPECTED_NUM_NODES}" ]]; then
    echo "Warning: HOSTFILE has ${NUM_NODES} nodes; expected ${EXPECTED_NUM_NODES} for the default 64-node profile."
fi

NODE_ADDR=${NODE_ADDR:-$(hostname -I 2>/dev/null | awk '{print $1}')}
if [[ -z "${NODE_ADDR}" ]]; then
    NODE_ADDR=$(hostname)
fi
MASTER_ADDR=${MASTER_ADDR:-$(head -n1 "${HOSTFILE}" | awk '{print $1}')}
NODE_RANK=${NODE_RANK:-$(awk -v node_addr="${NODE_ADDR}" '$1 == node_addr {print FNR - 1; found = 1} END {if (!found) exit 1}' "${HOSTFILE}" || true)}
if [[ -z "${NODE_RANK}" ]]; then
    NODE_RANK=$(awk -v node_name="$(hostname)" '$1 == node_name {print FNR - 1; found = 1} END {if (!found) exit 1}' "${HOSTFILE}" || true)
fi
if [[ -z "${NODE_RANK}" ]]; then
    echo "Failed to infer NODE_RANK from HOSTFILE=${HOSTFILE}; set NODE_RANK explicitly."
    exit 1
fi
export NODE_ADDR MASTER_ADDR NODE_RANK

MASTER_PORT=${MASTER_PORT:-12356}
RDZV_ID=${RDZV_ID:-${EXPNAME}}

TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-16}
EP_SIZE=${EP_SIZE:-4}
ETP_SIZE=${ETP_SIZE:-${TP_SIZE}}
CP_SIZE=${CP_SIZE:-1}

if (( WORLD_SIZE % (TP_SIZE * PP_SIZE * CP_SIZE) != 0 )); then
    echo "WORLD_SIZE=${WORLD_SIZE} must be divisible by TP*PP*CP=$((TP_SIZE * PP_SIZE * CP_SIZE))."
    exit 1
fi
if (( WORLD_SIZE % (ETP_SIZE * EP_SIZE * PP_SIZE) != 0 )); then
    echo "WORLD_SIZE=${WORLD_SIZE} must be divisible by ETP*EP*PP=$((ETP_SIZE * EP_SIZE * PP_SIZE))."
    exit 1
fi

NUM_LAYERS=${NUM_LAYERS:-${CFG_NUM_LAYERS}}
FIRST_K_DENSE_REPLACE=${FIRST_K_DENSE_REPLACE:-${CFG_FIRST_K_DENSE_REPLACE}}
MOE_LAYER_FREQ=${MOE_LAYER_FREQ:-"([0]*${FIRST_K_DENSE_REPLACE}+[1]*$((NUM_LAYERS - FIRST_K_DENSE_REPLACE)))"}
DECODER_FIRST_PIPELINE_NUM_LAYERS=${DECODER_FIRST_PIPELINE_NUM_LAYERS:-${FIRST_K_DENSE_REPLACE}}

HIDDEN_SIZE=${HIDDEN_SIZE:-${CFG_HIDDEN_SIZE}}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-${CFG_NUM_ATTENTION_HEADS}}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-${CFG_FFN_HIDDEN_SIZE}}
VOCAB_SIZE=${VOCAB_SIZE:-${CFG_VOCAB_SIZE}}
NORM_EPSILON=${NORM_EPSILON:-${CFG_NORM_EPSILON}}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-${CFG_ATTENTION_DROPOUT}}
HIDDEN_DROPOUT=${HIDDEN_DROPOUT:-0.0}

NUM_EXPERTS=${NUM_EXPERTS:-${CFG_NUM_EXPERTS}}
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-${CFG_MOE_FFN_HIDDEN_SIZE}}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-${CFG_MOE_TOPK}}
MOE_ROUTER_NUM_GROUPS=${MOE_ROUTER_NUM_GROUPS:-${CFG_MOE_ROUTER_NUM_GROUPS}}
MOE_ROUTER_GROUP_TOPK=${MOE_ROUTER_GROUP_TOPK:-${CFG_MOE_ROUTER_GROUP_TOPK}}
MOE_ROUTER_TOPK_SCALING_FACTOR=${MOE_ROUTER_TOPK_SCALING_FACTOR:-${CFG_MOE_ROUTER_TOPK_SCALING_FACTOR}}
MOE_ROUTER_SCORE_FUNCTION=${MOE_ROUTER_SCORE_FUNCTION:-${CFG_MOE_ROUTER_SCORE_FUNCTION}}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-${CFG_MOE_AUX_LOSS_COEFF}}
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-$((CFG_NUM_SHARED_EXPERTS * CFG_MOE_FFN_HIDDEN_SIZE))}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
MOE_ROUTER_DTYPE=${MOE_ROUTER_DTYPE:-fp32}
MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-1}
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-0}
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
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${CFG_MAX_POSITION_EMBEDDINGS}}

IMG_H=${IMG_H:-${CFG_IMG_H}}
IMG_W=${IMG_W:-${CFG_IMG_W}}
PATCH_DIM=${PATCH_DIM:-${CFG_PATCH_DIM}}
KIMI_VISION_NUM_LAYERS=${KIMI_VISION_NUM_LAYERS:-${CFG_KIMI_VISION_NUM_LAYERS}}
KIMI_VISION_HIDDEN_SIZE=${KIMI_VISION_HIDDEN_SIZE:-${CFG_KIMI_VISION_HIDDEN_SIZE}}
KIMI_VISION_FFN_HIDDEN_SIZE=${KIMI_VISION_FFN_HIDDEN_SIZE:-${CFG_KIMI_VISION_FFN_HIDDEN_SIZE}}
KIMI_VISION_NUM_ATTENTION_HEADS=${KIMI_VISION_NUM_ATTENTION_HEADS:-${CFG_KIMI_VISION_NUM_ATTENTION_HEADS}}
KIMI_VISION_MERGE_KERNEL_SIZE=${KIMI_VISION_MERGE_KERNEL_SIZE:-${CFG_KIMI_VISION_MERGE_KERNEL_SIZE}}
KIMI_VISION_INIT_POS_EMB_HEIGHT=${KIMI_VISION_INIT_POS_EMB_HEIGHT:-${CFG_KIMI_VISION_INIT_POS_EMB_HEIGHT}}
KIMI_VISION_INIT_POS_EMB_WIDTH=${KIMI_VISION_INIT_POS_EMB_WIDTH:-${CFG_KIMI_VISION_INIT_POS_EMB_WIDTH}}
KIMI_VISION_INIT_POS_EMB_TIME=${KIMI_VISION_INIT_POS_EMB_TIME:-${CFG_KIMI_VISION_INIT_POS_EMB_TIME}}
KIMI_MEDIA_PLACEHOLDER_TOKEN=${KIMI_MEDIA_PLACEHOLDER_TOKEN:-${CFG_KIMI_MEDIA_PLACEHOLDER_TOKEN}}

SEQ_LENGTH=${SEQ_LENGTH:-${CFG_ORIGINAL_MAX_POSITION_EMBEDDINGS}}
NUM_IMAGE_TOKENS=$((((IMG_H / PATCH_DIM) / KIMI_VISION_MERGE_KERNEL_SIZE) * (((IMG_W / PATCH_DIM) / KIMI_VISION_MERGE_KERNEL_SIZE))))
MIN_DECODER_SEQ_LENGTH=$((SEQ_LENGTH - 1 + NUM_IMAGE_TOKENS))
DECODER_SEQ_LENGTH=${DECODER_SEQ_LENGTH:-$((((MIN_DECODER_SEQ_LENGTH + 127) / 128) * 128))}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
TRAIN_ITERS=${TRAIN_ITERS:-1000000}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-}
SEED=${SEED:-42}
INIT_METHOD_STD=${INIT_METHOD_STD:-${CFG_INIT_METHOD_STD}}

LR=${LR:-2.0e-4}
MIN_LR=${MIN_LR:-2.0e-5}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2000}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}

SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
EVAL_ITERS=${EVAL_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-1}
NUM_WORKERS=${NUM_WORKERS:-2}

TOKENIZER_MODEL=${TOKENIZER_MODEL:-${MODEL_CONFIG_DIR}}
TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR:+${DATA_DIR}/train.jsonl}}
VALID_DATA=${VALID_DATA:-${DATA_DIR:+${DATA_DIR}/valid.jsonl}}
TEST_DATA=${TEST_DATA:-${DATA_DIR:+${DATA_DIR}/test.jsonl}}

if [[ -z "${TRAIN_DATA}" || ! -f "${TRAIN_DATA}" ]]; then
    echo "TRAIN_DATA is required and must point to a Kimi-K2.5-VL JSON/JSONL multimodal dataset."
    echo "Set TRAIN_DATA explicitly or set DATA_DIR containing train.jsonl."
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

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --log_dir "${TORCHRUN_LOG_PATH}"
    --redirects 3
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size "${ETP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --sequence-parallel
    --decoder-first-pipeline-num-layers "${DECODER_FIRST_PIPELINE_NUM_LAYERS}"
)

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
)

if [[ "${CFG_MOE_ROUTER_NORM_TOPK_PROB}" == "1" ]]; then
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
    --use-flash-attn
    --enable-experimental
    --distributed-backend nccl
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
)

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
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --min-lr "${MIN_LR}"
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
    --log-throughput
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --save "${CHECKPOINT_PATH}"
    --load "${LOAD_PATH}"
    --tensorboard-dir "${TB_PATH}"
)

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

printf '%q ' "${CMD[@]}" | tee "${LOG_PATH}/pretrain_kimi_k25_vl_64n8g.cmd"
printf '\n' | tee -a "${LOG_PATH}/pretrain_kimi_k25_vl_64n8g.cmd"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}/pretrain_kimi_k25_vl_64n8g.log"
