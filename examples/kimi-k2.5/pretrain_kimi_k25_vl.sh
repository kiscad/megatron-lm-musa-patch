#!/bin/bash
set -euo pipefail

# Simple multimodal JSONL examples:
# {"image":"images/sample1.jpg","text":"a red car parked by the road"}
# {"image":"images/sample2.jpg","prompt":"What is shown in the image?","text":"A dog running on grass."}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_HOME=$(cd "${SCRIPT_DIR}/../.." && pwd)
MEGATRON_PATH=${MEGATRON_PATH:-${PATCH_HOME}/../Megatron-LM}

export ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND:-musa}
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Single-node smoke-test defaults. Override these env vars to scale back up.
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
EP_SIZE=${EP_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}

TRAIN_ITERS=${TRAIN_ITERS:-50}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_ITERS=${EVAL_ITERS:-2}

SEQ_LENGTH=${SEQ_LENGTH:-256}
DECODER_SEQ_LENGTH=${DECODER_SEQ_LENGTH:-384}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-512}

IMG_H=${IMG_H:-224}
IMG_W=${IMG_W:-224}
PATCH_DIM=${PATCH_DIM:-14}

NUM_LAYERS=${NUM_LAYERS:-4}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-8}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-1024}

NUM_EXPERTS=${NUM_EXPERTS:-4}
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-256}
MOE_TOPK=${MOE_TOPK:-2}
MOE_LAYER_FREQ=${MOE_LAYER_FREQ:-"([0]+[1]*$((NUM_LAYERS - 1)))"}
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-${MOE_FFN_HIDDEN_SIZE}}
MOE_ROUTER_TOPK_SCALING_FACTOR=${MOE_ROUTER_TOPK_SCALING_FACTOR:-2.827}
MOE_ROUTER_NUM_GROUPS=${MOE_ROUTER_NUM_GROUPS:-1}
MOE_ROUTER_GROUP_TOPK=${MOE_ROUTER_GROUP_TOPK:-1}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}

Q_LORA_RANK=${Q_LORA_RANK:-128}
KV_LORA_RANK=${KV_LORA_RANK:-64}
QK_HEAD_DIM=${QK_HEAD_DIM:-64}
QK_POS_EMB_HEAD_DIM=${QK_POS_EMB_HEAD_DIM:-32}
V_HEAD_DIM=${V_HEAD_DIM:-64}

KIMI_VISION_NUM_LAYERS=${KIMI_VISION_NUM_LAYERS:-4}
KIMI_VISION_HIDDEN_SIZE=${KIMI_VISION_HIDDEN_SIZE:-256}
KIMI_VISION_FFN_HIDDEN_SIZE=${KIMI_VISION_FFN_HIDDEN_SIZE:-512}
KIMI_VISION_NUM_ATTENTION_HEADS=${KIMI_VISION_NUM_ATTENTION_HEADS:-4}
KIMI_VISION_MERGE_KERNEL_SIZE=${KIMI_VISION_MERGE_KERNEL_SIZE:-2}
KIMI_VISION_INIT_POS_EMB_HEIGHT=${KIMI_VISION_INIT_POS_EMB_HEIGHT:-64}
KIMI_VISION_INIT_POS_EMB_WIDTH=${KIMI_VISION_INIT_POS_EMB_WIDTH:-64}
KIMI_VISION_INIT_POS_EMB_TIME=${KIMI_VISION_INIT_POS_EMB_TIME:-4}
KIMI_MEDIA_PLACEHOLDER_TOKEN=${KIMI_MEDIA_PLACEHOLDER_TOKEN:-"<|media_pad|>"}

ROTARY_BASE=${ROTARY_BASE:-50000}
ROTARY_SCALING_FACTOR=${ROTARY_SCALING_FACTOR:-64}
MSCALE=${MSCALE:-1.0}
MSCALE_ALL_DIM=${MSCALE_ALL_DIM:-1.0}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.02}

TRAIN_DATA=${TRAIN_DATA:-"/home/dist/cchen/data/flickr30k_kimi_debug/train.simple.jsonl"}
VALID_DATA=${VALID_DATA:-"/home/dist/cchen/data/flickr30k_kimi_debug/valid.simple.jsonl"}
TEST_DATA=${TEST_DATA:-"/home/dist/cchen/data/flickr30k_kimi_debug/test.simple.jsonl"}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-${PATCH_HOME}/../kimi-k25-vl}
SAVE_DIR=${SAVE_DIR:-${SCRIPT_DIR}/checkpoints/kimi_k25_vl}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${SCRIPT_DIR}/tensorboard/kimi_k25_vl}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}

if [[ -z "${TRAIN_DATA}" ]]; then
    echo "Set TRAIN_DATA to a JSONL/JSON file with image-text samples."
    exit 1
fi

mkdir -p "${SAVE_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}"

DATA_ARGS=(
    --simple-mm-train-data "${TRAIN_DATA}"
    --simple-mm-image-key image
    --simple-mm-text-key text
    --simple-mm-prompt-key prompt
    --simple-mm-default-prompt "Describe the image."
)

if [[ -n "${VALID_DATA}" ]]; then
    DATA_ARGS+=(--simple-mm-valid-data "${VALID_DATA}")
fi
if [[ -n "${TEST_DATA}" ]]; then
    DATA_ARGS+=(--simple-mm-test-data "${TEST_DATA}")
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
)

MODEL_PARALLEL_ARGS=(
    --use-mcore-models
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --sequence-parallel
)

TEXT_MODEL_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --normalization RMSNorm
    --position-embedding-type rope
    --no-position-embedding
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

SEQUENCE_ARGS=(
    --seq-length "${SEQ_LENGTH}"
    --decoder-seq-length "${DECODER_SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
)

ROPE_ARGS=(
    --rotary-base "${ROTARY_BASE}"
    --rotary-scaling-factor "${ROTARY_SCALING_FACTOR}"
    --mscale "${MSCALE}"
    --mscale-all-dim "${MSCALE_ALL_DIM}"
)

MOE_ARGS=(
    --num-experts "${NUM_EXPERTS}"
    --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
    --moe-router-topk "${MOE_TOPK}"
    --moe-layer-freq "${MOE_LAYER_FREQ}"
    --moe-shared-expert-intermediate-size "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE}"
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF}"
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor "${MOE_ROUTER_TOPK_SCALING_FACTOR}"
    --moe-router-num-groups "${MOE_ROUTER_NUM_GROUPS}"
    --moe-router-group-topk "${MOE_ROUTER_GROUP_TOPK}"
    --moe-token-dispatcher-type allgather
)

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

TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --init-method-std "${INIT_METHOD_STD}"
    --use-flash-attn
    --use-distributed-optimizer
    --enable-experimental
)

LEARNING_RATE_ARGS=(
    --lr 2e-4
    --lr-decay-style cosine
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --min-lr 2e-5
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

MIXED_PRECISION_ARGS=(
    --attention-softmax-in-fp32
    --bf16
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)

EVAL_AND_LOGGING_ARGS=(
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --log-interval 10
    --tensorboard-dir "${TENSORBOARD_DIR}"
)

CMD=(
    torchrun
    "${DISTRIBUTED_ARGS[@]}"
    "${SCRIPT_DIR}/pretrain_kimi_k25_vl.py"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${TEXT_MODEL_ARGS[@]}"
    "${SEQUENCE_ARGS[@]}"
    "${ROPE_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${MLA_ARGS[@]}"
    "${VISION_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${EVAL_AND_LOGGING_ARGS[@]}"
    "$@"
)

"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/pretrain_kimi_k25_vl.log"
