#!/usr/bin/env bash
set -euo pipefail

# Convert Qwen3-VL-32B HuggingFace weights to Megatron-LM release checkpoints
# with tensor parallelism 4 and pipeline parallelism 8.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)
PATCH_HOME=${PATCH_HOME:-${PROJECT_ROOT}/megatron-lm-musa-patch}
MEGATRON_PATH=${MEGATRON_PATH:-${PROJECT_ROOT}/Megatron-LM}

HF_CKPT_PATH=${1:-${PROJECT_ROOT}/model/Qwen3vl-32b}
MCORE_CKPT_PATH=${2:-${PROJECT_ROOT}/tmp/qwen3vl-32b-mcore-tp4-pp8}

export PYTHONPATH="${PATCH_HOME}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
export ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND:-musa}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

if [[ ! -f "${HF_CKPT_PATH}/config.json" ]]; then
  echo "HF checkpoint config not found: ${HF_CKPT_PATH}/config.json" >&2
  exit 1
fi

mkdir -p "${MCORE_CKPT_PATH}"
find -L "${HF_CKPT_PATH}" -maxdepth 1 -type f \( -name "*.json" -o -name "merges.txt" -o -name "*.model" \) -exec cp {} "${MCORE_CKPT_PATH}/" \;

torchrun \
  --nproc_per_node 1 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr "${MASTER_ADDR:-localhost}" \
  --master_port "${MASTER_PORT:-29531}" \
  "${SCRIPT_DIR}/hf2mcore_qwen3_vl.py" \
  --load "${HF_CKPT_PATH}" \
  --save "${MCORE_CKPT_PATH}" \
  --pretrained-checkpoint "${HF_CKPT_PATH}" \
  --target-tensor-model-parallel-size 4 \
  --target-pipeline-model-parallel-size 8 \
  --use-cpu-initialization \
  --micro-batch-size 1 \
  --save-interval 1 \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --bf16 \
  --qk-layernorm \
  --swiglu \
  --disable-bias-linear \
  --no-bias-swiglu-fusion \
  --no-rope-fusion \
  --position-embedding-type mrope \
  --mrope-section 24 20 20 \
  --rotary-percent 1.0 \
  --rotary-base 5000000 \
  --rotary-seq-len-interpolation-factor 1 \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --num-layers 64 \
  --hidden-size 5120 \
  --ffn-hidden-size 25600 \
  --num-attention-heads 64 \
  --group-query-attention \
  --num-query-groups 8 \
  --kv-channels 128 \
  --seq-length 1 \
  --max-position-embeddings 262144 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${HF_CKPT_PATH}" \
  --vocab-size 151936 \
  --extra-vocab-size 293 \
  --untie-embeddings-and-output-weights \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --spatial-merge-size 2 \
  --patch-size 16 \
  --disable-vision-class-token \
  --no-async-tensor-model-parallel-allreduce

echo "Megatron checkpoint saved to: ${MCORE_CKPT_PATH}"
